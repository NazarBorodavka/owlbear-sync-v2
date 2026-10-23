"""The live processing loop: camera -> calibration rectify -> vision
pipeline -> published results. Deliberately independent of Flask or any
other web framework, so it can be driven directly in a test or a script
without spinning up a server, and so the web layer is a thin consumer of
this rather than the other place the actual tracking logic lives (which
is what made v1's single 3000-line app.py hard to reason about).

Runs its own background thread; callers read the latest results via
`latest()` (a cheap, lock-protected snapshot) rather than being blocked on
the processing loop's own pace.
"""
import threading
import time

import numpy as np

from .vision.calibration.session import CalibrationSession
from .vision.codebook import build_codebook
from .vision.geometry import RingGeometry


def _bgr_to_gray(bgr):
    """cv2.VideoCapture yields BGR frames; weighted by standard luma
    coefficients (same weights used throughout tracker/vision/simulate.py's
    synthetic test renders) rather than a flat channel mean, so real camera
    input and the validated synthetic tests are processed consistently.

    The pipeline's own math (Sobel-gradient voting, a quantile-based
    circle-fit trim, and a z-score-normalised matched-filter correlation
    in decode.py) is scale- and offset-invariant by construction -- every
    stage either divides by its own std or compares relative quantiles,
    never an absolute brightness level -- so this conversion only needs
    to be a REASONABLE grayscale, not calibrated to any particular range.
    """
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    return 0.114 * b + 0.587 * g + 0.299 * r


class TrackerRuntime:
    def __init__(self, calibration_dir, n_ids=24, n_chips=20,
                 target_fps=15.0, rescan_hz=3.0):
        self.geom = RingGeometry(n_chips=n_chips)
        self.geom.validate()
        self.codebook = build_codebook(n_chips, n_ids, seed=1)
        self.session = CalibrationSession(calibration_dir, geom=self.geom,
                                          codebook=self.codebook)
        self.target_fps = target_fps
        self.rescan_hz = rescan_hz

        self.capture = None
        self.pipeline = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_tokens = []
        self._stats = {"fps": 0.0, "frames": 0, "last_frame_time": 0.0,
                       "connected": False, "error": None}

    # ---- lifecycle -------------------------------------------------------
    def prepare_pipeline(self):
        """Builds the vision pipeline from the current calibration without
        touching the camera -- split out from start() so tests (and the
        web layer's calibration-preview tools) can exercise real
        detect/decode/track logic against injected frames without needing
        an RTSP connection at all."""
        if self.session.playarea is None:
            raise RuntimeError(
                "cannot build a pipeline before play-area calibration -- "
                "there is no scale (px_per_mm) to work with")
        with self._lock:
            self.pipeline = self.session.build_pipeline(rescan_hz=self.rescan_hz)
        return self.pipeline

    def start(self, camera_url):
        from .capture import RtspCapture
        self.prepare_pipeline()
        self.capture = RtspCapture(camera_url)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def rebuild_pipeline(self):
        """Call after any calibration stage changes while running, so a
        re-calibration (e.g. re-running detector auto-tune mid-session)
        takes effect without a full restart."""
        with self._lock:
            self.pipeline = self.session.build_pipeline(rescan_hz=self.rescan_hz)

    # ---- main loop ---------------------------------------------------------
    def _loop(self):
        frame_interval = 1.0 / self.target_fps
        last_process = 0.0
        frame_count = 0
        last_report = time.time()
        while self._running:
            got = self.capture.wait_for_frame(timeout=1.0)
            if not got:
                ok, _ = self.capture.read()
                with self._lock:
                    self._stats["connected"] = ok
                continue

            now = time.time()
            if now - last_process < frame_interval:
                continue   # throttle to target_fps even if the camera delivers faster
            last_process = now

            ok, raw = self.capture.read()
            if not ok or raw is None:
                with self._lock:
                    self._stats["connected"] = False
                continue

            ok = self.process_frame(raw, now=now)
            with self._lock:
                self._stats["connected"] = ok

            if not ok:
                continue

            frame_count += 1
            elapsed = now - last_report
            if elapsed >= 10.0:
                with self._lock:
                    self._stats["fps"] = frame_count / elapsed
                    self._stats["frames"] += frame_count
                frame_count = 0
                last_report = now

    def process_frame(self, raw, now=None):
        """Runs one raw camera frame through rectify -> pipeline ->
        publish. Pulled out of _loop() so it can be exercised directly in
        a test by injecting synthetic frames, with no RTSP capture or
        background thread involved -- see tests/test_runtime.py.

        Returns True on success, False (with self._stats["error"] set) on
        any failure, so a caller (the real loop, or a test) can tell a
        processed-but-nothing-found frame apart from a frame that
        actually failed to process.
        """
        now = time.time() if now is None else now
        try:
            gray = raw if raw.ndim == 2 else _bgr_to_gray(raw)
            rectified = self.session.rectify(gray)
            with self._lock:
                pipe = self.pipeline
            reported = pipe.process(rectified, now=now)
            tokens = pipe.tokens_mm(reported)
        except Exception as e:
            with self._lock:
                self._stats["error"] = str(e)
            return False

        with self._lock:
            self._latest_tokens = tokens
            self._stats["error"] = None
            self._stats["last_frame_time"] = now
        return True

    # ---- consumer-facing API -------------------------------------------------
    def latest(self):
        # play_area_mm reflects the CURRENT calibration, not "size of the
        # last processed frame" -- it needs to be available to API
        # consumers (the Owlbear Rodeo extension's auto-zoom, in
        # particular) as soon as play-area calibration exists, not only
        # after the camera has started and produced a first frame.
        play_area_mm = ((self.session.playarea.width_mm, self.session.playarea.height_mm)
                        if self.session.playarea is not None else None)
        with self._lock:
            return {
                "tokens": list(self._latest_tokens),
                "play_area_mm": play_area_mm,
                **self._stats,
            }

    def latest_frame_jpeg(self, quality=80):
        """Encode the most recent rectified frame for the web dashboard's
        live preview. Returns None if nothing has been processed yet."""
        import cv2
        with self._lock:
            pipe = self.pipeline
        if pipe is None or self.capture is None:
            return None
        ok, raw = self.capture.read()
        if not ok:
            return None
        gray = raw if raw.ndim == 2 else _bgr_to_gray(raw)
        try:
            rectified = self.session.rectify(gray)
        except Exception:
            return None
        disp = np.clip(rectified, 0, 255).astype(np.uint8) if rectified.max() > 1.5 \
            else np.clip(rectified * 255, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None
