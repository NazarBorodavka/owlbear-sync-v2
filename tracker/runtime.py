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

# Optional accelerator only -- every use has a numpy fallback, matching
# tracker/vision/detect.py's rule that the core stays importable without
# OpenCV. Here it is the BGR->gray conversion; see _bgr_to_gray.
try:
    import cv2 as _cv2
except ImportError:          # pragma: no cover - exercised by env, not tests
    _cv2 = None

from .vision.calibration.session import CalibrationSession
from .vision.codebook import codebook_metadata, load_codebook
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

    Returns uint8 on the cv2 path and float64 on the numpy fallback. That
    asymmetry is deliberate: callers must not assume a dtype (process_frame
    converts after rectifying, see there for why), and the numpy path
    exists for synthetic float frames in tests, which have no uint8 form
    to preserve.

    The pure-numpy expression below builds three float64 temporaries the
    size of the full 4K frame -- ~66MB each -- and measured 36ms/frame on
    real camera input, which was 38% of the ENTIRE live frame budget and
    the single largest cost in the system. cv2.cvtColor applies the
    identical BT.601 weights with SIMD. Verified bit-identical output.
    """
    if _cv2 is not None and bgr.dtype == np.uint8:
        return _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    return 0.114 * b + 0.587 * g + 0.299 * r


class TrackerRuntime:
    def __init__(self, calibration_dir, n_ids=6, n_chips=20,
                 target_fps=15.0, rescan_hz=0.0):
        self.geom = RingGeometry(n_chips=n_chips)
        self.geom.validate()
        # Loaded from the committed codebook, never regenerated here: the
        # decoder must expect exactly the codewords the physical markers
        # were PRINTED from. See load_codebook()'s comment in
        # tracker/vision/codebook.py.
        frozen_chips = codebook_metadata()["n_chips"]
        if frozen_chips != n_chips:
            raise ValueError(
                f"n_chips={n_chips} does not match the frozen codebook's "
                f"{frozen_chips}; the printed markers have {frozen_chips} "
                f"chips, so decoding with {n_chips} would be guaranteed to "
                f"fail. Mint a new codebook with tools/generate_codebook.py "
                f"and reprint all markers if you really want to change this.")
        self.codebook = load_codebook(n_ids=n_ids)
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
        # Cached preview material, produced as a by-product of the frame
        # the tracking loop ALREADY processed. Before this existed, the
        # dashboard's video_feed pulled its OWN fresh frame and redid the
        # full gray-convert + rectify + 4K JPEG encode at 10Hz --
        # duplicating the entire front half of the pipeline on a second
        # frame and starving the tracking loop of CPU. That is the
        # clearest single reason a server reported ~2fps while the same
        # code managed 20 on a laptop that was not also serving a preview.
        self._preview_rect = None     # downscaled rectified view
        self._preview_raw = None      # downscaled raw (full camera FOV)
        self._preview_tracks = []     # (x, y, r, id, coasting, age) in preview px
        self._preview_scale = 1.0
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
        """Starts capture. Deliberately does NOT require calibration.

        Capture and tracking are separate concerns, and conflating them
        created an unresolvable first-deploy deadlock: start() demanded a
        play-area calibration, but an operator can only click the four
        play-area corners on a LIVE camera preview -- so the camera could
        not start until calibration existed, and calibration could not be
        done until the camera started.

        So the pipeline is built if it can be, and left as None if not.
        The loop keeps capturing either way, which is what makes the
        preview (and therefore calibration) possible on a fresh install.
        """
        from .capture import RtspCapture
        try:
            self.prepare_pipeline()
        except RuntimeError as e:
            with self._lock:
                self.pipeline = None
                self._stats["error"] = str(e)
        self.capture = RtspCapture(camera_url)
        # Publish self._thread only AFTER it is actually started. Assigning
        # first left a window where a concurrent stop() -- which is
        # reachable, since auto-start runs on a background thread while the
        # operator can already use the dashboard -- saw a non-None thread
        # that had not started yet and died on
        # "cannot join thread before it is started".
        t = threading.Thread(target=self._loop, name="tracker-loop", daemon=True)
        self._running = True
        t.start()
        self._thread = t

    def is_running(self):
        return self._running

    def stop(self):
        self._running = False
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        self._thread = None
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
        with self._lock:
            pipe = self.pipeline
        if pipe is None:
            # Not calibrated yet: frames are still arriving and still
            # feed the live preview the operator calibrates against, so
            # this is a healthy connected state with nothing to track --
            # not an error.
            with self._lock:
                self._latest_tokens = []
                self._stats["last_frame_time"] = now
            return True
        try:
            gray = raw if raw.ndim == 2 else _bgr_to_gray(raw)
            # Warp FIRST, promote to float after. Both orders give the
            # same answer to within the camera's own 8-bit quantisation,
            # but warping uint8 moves 1/8th the bytes and lets OpenCV use
            # its integer path: measured end-to-end (gray + rectify) at
            # 60.2ms -> 10.8ms per 4K frame, a 49ms saving on a 94ms
            # budget.
            #
            # Checked rather than assumed, because the detector claims
            # 0.9px sub-pixel accuracy and this perturbs its input by up
            # to one intensity level. Across 2306 decodes on real
            # still/occluded footage: median centre shift 0.03px, p95
            # 0.35px, and ZERO decode disagreements.
            rectified = self.session.rectify(gray)
            if rectified.dtype != np.float64:
                rectified = rectified.astype(np.float64)
            reported = pipe.process(rectified, now=now)
            tokens = pipe.tokens_mm(reported)
            self._cache_preview(gray, rectified, reported, pipe)
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
                # Read directly off the live capture rather than mirrored
                # into _stats: it is a simple monotonic counter, and this
                # keeps it accurate even across a capture restart. Rising
                # steadily points at a flaky RTSP link (or a camera that
                # reboots itself); see RtspCapture._reopen()'s docstring.
                "reconnects": self.capture.reconnects if self.capture else 0,
                # Per-stage milliseconds, so "it is slow" can be answered
                # with WHICH stage rather than a guess. Without this the
                # only visible symptom on a deployed box is a low fps
                # number and an access log full of 200s.
                "timing_ms": (self.pipeline.timing_report()
                              if self.pipeline is not None else None),
                **self._stats,
            }

    PREVIEW_W = 960

    def _cache_preview(self, gray, rectified, reported, pipe):
        """Keep a small copy of what was just processed, for the dashboard.

        Downscaled to PREVIEW_W on purpose: a browser preview gains
        nothing from 3168px across, and JPEG-encoding that full width was
        itself a large share of the cost this cache exists to remove.

        Both views are kept -- the rectified one the tracker actually
        works on, AND the raw full-camera one, because an operator who
        can only see the cropped result cannot tell whether the crop is
        itself wrong, which is exactly when they most need to look.
        """
        if _cv2 is None:
            return
        try:
            def small(img):
                h, w = img.shape[:2]
                arr = np.clip(img, 0, 255).astype(np.uint8)
                if w <= self.PREVIEW_W:
                    return arr
                sc = self.PREVIEW_W / float(w)
                return _cv2.resize(arr, (int(w * sc), int(h * sc)),
                                   interpolation=_cv2.INTER_AREA)
            rect_small = small(rectified)
            scale = rect_small.shape[1] / float(rectified.shape[1])
            r_px = pipe.geom.r_out * pipe.px_per_mm * scale
            tracks = [(float(t.report_pos[0]) * scale,
                       float(t.report_pos[1]) * scale,
                       float(r_px), t.marker_id, bool(t.misses > 0),
                       float(t.last_seen - t.created))
                      for t in reported]
            with self._lock:
                self._preview_rect = rect_small
                self._preview_raw = small(gray)
                self._preview_tracks = tracks
                self._preview_scale = scale
        except Exception:
            # Preview is strictly cosmetic -- never let it break tracking.
            pass

    def latest_frame_jpeg(self, quality=80, view="rect", overlay=True):
        """Encode the dashboard preview.

        Serves the frame the tracking loop ALREADY processed rather than
        pulling and re-processing a new one (see _cache_preview). view
        selects the rectified image the tracker works on or the raw
        full-camera image; overlay draws what was detected.

        Falls back to the old read-and-rectify path only when no frame
        has been cached yet -- notably BEFORE play-area calibration,
        where there is no pipeline running to produce a cache but the
        operator still needs a live image to click the four corners on.
        """
        import cv2
        with self._lock:
            cached = self._preview_raw if view == "raw" else self._preview_rect
            tracks = list(self._preview_tracks)
        if cached is None:
            if self.capture is None:
                return None
            ok, raw = self.capture.read()
            if not ok:
                return None
            gray = raw if raw.ndim == 2 else _bgr_to_gray(raw)
            try:
                img = self.session.rectify(gray) if view != "raw" else gray
            except Exception:
                # No play-area calibration yet: serve the RAW frame,
                # because that is the image the corners get clicked on.
                # Raw and not lens-undistorted on purpose -- the
                # play-area endpoint undistorts the clicked POINTS
                # itself, so an already-undistorted image would
                # double-correct them.
                img = gray
            arr = np.clip(img, 0, 255).astype(np.uint8) if img.max() > 1.5 \
                else np.clip(img * 255, 0, 255).astype(np.uint8)
            h, w = arr.shape[:2]
            if w > self.PREVIEW_W:
                sc = self.PREVIEW_W / float(w)
                arr = cv2.resize(arr, (int(w*sc), int(h*sc)),
                                 interpolation=cv2.INTER_AREA)
            cached, tracks = arr, []

        disp = cv2.cvtColor(cached, cv2.COLOR_GRAY2BGR) if cached.ndim == 2 else cached.copy()
        # The overlay only means anything on the rectified view: track
        # positions are in rectified pixels, and mapping them back
        # through the homography onto the raw image would be a different
        # (and misleading-if-slightly-wrong) claim.
        if overlay and view != "raw":
            for (x, y, r, mid, coasting, age) in tracks:
                # green = measured this frame, amber = held position
                col = (0, 180, 230) if coasting else (0, 200, 0)
                cv2.circle(disp, (int(x), int(y)), max(3, int(r)), col, 2)
                cv2.drawMarker(disp, (int(x), int(y)), col, cv2.MARKER_CROSS,
                               max(4, int(r * 0.4)), 1)
                label = f"{mid}" if mid is not None else "?"
                if coasting:
                    label += " hold"
                cv2.putText(disp, label, (int(x - r), int(y - r - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None
