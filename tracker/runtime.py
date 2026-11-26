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
        # Native pixel size of the image each preview view was downscaled
        # FROM. The dashboard needs this to turn a click on the preview
        # back into a camera-native coordinate: the play-area corners are
        # defined in native pixels, and once the served preview stopped
        # being full resolution, <img>.naturalWidth silently stopped
        # meaning "camera width" -- which put every clicked corner at a
        # quarter of its true position and produced a badly wrong crop.
        self._preview_src = {}
        # When the cached preview was produced. A cache with no
        # staleness check is indistinguishable from a working feed right
        # up until it silently freezes.
        self._preview_at = 0.0
        # Guards start(): see there for why two concurrent starts used to
        # leave an orphaned capture holding the camera's only session.
        self._start_lock = threading.Lock()
        # Rolling per-stage cost for the stages OUTSIDE the pipeline.
        # Without these the dashboard could only account for detect +
        # decode + track, which on a deployed server added up to 72ms
        # against an observed 4.4fps -- leaving ~155ms/frame invisible
        # and the obvious conclusion ("detect is slow") wrong.
        self._stage_ms = {"gray": 0.0, "rectify": 0.0, "preview": 0.0,
                          "total": 0.0}
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
        # Idempotent, and serialised against a concurrent start().
        #
        # Without this, calling start() twice left the FIRST capture and
        # loop thread running: self.capture was simply overwritten
        # (never released) and the old thread still saw _running True.
        # The result was two live RTSP sessions against a camera that
        # serves exactly one, so both stalled -- observed on the deployed
        # server as "Stream timeout triggered after 30061ms", frames
        # frozen at 19, and reconnects stuck at 0 because the counter
        # being read belonged to the NEW object while the ABANDONED one
        # held the session.
        #
        # Two callers make this reachable in normal use: boot auto-start
        # runs on a background thread while the dashboard's Start button
        # is already clickable, and the button itself can be pressed
        # twice.
        with self._start_lock:
            self._start_locked(camera_url, RtspCapture)

    def _start_locked(self, camera_url, RtspCapture):
        self.stop()
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

            # Count every processed frame immediately, and report fps
            # from a short rolling window.
            #
            # The previous form accumulated into a local and flushed both
            # the counter and the rate only once 10s had elapsed, using
            # `now` -- a frame timestamp -- as the clock. On a deployed
            # server that reported 2.5fps while the frame counter was
            # demonstrably advancing at ~14fps: a 5x under-read that made
            # a working camera look broken and sent the investigation
            # after the wrong thing entirely. A statistic nobody can
            # trust is worse than no statistic.
            frame_count += 1
            wall = time.time()
            elapsed = wall - last_report
            if elapsed >= 1.0:
                with self._lock:
                    self._stats["fps"] = frame_count / elapsed
                    self._stats["frames"] += frame_count
                frame_count = 0
                last_report = wall

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
            #
            # The preview MUST still be refreshed here. It used to be
            # updated only further down, after the pipeline had run,
            # which created a catch-22 on a fresh install: no play-area
            # calibration means no pipeline, no pipeline means the
            # preview cache is never written, and the feed then served
            # whatever stale frame happened to be cached -- observed on
            # a deployed server as a live-looking image that was in fact
            # hours old, on the exact screen whose entire purpose is
            # clicking four corners on a CURRENT frame.
            try:
                gray = raw if raw.ndim == 2 else _bgr_to_gray(raw)
                self._cache_preview_raw(gray)
            except Exception:
                pass
            with self._lock:
                self._latest_tokens = []
                self._stats["last_frame_time"] = now
            return True
        try:
            t_all = time.perf_counter()
            t0 = time.perf_counter()
            gray = raw if raw.ndim == 2 else _bgr_to_gray(raw)
            t_gray = time.perf_counter() - t0
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
            t0 = time.perf_counter()
            rectified = self.session.rectify(gray)
            if rectified.dtype != np.float64:
                rectified = rectified.astype(np.float64)
            t_rect = time.perf_counter() - t0

            reported = pipe.process(rectified, now=now)

            t0 = time.perf_counter()
            self._cache_preview(gray, rectified, reported, pipe)
            t_prev = time.perf_counter() - t0
            tokens = pipe.tokens_mm(reported)

            # Exponential rolling average: one slow frame should show up
            # without a single outlier defining the reading forever.
            a = 0.1
            with self._lock:
                m = self._stage_ms
                m["gray"] = (1-a)*m["gray"] + a*1000*t_gray
                m["rectify"] = (1-a)*m["rectify"] + a*1000*t_rect
                m["preview"] = (1-a)*m["preview"] + a*1000*t_prev
                m["total"] = ((1-a)*m["total"]
                              + a*1000*(time.perf_counter() - t_all))
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
                "preview_src": dict(self._preview_src),
                "timing_ms": ({**self.pipeline.timing_report(),
                               **{f"t_{k}": v
                                  for k, v in self._stage_ms.items()}}
                              if self.pipeline is not None else None),
                **self._stats,
            }

    PREVIEW_W = 960
    PREVIEW_MAX_AGE_S = 2.0

    def _cache_preview_raw(self, gray):
        """Refresh ONLY the raw-camera preview. Used on the
        not-yet-calibrated path, where there is no rectified image and no
        tracks, but the operator still needs to see live video."""
        if _cv2 is None:
            return
        h, w = gray.shape[:2]
        arr = np.clip(gray, 0, 255).astype(np.uint8)
        if w > self.PREVIEW_W:
            sc = self.PREVIEW_W / float(w)
            arr = _cv2.resize(arr, (int(w * sc), int(h * sc)),
                              interpolation=_cv2.INTER_AREA)
        with self._lock:
            self._preview_raw = arr
            self._preview_tracks = []
            self._preview_at = time.time()
            self._preview_src = dict(self._preview_src)
            self._preview_src["raw"] = [int(w), int(h)]

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
            raw_small = small(gray)
            with self._lock:
                self._preview_rect = rect_small
                self._preview_raw = raw_small
                self._preview_tracks = tracks
                self._preview_scale = scale
                self._preview_at = time.time()
                self._preview_src = {
                    "rect": [int(rectified.shape[1]), int(rectified.shape[0])],
                    "raw": [int(gray.shape[1]), int(gray.shape[0])],
                }
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
            age = time.time() - self._preview_at
        # A stale cache is worse than no cache: it looks like a working
        # live feed. Past this age, fall through to reading a fresh frame.
        if cached is not None and age > self.PREVIEW_MAX_AGE_S:
            cached = None
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
            # Before play-area calibration there is no pipeline and so no
            # cache, yet this is precisely when the operator clicks the
            # four corners -- so record the source size here as well.
            with self._lock:
                self._preview_src = dict(self._preview_src)
                self._preview_src[view] = [int(w), int(h)]
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
