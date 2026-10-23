"""Orchestrates the three independent calibration stages (lens, play-area,
detector) behind one object: loads whatever is already persisted at
startup, lets any ONE stage be re-run and re-saved without touching the
others, and builds the rectification + detection pipeline the running
system actually uses from whatever is currently calibrated.

This is the concrete shape of "each part able to be rerun separately and
with persistent storage": each stage genuinely has its own file (see
store.py) and its own calibrate_* function, and this class only ever
reads/writes the ONE stage it was asked to touch.
"""
import numpy as np

from .store import CalibrationStore
from .lens import LensCalibration, calibrate_from_frames as calibrate_lens
from .playarea import PlayAreaCalibration
from .detector_tune import DetectorCalibration, calibrate_from_frames as calibrate_detector
from ..geometry import RingGeometry
from ..pipeline import Pipeline


class CalibrationSession:
    def __init__(self, store_dir, geom=None, codebook=None):
        self.store = CalibrationStore(store_dir)
        self.geom = geom or RingGeometry()
        self.codebook = codebook
        self.lens = None
        self.playarea = None
        self.detector = DetectorCalibration.defaults()
        self._load_all()

    # ---- loading -------------------------------------------------------
    def _load_all(self):
        lens_payload = self.store.load("lens")
        if lens_payload:
            self.lens = LensCalibration.from_dict(lens_payload["data"])
        area_payload = self.store.load("playarea")
        if area_payload:
            self.playarea = PlayAreaCalibration.from_dict(area_payload["data"])
        det_payload = self.store.load("detector")
        if det_payload:
            self.detector = DetectorCalibration.from_dict(det_payload["data"])

    # ---- stage 1: lens ---------------------------------------------------
    def calibrate_lens(self, gray_frames, pattern_size=(9, 6), square_mm=25.0):
        cal, per_frame_err = calibrate_lens(gray_frames, pattern_size, square_mm)
        self.lens = cal
        self.store.save("lens", cal.to_dict())
        return cal, per_frame_err

    def clear_lens(self):
        """Lens calibration is the one stage genuinely optional at
        startup (see lens.py) -- clearing it falls back to an identity
        transform rather than leaving the rest of the pipeline unable to
        run."""
        self.lens = None
        self.store.clear("lens")

    # ---- stage 2: play-area ----------------------------------------------
    def calibrate_playarea(self, corners_px, width_mm, height_mm, px_per_mm,
                           already_undistorted=False, image_size=None):
        """corners_px are in RAW camera pixel coordinates by default (what
        an operator clicks on the live, un-rectified feed) and are run
        through the current lens calibration first, so play-area
        calibration always operates in undistorted space regardless of
        whether the operator clicked before or after lens calibration was
        run. Pass already_undistorted=True to skip that if the caller has
        already done it."""
        pts = np.asarray(corners_px, dtype=np.float64)
        if not already_undistorted and self.lens is not None:
            pts = self.lens.undistort_points(pts)
        cal = PlayAreaCalibration(pts, width_mm, height_mm, px_per_mm)
        self.playarea = cal
        self.store.save("playarea", cal.to_dict())
        return cal

    def clear_playarea(self):
        self.playarea = None
        self.store.clear("playarea")

    # ---- stage 3: detector thresholds -------------------------------------
    def calibrate_detector(self, rectified_frames, expected_ids=None):
        if self.codebook is None:
            raise RuntimeError("CalibrationSession needs a codebook to calibrate "
                              "detector thresholds -- pass one at construction")
        cal, report = calibrate_detector(rectified_frames, self.geom,
                                         self.playarea.px_per_mm, self.codebook,
                                         expected_ids=expected_ids)
        self.detector = cal
        self.store.save("detector", cal.to_dict())
        return cal, report

    def reset_detector(self):
        self.detector = DetectorCalibration.defaults()
        self.store.clear("detector")

    # ---- runtime use -------------------------------------------------------
    def rectify(self, raw_frame):
        """Raw camera frame -> metric top-down frame at the calibrated
        px_per_mm. Requires play-area calibration (there is no meaningful
        fallback for "what scale is this"); lens undistortion is applied
        first if available, skipped otherwise."""
        if self.playarea is None:
            raise RuntimeError("play-area is not calibrated yet -- run "
                              "calibrate_playarea() before rectify()")
        img = raw_frame
        if self.lens is not None:
            img = self.lens.undistort_image(img)
        return self.playarea.warp(img)

    def build_pipeline(self, **pipeline_kw):
        if self.playarea is None:
            raise RuntimeError("play-area is not calibrated yet -- cannot "
                              "build a pipeline without a known scale")
        if self.codebook is None:
            raise RuntimeError("CalibrationSession needs a codebook to build a pipeline")
        pipe = Pipeline(self.geom, self.codebook, self.playarea.px_per_mm, **pipeline_kw)
        pipe.detector.max_resid_frac = self.detector.max_resid_frac
        pipe.matcher.min_peak = self.detector.min_peak
        pipe.matcher.min_margin = self.detector.min_margin
        return pipe

    def status(self):
        out = {
            "lens": {"calibrated": self.lens is not None,
                    "rms_error": self.lens.rms_error if self.lens else None},
            "playarea": None,
            "detector": {"calibrated": bool(self.detector.measured),
                        "min_peak": self.detector.min_peak,
                        "min_margin": self.detector.min_margin},
        }
        if self.playarea is not None:
            out["playarea"] = {
                "calibrated": True,
                "width_mm": self.playarea.width_mm, "height_mm": self.playarea.height_mm,
                "px_per_mm": self.playarea.px_per_mm,
                **self.playarea.coverage_summary(self.geom),
            }
        else:
            out["playarea"] = {"calibrated": False}
        return out
