"""Token-detection calibration: adapts decode-acceptance thresholds to
the operator's actual camera, lighting and print contrast.

Deliberately NOT a from-scratch parameter search. The defaults baked into
Detector and Matcher were validated extensively in simulation (see
docs/VALIDATION.md) and are a reasonable starting point anywhere. What
differs rig to rig is how much real-world margin exists around them --
this stage measures that margin directly against a handful of frames of
the operator's own printed markers and sets min_peak/min_margin with
headroom below what was ACTUALLY observed, rather than leaving the
simulation-derived defaults in place unverified. It is the right shape
for v1's actual lesson: tune against real captures, not synthetic
renders, and never trust a number you have not measured.

Only min_peak, min_margin and max_resid_frac are tuned here -- those are
the thresholds that directly gate a confident id against a refusal, and
the ones this measurement can speak to. Detector's lower-level vote
sensitivity was not a parameter that needed adjustment anywhere in
simulation testing, so it is left at its validated default rather than
pretending to tune a knob that was never actually exercised.
"""
import numpy as np

from ..detect import Detector
from ..decode import Matcher, MIN_PEAK, MIN_MARGIN


class DetectorCalibration:
    def __init__(self, min_peak, min_margin, max_resid_frac=0.06,
                 measured=None):
        self.min_peak = min_peak
        self.min_margin = min_margin
        self.max_resid_frac = max_resid_frac
        self.measured = measured or {}

    @classmethod
    def defaults(cls):
        return cls(min_peak=MIN_PEAK, min_margin=MIN_MARGIN, max_resid_frac=0.06)

    @classmethod
    def from_dict(cls, d):
        return cls(d["min_peak"], d["min_margin"], d.get("max_resid_frac", 0.06),
                  d.get("measured"))

    def to_dict(self):
        return {"min_peak": self.min_peak, "min_margin": self.min_margin,
                "max_resid_frac": self.max_resid_frac, "measured": self.measured}

    def build(self, geom, px_per_mm, codebook):
        """A (Detector, Matcher) pair configured with these calibrated
        thresholds and the SYSTEM's actual codebook -- this stage tunes
        acceptance thresholds, it does not own or regenerate identity."""
        det = Detector(geom, px_per_mm, max_resid_frac=self.max_resid_frac)
        mat = Matcher(codebook, geom, min_peak=self.min_peak,
                      min_margin=self.min_margin)
        return det, mat


def calibrate_from_frames(frames, geom, px_per_mm, codebook, expected_ids=None):
    """frames: list of rectified (metric top-down) grayscale images, each
    showing the operator's real markers on the table -- ideally several
    seconds' worth so camera noise and small movements are represented,
    not one lucky frame.

    Returns (DetectorCalibration, report); report includes enough detail
    (peaks found, detections per frame, which expected ids were never
    seen) to show the operator what calibration actually measured, not
    just a pass/fail.
    """
    det = Detector(geom, px_per_mm)
    # Ungated matcher so this measures the real peak/margin DISTRIBUTION,
    # not just whether frames happen to clear the current defaults.
    mat = Matcher(codebook, geom, min_peak=0.0, min_margin=0.0)

    all_peaks, all_margins, seen_ids = [], [], set()
    per_frame_counts = []
    for frame in frames:
        dets = det.detect(frame)
        count = 0
        for d in dets:
            did, margin, dbg = mat.decode(frame, d["x"], d["y"], d["px_per_mm"],
                                          return_debug=True)
            peak = dbg["best_peak"]
            if peak < 0.3:        # not a marker at all (background clutter) -- exclude
                continue
            all_peaks.append(peak)
            all_margins.append(margin)
            seen_ids.add(dbg["best_id"])
            count += 1
        per_frame_counts.append(count)

    if not all_peaks:
        raise ValueError(
            "no markers were detected in any calibration frame -- check "
            "that markers are placed in view, lighting is adequate, and "
            "px_per_mm matches the play-area calibration")

    peaks = np.array(all_peaks)
    margins = np.array(all_margins)
    # Headroom below the WORST (5th percentile, not average) observed
    # reading, so a single noisy frame does not set the threshold right
    # at the cliff edge.
    p5_peak = float(np.percentile(peaks, 5))
    p5_margin = float(np.percentile(margins, 5))
    min_peak = float(np.clip(p5_peak * 0.85, 0.5, 0.65))
    min_margin = float(np.clip(p5_margin * 0.7, 0.08, 0.15))

    report = {
        "frames": len(frames),
        "markers_seen": sorted(seen_ids),
        "expected_ids": sorted(expected_ids) if expected_ids else None,
        "missing_ids": sorted(set(expected_ids) - seen_ids) if expected_ids else [],
        "detections_per_frame": {"min": min(per_frame_counts), "max": max(per_frame_counts),
                                 "median": float(np.median(per_frame_counts))},
        "peak": {"p5": p5_peak, "median": float(np.median(peaks)), "max": float(peaks.max())},
        "margin": {"p5": p5_margin, "median": float(np.median(margins)), "max": float(margins.max())},
        "calibrated_min_peak": min_peak,
        "calibrated_min_margin": min_margin,
    }

    cal = DetectorCalibration(min_peak=min_peak, min_margin=min_margin,
                              measured=report)
    return cal, report
