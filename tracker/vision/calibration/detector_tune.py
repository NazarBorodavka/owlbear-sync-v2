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

Only min_peak and min_margin are tuned here -- those are
the thresholds that directly gate a confident id against a refusal, and
the ones this measurement can speak to. Detector's lower-level vote
sensitivity was not a parameter that needed adjustment anywhere in
simulation testing, so it is left at its validated default rather than
pretending to tune a knob that was never actually exercised.
"""
import numpy as np

from ..detect import Detector, MAX_RESID_FRAC
from ..decode import Matcher, MIN_PEAK, MIN_MARGIN


class DetectorCalibration:
    def __init__(self, min_peak, min_margin, max_resid_frac=MAX_RESID_FRAC,
                 measured=None):
        self.min_peak = min_peak
        self.min_margin = min_margin
        self.max_resid_frac = max_resid_frac
        self.measured = measured or {}

    @classmethod
    def defaults(cls):
        return cls(min_peak=MIN_PEAK, min_margin=MIN_MARGIN,
                   max_resid_frac=MAX_RESID_FRAC)

    @classmethod
    def from_dict(cls, d):
        return cls(d["min_peak"], d["min_margin"],
                   d.get("max_resid_frac", MAX_RESID_FRAC),
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
    bg_peaks = []
    per_frame_counts = []
    n_expected = len(expected_ids) if expected_ids else None
    rng = np.random.default_rng(0)
    for frame in frames:
        dets = det.detect(frame)
        scored = []
        for d in dets:
            did, margin, dbg = mat.decode(frame, d["x"], d["y"], d["px_per_mm"],
                                          return_debug=True)
            scored.append((dbg["best_peak"], margin, dbg["best_id"]))
        # Keep only the best n_expected reads per frame as "real".
        #
        # The old rule kept everything scoring above 0.3, which on a real
        # table is dominated by BACKGROUND: measured under IR, a read
        # taken on bare table has a median peak of 0.53 and reaches 0.85.
        # Mixing those into the distribution dragged its 5th percentile
        # down to ~0.70 and produced a min_peak of 0.60 -- a threshold
        # that 63% of bare-table reads passed, which is what let ghost
        # tokens appear on an empty table and hold a real marker's id.
        scored.sort(reverse=True)
        keep = scored[:n_expected] if n_expected else scored
        for peak, margin, bid in keep:
            if peak < 0.3:
                continue
            all_peaks.append(peak)
            all_margins.append(margin)
            seen_ids.add(bid)
        per_frame_counts.append(len(keep))

        # Sample the BACKGROUND too. A threshold can only be placed
        # sensibly if both distributions are known; measuring markers
        # alone says how low the gate may go, never how high it must be.
        h, w = frame.shape[:2]
        for _ in range(8):
            x = float(rng.uniform(0.1 * w, 0.9 * w))
            y = float(rng.uniform(0.1 * h, 0.9 * h))
            if any(np.hypot(x - d["x"], y - d["y"]) < 2.5 * geom.r_out * px_per_mm
                   for d in dets):
                continue
            _, _, g = mat.decode(frame, x, y, px_per_mm, return_debug=True)
            bg_peaks.append(g["best_peak"])

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

    # Place the gate BETWEEN what real markers score and what the bare
    # table scores, instead of only below the markers. The old form was
    # clip(p5 * 0.85, 0.5, 0.65): the 0.65 ceiling meant no amount of
    # image quality could ever buy a stricter gate, which is exactly
    # backwards -- the IR filter lifted real reads to ~0.96 median and
    # the gate stayed where it was for a projector-lit table.
    #
    # With both distributions measured, the midpoint is well separated
    # from each: real p5 ~0.88, background p99 ~0.80 gives ~0.84, and a
    # sweep on real footage put the optimum at 0.82-0.85 (bad publishes
    # 0.24-0.44%, against 0.87-0.89% at 0.60). The ceiling is raised to
    # 0.90 rather than removed, because above ~0.88 genuinely blurred
    # reads start being rejected and tracks go stale instead.
    if bg_peaks:
        bg_p99 = float(np.percentile(bg_peaks, 99))
        min_peak = float(np.clip(0.5 * (bg_p99 + p5_peak), 0.5, 0.90))
        # never above the markers' own floor -- better to admit some
        # background than to reject the markers we are calibrating for
        min_peak = min(min_peak, p5_peak * 0.97)
    else:
        bg_p99 = None
        min_peak = float(np.clip(p5_peak * 0.85, 0.5, 0.90))
    min_margin = float(np.clip(p5_margin * 0.7, 0.08, 0.15))

    report = {
        "frames": len(frames),
        "markers_seen": sorted(seen_ids),
        "expected_ids": sorted(expected_ids) if expected_ids else None,
        "missing_ids": sorted(set(expected_ids) - seen_ids) if expected_ids else [],
        "detections_per_frame": {"min": min(per_frame_counts), "max": max(per_frame_counts),
                                 "median": float(np.median(per_frame_counts))},
        "peak": {"p5": p5_peak, "median": float(np.median(peaks)), "max": float(peaks.max())},
        "background_peak": ({"n": len(bg_peaks), "p99": bg_p99,
                             "median": float(np.median(bg_peaks))}
                            if bg_peaks else None),
        "margin": {"p5": p5_margin, "median": float(np.median(margins)), "max": float(margins.max())},
        "calibrated_min_peak": min_peak,
        "calibrated_min_margin": min_margin,
    }

    cal = DetectorCalibration(min_peak=min_peak, min_margin=min_margin,
                              measured=report)
    return cal, report
