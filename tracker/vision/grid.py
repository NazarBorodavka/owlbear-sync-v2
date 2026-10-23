"""Measures the real-world spacing of a projected grid, directly from the
calibrated camera feed -- what makes "automatic map zooming so the grid
aligns with token base size" an actually automatic feature rather than a
manual trial-and-error zoom adjustment.

The play-area calibration already gives an exact px_per_mm for the
camera's rectified view (see calibration/playarea.py). If the DM turns on
Owlbear Rodeo's own grid overlay, the ceiling projector displays that
grid on the table, and the camera sees it as a periodic pattern of thin
dark lines. Measuring that pattern's real spacing in millimetres and
comparing it to the known token base diameter is enough to compute the
exact OBR viewport zoom correction needed, with no manual guesswork and
no separate projector calibration step.
"""
import numpy as np


def _otsu_threshold_1d(values, n_bins=256):
    """Otsu's method: the threshold maximising between-class variance.
    Unlike a fixed percentile, this correctly separates two clusters of
    very UNEQUAL size -- which matters here specifically, since a thin
    grid line can be a tiny fraction (a couple of percent) of the pixels
    in its profile, not anywhere near a round percentile like 30%. Found
    directly: a percentile-based threshold merged every single line in
    the signal into one giant run, because the 30th percentile of a
    signal that is ~98% background sits inside the background cluster,
    not between the two clusters at all.
    """
    v = np.asarray(values, dtype=np.float64)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return float(lo)
    hist, edges = np.histogram(v, bins=n_bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    w_bg = np.cumsum(hist)
    w_fg = total - w_bg
    sum_total = np.sum(hist * centers)
    sum_bg = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = np.where(w_bg > 0, sum_bg / np.where(w_bg > 0, w_bg, 1), 0)
        mean_fg = np.where(w_fg > 0, (sum_total - sum_bg) / np.where(w_fg > 0, w_fg, 1), 0)
        between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
    valid = (w_bg > 0) & (w_fg > 0)
    if not np.any(valid):
        return float(np.median(v))
    between = np.where(valid, between, -1)
    return float(centers[int(np.argmax(between))])


def _line_positions(profile, min_period_px):
    """Finds the center position of each dark line (a contiguous run
    below an Otsu-derived threshold) in a 1D brightness profile."""
    thr = _otsu_threshold_1d(profile)
    is_line = profile <= thr
    # Merge any gap shorter than a tenth of the minimum plausible period --
    # antialiasing or noise can otherwise split one real line into two
    # detections a pixel or two apart.
    min_gap = max(1, int(min_period_px * 0.1))
    idx = np.flatnonzero(is_line)
    if len(idx) == 0:
        return []
    runs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return [(a + b) / 2.0 for a, b in runs]


def _dominant_period_from_positions(positions, min_period_px, max_period_px):
    """Period = the MEDIAN consecutive spacing between detected line
    centers, not an autocorrelation peak.

    An earlier version used autocorrelation, which has a real, confirmed
    problem for a sparse thin-line signal specifically: integer multiples
    of the true period also align every line with a line, so harmonics
    produce autocorrelation peaks that can be as strong as or STRONGER
    than the true fundamental's (measured directly: a 2x harmonic scored
    0.875 against the fundamental's 0.714 for one realistic case) --
    magnitude-threshold heuristics to prefer the fundamental were tried
    and found not reliable enough across different line widths/periods.

    Working from the actual line POSITIONS instead sidesteps the harmonic
    ambiguity entirely: consecutive real lines are always one true period
    apart, so the raw, unprocessed consecutive-gap sequence is already
    dominated by the correct answer with no aliasing-prone transform in
    between. The median single-gap in that sequence IS the period.
    """
    if len(positions) < 3:
        return None, 0.0
    gaps = np.diff(positions)
    gaps = gaps[(gaps >= min_period_px) & (gaps <= max_period_px)]
    if len(gaps) < 2:
        return None, 0.0
    period = float(np.median(gaps))
    # Confidence: what fraction of gaps agree with the median within 20%
    # -- a few missed/merged line detections should not tank this, but a
    # genuinely non-periodic signal (no real grid) will have scattered,
    # disagreeing gaps and score low.
    agree = np.abs(gaps - period) <= 0.2 * period
    confidence = float(agree.mean())
    return period, confidence


def measure_grid_spacing_mm(gray, px_per_mm, min_cell_mm=15.0, max_cell_mm=200.0,
                            min_confidence=0.5):
    """Measures the dominant grid period along both image axes and
    returns their average in millimetres, or None if no sufficiently
    periodic line pattern was found (e.g. the grid overlay is off, or
    the table is showing a busy map with no visible grid lines).

    Looking along BOTH axes independently and averaging is deliberate: a
    SMALL residual misalignment between the rectified view's axes and the
    true grid (expected to be minor, since the play-area homography
    already aligns the physical table to the rectified frame by
    construction -- see calibration/playarea.py) barely changes the row/
    column spacing for a true rectangular grid. Verified directly up to
    about 1.5 degrees of residual misalignment; this is NOT a general
    rotation-invariant technique -- at 2+ degrees a tall frame's top-to-
    bottom pixel drift becomes wide enough to smear one line's column-
    profile contribution across many columns and defeat per-line
    detection (see tests/test_grid.py for the exact numbers).
    """
    h, w = gray.shape
    row_profile = gray.mean(axis=1)   # averaging across the axis being measured
    col_profile = gray.mean(axis=0)   # cancels non-grid texture, keeps the periodic part

    min_px = min_cell_mm * px_per_mm
    max_px = min(max_cell_mm * px_per_mm, min(h, w) / 2.0)

    row_positions = _line_positions(row_profile, min_px)
    col_positions = _line_positions(col_profile, min_px)
    row_period, row_conf = _dominant_period_from_positions(row_positions, min_px, max_px)
    col_period, col_conf = _dominant_period_from_positions(col_positions, min_px, max_px)

    candidates = [(p, c) for p, c in [(row_period, row_conf), (col_period, col_conf)]
                 if p is not None and c >= min_confidence]
    if not candidates:
        return None

    period_px = float(np.mean([p for p, _ in candidates]))
    confidence = float(np.mean([c for _, c in candidates]))
    return {"mm_per_cell": period_px / px_per_mm, "confidence": confidence,
            "axes_used": len(candidates)}
