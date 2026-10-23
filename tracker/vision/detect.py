"""Finding tokens in a metric-rectified frame.

Because the frame is rectified to a known millimetres-per-pixel scale
before this runs, every token is a circle of EXACTLY known radius. v1
threw that constraint away by running cv2.HoughCircles over a radius
RANGE, which is both slower and much less accurate -- measured on real v1
captures, its radius estimate was 10-13% off, which is right at the point
where decoding collapses.

Detection here is a single-radius gradient vote: each edge pixel votes for
a centre one known radius away along its own gradient direction. Using
gradient DIRECTION rather than intensity makes the vote invariant to
absolute brightness and to smooth lighting gradients, and being a vote
accumulation it degrades gracefully under occlusion.

The peak is then refined to sub-pixel accuracy against the detection
ring's inner edge. Measured on synthetic scenes: 0.9px median centre
error and 0.8-2% radius error, against a decoder that tolerates 4px and
6% -- an order of magnitude inside spec rather than sitting on the cliff.
"""
import numpy as np

from .decode import bilinear


def sobel(img):
    kx = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
    p = np.pad(img, 1, mode='edge')
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    for i in range(3):
        for j in range(3):
            if kx[i, j]:
                gx += kx[i, j] * p[i:i + img.shape[0], j:j + img.shape[1]]
            if kx[j, i]:
                gy += kx[j, i] * p[i:i + img.shape[0], j:j + img.shape[1]]
    return gx, gy


def gaussian_blur(img, sigma):  # noqa: D401
    """Separable Gaussian via shifted adds. Pure numpy on purpose: the core
    algorithms stay importable and testable without OpenCV, which is only
    pulled in for video I/O and lens calibration."""
    img = np.asarray(img, dtype=np.float64)
    if sigma <= 0:
        return img
    rad = max(1, int(3 * sigma))
    x = np.arange(-rad, rad + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    pad = np.pad(img, rad, mode='edge')
    out = np.zeros_like(img)
    for i, wgt in enumerate(k):
        out += wgt * pad[rad:rad + img.shape[0], i:i + img.shape[1]]
    pad = np.pad(out, rad, mode='edge')
    out2 = np.zeros_like(img)
    for i, wgt in enumerate(k):
        out2 += wgt * pad[i:i + img.shape[0], rad:rad + img.shape[1]]
    return out2


def downsample(img, f):
    if f <= 1:
        return img
    h = (img.shape[0] // f) * f
    w = (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def vote_centers(img, r_px, grad_thresh_rel=0.25, polarity=+1):
    """Single-radius gradient vote.

    polarity=+1 suits the detection ring's inner edge: crossing it outward
    goes white paper -> black ink, so the intensity gradient points inward,
    toward the centre.
    """
    gx, gy = sobel(img)
    mag = np.hypot(gx, gy)
    mmax = mag.max()
    if mmax <= 0:
        return np.zeros_like(img)
    ys, xs = np.nonzero(mag >= grad_thresh_rel * mmax)
    if len(xs) == 0:
        return np.zeros_like(img)
    m = mag[ys, xs]
    cx = np.round(xs + polarity * r_px * gx[ys, xs] / m).astype(np.intp)
    cy = np.round(ys + polarity * r_px * gy[ys, xs] / m).astype(np.intp)
    h, w = img.shape
    ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    # bincount rather than np.add.at: identical result, ~10x faster,
    # because np.add.at falls back to an unbuffered element-wise path.
    # bincount on an empty selection returns an integer array, which then
    # poisons the dtype of everything downstream -- force float.
    return np.bincount((cy[ok] * w + cx[ok]), weights=m[ok],
                       minlength=h * w).astype(np.float64, copy=False).reshape(h, w)


def find_peaks(acc, min_sep_px, n_max=12, rel_thresh=0.22):
    a = gaussian_blur(acc, max(1.0, min_sep_px * 0.06))
    amax = a.max()
    if amax <= 0:
        return []
    peaks, work = [], a.copy()
    for _ in range(n_max):
        idx = int(np.argmax(work))
        y, x = np.unravel_index(idx, work.shape)
        v = float(work[y, x])
        if v < rel_thresh * amax:
            break
        peaks.append((float(x), float(y), v))
        y0, y1 = max(0, int(y - min_sep_px)), min(work.shape[0], int(y + min_sep_px))
        x0, x1 = max(0, int(x - min_sep_px)), min(work.shape[1], int(x + min_sep_px))
        work[y0:y1, x0:x1] = 0.0
    return peaks


def refine(img, cx, cy, r_px, n_ang=180, search_frac=0.18, trim=0.35):
    """Sub-pixel centre and radius from the detection ring's inner edge.

    That edge is chosen over the token's outer silhouette because both of
    its sides are printed, so its contrast is set by the artwork and does
    not depend on the table, the lighting or what the token sits on.

    The weakest `trim` fraction of rays is discarded before fitting, which
    is what stops an occluded arc -- a hand, an overhanging miniature --
    from dragging the circle off centre.
    """
    th = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    lo, hi = r_px * (1 - search_frac), r_px * (1 + search_frac)
    n_s = 40
    rr = np.linspace(lo, hi, n_s)
    X = cx + rr[None, :] * np.cos(th)[:, None]
    Y = cy + rr[None, :] * np.sin(th)[:, None]
    prof = bilinear(img, X, Y)
    d = -np.diff(prof, axis=1)                 # light -> dark = positive
    k = np.argmax(d, axis=1)
    strength = d[np.arange(n_ang), k]

    kc = np.clip(k, 1, d.shape[1] - 2)
    y0 = d[np.arange(n_ang), kc - 1]
    y1 = d[np.arange(n_ang), kc]
    y2 = d[np.arange(n_ang), kc + 1]
    den = y0 - 2 * y1 + y2
    delta = np.where(np.abs(den) > 1e-9,
                     0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
    redge = lo + (kc + np.clip(delta, -1, 1)) * (hi - lo) / (n_s - 1)

    px_ = cx + redge * np.cos(th)
    py_ = cy + redge * np.sin(th)
    keep = strength >= np.quantile(strength, trim)
    px_, py_ = px_[keep], py_[keep]
    if len(px_) < 12:
        return cx, cy, r_px, float('inf')
    A = np.c_[2 * px_, 2 * py_, np.ones(len(px_))]
    b = px_ ** 2 + py_ ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    ncx, ncy = float(sol[0]), float(sol[1])
    nr = float(np.sqrt(max(sol[2] + ncx ** 2 + ncy ** 2, 1e-9)))
    resid = float(np.median(np.abs(np.hypot(px_ - ncx, py_ - ncy) - nr)))
    return ncx, ncy, nr, resid


class Detector:
    def __init__(self, geom, px_per_mm, n_max=12, coarse=2,
                 max_resid_frac=0.06, max_scale_err=0.20):
        self.geom = geom
        self.px_per_mm = px_per_mm
        self.n_max = n_max
        self.coarse = coarse
        self.max_resid_frac = max_resid_frac
        self.max_scale_err = max_scale_err

    @property
    def r_ref(self):
        """Expected radius, in pixels, of the detection ring's inner edge."""
        return self.geom.det_inner * self.px_per_mm

    def detect(self, img):
        """Full-frame search. Returns dicts with sub-pixel centre, the
        refined reference radius, and the implied metric scale."""
        r_ref = self.r_ref
        # Vote on a downsampled copy -- the accumulator only needs to place
        # a peak within a few pixels because refine() re-finds the edge at
        # full resolution. Voting is the expensive stage and scales with
        # pixel count, so this is close to a free 4x.
        small = downsample(img, self.coarse)
        acc = vote_centers(small, r_ref / self.coarse)
        peaks = find_peaks(acc, min_sep_px=(r_ref / self.coarse) * 1.2,
                           n_max=self.n_max)
        out = []
        for (x, y, v) in peaks:
            out_i = self.refine_at(img, x * self.coarse, y * self.coarse, vote=v)
            if out_i is not None:
                out.append(out_i)
        return out

    def search_near(self, img, cx, cy, radius_px):
        """Vote-based search restricted to a window around a prediction.

        Used to reacquire a track the cheap edge fit lost, without paying
        for a full-frame rescan."""
        r_ref = self.r_ref
        pad = int(max(radius_px, r_ref * 1.4)) + 4
        x0 = max(0, int(cx) - pad); x1 = min(img.shape[1], int(cx) + pad)
        y0 = max(0, int(cy) - pad); y1 = min(img.shape[0], int(cy) + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        roi = img[y0:y1, x0:x1]
        # Lower gradient and peak thresholds than a full-frame scan: we
        # already know a token was here a moment ago, so the question is
        # "where did it go", not "is there anything here". A fast-moving
        # token is motion-blurred and its edges are genuinely weaker, so
        # full-frame thresholds would reject exactly the case this exists
        # to recover. False peaks are harmless here because the caller
        # still gates on the refined fit and on distance.
        acc = vote_centers(roi, r_ref, grad_thresh_rel=0.12)
        peaks = find_peaks(acc, min_sep_px=r_ref * 1.2, n_max=4,
                           rel_thresh=0.12)
        best = None
        for (px_, py_, v) in peaks:
            cand = self.refine_at(img, px_ + x0, py_ + y0, vote=v)
            if cand is None:
                continue
            dist = np.hypot(cand["x"] - cx, cand["y"] - cy)
            if dist > radius_px:
                continue
            if best is None or dist < best[0]:
                best = (dist, cand)
        return best[1] if best else None

    def refine_at(self, img, cx, cy, vote=0.0):
        """Refine a single candidate. Used both by full-frame detection and,
        every frame, to re-localise an already-tracked token from its
        predicted position -- which is the cheap steady-state path."""
        r_ref = self.r_ref
        rx, ry, rr, resid = refine(img, cx, cy, r_ref)
        if not np.isfinite(resid) or resid > self.max_resid_frac * r_ref:
            return None
        if abs(rr - r_ref) > self.max_scale_err * r_ref:
            return None
        return {"x": rx, "y": ry, "r_ref": rr,
                "px_per_mm": rr / self.geom.det_inner,
                "vote": vote, "resid": resid}
