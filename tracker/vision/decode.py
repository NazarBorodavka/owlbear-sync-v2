"""Matched-filter decoding of the angular ring code.

The band is polar-unwrapped into a 1D strip and circularly correlated
against every codeword. That is the noise-optimal linear detector, and it
replaces v1's per-dot thresholding, which collapsed each dot to a single
trit and threw away almost all the available signal -- which is why v1
was destroyed by motion blur that this approach barely notices.

Three properties fall out of doing it this way:

  * Rotation is free. The argmax over circular shift IS the marker's
    rotation, so there is no separate sync-word stage to fail. v1's sync
    stage demanded a PERFECT 4-bit match and refused the whole read
    otherwise, which was a large fraction of its refusals.
  * Blur degrades the correlation peak smoothly instead of flipping bits.
  * Occlusion removes signal rather than corrupting it, so a partly
    covered ring loses confidence instead of producing a wrong answer.
"""
import numpy as np

MIN_PEAK = 0.65     # "is this a marker at all"
MIN_MARGIN = 0.15   # "is the winner clearly better than the runner-up"


def bilinear(img, x, y):
    h, w = img.shape
    x0 = np.clip(np.floor(x).astype(np.intp), 0, w - 2)
    y0 = np.clip(np.floor(y).astype(np.intp), 0, h - 2)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy)
            + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy
            + img[y0 + 1, x0 + 1] * fx * fy)


def unwrap_band(img, cx, cy, px_per_mm, geom, n_ang=256, n_rad=8, inset=0.15):
    """Polar-unwrap the code band to an (n_rad, n_ang) strip.

    Radii come from the geometry and the measured scale, so any error in
    the detector's centre/scale estimate shows up here as the sampling
    ring sliding off the printed band. That coupling is deliberate: it is
    the dominant real-world failure path, so it belongs where the tests
    can see it rather than being hidden behind a tolerance.
    """
    r_i = geom.code_inner * px_per_mm
    r_o = geom.code_outer * px_per_mm
    pad = inset * (r_o - r_i)
    rr = np.linspace(r_i + pad, r_o - pad, n_rad)
    th = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    X = cx + rr[:, None] * np.cos(th)[None, :]
    Y = cy + rr[:, None] * np.sin(th)[None, :]
    return bilinear(img, X, Y)


def _templates(codebook, n_ang):
    """Codewords upsampled to n_ang samples and zero-meaned, cached by the
    caller via build_matcher()."""
    n_chips = codebook.shape[1]
    idx = (np.arange(n_ang) * n_chips) // n_ang
    tpl = codebook[:, idx].astype(np.float64)
    tpl -= tpl.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(tpl, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return tpl / norms


class Matcher:
    """Precomputes codeword spectra so each decode is a handful of FFTs."""

    def __init__(self, codebook, geom, n_ang=256, n_rad=8,
                 min_peak=MIN_PEAK, min_margin=MIN_MARGIN):
        self.codebook = np.asarray(codebook, dtype=np.float64)
        self.geom = geom
        self.n_ang = n_ang
        self.n_rad = n_rad
        self.min_peak = min_peak
        self.min_margin = min_margin
        self._tpl = _templates(self.codebook, n_ang)
        self._tpl_f = np.conj(np.fft.rfft(self._tpl, axis=1))

    def decode(self, img, cx, cy, px_per_mm, return_debug=False):
        """Returns (marker_id or None, margin, debug).

        Two independent gates must both pass:

        The PEAK FLOOR answers "does this look like any of our codewords at
        all". v1 never asked this -- it only ever asked which codeword fit
        best, so an arbitrary round object always won something, which is
        how lamp bases and bottle caps became tokens. Measured separation
        is wide: real markers under blur, occlusion and lighting score
        >=0.68 while blank discs, blank rings, wrong-size rings and pure
        noise top out at 0.61.

        The MARGIN answers "is the winner clearly ahead of the runner-up",
        which is what prevents confusion between two genuine codewords.
        """
        strip2d = unwrap_band(img, cx, cy, px_per_mm, self.geom,
                              self.n_ang, self.n_rad)
        strip = strip2d.mean(axis=0)   # radial average, ~sqrt(n_rad) SNR gain

        s = strip - strip.mean()
        sd = s.std()
        if sd < 1e-6:
            dbg = {"reason": "flat strip", "best_peak": 0.0, "margin": 0.0}
            return (None, 0.0, dbg) if return_debug else (None, 0.0)
        s = s / (sd * np.sqrt(self.n_ang))

        cc = np.fft.irfft(np.fft.rfft(s)[None, :] * self._tpl_f,
                          self.n_ang, axis=1)
        peaks = cc.max(axis=1)
        shifts = cc.argmax(axis=1)

        order = np.argsort(peaks)[::-1]
        best = int(order[0])
        second_peak = float(peaks[order[1]]) if len(order) > 1 else 0.0
        best_peak = float(peaks[best])
        margin = best_peak - second_peak

        reason = None
        if best_peak < self.min_peak:
            reason = "peak below floor (not a marker)"
        elif margin < self.min_margin:
            reason = "margin too small (ambiguous between codewords)"

        rotation_deg = 360.0 * float(shifts[best]) / self.n_ang
        dbg = {"reason": reason, "best_peak": best_peak, "margin": margin,
               "best_id": best, "rotation_deg": rotation_deg,
               "peaks": peaks}
        out = None if reason else best
        return (out, margin, dbg) if return_debug else (out, margin)
