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
# Default OFF. This was added to cancel a projector's multiplicative
# illumination gradient across a marker (see the docstring on the
# homomorphic branch of decode() below), and was a clear win while the
# camera saw visible light: it took real projector-lit footage from 34%
# to 51% all-six-decoded on a moving clip.
#
# An IR bandpass filter was installed after that measurement, and it
# removes the projector from the image before any of this runs -- a
# `map-change` clip (switch maps while tokens sit still) now shows zero
# id churn either way, proving there is no gradient left to cancel. With
# nothing left to fix, the cost became pure cost: A/B'd on four real IR
# clips with the CURRENT pipeline (min_peak auto-tuned from measured
# background separation, not the old fixed ceiling), homomorphic on
# never wins on decode rate and lowers margin by 35-40% on every single
# clip (e.g. still: 0.516 vs 0.701; map-change: 0.515 vs 0.722). Lower
# margin is lower protection against confusing two codewords, paid for
# nothing.
#
# This is also where a concrete regression showed up: one synthetic
# token in tests/test_replay.py's fixture decoded at 0.57 peak with this
# on, against 0.68 off -- enough to drop it below MIN_PEAK and fail the
# test on a token sitting on a perfectly clean, noise-free rendered
# marker. That the illumination-rejection filter was costing a read on a
# surface with no illumination problem to reject is exactly the signature
# of a fix whose reason stopped applying.
#
# Kept available (not deleted) behind this flag: a lamp-based projector
# (rather than LED/laser) can still leak real IR, and if that is ever
# the setup this is the lever to pull back on, with the measurements
# above as the baseline to beat.
#
# Angular harmonics discarded as illumination rather than code. The
# codebook search already capped codeword energy in harmonics 1-2
# (the "shading subspace"), so dropping them costs the code almost
# nothing while removing where projector gradients live.
HP_HARMONICS = 2
# Below this peak, re-read at a corrected centre (see
# Matcher._refine_centre). Above it the read is already clean and
# the search is skipped entirely.
REFINE_BELOW = 0.85
REFINE_STEPS = (3.0, 1.5)

# Occlusion-masked decoding (excluding a finger-covered arc from the
# correlation, scoring only the visible samples) was implemented and
# measured against real footage, then REMOVED. The measurement, not a
# guess: scored masked-path output against (a) ground truth from a clean
# nearby read on the same physical token and (b) background noise, on
# the real occluded-footage clip. Correct and background peaks occupy
# almost the same range (correct p5-p95 0.557-0.751 vs background
# 0.514-0.746); a soft-weighted variant (downweight high-variance
# samples instead of hard-erasing them) was no better (correct peak p5
# 0.490 sits BELOW background's own median 0.631). There is no threshold
# that accepts genuine partial occlusion without also accepting table
# noise at a similar rate, and the conservative version that was live
# briefly still produced 2 false accepts in 400 attempts on pure
# background.
#
# This matches, rather than contradicts, the published literature: Koehler
# et al. (VISAPP 2010) make the point directly -- "correlation based
# trackers are not able to handle occlusions robustly, because the
# fiducialness of an observed [pattern] is a threshold decision and the
# correlation measure cannot be used to identify and correct read
# errors." Every system that genuinely tolerates partial occlusion (ARTag,
# RuneTag) does it with an actual error-correcting CODE -- redundant,
# individually-addressable symbols plus a checksum with a verifiable
# pass/fail, not a continuous confidence score -- which is a marker
# REDESIGN, not a decoder change, and not justified against markers that
# are already printed and otherwise meeting the accuracy target.


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


def highpass(x, n_low):
    """Circular high-pass: drop DC and the first n_low angular harmonics.

    Done by zeroing Fourier coefficients rather than by subtracting a
    smoothed copy, because the strip is periodic in angle -- a windowed
    smoother would wrap incorrectly at the seam and leave an artefact
    exactly where the correlation has to be seamless.
    """
    F = np.fft.rfft(x, axis=-1)
    F[..., :n_low + 1] = 0.0
    return np.fft.irfft(F, x.shape[-1], axis=-1)


def _templates(codebook, n_ang, n_low=0):
    """Codewords upsampled to n_ang samples, high-passed to match whatever
    was done to the signal, and unit-normalised. Cached by the caller via
    build_matcher().

    The templates get the SAME high-pass as the strip on purpose. The peak
    is read as a correlation coefficient and compared against fixed gates
    (MIN_PEAK/MIN_MARGIN); if the signal lost its low harmonics but the
    templates kept theirs, every peak would be depressed by the
    unmatchable template energy and those gates would silently mean
    something different than they were calibrated to mean.
    """
    n_chips = codebook.shape[1]
    idx = (np.arange(n_ang) * n_chips) // n_ang
    tpl = codebook[:, idx].astype(np.float64)
    if n_low > 0:
        tpl = highpass(tpl, n_low)
    else:
        tpl -= tpl.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(tpl, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return tpl / norms


class Matcher:
    """Precomputes codeword spectra so each decode is a handful of FFTs."""

    def __init__(self, codebook, geom, n_ang=256, n_rad=8,
                 min_peak=MIN_PEAK, min_margin=MIN_MARGIN,
                 homomorphic=False, hp_harmonics=HP_HARMONICS,
                 refine_below=REFINE_BELOW,
                 refine_steps=REFINE_STEPS):
        self.codebook = np.asarray(codebook, dtype=np.float64)
        self.geom = geom
        self.n_ang = n_ang
        self.n_rad = n_rad
        self.min_peak = min_peak
        self.min_margin = min_margin
        self.refine_below = refine_below
        self._offset_cache = {}
        self.refine_steps = refine_steps
        self.homomorphic = homomorphic
        self.hp_harmonics = hp_harmonics if homomorphic else 0
        self._tpl = _templates(self.codebook, n_ang, self.hp_harmonics)
        self._tpl_f = np.conj(np.fft.rfft(self._tpl, axis=1))

    def _score(self, img, cx, cy, px_per_mm):
        """Correlation peak and rotation for every codeword at one
        candidate centre. Returns (None, None) for a featureless strip."""
        strip2d = unwrap_band(img, cx, cy, px_per_mm, self.geom,
                              self.n_ang, self.n_rad)
        strip = strip2d.mean(axis=0)   # radial average, ~sqrt(n_rad) SNR gain

        if self.homomorphic:
            # simulate.py models the projector the way optics actually
            # works: observed = incident x reflectance. A log turns that
            # product into a sum, so the projected pattern becomes an
            # ADDITIVE term -- and because projected light varies slowly
            # across a 25mm token compared with the 20 chips around it,
            # that term lives almost entirely in the lowest angular
            # harmonics, where a high-pass removes it outright.
            #
            # Subtracting the mean (what this did before) only cancels a
            # CONSTANT illumination offset. It is exactly right for a
            # uniformly lit token and does nothing for the case that
            # actually hurts: a bright edge or gradient falling across
            # part of the ring.
            #
            # The floor is relative, not absolute, so this is independent
            # of whether frames arrive as 0-255 or 0-1: a global scale
            # factor a becomes log(a), a pure DC term, which the
            # high-pass discards anyway.
            peak_v = float(np.max(strip)) if strip.size else 0.0
            s = highpass(np.log(np.maximum(strip, max(1e-6, 1e-4 * peak_v))),
                         self.hp_harmonics)
        else:
            s = strip - strip.mean()
        nrm = float(np.linalg.norm(s))
        if nrm < 1e-9:
            return None, None
        s = s / nrm

        cc = np.fft.irfft(np.fft.rfft(s)[None, :] * self._tpl_f,
                          self.n_ang, axis=1)
        peaks = cc.max(axis=1)
        shifts = cc.argmax(axis=1)

        return peaks, cc.argmax(axis=1)

    def _cache_key(self, cx, cy):
        # One cell per token-ish area. Coarse on purpose: the point is to
        # find the offset that worked for whatever sits HERE, and a token
        # that has moved to a new cell simply misses and pays for one
        # fresh search.
        c = max(32.0, self.geom.r_out * 4.0)
        return (int(cx // c), int(cy // c))

    def _refine_centre(self, img, cx, cy, px_per_mm, peaks, shifts):
        """Local hill-climb on the centre, keeping the best-scoring read.

        Coarse-then-fine rather than a full grid: two passes over the 8
        neighbours at 3px then 1.5px reach ~4.5px of offset with ~1.5px
        resolution in 16 evaluations, against 81 for the equivalent grid.

        The offset is then remembered per table region, because the error
        it corrects is SYSTEMATIC -- a marker whose code band sits a few
        pixels off its detection ring does so every frame, so re-deriving
        the same offset 20 times a second is pure waste. Measured: the
        search cost 7.9ms/frame when it ran every frame against 0.6ms
        when the previous answer is tried first. A cached offset is only
        ever ACCEPTED if it actually scores well, so a stale entry (a
        token moved away, a different marker now here) costs one wasted
        evaluation and then falls through to the real search.
        """
        best = (float(peaks.max()), peaks, shifts, cx, cy)
        key = self._cache_key(cx, cy)
        hint = self._offset_cache.get(key)
        if hint is not None:
            p, sh = self._score(img, cx + hint[0], cy + hint[1], px_per_mm)
            if p is not None and float(p.max()) > best[0]:
                best = (float(p.max()), p, sh, cx + hint[0], cy + hint[1])
                if best[0] >= self.refine_below:
                    return best[1], best[2], best[3], best[4]
        for step in self.refine_steps:
            for dx in (-step, 0.0, step):
                for dy in (-step, 0.0, step):
                    if dx == 0.0 and dy == 0.0:
                        continue
                    p, sh = self._score(img, best[3] + dx, best[4] + dy,
                                        px_per_mm)
                    if p is None:
                        continue
                    if float(p.max()) > best[0]:
                        best = (float(p.max()), p, sh, best[3] + dx,
                                best[4] + dy)
        if best[0] >= self.refine_below:
            if len(self._offset_cache) > 64:
                self._offset_cache.clear()   # bounded; cheap to relearn
            self._offset_cache[key] = (best[3] - cx, best[4] - cy)
        return best[1], best[2], best[3], best[4]

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
        peaks, shifts = self._score(img, cx, cy, px_per_mm)
        if peaks is None:
            dbg = {"reason": "flat strip", "best_peak": 0.0, "margin": 0.0}
            return (None, 0.0, dbg) if return_debug else (None, 0.0)

        # A weak read is usually a MISPLACED read, not an unreadable
        # marker. The detector locks onto the detection ring, and when
        # that estimate is off the unwrap samples the code band along the
        # wrong circle -- which smears the chips and collapses the
        # correlation even though the printed code is perfect.
        #
        # Measured on real IR footage with six stationary markers: three
        # of them sat 2-5px off, right at the ~4px the decoder tolerates.
        # Correcting the centre alone took id1 from 0.650 to 0.948, id2
        # from 0.853 to 0.972 and id5 from 0.792 to 0.901, while the three
        # already-good markers were untouched (their best offset was
        # 0,0). A matching scale search was tried at the same time and
        # added nothing once the centre was right, so it is not done here.
        #
        # Only runs when the first read is weak, so well-placed markers
        # pay nothing for it.
        if float(peaks.max()) < self.refine_below:
            peaks, shifts, cx, cy = self._refine_centre(
                img, cx, cy, px_per_mm, peaks, shifts)


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
