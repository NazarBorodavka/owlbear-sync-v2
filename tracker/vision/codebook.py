"""Codeword design for the angular ring code.

v1 picked codewords by maximising Hamming distance between ALIGNED
codewords. That is the wrong objective for a rotation-invariant marker
read by correlation: it says nothing about how a codeword correlates with
a ROTATED copy of itself or of another codeword, and since the marker's
rotation on the table is arbitrary, that is exactly where confusions come
from. v1 then needed a separate sync word to pin down rotation, which
became its own brittle failure mode.

Here the objective is the quantity the matched filter actually measures:
the worst-case periodic cross-correlation between any two codewords at
ANY relative rotation, and the worst out-of-phase autocorrelation sidelobe
of each codeword. Optimising that directly removes the need for a sync
word at all -- rotation falls out of the correlation peak.
"""
import numpy as np


def periodic_xcorr_max(a, b, exclude_zero_shift=None):
    """Max |circular cross-correlation| of two +/-1 sequences, normalised.

    When a and b are the same sequence this excludes zero shift, so the
    result is the worst autocorrelation SIDELOBE -- the thing that decides
    whether a codeword can be confused with a rotation of itself.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    if exclude_zero_shift is None:
        exclude_zero_shift = np.array_equal(a, b)
    cc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n) / n
    if exclude_zero_shift:
        cc = cc.copy()
        cc[0] = 0.0
    return float(np.max(np.abs(cc)))


def _canonical(v):
    return min(tuple(np.roll(v, k)) for k in range(len(v)))


def build_codebook(n_chips=20, n_ids=8, seed=1, max_attempts=12000,
                   auto_thr=0.45, cross_thr=0.40):
    """Greedy search for n_ids balanced binary cyclic sequences.

    Balanced (equal black and white chips) is a hard requirement, not a
    preference: it makes the unwrapped strip zero-mean, so the correlation
    is immune to any overall brightness offset. Most of what "variable
    lighting" does to a patch this small is exactly such an offset.
    """
    if n_chips % 2:
        raise ValueError("n_chips must be even so codewords can be balanced")
    rng = np.random.default_rng(seed)
    pool, seen = [], set()
    for _ in range(max_attempts):
        v = np.ones(n_chips, dtype=np.int8)
        v[rng.choice(n_chips, n_chips // 2, replace=False)] = -1
        canon = _canonical(v)
        if canon in seen:
            continue
        seen.add(canon)
        arr = np.array(canon, dtype=np.float64)
        if periodic_xcorr_max(arr, arr) <= auto_thr:
            pool.append(arr)
    if not pool:
        raise ValueError(f"no usable sequences at n_chips={n_chips}")

    chosen = [pool[0]]
    thr = cross_thr
    while len(chosen) < n_ids and thr <= 1.0:
        for cand in pool:
            if len(chosen) >= n_ids:
                break
            if any(np.array_equal(cand, c) for c in chosen):
                continue
            if max(periodic_xcorr_max(cand, c) for c in chosen) <= thr:
                chosen.append(cand)
        thr += 0.05
    if len(chosen) < n_ids:
        raise ValueError(f"only found {len(chosen)}/{n_ids} codewords "
                         f"at n_chips={n_chips}")
    return np.array(chosen[:n_ids])


def quality(cb):
    """(worst autocorrelation sidelobe, worst pairwise cross-correlation)."""
    worst_auto = max(periodic_xcorr_max(c, c) for c in cb)
    worst_cross = 0.0
    for i in range(len(cb)):
        for j in range(i + 1, len(cb)):
            worst_cross = max(worst_cross, periodic_xcorr_max(cb[i], cb[j]))
    return worst_auto, worst_cross


def to_bits(cb):
    """+/-1 -> 0/1, for human-readable display and printing."""
    return ((np.asarray(cb) > 0).astype(int)).tolist()
