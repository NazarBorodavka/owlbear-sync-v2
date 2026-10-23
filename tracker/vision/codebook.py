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


# ---------------------------------------------------------------------------
# The frozen, shipped codebook
# ---------------------------------------------------------------------------
# The codebook is deliberately a COMMITTED DATA FILE, not something
# regenerated at runtime from a seed, for one reason that matters far more
# than the (real, ~4s) startup cost it also saves:
#
# The physical markers are PRINTED from this codebook. If the decoder
# derives its expected codewords by re-running a search at import time,
# then any change to that search -- a numpy RNG implementation change
# across versions, a tweaked default, a reordered candidate loop -- would
# silently desync the decoder from markers already printed and sitting on
# someone's table, with no error and no way to notice except tokens
# mysteriously refusing to decode.
#
# That is exactly the failure v1 hit when a whole print run turned out
# mirrored relative to what the decoder expected: artwork and decoder
# expectations living in separate places and drifting apart. Freezing the
# codebook makes the artwork generator and the decoder read the SAME
# bytes, so they cannot disagree.
#
# To mint a NEW codebook (which invalidates every printed marker -- they
# must all be reprinted), run tools/generate_codebook.py deliberately.
import json
import os

# Deliberately NOT in a "data" directory. The codebook is a shipped
# CONSTANT -- part of the code, fixed to the printed markers -- not
# mutable state, so it lives beside the module that reads it.
#
# It was briefly in tracker/data/, which collided with the default
# TRACKER_DATA_DIR: bind-mounting a host directory there for calibration
# persistence (exactly what ZimaOS/CasaOS does) shadowed this file and
# the app died at startup with FileNotFoundError. Named volumes hid the
# bug by copying image contents in on first mount; bind mounts do not.
# Keep shipped constants and user state in separate trees.
_CODEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "codebook_data.json")

_cached = None


def load_codebook(path=None, n_ids=None):
    """Loads the frozen, shipped codebook as a +/-1 float array.

    n_ids optionally truncates to the first n_ids codewords -- the full
    set is sized for the largest supported token library, and a smaller
    deployment pays nothing for the unused ones except a marginally
    tighter decode margin from having more candidates to rule out.
    """
    global _cached
    if path is None and _cached is not None:
        payload = _cached
    else:
        with open(path or _CODEBOOK_PATH) as f:
            payload = json.load(f)
        if path is None:
            _cached = payload
    bits = payload["codewords"]
    if n_ids is not None:
        if n_ids > len(bits):
            raise ValueError(
                f"requested {n_ids} ids but the frozen codebook only has "
                f"{len(bits)}; mint a larger one with tools/generate_codebook.py "
                f"(this invalidates all printed markers)")
        bits = bits[:n_ids]
    arr = np.asarray(bits, dtype=np.float64)
    return np.where(arr > 0, 1.0, -1.0)


def codebook_metadata(path=None):
    with open(path or _CODEBOOK_PATH) as f:
        return json.load(f)
