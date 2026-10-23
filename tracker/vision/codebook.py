"""Codeword design for the angular ring code.

v1 picked codewords by maximising Hamming distance between ALIGNED
codewords. That is the wrong objective for a rotation-invariant marker
read by correlation: it says nothing about how a codeword correlates with
a ROTATED copy of itself or of another codeword, and since the marker's
rotation on the table is arbitrary, that is exactly where confusions come
from. v1 then needed a separate sync word to pin down rotation, which
became its own brittle failure mode.

The objective here is the quantity the matched filter actually measures:
the worst-case periodic cross-correlation between any two codewords at
ANY relative rotation. Optimising that directly removes the need for a
sync word at all -- rotation falls out of the correlation peak.

Two corrections to that objective, both found by measurement and both
load-bearing:

SIGNED, NOT ABSOLUTE. The first version of this module scored pairs on
|correlation|. The decoder does not: it takes cc.max() per codeword and
compares those peaks, so a strongly ANTI-correlated pair is not
confusable -- a correlation of -0.6 means "definitely not this one".
Scoring on magnitude needlessly discarded half the design space and
capped the achievable separation at 4 chips out of 20. On the signed
metric the same 20-chip ring supports 8.

SPECTRALLY CONSTRAINED. Maximising separation ALONE produces degenerate
codewords. The single most separable 20-chip marker is ten black chips
followed by ten white ones -- and that is a brightness ramp, which is
also what a lighting gradient, a shadow edge or vignetting looks like.
Measured: a BLANK ring under a mild gradient correlated 0.822 with such a
codeword and decoded as a confident token. The mirror failure is a
near-alternating codeword, whose signal lives entirely at the chip scale
and is the first thing defocus destroys. So candidates are filtered on
their angular power spectrum before separation is considered at all.

The search is exhaustive and deterministic: it enumerates every balanced
necklace of the given length, so there is no RNG whose implementation
could drift across numpy versions and silently desync the decoder from
markers already printed. That drift risk was the original reason the
codebook had to be frozen to a committed file; the file is still frozen,
but the generator behind it no longer depends on random sampling.
"""
import itertools

import numpy as np


def periodic_xcorr_peak(a, b, exclude_zero_shift=None):
    """Max circular cross-correlation of two +/-1 sequences, normalised.

    SIGNED, matching what the decoder compares. When a and b are the same
    sequence this excludes zero shift, so the result is the worst
    autocorrelation SIDELOBE -- the thing that decides whether a codeword
    can be confused with a rotation of itself.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    if exclude_zero_shift is None:
        exclude_zero_shift = np.array_equal(a, b)
    cc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n) / n
    if exclude_zero_shift:
        cc = cc.copy()
        cc[0] = -np.inf
    return float(cc.max())


def spectral_profile(c):
    """(low, top, peak) fractions of a codeword's angular power spectrum.

    low  -- harmonics 1 and 2, the shading subspace. A brightness ramp
            across the token is harmonic 1 almost exactly, so a codeword
            with energy here is confusable with illumination itself
            rather than with ink.
    top  -- the Nyquist harmonic, i.e. chip-scale alternation. This is
            what defocus and low px/mm attenuate first.
    peak -- the largest single harmonic. Bounding it keeps a codeword's
            signal spread across the band, so no single narrowband
            disturbance can take all of it at once.

    DC is excluded: balanced codewords have none, which is what makes the
    decode immune to a flat brightness offset.
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    p = np.abs(np.fft.rfft(c))[1:n // 2 + 1] ** 2
    p = p / p.sum()
    return float(p[0] + p[1]), float(p[-1]), float(p.max())


def balanced_necklaces(n):
    """Every balanced +/-1 sequence of length n, one per rotation class.

    All rotations of a sequence are the SAME physical marker -- the
    marker's orientation on the table is arbitrary -- so only one
    representative of each rotation class is a distinct design. Returned
    in sorted order, which is what makes the search deterministic.
    """
    if n % 2:
        raise ValueError("n_chips must be even so codewords can be balanced")
    seen = set()
    for ones in itertools.combinations(range(n), n // 2):
        v = [-1] * n
        for i in ones:
            v[i] = 1
        seen.add(min(tuple(v[k:] + v[:k]) for k in range(n)))
    return np.array(sorted(seen), dtype=np.float64)


def build_codebook(n_chips=20, n_ids=6, max_cross=0.20, low_cap=0.35,
                   top_cap=0.30, harm_cap=0.60, auto_cap=0.60):
    """Exhaustive deterministic search for n_ids codewords.

    Candidates are first filtered on the physical constraints (balance is
    structural, the rest spectral), then a depth-first search finds the
    lexicographically first set whose every pair stays below max_cross at
    every relative rotation.

    max_cross is the design's headline property: 0.20 at 20 chips means
    no two markers come within 8 chips of each other under any rotation.
    It is a target, not an optimum discovered at runtime -- the optimum
    was established offline by searching every achievable separation
    level, and baking the resulting value in keeps generation fast and
    its output stable.

    Raises if no such set exists, rather than quietly returning fewer
    codewords than asked for.
    """
    pool = balanced_necklaces(n_chips)
    # Same quantities as spectral_profile() and periodic_xcorr_peak(), but
    # over the whole pool at once -- the per-candidate versions are clearer
    # to read and are what the tests check against, while this keeps
    # regeneration to a couple of seconds instead of half a minute.
    spec = np.fft.rfft(pool, axis=1)
    power = np.abs(spec[:, 1:n_chips // 2 + 1]) ** 2
    power = power / power.sum(axis=1, keepdims=True)
    low = power[:, 0] + power[:, 1]
    top = power[:, -1]
    harm = power.max(axis=1)
    auto = (np.fft.irfft(spec * np.conj(spec), n_chips, axis=1)
            / n_chips)[:, 1:].max(axis=1)
    keep = np.flatnonzero((low <= low_cap + 1e-9) & (top <= top_cap + 1e-9)
                          & (harm <= harm_cap + 1e-9)
                          & (auto <= auto_cap + 1e-9))
    sub = pool[keep]
    if len(sub) < n_ids:
        raise ValueError(
            f"only {len(sub)} of {len(pool)} {n_chips}-chip markers survive the "
            f"spectral constraints; cannot pick {n_ids}")

    # Adjacency as bitmasks: bit j of adj[i] means i and j can coexist.
    spectrum = spec[keep]
    adj = []
    for i in range(len(sub)):
        cc = np.fft.irfft(spectrum[i][None, :] * np.conj(spectrum),
                          n_chips, axis=1) / n_chips
        ok = cc.max(axis=1) <= max_cross + 1e-9
        ok[i] = False
        adj.append(int.from_bytes(np.packbits(ok, bitorder="little").tobytes(),
                                  "little"))

    def extend(chosen, candidates):
        if len(chosen) == n_ids:
            return chosen
        if len(chosen) + int(bin(candidates).count("1")) < n_ids:
            return None
        rest = candidates
        while rest:
            lowbit = rest & -rest
            v = lowbit.bit_length() - 1
            rest ^= lowbit
            got = extend(chosen + [v], rest & adj[v])
            if got:
                return got
        return None

    sol = extend([], (1 << len(sub)) - 1)
    if sol is None:
        raise ValueError(
            f"no {n_ids} markers of {n_chips} chips stay below "
            f"max_cross={max_cross} pairwise; loosen it or add chips")
    return sub[sol]


def quality(cb):
    """(worst autocorrelation sidelobe, worst pairwise cross-correlation)."""
    worst_auto = max(periodic_xcorr_peak(c, c, True) for c in cb)
    worst_cross = -1.0
    for i in range(len(cb)):
        for j in range(i + 1, len(cb)):
            worst_cross = max(worst_cross, periodic_xcorr_peak(cb[i], cb[j]))
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
