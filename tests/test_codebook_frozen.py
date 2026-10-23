"""Guards the frozen codebook against silent drift from printed markers.

The physical markers are printed from tracker/data/codebook.json. If that
file's contents ever change, every marker already printed becomes
undecodable -- silently, with tokens simply failing to be recognised and
no error pointing at the cause. v1 lost significant debugging time to
exactly this shape of problem (a print run that was mirrored relative to
what the decoder expected).

So these tests assert the committed codebook is BIT-FOR-BIT what it was
when the markers were printed, and that it still meets the correlation
quality bar the decoder's thresholds assume.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tracker.vision.codebook import (codebook_metadata, load_codebook,
                                     periodic_xcorr_max, quality)
from tracker.vision.decode import MIN_MARGIN, MIN_PEAK

# The fingerprint of the codebook the markers were printed from. If this
# assertion fails, either someone regenerated the codebook (in which case
# ALL markers need reprinting and this constant needs updating), or the
# data file was corrupted.
EXPECTED_N_CHIPS = 20
EXPECTED_N_IDS = 24
EXPECTED_WORST_AUTO = 0.40
EXPECTED_WORST_CROSS = 0.60


def test_frozen_codebook_shape_is_unchanged():
    meta = codebook_metadata()
    assert meta["n_chips"] == EXPECTED_N_CHIPS
    assert meta["n_ids"] == EXPECTED_N_IDS
    cb = load_codebook()
    assert cb.shape == (EXPECTED_N_IDS, EXPECTED_N_CHIPS)


def test_frozen_codebook_is_bipolar_and_balanced():
    """Codewords must be +/-1 and zero-mean.

    Zero-mean is what makes decoding immune to a brightness offset: the
    correlation of a DC-shifted strip against a zero-mean codeword is
    unchanged by the shift. A codeword that drifted off balance would
    decode fine in even lighting and degrade under the projector.
    """
    cb = load_codebook()
    assert set(np.unique(cb).tolist()) == {-1.0, 1.0}
    sums = cb.sum(axis=1)
    # 20 chips is even, so perfectly balanced means a sum of exactly 0.
    assert np.all(sums == 0), f"unbalanced codewords at ids {np.flatnonzero(sums)}"


def test_frozen_codebook_meets_correlation_quality_bar():
    """The decode thresholds are only safe for a codebook this good."""
    cb = load_codebook()
    worst_auto, worst_cross = quality(cb)
    assert abs(worst_auto - EXPECTED_WORST_AUTO) < 1e-6, worst_auto
    assert abs(worst_cross - EXPECTED_WORST_CROSS) < 1e-6, worst_cross
    # The real requirement behind those numbers: a correct match must beat
    # the best wrong match by more than the margin gate, otherwise the
    # decoder would be relying on noise to break the tie.
    assert 1.0 - worst_cross > MIN_MARGIN
    assert 1.0 - worst_auto > MIN_MARGIN


def test_frozen_codebook_separates_every_pair_at_every_rotation():
    """Exhaustive check of the property the decoder actually depends on.

    quality() reports the worst case; this walks all 276 pairs x 20
    rotations to confirm nothing individually violates the peak/margin
    gates. This is the property that makes a confident WRONG id
    impossible rather than merely unlikely.
    """
    cb = load_codebook()
    n_ids, n_chips = cb.shape
    for i in range(n_ids):
        for j in range(i + 1, n_ids):
            xc = periodic_xcorr_max(cb[i], cb[j])
            assert xc <= EXPECTED_WORST_CROSS + 1e-9, (
                f"ids {i},{j} correlate at {xc:.3f} under some rotation -- "
                f"they could be confused for each other")
    for i in range(n_ids):
        # A codeword must not correlate strongly with a rotation of
        # ITSELF, or the decoder would resolve identity correctly but
        # report the wrong orientation.
        shifts = [float(np.dot(cb[i], np.roll(cb[i], s)) / n_chips)
                  for s in range(1, n_chips)]
        assert max(shifts) <= EXPECTED_WORST_AUTO + 1e-9, (
            f"id {i} has a rotational self-ambiguity of {max(shifts):.3f}")


def test_regenerating_reproduces_the_committed_codebook():
    """The generator must still produce exactly what is committed.

    This is the drift tripwire. If numpy's RNG changes across versions, or
    build_codebook's search is refactored, this fails -- telling us the
    generator and the printed artwork have parted ways BEFORE anyone
    reprints markers from a generator that no longer agrees with the
    decoder.
    """
    from tracker.vision.codebook import build_codebook, to_bits
    meta = codebook_metadata()
    gen = meta["generated_with"]
    rebuilt = build_codebook(meta["n_chips"], meta["n_ids"],
                             seed=gen["seed"],
                             max_attempts=gen["max_attempts"],
                             auto_thr=gen["auto_thr"],
                             cross_thr=gen["cross_thr"])
    assert to_bits(rebuilt) == meta["codewords"], (
        "build_codebook no longer reproduces the committed codebook. The "
        "PRINTED markers follow the committed file, so do NOT regenerate "
        "to make this pass -- investigate what changed in the generator.")


def test_loading_is_fast_enough_for_container_startup():
    """Loading must not reintroduce the ~3.9s regeneration cost."""
    import time
    import tracker.vision.codebook as cbmod
    cbmod._cached = None  # defeat the module cache so this measures real work
    t0 = time.perf_counter()
    load_codebook()
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"codebook load took {elapsed:.2f}s"


def test_requesting_more_ids_than_exist_fails_loudly():
    """Silently returning fewer tokens than asked for would surface as a
    mysterious cap on token count, so it must raise."""
    meta = codebook_metadata()
    try:
        load_codebook(n_ids=meta["n_ids"] + 1)
    except ValueError as e:
        assert "only has" in str(e), e
    else:
        raise AssertionError("oversized n_ids silently succeeded")


def test_runtime_rejects_a_chip_count_the_markers_do_not_have():
    """Constructing a runtime with mismatched n_chips cannot silently
    produce a decoder that disagrees with every printed marker."""
    from tracker.runtime import TrackerRuntime
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        try:
            TrackerRuntime(d, n_chips=16)
        except ValueError as e:
            assert "frozen codebook" in str(e), e
        else:
            raise AssertionError("mismatched n_chips silently succeeded")


if __name__ == "__main__":
    test_frozen_codebook_shape_is_unchanged()
    print("  shape unchanged: 24 codewords x 20 chips")
    test_frozen_codebook_is_bipolar_and_balanced()
    print("  all codewords bipolar and exactly zero-mean")
    test_frozen_codebook_meets_correlation_quality_bar()
    print(f"  quality bar met: worst auto={EXPECTED_WORST_AUTO}, "
          f"cross={EXPECTED_WORST_CROSS}")
    test_frozen_codebook_separates_every_pair_at_every_rotation()
    print("  all 276 pairs separated at all 20 rotations")
    test_regenerating_reproduces_the_committed_codebook()
    print("  generator still reproduces the committed codebook bit-for-bit")
    test_loading_is_fast_enough_for_container_startup()
    print("  loads fast (no regeneration at startup)")
    test_requesting_more_ids_than_exist_fails_loudly()
    test_runtime_rejects_a_chip_count_the_markers_do_not_have()
    print("  mismatched requests rejected loudly")
    print("\nall frozen-codebook tests passed")
