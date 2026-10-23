"""Guards the frozen codebook against silent drift from printed markers.

The physical markers are printed from tracker/vision/codebook_data.json. If that
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
                                     periodic_xcorr_peak, spectral_profile,
                                     quality)
from tracker.vision.decode import MIN_MARGIN, MIN_PEAK

# The fingerprint of the codebook the markers were printed from. If this
# assertion fails, either someone regenerated the codebook (in which case
# ALL markers need reprinting and this constant needs updating), or the
# data file was corrupted.
EXPECTED_N_CHIPS = 20
EXPECTED_N_IDS = 6
EXPECTED_WORST_AUTO = 0.60
EXPECTED_WORST_CROSS = 0.20

# Energy a codeword may put in harmonics 1-2 (the shading subspace) and in
# any single harmonic. These are not cosmetic: optimising separation with
# no spectral constraint picks ten-black-then-ten-white, which IS a
# brightness ramp. A blank ring under a mild lighting gradient was
# measured correlating 0.822 with such a codeword and decoding as a
# confident token.
MAX_LOW_FREQ_ENERGY = 0.35
MAX_SINGLE_HARMONIC = 0.60


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

    quality() reports the worst case; this walks all 15 pairs x 20
    rotations to confirm nothing individually violates the peak/margin
    gates. This is the property that makes a confident WRONG id
    impossible rather than merely unlikely.
    """
    cb = load_codebook()
    n_ids, n_chips = cb.shape
    for i in range(n_ids):
        for j in range(i + 1, n_ids):
            xc = periodic_xcorr_peak(cb[i], cb[j])
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

    This is the drift tripwire. If build_codebook's search is refactored,
    this fails -- telling us the generator and the printed artwork have
    parted ways BEFORE anyone reprints markers from a generator that no
    longer agrees with the decoder.

    The search is exhaustive and deterministic, so unlike the original
    random one this no longer depends on numpy's RNG staying stable
    across versions -- which was itself a live drift risk.
    """
    from tracker.vision.codebook import build_codebook, to_bits
    meta = codebook_metadata()
    gen = meta["generated_with"]
    rebuilt = build_codebook(meta["n_chips"], meta["n_ids"],
                             max_cross=gen["max_cross"],
                             low_cap=gen["low_cap"], top_cap=gen["top_cap"],
                             harm_cap=gen["harm_cap"],
                             auto_cap=gen["auto_cap"])
    assert to_bits(rebuilt) == meta["codewords"], (
        "build_codebook no longer reproduces the committed codebook. The "
        "PRINTED markers follow the committed file, so do NOT regenerate "
        "to make this pass -- investigate what changed in the generator.")


def test_frozen_codebook_cannot_be_confused_with_illumination():
    """No codeword may look like a lighting gradient.

    Harmonics 1-2 are the shading subspace: a brightness ramp across a
    token is harmonic 1 almost exactly, and that is what a projector
    edge, a shadow or vignetting puts on the marker. A codeword with its
    energy there is competing with illumination rather than with other
    markers.

    This guards a failure that a pure separation search walks straight
    into: the MOST separable 20-chip codeword is ten black chips then ten
    white ones, and a BLANK ring under a mild gradient was measured
    scoring 0.822 against it -- a confident id for an object with no code
    on it at all. Bounding the largest single harmonic too keeps a
    codeword's signal spread across the band, so no one narrowband
    disturbance can take all of it.
    """
    cb = load_codebook()
    for i, c in enumerate(cb):
        low, top, peak = spectral_profile(c)
        assert low <= MAX_LOW_FREQ_ENERGY + 1e-9, (
            f"id {i} puts {low:.2f} of its energy in harmonics 1-2; it is "
            f"confusable with a lighting gradient")
        assert peak <= MAX_SINGLE_HARMONIC + 1e-9, (
            f"id {i} concentrates {peak:.2f} of its energy in one harmonic")


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


def test_codebook_does_not_live_in_a_mountable_data_directory():
    """The codebook must not sit where user state gets bind-mounted.

    It was briefly in tracker/data/, which is the default
    TRACKER_DATA_DIR. Bind-mounting a host directory there for
    calibration persistence -- exactly what ZimaOS/CasaOS does -- shadowed
    the codebook and the app died at startup with FileNotFoundError.
    Named Docker volumes masked it, because an empty named volume is
    seeded from the image on first mount; a bind mount is not.

    So: shipped constants and mutable user state must stay in separate
    trees, and this asserts the codebook is on the code side of that line.
    """
    import tracker.vision.codebook as cbmod
    path = cbmod._CODEBOOK_PATH
    assert os.path.isfile(path), path
    parts = os.path.normpath(path).split(os.sep)
    assert "data" not in parts[:-1], (
        f"codebook lives under a 'data' directory ({path}); a bind mount "
        f"for persistence would shadow it and break startup")

    # And the app's default data dir must not contain it either, whatever
    # that default happens to be.
    from tracker.web import app as appmod
    default_data_dir = os.environ.get(
        "TRACKER_DATA_DIR", os.path.join(appmod.BASE_DIR, "data"))
    assert not os.path.abspath(path).startswith(
        os.path.abspath(default_data_dir) + os.sep), (
        f"codebook {path} is inside the default data dir {default_data_dir}")


if __name__ == "__main__":
    test_frozen_codebook_shape_is_unchanged()
    print(f"  shape unchanged: {EXPECTED_N_IDS} codewords x "
          f"{EXPECTED_N_CHIPS} chips")
    test_frozen_codebook_is_bipolar_and_balanced()
    print("  all codewords bipolar and exactly zero-mean")
    test_frozen_codebook_meets_correlation_quality_bar()
    print(f"  quality bar met: worst auto={EXPECTED_WORST_AUTO}, "
          f"cross={EXPECTED_WORST_CROSS}")
    test_frozen_codebook_separates_every_pair_at_every_rotation()
    n_pairs = EXPECTED_N_IDS * (EXPECTED_N_IDS - 1) // 2
    print(f"  all {n_pairs} pairs separated at all {EXPECTED_N_CHIPS} rotations")
    test_frozen_codebook_cannot_be_confused_with_illumination()
    print("  no codeword is confusable with a lighting gradient")
    test_regenerating_reproduces_the_committed_codebook()
    print("  generator still reproduces the committed codebook bit-for-bit")
    test_loading_is_fast_enough_for_container_startup()
    print("  loads fast (no regeneration at startup)")
    test_codebook_does_not_live_in_a_mountable_data_directory()
    print("  codebook is outside any mountable data directory")
    test_requesting_more_ids_than_exist_fails_loudly()
    test_runtime_rejects_a_chip_count_the_markers_do_not_have()
    print("  mismatched requests rejected loudly")
    print("\nall frozen-codebook tests passed")
