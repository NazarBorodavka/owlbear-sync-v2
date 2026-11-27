"""Detector voting-decimation tests.

detect() is about half the per-frame budget on the server, and almost all
of that is the vote accumulator, whose cost scales with pixel count. The
factor it is decimated by is therefore the single biggest speed dial in
the pipeline -- and the one most able to quietly break detection if it is
set past what the ring size supports.

So it is derived (Detector.vote_coarse) rather than fixed, and what is
pinned here is the invariant that matters: whatever px/mm the operator
dials in, the ring must still be at least MIN_VOTE_R_PX in the voting
image. A hard-coded factor satisfies that at one resolution and violates
it at every lower one.

The corpus replays are what established the 20px floor empirically; this
file guards the arithmetic that keeps the shipped config on the right side
of it.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tracker.vision.geometry import RingGeometry
from tracker.vision.detect import Detector, MIN_VOTE_R_PX

GEOM = RingGeometry()
# Spans well below the lowest play-area scale that still decodes and well
# above the highest a 4K sensor can deliver over a 1m table.
PX_MM_RANGE = [1.5, 2.0, 2.5, 3.0, 3.2, 3.5, 4.0, 5.0, 6.0, 8.0]


def test_vote_ring_never_decimated_below_the_floor():
    for ppm in PX_MM_RANGE:
        d = Detector(GEOM, ppm)
        c = d.vote_coarse
        assert c >= 1, f"px/mm={ppm}: non-positive decimation {c}"
        assert isinstance(c, int), f"px/mm={ppm}: {c!r} is not an int"
        # The whole point: decimation may never shrink the ring past the
        # radius refine() needs to find the real edge from the seed.
        vote_r = d.r_ref / c
        assert vote_r >= MIN_VOTE_R_PX or c == 1, (
            f"px/mm={ppm}: vote radius {vote_r:.1f}px is below the "
            f"{MIN_VOTE_R_PX}px floor at coarse={c}")
    print(f"  vote radius stays >= {MIN_VOTE_R_PX}px across "
          f"px/mm {PX_MM_RANGE[0]}-{PX_MM_RANGE[-1]}")


def test_decimation_grows_with_resolution():
    """Raising px/mm must not cost quadratically -- that was the bug a
    fixed factor had. Each step up in scale may only hold or increase the
    factor, never decrease it."""
    factors = [Detector(GEOM, p).vote_coarse for p in PX_MM_RANGE]
    for a, b, pa, pb in zip(factors, factors[1:], PX_MM_RANGE, PX_MM_RANGE[1:]):
        assert b >= a, (f"px/mm {pa}->{pb}: decimation went backwards "
                        f"{a}->{b}")
    assert factors[-1] > factors[0], "factor never grows with resolution"
    print("  decimation rises monotonically with px/mm: "
          + ", ".join(f"{p}->{c}" for p, c in zip(PX_MM_RANGE, factors)))


def test_explicit_factor_still_wins():
    """Pinning it must remain possible, or the corpus A/B that set the
    floor becomes unreproducible."""
    for c in (1, 2, 3, 4):
        assert Detector(GEOM, 3.5, coarse=c).vote_coarse == c
    # ... and it must differ from the derived value, else this proves
    # nothing.
    assert Detector(GEOM, 3.5).vote_coarse != 1
    print("  an explicit coarse= overrides the derived factor")


def test_shipped_scale_gets_the_measured_factor():
    """3.5 px/mm is what the calibrated play area actually runs at, and
    /3 is what the corpus measured as free there (detect 29.9 -> 15.7ms on
    `slow`, churn 0, published/frame 4.08 -> 4.10)."""
    assert Detector(GEOM, 3.5).vote_coarse == 3
    print("  the calibrated 3.5 px/mm resolves to the measured coarse=3")


if __name__ == "__main__":
    test_vote_ring_never_decimated_below_the_floor()
    test_decimation_grows_with_resolution()
    test_explicit_factor_still_wins()
    test_shipped_scale_gets_the_measured_factor()
    print("\nall detector decimation tests passed")
