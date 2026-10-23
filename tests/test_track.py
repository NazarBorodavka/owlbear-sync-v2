"""Tracker behaviour tests, written against v1's actual field failures.

Each test names the v1 symptom it is guarding against, so a future
regression is recognisable as "that bug is back" rather than just a
failing assertion.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tracker.vision.geometry import RingGeometry
from tracker.vision.track import Tracker

GEOM = RingGeometry()
PX_MM = 3.2


def mk(x, y):
    return {"x": x, "y": y, "px_per_mm": PX_MM}


def test_identity_survives_occlusion_gap():
    """v1 symptom: a token that was briefly hidden lost its id and had to
    be re-identified from scratch."""
    tr = Tracker(GEOM, PX_MM)
    t = 0.0
    for _ in range(5):                       # establish
        tr.step([mk(100, 100)], [(3, 0.5)], now=t); t += 1 / 15
    assert tr.step([mk(100, 100)], [(3, 0.5)], now=t)[0].marker_id == 3
    # 15 frames fully occluded: nothing detected at all
    for _ in range(15):
        t += 1 / 15
        tr.step([], [], now=t)
    # reappears, but too blurred to decode (id=None)
    t += 1 / 15
    out = tr.step([mk(104, 101)], [(None, 0.0)], now=t)
    assert len(out) == 1 and out[0].marker_id == 3, \
        f"identity lost across occlusion: {[(o.marker_id) for o in out]}"
    print("  identity survives a 1s occlusion gap + undecodable reacquisition")


def test_undecodable_blob_never_becomes_a_token():
    """v1 symptom: lamp bases, bottle caps and motion-blur artefacts became
    visible 'phantom' tokens (the trail of red ghost disks)."""
    tr = Tracker(GEOM, PX_MM)
    t = 0.0
    for _ in range(40):
        tr.step([mk(300 + t * 20, 200)], [(None, 0.0)], now=t)
        t += 1 / 15
    assert tr.step([mk(400, 200)], [(None, 0.0)], now=t) == [], \
        "an object that never decoded was reported as a token"
    print("  blobs that never decode are never reported (no phantoms)")


def test_two_tokens_crossing_do_not_swap():
    """v1 symptom: token 2 'stole' token 3's id when they passed close."""
    tr = Tracker(GEOM, PX_MM)
    t = 0.0
    for _ in range(6):                       # establish both
        tr.step([mk(100, 200), mk(400, 200)], [(1, 0.5), (5, 0.5)], now=t)
        t += 1 / 15
    # they approach, cross, and separate; decoding fails mid-crossing
    for k in range(14):
        xa, xb = 100 + k * 20, 400 - k * 20
        dec = [(1, 0.4), (5, 0.4)] if k < 4 or k > 9 else [(None, 0.0)] * 2
        tr.step([mk(xa, 200), mk(xb, 200)], dec, now=t)
        t += 1 / 15
    out = {o.marker_id: o.pos[0] for o in tr.step(
        [mk(380, 200), mk(120, 200)], [(1, 0.4), (5, 0.4)], now=t)}
    assert set(out) == {1, 5}, f"ids corrupted while crossing: {out}"
    assert out[1] > out[5], f"ids swapped during crossing: {out}"
    print("  two tokens crossing keep their own ids (no id theft)")


def test_no_duplicate_identity():
    """v1 symptom: two tracks simultaneously claimed one id, and the
    display flickered between them."""
    tr = Tracker(GEOM, PX_MM)
    t = 0.0
    for _ in range(6):
        tr.step([mk(100, 100)], [(2, 0.6)], now=t); t += 1 / 15
    for _ in range(6):
        out = tr.step([mk(100, 100), mk(400, 400)], [(2, 0.6), (2, 0.3)], now=t)
        t += 1 / 15
    ids = [o.marker_id for o in out]
    assert len(ids) == len(set(ids)), f"same id held by two tracks: {ids}"
    assert 2 in ids
    print("  one identity is never held by two tracks at once")


def test_fast_motion_keeps_track():
    """v1 symptom: 'moving faster makes it re-identify' -- the track was
    lost because the catchment could not keep up."""
    tr = Tracker(GEOM, PX_MM)
    t, x = 0.0, 100.0
    for _ in range(5):
        tr.step([mk(x, 200)], [(4, 0.5)], now=t); t += 1 / 15; x += 4
    keys = set()
    for _ in range(20):                      # ~55px/frame = fast drag
        out = tr.step([mk(x, 200)], [(4, 0.5)], now=t)
        if out:
            keys.add(out[0].key)
        t += 1 / 15
        x += 55
    assert len(keys) == 1, f"track was broken and recreated {len(keys)} times"
    print("  a fast drag stays on one continuous track")


if __name__ == "__main__":
    for fn in [test_identity_survives_occlusion_gap,
               test_undecodable_blob_never_becomes_a_token,
               test_two_tokens_crossing_do_not_swap,
               test_no_duplicate_identity,
               test_fast_motion_keeps_track]:
        fn()
    print("\nall tracker tests passed")
