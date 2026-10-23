"""Exercises the validation-corpus replay harness on synthetic footage.

tools/replay.py is how real recorded footage will be scored, so the
harness itself needs to be known-good BEFORE it is pointed at a real
recording -- otherwise a harness bug is indistinguishable from a tracker
regression, and the whole point of the corpus is lost.

So: render a sequence whose contents are known exactly, encode it to a
real video file, replay it through the harness, and check the harness
reports what is actually in the recording. The encode is deliberately
included rather than feeding frames in directly, because codec
compression is part of what real footage will have been through.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import tracker.vision.simulate as sim
from tracker.vision.calibration.session import CalibrationSession
from tracker.vision.codebook import load_codebook
from tracker.vision.geometry import RingGeometry

GEOM = RingGeometry()
CB = load_codebook(n_ids=6)
PX_MM = 3.5
N_TOKENS = 6          # the stated requirement: 6 active tokens
N_FRAMES = 90
FRAME_H, FRAME_W = 520, 920


def _write_sequence(path, calibration_dir):
    """Renders N_TOKENS tokens on smooth paths, encodes to an mp4, and
    sets up a matching play-area calibration."""
    import cv2

    sess = CalibrationSession(calibration_dir, geom=GEOM, codebook=CB)
    # Corners of the whole frame: the synthetic footage is already
    # top-down, so the play area is the frame and rectify() is a no-op
    # scale. Real footage gets a real perspective homography here.
    corners = np.array([[0, 0], [FRAME_W, 0], [FRAME_W, FRAME_H], [0, FRAME_H]],
                       dtype=np.float64)
    sess.calibrate_playarea(corners, width_mm=FRAME_W / PX_MM,
                            height_mm=FRAME_H / PX_MM, px_per_mm=PX_MM,
                            already_undistorted=True)

    rng = np.random.default_rng(11)
    phases = rng.uniform(0, 2 * np.pi, N_TOKENS)
    writer = None
    for f in range(N_FRAMES):
        t = f / N_FRAMES
        tokens = []
        for i in range(N_TOKENS):
            # Slow lissajous drifts: continuous, well separated, and
            # inside the frame with room for the marker radius.
            cx = 130 + i * 130 + 28 * np.sin(2 * np.pi * t + phases[i])
            cy = 150 + 90 * np.sin(4 * np.pi * t + phases[i]) + 60 * (i % 2)
            tokens.append({"id": i, "x": float(cx), "y": float(cy),
                           "rot_deg": float((f * 3 + i * 47) % 360)})
        img = sim.render_frame(tokens, GEOM, CB, PX_MM, (FRAME_H, FRAME_W),
                               rng, noise=0.02)
        u8 = np.clip(img * 255.0 if img.max() <= 1.5 else img, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        if writer is None:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     15.0, (FRAME_W, FRAME_H))
            assert writer.isOpened(), "no mp4v encoder available"
        writer.write(bgr)
    writer.release()
    assert os.path.getsize(path) > 0


def test_replay_scores_a_known_recording():
    from tools.replay import replay, verdict

    d = tempfile.mkdtemp()
    try:
        video = os.path.join(d, "seq.mp4")
        caldir = os.path.join(d, "calibration")
        _write_sequence(video, caldir)

        r = replay(video, caldir, n_ids=6, progress=False)

        assert r["frames"] >= N_FRAMES - 2, \
            f"harness read {r['frames']} of {N_FRAMES} frames"
        assert r["source_fps"] == 15.0, r["source_fps"]
        assert abs(r["px_per_mm"] - PX_MM) < 1e-6, r["px_per_mm"]

        # The harness must find what is genuinely in the recording.
        assert set(r["distinct_ids_seen"]) == set(range(N_TOKENS)), (
            f"expected ids {list(range(N_TOKENS))}, harness reported "
            f"{r['distinct_ids_seen']}")
        assert r["max_ids_per_frame"] == N_TOKENS, (
            f"never saw all {N_TOKENS} tokens at once "
            f"(max {r['max_ids_per_frame']})")

        # And it must not invent failures on clean footage.
        assert r["identity_flips"] == 0, r["flip_detail"]
        assert r["phantom_tracks"] == 0, r["phantom_tracks"]
        assert not r["id_churn"], r["id_churn"]

        ok, lines = verdict(r, target_fps=15.0, n_expected=N_TOKENS)
        assert ok, "verdict failed on clean footage:\n" + "\n".join(lines)

        print(f"  {r['frames']} frames, {N_TOKENS} tokens, ids "
              f"{r['distinct_ids_seen']}")
        print(f"  {r['timing_ms']['mean']}ms/frame mean, "
              f"{r['timing_ms']['p95']}ms p95 -> "
              f"{r['sustainable_fps']}fps sustainable")
        print(f"  coasting {r['coasting_fraction']:.1%}, "
              f"0 flips, 0 phantoms, 0 churn")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_replay_refuses_without_a_play_area():
    """Replay has no meaningful fallback for "what scale is this", so it
    must refuse rather than silently report positions in pixels."""
    from tools.replay import replay

    d = tempfile.mkdtemp()
    try:
        video = os.path.join(d, "seq.mp4")
        caldir = os.path.join(d, "calibration")
        _write_sequence(video, os.path.join(d, "real_cal"))
        os.makedirs(caldir, exist_ok=True)   # present but empty
        try:
            replay(video, caldir, n_ids=6, progress=False)
        except SystemExit as e:
            assert "play-area" in str(e), e
        else:
            raise AssertionError("replay ran without a play-area calibration")
        print("  refuses to score footage with no play-area calibration")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_annotated_output_is_written():
    """The overlay video is the eyeball check on a real recording, so it
    needs to actually be produced."""
    from tools.replay import replay

    d = tempfile.mkdtemp()
    try:
        video = os.path.join(d, "seq.mp4")
        caldir = os.path.join(d, "calibration")
        out = os.path.join(d, "annotated.mp4")
        _write_sequence(video, caldir)
        replay(video, caldir, n_ids=6, limit=20, annotate=out, progress=False)
        assert os.path.exists(out) and os.path.getsize(out) > 0
        print(f"  annotated overlay written ({os.path.getsize(out)} bytes)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _clean_report():
    """A report representing a recording with nothing wrong with it."""
    return {
        "identity_flips": 0, "flip_detail": [], "phantom_tracks": 0,
        "id_churn": {}, "sustainable_fps": 40.0, "coasting_fraction": 0.01,
        "max_ids_per_frame": N_TOKENS,
        "timing_ms": {"mean": 25.0, "p50": 24.0, "p95": 30.0, "max": 40.0},
    }


def test_verdict_flags_every_failure_mode():
    """The scorer is the part that decides whether a recording is good, so
    it must actually catch each failure -- not just stay quiet on clean
    footage. Each case below is a failure this project has already seen in
    v1, injected one at a time so no single check can mask another.
    """
    from tools.replay import verdict

    ok, _ = verdict(_clean_report(), 15.0, N_TOKENS)
    assert ok, "clean report should pass"

    cases = {
        "identity flip": {"identity_flips": 2,
                          "flip_detail": [{"track": 1, "ids": [3, 5]}]},
        "phantom track": {"phantom_tracks": 1},
        "id churn": {"id_churn": {4: 3}},
        "too slow": {"sustainable_fps": 9.0},
        "missing tokens": {"max_ids_per_frame": N_TOKENS - 2},
    }
    for name, override in cases.items():
        r = _clean_report() | override
        ok, lines = verdict(r, 15.0, N_TOKENS)
        assert not ok, f"scorer did NOT flag {name}: " + " | ".join(lines)
        assert any("FAIL" in l for l in lines), name

    # Heavy coasting is a warning, not a failure: the filter covering a
    # brief detector gap is the system working as designed, so it must
    # not fail an otherwise-good recording.
    r = _clean_report() | {"coasting_fraction": 0.40}
    ok, lines = verdict(r, 15.0, N_TOKENS)
    assert ok, "heavy coasting should warn, not fail"
    assert any("WARN" in l and "extrapolated" in l for l in lines), lines

    # expect_tokens is optional; omitting it must not silently pass a
    # recording where tokens were missing, it just cannot know.
    r = _clean_report() | {"max_ids_per_frame": 1}
    ok, lines = verdict(r, 15.0, None)
    assert ok and not any("expected tokens" in l for l in lines)

    print(f"  scorer flags all {len(cases)} failure modes, "
          f"warns (not fails) on heavy coasting")


if __name__ == "__main__":
    test_verdict_flags_every_failure_mode()
    test_replay_scores_a_known_recording()
    test_replay_refuses_without_a_play_area()
    test_annotated_output_is_written()
    print("\nall replay-harness tests passed")
