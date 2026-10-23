"""TrackerRuntime integration test: exercises the real rectify -> pipeline
-> publish path by injecting synthetic frames directly via
process_frame(), with no RTSP capture or background thread involved.
Requires cv2 (play-area calibration does) -- skips cleanly if absent.
"""
import sys, os, shutil
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from tracker.runtime import TrackerRuntime
from tracker.vision import simulate as sim

TEST_DIR = "/tmp/v2_runtime_test"


def test_process_frame_reports_tokens_with_correct_mm_positions():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    rt = TrackerRuntime(TEST_DIR, n_ids=8)
    px_mm = 3.5
    width_mm, height_mm = 257.0, 143.0   # matches a 900x500px raw frame at this scale
    corners = np.array([[0, 0], [900, 0], [900, 500], [0, 500]], dtype=np.float64)
    rt.session.calibrate_playarea(corners, width_mm, height_mm, px_mm,
                                  already_undistorted=True)
    rt.prepare_pipeline()

    rng = np.random.default_rng(3)
    true_x_mm, true_y_mm = 120.0, 80.0
    for _ in range(4):   # a couple of frames so the track confirms (confirm_hits)
        tokens = [{"id": 2, "x": true_x_mm * px_mm, "y": true_y_mm * px_mm,
                  "rot_deg": 30.0}]
        raw = sim.render_frame(tokens, rt.geom, rt.codebook, px_mm, (500, 900), rng,
                               noise=0.02)
        ok = rt.process_frame(raw)
        assert ok, rt.latest()["error"]

    out = rt.latest()
    assert out["play_area_mm"] == (width_mm, height_mm)
    assert len(out["tokens"]) == 1, f"expected 1 reported token, got {out['tokens']}"
    t = out["tokens"][0]
    assert t["marker_id"] == 2
    err_mm = np.hypot(t["x_mm"] - true_x_mm, t["y_mm"] - true_y_mm)
    assert err_mm < 1.0, f"position error {err_mm:.2f}mm"
    print(f"  TrackerRuntime.process_frame reports id={t['marker_id']} at "
          f"({t['x_mm']:.1f},{t['y_mm']:.1f})mm, true ({true_x_mm},{true_y_mm})mm "
          f"-- {err_mm:.2f}mm error")


def test_process_frame_handles_bgr_input_like_a_real_camera():
    """cv2.VideoCapture hands back BGR frames, not the grayscale this
    module tests with elsewhere -- verifies the 3-channel path actually
    works, not just that it does not crash."""
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    rt = TrackerRuntime(TEST_DIR, n_ids=8)
    px_mm = 3.5
    corners = np.array([[0, 0], [900, 0], [900, 500], [0, 500]], dtype=np.float64)
    rt.session.calibrate_playarea(corners, 257.0, 143.0, px_mm, already_undistorted=True)
    rt.prepare_pipeline()

    rng = np.random.default_rng(9)
    tokens = [{"id": 5, "x": 100 * px_mm, "y": 70 * px_mm, "rot_deg": 10.0}]
    gray = sim.render_frame(tokens, rt.geom, rt.codebook, px_mm, (500, 900), rng, noise=0.02)
    bgr = np.stack([gray * 255, gray * 255, gray * 255], axis=-1).astype(np.uint8)
    for _ in range(4):
        assert rt.process_frame(bgr), rt.latest()["error"]
    out = rt.latest()
    assert len(out["tokens"]) == 1 and out["tokens"][0]["marker_id"] == 5
    print("  process_frame correctly handles 3-channel BGR input (as cv2.VideoCapture yields)")


def test_process_frame_before_calibration_fails_clearly():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    rt = TrackerRuntime(TEST_DIR, n_ids=8)
    try:
        rt.prepare_pipeline()
        assert False, "should have raised -- no play-area calibration exists"
    except RuntimeError as e:
        assert "play-area" in str(e)
        print(f"  prepare_pipeline() fails with a clear message before calibration: {e}")


if __name__ == "__main__":
    if not CV2_AVAILABLE:
        print("cv2 not available -- skipping runtime tests")
        sys.exit(0)
    test_process_frame_before_calibration_fails_clearly()
    test_process_frame_reports_tokens_with_correct_mm_positions()
    test_process_frame_handles_bgr_input_like_a_real_camera()
    print("\nall runtime tests passed")
