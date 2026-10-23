"""Calibration pipeline tests. Require cv2 (lens/homography use it; see
those modules' docstrings for why that is the right dependency boundary)
-- skip cleanly if it is not installed rather than failing the whole
suite, since the pure-vision tests must stay runnable without it.
"""
import sys, os, shutil
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from tracker.vision.geometry import RingGeometry
from tracker.vision.codebook import load_codebook
from tracker.vision import simulate as sim
from tracker.vision.calibration.store import CalibrationStore
from tracker.vision.calibration.session import CalibrationSession

GEOM = RingGeometry()
CB = load_codebook(n_ids=8)
TEST_DIR = "/tmp/v2_calibration_test"


def _render_chessboard(rvec, tvec, K, dist, W, H, cols, rows, sq):
    img = np.full((H, W), 210, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 1:
                continue
            quad = np.array([[c * sq, r * sq, 0], [(c + 1) * sq, r * sq, 0],
                             [(c + 1) * sq, (r + 1) * sq, 0], [c * sq, (r + 1) * sq, 0]],
                            dtype=np.float64)
            proj, _ = cv2.projectPoints(quad, rvec, tvec, K, dist)
            cv2.fillConvexPoly(img, proj.reshape(-1, 2).astype(np.int32), 20)
    return cv2.GaussianBlur(img, (3, 3), 0.5)


def test_lens_calibration_recovers_true_intrinsics():
    cols, rows, sq = 9, 6, 25.0
    W, H = 1280, 960
    true_K = np.array([[900., 0, W / 2], [0, 900., H / 2], [0, 0, 1.]])
    true_dist = np.array([-0.25, 0.08, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(14):
        rvec = rng.normal(0, 0.3, 3); rvec[2] = rng.uniform(-0.4, 0.4)
        tvec = np.array([rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(450, 750)])
        frames.append(_render_chessboard(rvec, tvec, true_K, true_dist, W, H, cols, rows, sq))

    shutil.rmtree(TEST_DIR, ignore_errors=True)
    sess = CalibrationSession(TEST_DIR, geom=GEOM, codebook=CB)
    cal, errs = sess.calibrate_lens(frames, (cols, rows), sq)

    assert max(errs) < 1.0, f"a chessboard frame reprojected with {max(errs):.2f}px error"
    assert abs(cal.camera_matrix[0, 0] - 900.0) / 900.0 < 0.03, "recovered fx is off by >3%"
    assert abs(cal.dist_coeffs[0] - (-0.25)) < 0.02, "recovered distortion is off"
    print("  lens calibration recovers true intrinsics within 3%")

    # reload in a FRESH session and confirm it persisted correctly
    sess2 = CalibrationSession(TEST_DIR, geom=GEOM, codebook=CB)
    assert sess2.lens is not None
    assert abs(sess2.lens.camera_matrix[0, 0] - cal.camera_matrix[0, 0]) < 1e-6
    print("  lens calibration survives a fresh session reload")


def test_playarea_homography_roundtrip():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    sess = CalibrationSession(TEST_DIR, geom=GEOM, codebook=CB)
    raw_corners = np.array([[140, 90], [1180, 60], [1250, 760], [60, 800]], dtype=np.float64)
    cal = sess.calibrate_playarea(raw_corners, width_mm=800, height_mm=600,
                                  px_per_mm=3.5, already_undistorted=True)
    assert cal.out_w == 2800 and cal.out_h == 2100

    H_inv = np.linalg.inv(cal.H)
    disc_out = np.full((cal.out_h, cal.out_w), 180, dtype=np.uint8)
    cx_out, cy_out, r_px = cal.out_w / 2, cal.out_h / 2, 19.0 * 3.5
    cv2.circle(disc_out, (int(cx_out), int(cy_out)), int(r_px), 30, -1)
    raw = cv2.warpPerspective(disc_out, H_inv, (1300, 900), flags=cv2.INTER_LINEAR,
                              borderValue=180)
    rectified = sess.rectify(raw)
    mask = rectified < 100
    ys, xs = np.nonzero(mask)
    fx, fy = xs.mean(), ys.mean()
    err_mm = np.hypot(fx - cx_out, fy - cy_out) / 3.5
    assert err_mm < 0.5, f"homography round-trip error {err_mm:.2f}mm"
    print(f"  play-area homography round-trips a known point to {err_mm:.3f}mm")

    # independent rerun: re-calibrating LENS must not disturb play-area
    K = np.array([[900., 0, 640.], [0, 900., 480.], [0, 0, 1.]])
    dist = np.zeros(5)
    rng = np.random.default_rng(1)
    lens_frames = []
    for _ in range(14):
        rvec = rng.normal(0, 0.3, 3); rvec[2] = rng.uniform(-0.4, 0.4)
        tvec = np.array([rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(450, 750)])
        lens_frames.append(_render_chessboard(rvec, tvec, K, dist, 1280, 960, 9, 6, 25.0))
    sess.calibrate_lens(lens_frames, (9, 6), 25.0)
    assert sess.playarea is not None and sess.playarea.px_per_mm == 3.5, \
        "running lens calibration disturbed an already-calibrated play area"
    print("  re-running lens calibration does not disturb an existing play-area calibration")


def test_detector_calibration_and_pipeline_build():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    sess = CalibrationSession(TEST_DIR, geom=GEOM, codebook=CB)
    raw_corners = np.array([[0, 0], [900, 0], [900, 500], [0, 500]], dtype=np.float64)
    sess.calibrate_playarea(raw_corners, width_mm=257, height_mm=143,
                            px_per_mm=3.5, already_undistorted=True)

    rng = np.random.default_rng(7)
    frames = []
    for _ in range(15):
        tokens = [{"id": i, "x": 150 + i * 180, "y": 200, "rot_deg": float(rng.uniform(0, 360))}
                  for i in range(3)]
        frames.append(sim.render_frame(tokens, GEOM, CB, 3.5, (500, 900), rng, noise=0.02))

    cal, report = sess.calibrate_detector(frames, expected_ids={0, 1, 2})
    assert report["missing_ids"] == []
    assert cal.min_peak > 0 and cal.min_margin > 0
    print(f"  detector calibration: {report['frames']} frames, "
          f"{report['detections_per_frame']['median']:.0f} markers/frame median")

    pipe = sess.build_pipeline(rescan_hz=3.0)
    assert pipe.matcher.min_peak == cal.min_peak
    test_img = sim.render_frame([{"id": 4, "x": 300, "y": 250, "rot_deg": 20.0}],
                                GEOM, CB, 3.5, (500, 600), rng, noise=0.02)
    out = pipe.process(test_img, now=0.0)
    print(f"  pipeline built from calibration session processed a frame "
          f"({len(out)} reportable tracks on first frame, as expected before confirm_hits)")

    sess2 = CalibrationSession(TEST_DIR, geom=GEOM, codebook=CB)
    assert sess2.detector.min_peak == cal.min_peak
    assert sess2.playarea.px_per_mm == 3.5
    print("  all three calibration stages survive a fresh session reload together")


if __name__ == "__main__":
    if not CV2_AVAILABLE:
        print("cv2 not available -- skipping calibration tests "
              "(these require the same cv2 the Docker image installs)")
        sys.exit(0)
    test_lens_calibration_recovers_true_intrinsics()
    test_playarea_homography_roundtrip()
    test_detector_calibration_and_pipeline_build()
    print("\nall calibration tests passed")
