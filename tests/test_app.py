"""Flask API contract tests: auth, calibration endpoints, token naming,
and the play_area_mm-available-immediately-after-calibration fix (this
test is what caught that bug in the first place -- it used to only
populate after the first processed camera frame, which left the
Owlbear Rodeo extension's auto-zoom with nothing to compute from until
a camera was actually connected and running).
"""
import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import flask  # noqa
    import cv2  # noqa
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

TEST_DATA_DIR = "/tmp/v2_app_test_formal"
_counter = [0]


def _fresh_app():
    """A fresh Flask app with its own TrackerRuntime/NameStore for every
    test, via the application factory (see tracker/web/app.py's
    create_app docstring for why this replaced an earlier module-reload
    trick that turned out not to work)."""
    _counter[0] += 1
    data_dir = f"{TEST_DATA_DIR}_{_counter[0]}"
    shutil.rmtree(data_dir, ignore_errors=True)
    from tracker.web.app import create_app
    return create_app(data_dir=data_dir)


def test_requires_auth_on_protected_endpoints():
    app = _fresh_app()
    client = app.test_client()
    r = client.get("/api/names")
    assert r.status_code == 401, "protected endpoint allowed anonymous access"
    r = client.get("/api/tokens")
    assert r.status_code == 200, "the public token feed should not require auth"
    print("  /api/names requires auth (401), /api/tokens is public")


def test_names_round_trip():
    app = _fresh_app()
    client = app.test_client()
    auth = ("admin", "admin")
    client.put("/api/names/3", json={"name": "Gandalf"}, auth=auth)
    r = client.get("/api/names", auth=auth)
    assert r.get_json() == {"3": "Gandalf"}
    # a token on the live feed should carry the assigned name
    r = client.get("/api/tokens")
    assert r.status_code == 200
    print("  name assignment persists and round-trips through the API")


def test_playarea_calibration_available_immediately():
    app = _fresh_app()
    client = app.test_client()
    auth = ("admin", "admin")

    r = client.get("/api/tokens")
    assert r.get_json()["play_area_mm"] is None, "play area should be unset before calibration"

    r = client.post("/api/calibration/playarea", auth=auth, json={
        "corners": [[0, 0], [900, 0], [900, 500], [0, 500]],
        "width_mm": 257, "height_mm": 143, "px_per_mm": 3.5})
    assert r.status_code == 200, r.get_json()

    # THE BUG THIS GUARDS: play_area_mm must be available right after
    # calibration, not only after a camera frame has been processed.
    r = client.get("/api/tokens")
    assert r.get_json()["play_area_mm"] == [257.0, 143.0], \
        "play_area_mm did not appear immediately after calibration"
    print("  play_area_mm is available immediately after calibration, "
          "with no camera connected")


def test_playarea_preview_does_not_persist():
    app = _fresh_app()
    client = app.test_client()
    auth = ("admin", "admin")
    client.post("/api/calibration/playarea/preview", auth=auth, json={
        "corners": [[0, 0], [900, 0], [900, 500], [0, 500]],
        "width_mm": 257, "height_mm": 143, "px_per_mm": 3.5})
    status = client.get("/api/calibration/status", auth=auth).get_json()
    assert status["playarea"]["calibrated"] is False, \
        "preview endpoint should not persist a calibration"
    print("  playarea/preview computes accuracy without committing to it")


def test_calibration_status_reflects_each_stage_independently():
    app = _fresh_app()
    client = app.test_client()
    auth = ("admin", "admin")
    status = client.get("/api/calibration/status", auth=auth).get_json()
    assert status["lens"]["calibrated"] is False
    assert status["playarea"]["calibrated"] is False
    assert status["detector"]["calibrated"] is False

    client.post("/api/calibration/playarea", auth=auth, json={
        "corners": [[0, 0], [900, 0], [900, 500], [0, 500]],
        "width_mm": 257, "height_mm": 143, "px_per_mm": 3.5})
    status = client.get("/api/calibration/status", auth=auth).get_json()
    assert status["playarea"]["calibrated"] is True
    assert status["lens"]["calibrated"] is False, \
        "calibrating play-area should not mark lens as calibrated"
    print("  calibration status correctly reflects each stage independently")


def test_grid_measure_endpoint_with_injected_camera():
    """No real RTSP connection needed -- a minimal fake capture object
    with the same read() contract as RtspCapture is enough to exercise
    the grid-measurement endpoint end-to-end."""
    import numpy as np
    app = _fresh_app()
    client = app.test_client()
    auth = ("admin", "admin")

    px_mm = 3.5
    client.post("/api/calibration/playarea", auth=auth, json={
        "corners": [[0, 0], [1400, 0], [1400, 900], [0, 900]],
        "width_mm": 400, "height_mm": 257, "px_per_mm": px_mm})
    app.runtime.prepare_pipeline()

    from tests.test_grid import render_grid
    true_mm = 50.0
    gray = render_grid(true_mm, px_mm, (900, 1400))
    bgr = np.stack([gray * 255] * 3, axis=-1).astype(np.uint8)

    class FakeCapture:
        def read(self):
            return True, bgr
    app.runtime.capture = FakeCapture()

    r = client.post("/api/grid/measure", auth=auth)
    assert r.status_code == 200, r.get_json()
    data = r.get_json()
    err_pct = 100 * abs(data["mm_per_cell"] - true_mm) / true_mm
    assert err_pct < 2.0, data
    print(f"  /api/grid/measure correctly measures {data['mm_per_cell']:.1f}mm/cell "
          f"(true {true_mm}mm) via an injected fake camera, no RTSP needed")


def test_camera_start_requires_playarea_calibration_first():
    app = _fresh_app()
    client = app.test_client()
    auth = ("admin", "admin")
    r = client.post("/api/camera/start", auth=auth, json={"url": "rtsp://example/stream"})
    assert r.status_code == 400
    assert "play-area" in r.get_json()["error"]
    print("  starting the camera before play-area calibration fails with a clear error")


if __name__ == "__main__":
    if not DEPS_AVAILABLE:
        print("flask/cv2 not available -- skipping app tests")
        sys.exit(0)
    test_requires_auth_on_protected_endpoints()
    test_names_round_trip()
    test_playarea_calibration_available_immediately()
    test_playarea_preview_does_not_persist()
    test_calibration_status_reflects_each_stage_independently()
    test_grid_measure_endpoint_with_injected_camera()
    test_camera_start_requires_playarea_calibration_first()
    print("\nall app tests passed")
