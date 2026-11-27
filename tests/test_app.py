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


def test_camera_start_does_not_require_calibration():
    """Capture must be startable on a fresh install.

    This test previously asserted the OPPOSITE -- that starting the camera
    without a play-area calibration fails with a 400. That was enforcing a
    first-deploy deadlock: the operator calibrates the play area by
    clicking its four corners on a LIVE camera preview, so requiring
    calibration before capture meant the camera could never start on a new
    install, and the dashboard could never show anything to click.

    Capture and tracking are separate concerns. Capture starts; tracking
    begins once a pipeline can be built. See TrackerRuntime.start().
    """
    app = _fresh_app()
    client = app.test_client()
    auth = ("admin", "admin")

    # With nothing configured, a missing URL is still an error: there is
    # genuinely nothing to connect to.
    bad = client.post("/api/camera/start", auth=auth, json={})
    assert bad.status_code == 400 and "url" in bad.get_json()["error"], bad.get_json()

    r = client.post("/api/camera/start", auth=auth, json={"url": "rtsp://127.0.0.1:1/x"})
    try:
        assert r.status_code == 200, r.get_json()
        assert app.runtime.pipeline is None, \
            "no pipeline should exist without calibration"
        # Now that a URL is remembered, Start with an empty body reuses it
        # rather than making the operator retype a credentialed URL.
        again = client.post("/api/camera/start", auth=auth, json={})
        assert again.status_code == 200, again.get_json()
        print("  camera starts without calibration; Start reuses the saved URL")
    finally:
        app.runtime.stop()


def test_tokens_payload_matches_what_the_extension_reads():
    """The /api/tokens contract, pinned field by field.

    extension/main.js reads these names directly off the JSON. Nothing in
    Python fails if one is renamed -- the extension just silently stops
    syncing, which is indistinguishable from "the camera is not working"
    from the operator's chair. So the exact field names the extension
    depends on are asserted here, with a real Track pushed through the
    real pipeline conversion rather than a hand-built dict.
    """
    import numpy as np
    from tracker.vision.geometry import RingGeometry
    from tracker.vision.codebook import load_codebook
    from tracker.vision.pipeline import Pipeline
    from tracker.vision.track import Track

    app = _fresh_app()
    client = app.test_client()
    auth = ("admin", "admin")

    # Assert the SETUP worked. Without this the whole test silently
    # degrades into asserting against an uncalibrated app if the
    # calibration payload shape ever drifts.
    r = client.post("/api/calibration/playarea", auth=auth, json={
        "corners": [[0, 0], [1000, 0], [1000, 600], [0, 600]],
        "width_mm": 905.0, "height_mm": 510.0, "px_per_mm": 3.5,
    })
    assert r.status_code == 200, r.get_json()

    # Build one token through the same code path the runtime uses.
    pipe = Pipeline(RingGeometry(), load_codebook(n_ids=6), 3.5)
    t = Track(350.0, 210.0, 3.5, 0.0)
    t.marker_id = 4
    app.runtime._latest_tokens = pipe.tokens_mm([t])
    client.put("/api/names/4", json={"name": "Aragorn"}, auth=auth)

    body = client.get("/api/tokens").get_json()

    # --- top level, read in pollOnce() -----------------------------
    for field in ("tokens", "play_area_mm", "connected", "fps"):
        assert field in body, f"/api/tokens no longer returns {field!r}"
    assert isinstance(body["play_area_mm"], list) and len(body["play_area_mm"]) == 2, \
        "play_area_mm must be a [width_mm, height_mm] pair -- the extension "\
        "destructures it and the auto-zoom divides by both"
    assert body["play_area_mm"] == [905.0, 510.0]
    # .toFixed(0) is called on fps unconditionally
    assert isinstance(body["fps"], (int, float)), "fps must be numeric"

    # --- per token, read in pushPositions()/autoMatchByName() ------
    tok = body["tokens"][0]
    for field in ("marker_id", "x_mm", "y_mm", "coasting", "name"):
        assert field in tok, f"token objects no longer carry {field!r}"
    assert tok["marker_id"] == 4
    assert tok["name"] == "Aragorn"
    # The extension multiplies these by px/mm; a string would concatenate.
    assert isinstance(tok["x_mm"], (int, float)), "x_mm must be numeric"
    assert isinstance(tok["y_mm"], (int, float)), "y_mm must be numeric"
    # `if (pt.coasting) continue` -- must be a real bool, since a
    # non-empty string like "false" is truthy in JS and would suppress
    # every update forever.
    assert isinstance(tok["coasting"], bool), \
        "coasting must be a JSON bool; any non-empty string is truthy in JS"
    assert tok["coasting"] is False, "a freshly measured track is not coasting"
    # mm are relative to the play-area origin and must land inside it,
    # or the extension maps tokens off the map.
    assert 0 <= tok["x_mm"] <= 905.0 and 0 <= tok["y_mm"] <= 510.0, \
        f"token at ({tok['x_mm']}, {tok['y_mm']})mm is outside the play area"
    print("  /api/tokens carries every field extension/main.js reads, "
          "with the right types")


def test_coasting_is_true_when_the_position_is_not_a_fresh_read():
    """The Owlbear requirement in one assertion: a token whose code is not
    being read must be flagged, so its stale position is never pushed
    over a good one. If this inverts, the extension cannot tell a held
    position from a measured one."""
    from tracker.vision.geometry import RingGeometry
    from tracker.vision.codebook import load_codebook
    from tracker.vision.pipeline import Pipeline
    from tracker.vision.track import Track

    pipe = Pipeline(RingGeometry(), load_codebook(n_ids=6), 3.5)
    t = Track(100.0, 100.0, 3.5, 0.0)
    t.marker_id = 2
    assert pipe.tokens_mm([t])[0]["coasting"] is False
    t.misses = 1                      # one frame without a confirming read
    assert pipe.tokens_mm([t])[0]["coasting"] is True, \
        "an unread token must report coasting=True"
    print("  coasting flips exactly with 'was this position read this frame'")



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
    test_camera_start_does_not_require_calibration()
    test_tokens_payload_matches_what_the_extension_reads()
    test_coasting_is_true_when_the_position_is_not_a_fresh_read()
    print("\nall app tests passed")
