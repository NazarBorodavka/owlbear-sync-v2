"""Camera configuration, persistence and auto-start.

These cover the gap between "the code works" and "it is deployed": an
appliance that needs a human to retype an RTSP URL after every reboot is
not deployed, and a first install that cannot be calibrated is not
installable.

RtspCapture is stubbed throughout. The point here is the configuration
and lifecycle logic, and a real RTSP attempt against a nonexistent
camera costs a 30s FFmpeg timeout per test.
"""
import base64
import contextlib
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import tracker.capture
from tracker.settings import SettingsStore, redact

AUTH = {"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()}
URL = "rtsp://admin:s3cret@192.168.1.50:554/h264Preview_01_main"
SECRET = "s3cret"


class _StubCapture:
    """Stands in for RtspCapture: connects instantly, yields grey frames."""
    instances = []

    def __init__(self, url, stale_after_s=5.0):
        self.url = url
        self.is_running = True
        self.reconnects = 0
        _StubCapture.instances.append(self)

    def wait_for_frame(self, timeout=1.0):
        return True

    def read(self):
        return True, np.full((60, 80), 128, dtype=np.uint8)

    def isOpened(self):
        return True

    def release(self):
        self.is_running = False


# Patched for the whole module, not per-call: start() resolves
# RtspCapture at call time, so a patch scoped only to create_app() would
# leave later /api/camera/start calls hitting the real one and waiting out
# a 30s FFmpeg timeout against a camera that does not exist.
tracker.capture.RtspCapture = _StubCapture


@contextlib.contextmanager
def _app(data_dir, **env):
    """An app whose env stays applied for the whole test.

    A context manager rather than a factory because the app reads
    TRACKER_CAMERA_URL at REQUEST time, not just at construction: an env
    var restored as soon as create_app() returned would be absent during
    the very requests under test.
    """
    from tracker.web.app import create_app
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    app = None
    try:
        app = create_app(data_dir=data_dir, users={"admin": "pw"})
        yield app
    finally:
        if app is not None:
            app.runtime.stop()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_camera_starts_without_calibration():
    """The first-deploy deadlock.

    start() used to require a play-area calibration, but the operator can
    only click the four play-area corners on a LIVE preview -- so the
    camera could not start until calibrated, and could not be calibrated
    until started. Capture must therefore be independent of calibration.
    """
    d = tempfile.mkdtemp()
    with _app(d) as app:
        c = app.test_client()
        r = c.post("/api/camera/start", json={"url": URL}, headers=AUTH)
        assert r.status_code == 200, r.get_json()
        assert app.runtime.is_running()
        # Nothing to track yet, but the feed is healthy rather than errored.
        assert app.runtime.pipeline is None
        assert app.runtime.process_frame(
            np.full((60, 80), 128, dtype=np.uint8)) is True
        print("  camera starts and streams with no calibration present")

def test_preview_is_served_before_calibration():
    """Without a preview there is nothing to click, so calibration is
    impossible -- the preview must work in exactly the uncalibrated state."""
    d = tempfile.mkdtemp()
    with _app(d) as app:
        app.test_client().post("/api/camera/start", json={"url": URL}, headers=AUTH)
        jpeg = app.runtime.latest_frame_jpeg()
        assert jpeg is not None and len(jpeg) > 0, \
            "no preview frame available before calibration"
        assert jpeg[:2] == b"\xff\xd8", "not a JPEG"
        print(f"  raw preview served before calibration ({len(jpeg)} bytes)")

def test_url_persists_and_autostarts_after_restart():
    """The reboot case: a restart must not need a human to retype it."""
    d = tempfile.mkdtemp()
    with _app(d) as app:
        app.test_client().post("/api/camera/start", json={"url": URL}, headers=AUTH)

    with open(os.path.join(d, "settings.json")) as f:
        assert json.load(f)["camera_url"] == URL

    with _app(d) as app2:          # simulates the container restarting
        assert app2.runtime.is_running(), \
            "camera did not auto-start from the remembered URL"
        assert _StubCapture.instances[-1].url == URL
        print("  remembered URL auto-starts on a fresh process")

def test_env_url_wins_and_is_not_overwritten():
    """TRACKER_CAMERA_URL is declarative compose config, so it must win
    over a remembered value and must not be silently replaced by one."""
    d = tempfile.mkdtemp()
    SettingsStore(os.path.join(d, "settings.json")).set("camera_url", "rtsp://old/stream")
    env_url = "rtsp://user:pw@10.0.0.9/live"
    with _app(d, TRACKER_CAMERA_URL=env_url) as app:
        assert _StubCapture.instances[-1].url == env_url
        c = app.test_client()
        body = c.get("/api/camera", headers=AUTH).get_json()
        assert body["source"] == "env", body
        assert body["editable"] is False, \
            "UI must not offer to edit a URL the deployment pins"
        # Starting again must not persist over the env value.
        c.post("/api/camera/start", json={"url": "rtsp://sneaky/stream"}, headers=AUTH)
        with open(os.path.join(d, "settings.json")) as f:
            assert json.load(f)["camera_url"] == "rtsp://old/stream"
        print("  env URL wins, is marked non-editable, and is never overwritten")

def test_autostart_can_be_disabled():
    d = tempfile.mkdtemp()
    SettingsStore(os.path.join(d, "settings.json")).set("camera_url", URL)
    with _app(d, TRACKER_AUTOSTART="0") as app:
        assert not app.runtime.is_running(), "TRACKER_AUTOSTART=0 was ignored"
        print("  TRACKER_AUTOSTART=0 suppresses auto-start")

def test_credentials_never_reach_a_response():
    """RTSP URLs embed the camera login. It is stored, so it must not be
    echoed -- least of all on the public endpoints the extension polls."""
    d = tempfile.mkdtemp()
    with _app(d) as app:
        c = app.test_client()
        r = c.post("/api/camera/start", json={"url": URL}, headers=AUTH)
        assert SECRET not in r.get_data(as_text=True)
        for path in ("/api/tokens", "/api/version"):
            assert SECRET not in c.get(path).get_data(as_text=True), path
        authed = c.get("/api/camera", headers=AUTH).get_data(as_text=True)
        assert SECRET not in authed, "credential leaked on /api/camera"
        assert "***" in authed
        # Unauthenticated access to the camera config must be refused.
        assert c.get("/api/camera").status_code == 401
        print("  credentials redacted everywhere; /api/camera requires auth")

def test_stored_settings_are_not_world_readable():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "settings.json")
    SettingsStore(path).set("camera_url", URL)
    mode = os.stat(path).st_mode & 0o777
    assert mode == 0o600, f"settings file mode is {oct(mode)}, expected 0o600"
    print("  settings.json is 0600 (it holds camera credentials)")


def test_delete_forgets_the_camera():
    d = tempfile.mkdtemp()
    with _app(d) as app:
        c = app.test_client()
        c.post("/api/camera/start", json={"url": URL}, headers=AUTH)
        # stop() is temporary and must NOT forget the credential.
        c.post("/api/camera/stop", headers=AUTH)
        assert c.get("/api/camera", headers=AUTH).get_json()["configured"]
        c.delete("/api/camera", headers=AUTH)
        assert not c.get("/api/camera", headers=AUTH).get_json()["configured"]
        print("  stop keeps the URL, DELETE forgets it")

def test_boot_survives_a_bad_url_and_corrupt_settings():
    """Auto-start failures must never stop the dashboard serving -- it is
    the only place the operator can go to fix the problem."""
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "settings.json"), "w") as f:
        f.write("{not json at all")
    with _app(d) as app:                      # must not raise
        assert app.test_client().get("/api/version").status_code == 200

    d2 = tempfile.mkdtemp()
    SettingsStore(os.path.join(d2, "settings.json")).set("camera_url", URL)

    class _Exploding(_StubCapture):
        def __init__(self, url, stale_after_s=5.0):
            raise OSError("camera unreachable at boot")

    tracker.capture.RtspCapture = _Exploding
    try:
        with _app(d2) as app2:                # must not raise
            assert app2.test_client().get("/api/version").status_code == 200
            assert not app2.runtime.is_running()
    finally:
        tracker.capture.RtspCapture = _StubCapture
    print("  corrupt settings and an unreachable camera still boot the dashboard")

def test_redact_handles_awkward_input():
    assert redact(URL) == "rtsp://***@192.168.1.50:554/h264Preview_01_main"
    assert redact("rtsp://192.168.1.50/stream") == "rtsp://192.168.1.50/stream"
    assert redact(None) is None
    assert redact("") is None
    # Unparseable input must mask rather than echo: leaking a password is
    # worse than showing the operator something unhelpful.
    assert redact("not a url") == "***"
    assert SECRET not in (redact(f"://{SECRET}@") or "")
    print("  redact() masks credentials and fails closed")


def test_dashboard_serves_while_an_unreachable_camera_connects():
    """Auto-start must never block the web server from coming up.

    Opening an RTSP stream blocks until it connects or FFmpeg times out,
    and that timeout is 30s per attempt. An inline auto-start therefore
    meant an unreachable camera -- switched off, or moved to a new IP --
    stopped the dashboard serving at all: measured against the real
    container, it never served within 70 seconds. That is the worst
    possible failure mode here, because the dashboard is the only place
    the operator can correct the URL, and restart: unless-stopped would
    loop on it forever.

    The earlier stub connected instantly, which is exactly why this was
    missed, so this one blocks the way a real unreachable camera does.
    """
    import threading
    import time

    connecting = threading.Event()
    release = threading.Event()

    class _BlockingCapture(_StubCapture):
        def __init__(self, url, stale_after_s=5.0):
            connecting.set()
            # Stands in for FFmpeg's connect timeout.
            if not release.wait(timeout=10.0):
                raise OSError("timed out connecting")
            super().__init__(url)

    d = tempfile.mkdtemp()
    SettingsStore(os.path.join(d, "settings.json")).set("camera_url", URL)
    tracker.capture.RtspCapture = _BlockingCapture
    try:
        t0 = time.perf_counter()
        with _app(d) as app:
            construct_s = time.perf_counter() - t0
            assert connecting.wait(timeout=5.0), "auto-start never attempted"
            # The key assertion: the app is usable while the connect is
            # still outstanding.
            assert construct_s < 2.0, (
                f"create_app() blocked {construct_s:.1f}s on the camera "
                f"connect; it must not wait for the camera at all")
            c = app.test_client()
            assert c.get("/api/version").status_code == 200
            assert c.get("/api/camera", headers=AUTH).status_code == 200
            assert not app.runtime.is_running(), \
                "should not report running while still connecting"
            print(f"  dashboard served in {construct_s*1000:.0f}ms with the "
                  f"camera connect still outstanding")
            release.set()
    finally:
        release.set()
        tracker.capture.RtspCapture = _StubCapture


if __name__ == "__main__":
    test_redact_handles_awkward_input()
    test_stored_settings_are_not_world_readable()
    test_camera_starts_without_calibration()
    test_preview_is_served_before_calibration()
    test_url_persists_and_autostarts_after_restart()
    test_env_url_wins_and_is_not_overwritten()
    test_autostart_can_be_disabled()
    test_credentials_never_reach_a_response()
    test_delete_forgets_the_camera()
    test_boot_survives_a_bad_url_and_corrupt_settings()
    test_dashboard_serves_while_an_unreachable_camera_connects()
    print("\nall camera-config tests passed")
