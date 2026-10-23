"""Flask application: REST API for calibration, live tokens, and token
naming; serves the dashboard and the Owlbear Rodeo extension's static
files.

Deliberately thin. The actual tracking logic lives in tracker.runtime
and tracker.vision -- this module's job is translating HTTP requests
into calls on a TrackerRuntime instance and JSON-encoding the result,
not implementing anything itself. That split is what v1's single
3000-line app.py did not have.
"""
import base64
import os
import time

import numpy as np
from flask import Flask, Response, current_app, jsonify, request, send_from_directory
from flask_httpauth import HTTPBasicAuth

from ..runtime import TrackerRuntime, _bgr_to_gray
from ..names import NameStore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTENSION_DIR = os.path.join(os.path.dirname(BASE_DIR), "extension")

auth = HTTPBasicAuth()


def create_app(data_dir=None, users=None):
    """Flask application factory, not a module-level singleton.

    This is what actually makes "each calibration stage independently
    rerunnable and persisted" testable: a module-level `app = Flask(...)`
    built once at import time forces every test to share ONE
    TrackerRuntime/NameStore for the whole process, with no clean way to
    reset between tests -- found directly while writing
    tests/test_app.py, where a module-reload trick to work around that
    silently returned Python's CACHED package attribute instead of a
    fresh module (a genuine `from package import submodule` gotcha: it
    resolves via getattr on the already-imported package before ever
    re-checking sys.modules). The factory pattern sidesteps that
    entirely -- every call gets a genuinely fresh app, runtime and
    name store.

    data_dir defaults to TRACKER_DATA_DIR or ./data, read at CALL time
    (not import time), so it is no longer fixed the moment this module
    is first imported.
    """
    data_dir = data_dir or os.environ.get(
        "TRACKER_DATA_DIR", os.path.join(BASE_DIR, "data"))
    calibration_dir = os.path.join(data_dir, "calibration")
    names_path = os.path.join(data_dir, "names.json")
    os.makedirs(data_dir, exist_ok=True)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.runtime = TrackerRuntime(calibration_dir)
    app.names = NameStore(names_path)
    app.extension_users = users or {
        os.environ.get("TRACKER_USER", "admin"):
        os.environ.get("TRACKER_PASSWORD", "admin")}

    _register_routes(app)
    return app


@auth.verify_password
def verify_password(username, password):
    return username if current_app.extension_users.get(username) == password else None


def _register_routes(app):
    # -----------------------------------------------------------------
    # Dashboard + static assets
    # -----------------------------------------------------------------
    @app.route("/")
    def index():
        return send_from_directory(app.template_folder, "index.html")

    @app.route("/extension/<path:filename>")
    def extension_files(filename):
        """Serves the Owlbear Rodeo extension from the SAME origin as the
        API it calls, so the extension's fetch() calls need no CORS
        configuration -- the manifest URL given to Owlbear Rodeo is just
        this tracker's own /extension/manifest.json."""
        return send_from_directory(EXTENSION_DIR, filename)

    @app.route("/api/version")
    def api_version():
        """Public so "did my update actually deploy" is answerable without
        logging in. The Dockerfile accepts these as build args and the CI
        workflow fills them from the run number and commit; a local run
        reports the dev/local/unknown defaults.

        Also reports the codebook fingerprint, because that is the one
        thing that must match the PHYSICAL markers on the table -- if a
        build ever ships a different codebook, the printed markers need
        reprinting, and this is where that becomes visible.
        """
        from ..vision.codebook import codebook_metadata
        meta = codebook_metadata()
        return jsonify({
            "version": os.environ.get("BUILD_VERSION", "dev"),
            "branch": os.environ.get("BUILD_BRANCH", "local"),
            "commit": os.environ.get("BUILD_COMMIT", "unknown"),
            "codebook": {
                "n_chips": meta["n_chips"],
                "n_ids": meta["n_ids"],
                "worst_crosscorrelation": meta["worst_pairwise_crosscorrelation"],
            },
        })

    # -----------------------------------------------------------------
    # Live video + tokens
    # -----------------------------------------------------------------
    @app.route("/api/tokens")
    def api_tokens():
        """Polled by the Owlbear Rodeo extension. Positions are in
        millimetres relative to the calibrated play-area origin, plus the
        play area's own size, so the extension can convert to its own
        grid without needing to know anything about the camera."""
        out = app.runtime.latest()
        out["tokens"] = [
            {**t, "name": app.names.get(t["marker_id"])}
            for t in out["tokens"]
        ]
        return jsonify(out)

    @app.route("/video_feed")
    @auth.login_required
    def video_feed():
        def gen():
            while True:
                jpg = app.runtime.latest_frame_jpeg()
                if jpg is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1.0 / 10.0)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # -----------------------------------------------------------------
    # Token naming (persistent, shared with the extension via /api/tokens)
    # -----------------------------------------------------------------
    @app.route("/api/names", methods=["GET"])
    @auth.login_required
    def get_names():
        return jsonify(app.names.all())

    @app.route("/api/names/<int:marker_id>", methods=["PUT"])
    @auth.login_required
    def set_name(marker_id):
        data = request.json or {}
        app.names.set(marker_id, data.get("name"))
        return jsonify({"marker_id": marker_id, "name": app.names.get(marker_id)})

    # -----------------------------------------------------------------
    # Calibration: status
    # -----------------------------------------------------------------
    @app.route("/api/calibration/status")
    @auth.login_required
    def calibration_status():
        return jsonify(app.runtime.session.status())

    # -----------------------------------------------------------------
    # Calibration: lens
    # -----------------------------------------------------------------
    @app.route("/api/calibration/lens", methods=["POST"])
    @auth.login_required
    def calibrate_lens():
        """Body: {"frames": [base64 PNG/JPEG, ...], "pattern_cols": 9,
        "pattern_rows": 6, "square_mm": 25.0}. Frames are the operator's
        own chessboard-at-different-angles captures, taken via the
        dashboard's camera preview."""
        import cv2
        data = request.json or {}
        frames_b64 = data.get("frames", [])
        if len(frames_b64) < 4:
            return jsonify({"error": "need at least 4 chessboard captures at "
                           "different angles/positions"}), 400
        grays = []
        for b64 in frames_b64:
            raw = base64.b64decode(b64.split(",")[-1])
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                grays.append(img)
        try:
            cal, per_frame_err = app.runtime.session.calibrate_lens(
                grays, (int(data.get("pattern_cols", 9)), int(data.get("pattern_rows", 6))),
                float(data.get("square_mm", 25.0)))
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"rms_error": cal.rms_error, "per_frame_errors": per_frame_err,
                       "frames_used": len(per_frame_err)})

    @app.route("/api/calibration/lens", methods=["DELETE"])
    @auth.login_required
    def clear_lens():
        app.runtime.session.clear_lens()
        return jsonify({"ok": True})

    # -----------------------------------------------------------------
    # Calibration: play-area
    # -----------------------------------------------------------------
    @app.route("/api/calibration/playarea", methods=["POST"])
    @auth.login_required
    def calibrate_playarea():
        """Body: {"corners": [[x,y],[x,y],[x,y],[x,y]], "width_mm": 800,
        "height_mm": 600, "px_per_mm": 3.5}. Corners are clicked by the
        operator on the live (raw, lens-corrected-if-available) camera
        preview, in order: top-left, top-right, bottom-right,
        bottom-left."""
        data = request.json or {}
        try:
            corners = data["corners"]
            cal = app.runtime.session.calibrate_playarea(
                corners, float(data["width_mm"]), float(data["height_mm"]),
                float(data["px_per_mm"]))
        except (KeyError, ValueError) as e:
            return jsonify({"error": f"missing or invalid field: {e}"}), 400
        if app.runtime.pipeline is not None:
            app.runtime.rebuild_pipeline()
        return jsonify({"ok": True, **cal.coverage_summary(app.runtime.geom)})

    @app.route("/api/calibration/playarea/preview", methods=["POST"])
    @auth.login_required
    def preview_playarea():
        """Lets the dashboard show the predicted accuracy for a candidate
        px_per_mm/corners choice BEFORE committing to it, so zoom/coverage
        trade-offs (see docs/VALIDATION.md) are visible while the
        operator is still adjusting, not only after calibrating."""
        from ..vision.calibration.playarea import PlayAreaCalibration
        data = request.json or {}
        try:
            corners = np.asarray(data["corners"], dtype=np.float64)
            width_mm, height_mm = float(data["width_mm"]), float(data["height_mm"])
            px_per_mm = float(data["px_per_mm"])
        except (KeyError, ValueError) as e:
            return jsonify({"error": f"missing or invalid field: {e}"}), 400
        cal = PlayAreaCalibration(corners, width_mm, height_mm, px_per_mm)
        return jsonify(cal.coverage_summary(app.runtime.geom))

    @app.route("/api/calibration/playarea", methods=["DELETE"])
    @auth.login_required
    def clear_playarea():
        app.runtime.session.clear_playarea()
        return jsonify({"ok": True})

    # -----------------------------------------------------------------
    # Calibration: detector auto-tune
    # -----------------------------------------------------------------
    @app.route("/api/calibration/detector", methods=["POST"])
    @auth.login_required
    def calibrate_detector():
        """Captures a short burst of LIVE rectified frames (the operator
        has already placed their real markers on the table) and measures
        actual peak/margin statistics from them -- see
        tracker/vision/calibration/detector_tune.py for why this is
        measured from real captures rather than searched blind."""
        data = request.json or {}
        duration_s = float(data.get("duration_s", 3.0))
        expected_ids = set(data.get("expected_ids", [])) or None

        if app.runtime.capture is None:
            return jsonify({"error": "camera is not connected -- start tracking first"}), 400

        frames = []
        deadline = time.time() + duration_s
        while time.time() < deadline:
            ok, raw = app.runtime.capture.read()
            if ok and raw is not None:
                gray = raw if raw.ndim == 2 else _bgr_to_gray(raw)
                try:
                    frames.append(app.runtime.session.rectify(gray))
                except Exception:
                    pass
            time.sleep(1.0 / 15.0)

        try:
            cal, report = app.runtime.session.calibrate_detector(
                frames, expected_ids=expected_ids)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        app.runtime.rebuild_pipeline()
        return jsonify(report)

    @app.route("/api/calibration/detector", methods=["DELETE"])
    @auth.login_required
    def reset_detector():
        app.runtime.session.reset_detector()
        app.runtime.rebuild_pipeline()
        return jsonify({"ok": True})

    # -----------------------------------------------------------------
    # Grid measurement (auto-zoom support)
    # -----------------------------------------------------------------
    @app.route("/api/grid/measure", methods=["POST"])
    @auth.login_required
    def measure_grid():
        """Measures the real-world mm/cell of whatever grid is currently
        visible in the camera feed -- the operator turns on Owlbear
        Rodeo's grid overlay first, the ceiling projector displays it on
        the table, and this is what makes "automatic map zooming" actually
        automatic: the extension compares this measurement to the known
        token base size and computes the exact OBR viewport zoom needed,
        no manual trial-and-error required. See tracker/vision/grid.py.
        """
        from ..vision.grid import measure_grid_spacing_mm
        if app.runtime.capture is None:
            return jsonify({"error": "camera is not connected"}), 400
        if app.runtime.session.playarea is None:
            return jsonify({"error": "play-area is not calibrated"}), 400

        ok, raw = app.runtime.capture.read()
        if not ok or raw is None:
            return jsonify({"error": "no frame available from camera"}), 400
        gray = raw if raw.ndim == 2 else _bgr_to_gray(raw)
        try:
            rectified = app.runtime.session.rectify(gray)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        result = measure_grid_spacing_mm(rectified, app.runtime.session.playarea.px_per_mm)
        if result is None:
            return jsonify({"error": "no grid pattern found -- make sure Owlbear "
                           "Rodeo's grid overlay is turned on and visible on the table"}), 404
        return jsonify(result)

    # -----------------------------------------------------------------
    # Camera lifecycle
    # -----------------------------------------------------------------
    @app.route("/api/camera/start", methods=["POST"])
    @auth.login_required
    def start_camera():
        data = request.json or {}
        url = data.get("url")
        if not url:
            return jsonify({"error": "missing camera url"}), 400
        try:
            app.runtime.start(url)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True})

    @app.route("/api/camera/stop", methods=["POST"])
    @auth.login_required
    def stop_camera():
        app.runtime.stop()
        return jsonify({"ok": True})


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, threaded=True)
