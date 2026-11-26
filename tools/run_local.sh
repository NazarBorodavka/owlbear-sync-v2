#!/usr/bin/env bash
# Runs the dashboard straight on this machine against the real camera --
# no Docker, no ZimaOS, no deploy. For trying an uncommitted change (or a
# hardware change like the IR filter) against live video before it goes
# anywhere near the server.
#
# Uses TRACKER_DATA_DIR=./data by default, the same directory
# tools/replay.py and tests/ already calibrate against -- so the lens,
# play-area and detector calibration already on this machine is reused
# as-is, with nothing to redo.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -x "$VENV_DIR/bin/python3" ]; then
    echo "No venv at $VENV_DIR -- creating one (first run only) ..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r tracker/requirements.txt
fi
PY="$VENV_DIR/bin/python3"

export TRACKER_DATA_DIR="${TRACKER_DATA_DIR:-$(pwd)/data}"
export TRACKER_USER="${TRACKER_USER:-admin}"
TRACKER_PASSWORD_WAS_SET="${TRACKER_PASSWORD:+yes}"
export TRACKER_PASSWORD="${TRACKER_PASSWORD:-change-me}"
# TRACKER_CAMERA_URL is read from the environment if you export it before
# calling this script (rtsp://user:pass@host:554/...). Left unset here on
# purpose -- this script never prompts for or stores a camera password;
# leave it unset and type the URL into the dashboard instead, which saves
# it to $TRACKER_DATA_DIR/settings.json (0600) for next time.

echo "data dir  : $TRACKER_DATA_DIR  (calibration there is reused, not redone)"
# Print the password when it is this script's own default. A default
# the operator cannot see is worse than a weak one -- it just produces a
# dashboard nobody can log into. An explicitly exported password is NOT
# echoed, since that one may be real.
if [ -n "${TRACKER_PASSWORD_WAS_SET:-}" ]; then
    echo "dashboard : http://localhost:5000  (user: $TRACKER_USER, password: as exported)"
else
    echo "dashboard : http://localhost:5000  (user: $TRACKER_USER, password: $TRACKER_PASSWORD)"
fi
if [ -n "${TRACKER_CAMERA_URL:-}" ]; then
    echo "camera    : pinned via \$TRACKER_CAMERA_URL -- will auto-start"
else
    echo "camera    : not pinned -- enter the RTSP URL in the dashboard"
fi
echo

exec "$PY" -m tracker.web.app
