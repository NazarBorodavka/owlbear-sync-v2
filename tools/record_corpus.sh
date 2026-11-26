#!/usr/bin/env bash
# Walks through recording the whole validation corpus in one sitting,
# prompting for the physical action before each clip.
#
# Exists because the corpus is only meaningful if every clip is the
# scenario docs/VALIDATION.md says it is -- recording them ad hoc, weeks
# apart, from memory, is how you end up comparing a "fast" clip against
# a differently-fast one and reading the difference as a code change.
#
# Requires $TRACKER_CAMERA_URL. Stop any other viewer of this camera
# first (the ZimaOS container, the Reolink app): the camera serves ONE
# RTSP session, and a second viewer corrupts the stream for both.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${TRACKER_CAMERA_URL:?set TRACKER_CAMERA_URL first (rtsp://user:pass@host:554/...)}"
DURATION="${DURATION:-30}"
PY="${PY:-.venv/bin/python3}"
[ -x "$PY" ] || PY=python3

clip () {
    local label="$1" instruction="$2"
    echo
    echo "────────────────────────────────────────────────────────────"
    echo "  CLIP: $label   (${DURATION}s)"
    echo "  DO:   $instruction"
    echo "────────────────────────────────────────────────────────────"
    read -r -p "  Ready? [Enter to record, s to skip] " ans
    [ "$ans" = "s" ] && { echo "  skipped"; return 0; }
    echo "  recording ${DURATION}s ..."
    "$PY" tools/record_rtsp.py --label "$label" --duration "$DURATION" --force
    echo "  done: $label"
}

# Fail fast on a bad URL. Without this the first failure arrives AFTER a
# prompt and a 30s wait, and the most common cause -- a '!' in the
# password being eaten by interactive history expansion when the URL was
# exported in double quotes -- looks like a camera fault rather than a
# shell quoting mistake.
echo "Checking camera credentials ..."
if ! "$PY" tools/record_rtsp.py --out /tmp/.corpus_preflight.mp4 \
        --duration 2 --force >/tmp/.corpus_preflight.log 2>&1; then
    echo
    echo "  Could not open the camera. Last lines:"
    sed 's/^/    /' /tmp/.corpus_preflight.log | tail -5
    echo
    echo "  If that says 401 Unauthorized, re-export the URL in SINGLE"
    echo "  quotes -- a '!' in the password is history-expanded by an"
    echo "  interactive shell inside double quotes:"
    echo "    export TRACKER_CAMERA_URL='rtsp://user:pass@host:554/Preview_01_main'"
    echo "  Also check nothing else holds the camera (this one serves a"
    echo "  single RTSP session): stop the ZimaOS container first."
    rm -f /tmp/.corpus_preflight.mp4
    exit 1
fi
rm -f /tmp/.corpus_preflight.mp4
echo "  camera OK"
echo

echo "All 6 markers should be on the table for every clip."
echo "Recording ${DURATION}s per clip into recordings/."

# Ordered so the two that matter most for the current work come first,
# in case the session gets cut short.
clip occluded   "Move tokens around WITH YOUR HAND OVER THEM -- cover each marker partially, briefly fully, then uncover."
clip slow       "Drag each token slowly across the play area, one at a time."
clip fast       "Flick tokens quickly between spots. This is the motion-blur limit test."
clip still      "Place all 6 and DO NOT TOUCH anything. Baseline for false positives and id stability."
clip crossing   "Pass two tokens over/past each other repeatedly."
clip map-change "Tokens still; switch projected maps underneath them. With the IR filter this should now be a NO-OP -- that is what it is testing."
clip clutter    "Add dice, a mug, a d20 alongside the tokens. Tests that non-markers are still rejected."

echo
echo "Corpus recorded:"
ls -la recordings/*.mp4 2>/dev/null | awk '{print "  ", $9, $5}'
echo
echo "Next: replay them through the pipeline to score the result."
