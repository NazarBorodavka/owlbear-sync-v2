"""RtspCapture's reconnect behaviour.

Found against the real deployed hardware, not in review: once FFmpeg's
RTSP/UDP demuxer loses sync -- a dropped keyframe, a brief Wi-Fi hiccup,
the camera's own momentary rekey -- cap.read() can return False forever
on the SAME VideoCapture object, with no exception. The original _reader
loop just slept and retried on the dead object indefinitely, which
surfaced as "the camera connection keeps dropping" and "frames stop
updating" with no recovery short of restarting the whole container.

cv2.VideoCapture is stubbed, not the real network: a genuine RTSP
connection costs 30s FFmpeg timeouts and needs a real camera, and the
property under test -- does a sustained read failure get the capture
object recreated -- does not depend on real video at all.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np

import tracker.capture as capture_mod
from tracker.capture import RtspCapture

_REAL_VIDEOCAPTURE = cv2.VideoCapture


class _FakeCV2Capture:
    """Stands in for cv2.VideoCapture.

    Generation 0 (whatever RtspCapture.__init__ constructs first) is
    scripted to fail forever. Any LATER generation succeeds immediately.
    So a frame can only ever arrive because _reopen() actually
    constructed a new instance -- a test that merely waited out a flaky
    read() that recovers on its own would not be exercising the fix at
    all, which is exactly the stub-fidelity mistake this project has
    been bitten by before (see HANDOFF on the auto-start capture test).
    """
    instances = []

    def __init__(self, url, backend=None):
        self.url = url
        self.generation = len(_FakeCV2Capture.instances)
        _FakeCV2Capture.instances.append(self)
        self.released = False
        self.read_calls = 0

    def read(self):
        self.read_calls += 1
        if self.generation == 0:
            return False, None
        return True, np.full((48, 64), 100, dtype=np.uint8)

    def set(self, *a, **kw):
        return True

    def isOpened(self):
        return not self.released

    def release(self):
        self.released = True


class _AlwaysDeadCV2Capture(_FakeCV2Capture):
    """Every generation fails -- models a camera that is genuinely
    unreachable, not merely desynced. Reconnect attempts must still be
    rate-limited rather than spinning a tight construction loop."""

    def read(self):
        self.read_calls += 1
        return False, None


def _patch(cls):
    cv2.VideoCapture = cls
    cls.instances = []


def _restore():
    cv2.VideoCapture = _REAL_VIDEOCAPTURE


def test_sustained_failure_triggers_a_reopen_and_recovers():
    _patch(_FakeCV2Capture)
    try:
        cap = RtspCapture("rtsp://fake", stale_after_s=2.0,
                          reconnect_after_s=0.15, reconnect_backoff_s=0.05)
        try:
            deadline = time.time() + 3.0
            got = False
            while time.time() < deadline:
                ok, frame = cap.read()
                if ok:
                    got = True
                    break
                time.sleep(0.02)
            assert got, "capture never recovered after a sustained failure"
            assert cap.reconnects >= 1, "no reopen was recorded"
            assert len(_FakeCV2Capture.instances) >= 2, (
                "a second cv2.VideoCapture was never constructed -- the "
                "frame must have come from the original (dead-scripted) "
                "instance, so this did not test the reconnect path")
        finally:
            cap.is_running = False
            cap.thread.join(timeout=2.0)
    finally:
        _restore()


def test_brief_failure_does_not_reopen():
    """A single bad read (well under reconnect_after_s) is the normal,
    expected cost of UDP transport -- see capture.py's module docstring
    on why UDP is used at all. Reopening on every one of those would
    defeat the point of tolerating them cheaply."""
    _patch(_FakeCV2Capture)
    # generation 0 never recovers in this stub, so "no reopen happened"
    # is checked by elapsed time staying well under reconnect_after_s,
    # not by a frame arriving.
    try:
        cap = RtspCapture("rtsp://fake", stale_after_s=2.0,
                          reconnect_after_s=1.0, reconnect_backoff_s=0.05)
        try:
            time.sleep(0.3)
            assert cap.reconnects == 0, (
                "reopened before reconnect_after_s elapsed")
        finally:
            cap.is_running = False
            cap.thread.join(timeout=2.0)
    finally:
        _restore()


def test_unreachable_camera_does_not_spin_reconnects():
    """A camera that is simply off/unreachable must still get occasional
    reconnect attempts (so it is picked back up when it returns), but
    must not busy-loop constructing VideoCapture objects."""
    _patch(_AlwaysDeadCV2Capture)
    try:
        cap = RtspCapture("rtsp://fake", stale_after_s=10.0,
                          reconnect_after_s=0.1, reconnect_backoff_s=0.2)
        try:
            time.sleep(1.0)
            # In 1s with a 0.1s trigger + 0.2s backoff, at most ~4
            # reopen cycles are possible; a busy loop would produce
            # hundreds.
            assert 1 <= cap.reconnects <= 8, (
                f"expected a handful of rate-limited reconnects, got "
                f"{cap.reconnects}")
        finally:
            cap.is_running = False
            cap.thread.join(timeout=2.0)
    finally:
        _restore()


def test_reconnect_after_s_must_be_shorter_than_stale_after_s():
    """Enforced because the ordering matters, not just as a nice-to-have:
    if the external staleness check could fire before this class even
    attempts to self-heal, callers would see a disconnected stream that
    a moment later fixes itself for reasons they cannot observe."""
    _patch(_FakeCV2Capture)
    try:
        try:
            RtspCapture("rtsp://fake", stale_after_s=1.0,
                       reconnect_after_s=2.0)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                "accepted reconnect_after_s >= stale_after_s")
    finally:
        _restore()


if __name__ == "__main__":
    test_sustained_failure_triggers_a_reopen_and_recovers()
    print("  sustained failure triggers a reopen and recovers")
    test_brief_failure_does_not_reopen()
    print("  a single brief failure does not trigger a reopen")
    test_unreachable_camera_does_not_spin_reconnects()
    print("  an unreachable camera gets rate-limited reconnects, not a busy loop")
    test_reconnect_after_s_must_be_shorter_than_stale_after_s()
    print("  reconnect_after_s >= stale_after_s is rejected")
    print("\nall capture reconnect tests passed")
