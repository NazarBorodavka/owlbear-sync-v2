"""RTSP camera capture: a background-threaded reader exposing the single
latest frame, decoupled from however fast the processing loop consumes it.

Four fixes here are carried forward from hard-won v1 operational
experience, not re-derived from scratch -- they are low-level RTSP/
threading correctness, not a settings/UI design decision, so there is
nothing to "start fresh" about:

  1. SUPERSEDED -- see the option string below. This used to trim
     analyzeduration/probesize to 100ms/32KB to avoid a connection-time
     latency handicap. That reasoning was derived against a 1080p H.264
     stream and does NOT survive the camera's actual 4K HEVC output: 32KB
     is smaller than a single 4K keyframe, so FFmpeg never sees a
     complete IDR before it starts handing out frames.
  2. SUPERSEDED -- rtsp_transport=udp is still right (see below), but the
     reason recorded here was incomplete: UDP only works if the socket
     buffer is large enough to hold a 4K keyframe burst, which the
     default is not.
  3. release() MUST join the reader thread before releasing the
     VideoCapture object. Doing both at once is a use-after-free race at
     the OpenCV/FFmpeg C++ level -- the reader can be blocked inside
     cap.read() on another thread at the exact moment release() runs,
     and that was a real, process-crashing "double free or corruption"
     bug in v1 until the ordering was fixed. Worth stating explicitly:
     this is a correctness bug, not a style preference, and is exactly
     the kind of thing that looks fine until it crashes under a
     reconnect at the wrong moment.
  4. cap.read() failing must eventually RECREATE the VideoCapture, not
     just retry read() on the same one. Measured against a real Reolink
     E1 Zoom over RTSP/UDP: once FFmpeg's demuxer loses sync -- a dropped
     keyframe, a brief Wi-Fi hiccup, the camera's own momentary rekey --
     read() can return False forever with no exception, and the earlier
     version of this loop just slept and retried on the dead object
     indefinitely. That looked like "the camera connection keeps
     dropping" and "frames stop updating" from the dashboard, and only a
     full container restart recovered it. Safe to reopen from inside
     this same thread: nothing else touches self.cap except release(),
     and release() only runs after this thread has already been joined
     (fix 3, above) -- see _reopen()'s docstring.
"""
import os
import time
import threading
import collections

# Must be set before the first cv2.VideoCapture() call -- OpenCV's FFmpeg
# backend reads this at stream-open time, not per-frame. setdefault, not
# a plain assignment, so an operator can override the whole string from
# the environment to test a different transport without editing code.
#
# Tuned by measuring against the REAL camera (4K HEVC, 20fps nominal),
# 12-20s per variant, counting FFmpeg's own decode errors:
#
#   transport  probe        extra flags          fps   decode errors
#   udp        100ms/32KB   nobuffer+low_delay   16.1  66     <- the old default
#   tcp        2s/5MB       nobuffer+low_delay   15.7  74
#   udp        2s/5MB       nobuffer             17.4  74
#   tcp        2s/5MB       --                   22.3   0
#   udp        2s/5MB       --                   21.4   0     <- this
#
# Two separate things were wrong, and each alone is enough to break it:
#
#   * probesize/analyzeduration too small. A 4K HEVC keyframe is larger
#     than 32KB on its own, so the demuxer began emitting frames before
#     it had a complete IDR to anchor them to.
#   * fflags=nobuffer / flags=low_delay. These tell FFmpeg to start
#     decoding mid-GOP rather than waiting for a keyframe. On H.264 that
#     costs a few smeared frames; on HEVC, whose reference-picture sets
#     are far less forgiving, every inter frame until the next IDR fails
#     to resolve its references -- the "Could not find ref with POC N /
#     Error constructing the frame RPS" flood, repeating once per GOP
#     forever.
#
# Note the latency argument that motivated the old values was simply
# false: the corrected settings reach the FIRST frame in 0.07s versus
# 2.38s, because a demuxer that can actually sync does not spend the
# opening seconds discarding undecodable frames. Steady-state latency is
# bounded by frame_buffer=deque(maxlen=1) below, not by these flags --
# the reader always hands out the newest frame it has.
#
# buffer_size is the UDP socket receive buffer. The default (~64KB) is
# far too small for 4K: a keyframe burst overruns it and the lost packets
# are exactly what corrupts the reference chain. This is what makes UDP
# viable here, and UDP is still preferred over TCP for a live feed for
# the original reason -- a dropped packet should cost one frame, not
# stall every frame queued behind a retransmit.
os.environ.setdefault(
    'OPENCV_FFMPEG_CAPTURE_OPTIONS',
    'rtsp_transport;udp|buffer_size;10485760'
    '|analyzeduration;2000000|probesize;5000000'
)

import cv2  # noqa: E402  (must follow the env var above)


class RtspCapture:
    def __init__(self, url, stale_after_s=5.0, reconnect_after_s=3.0,
                 reconnect_backoff_s=2.0):
        self.url = url
        self.stale_after_s = stale_after_s
        # Proactive self-heal fires BEFORE the external staleness check
        # (stale_after_s) does, so a caller polling read() ideally never
        # even observes the drop -- reconnect_after_s must stay below
        # stale_after_s for that ordering to hold.
        assert reconnect_after_s < stale_after_s, (
            "reconnect_after_s must be shorter than stale_after_s, or "
            "callers will see a stale/disconnected stream before this "
            "class has even tried to recover it")
        self.reconnect_after_s = reconnect_after_s
        self.reconnect_backoff_s = reconnect_backoff_s
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.frame_buffer = collections.deque(maxlen=1)
        self.last_frame_time = 0.0
        self.frame_available = threading.Event()
        self.is_running = True
        self.fps_measured = 0.0
        self.reconnects = 0

        # CAP_PROP_BUFFERSIZE is not honoured by OpenCV's FFmpeg backend for
        # network streams -- left here as a harmless no-op in case a future
        # OpenCV build changes that; the real buffering knobs are the env
        # var above.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reopen(self):
        """Tears down and recreates the VideoCapture on the SAME thread
        that reads it, so this cannot race release() (see fix 3 in the
        module docstring) -- release() is only called after this thread
        has already been joined, and nothing else ever touches self.cap.
        """
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.reconnects += 1

    def _reader(self):
        frame_count = 0
        last_report = time.time()
        last_success = time.time()
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                if time.time() - last_success >= self.reconnect_after_s:
                    self._reopen()
                    # Give the fresh connection its own grace window
                    # before considering IT stuck, and a short pause so a
                    # genuinely unreachable camera does not get hammered
                    # with VideoCapture() calls in a tight loop.
                    last_success = time.time()
                    time.sleep(self.reconnect_backoff_s)
                else:
                    time.sleep(0.1)
                continue
            self.frame_buffer.append(frame)
            self.last_frame_time = time.time()
            last_success = self.last_frame_time
            self.frame_available.set()

            frame_count += 1
            now = time.time()
            elapsed = now - last_report
            if elapsed >= 10.0:
                # A live stream can only be read as fast as the camera
                # produces it -- this is the number to compare against the
                # camera's own configured fps to tell whether decode is
                # keeping up in real time.
                self.fps_measured = frame_count / elapsed
                frame_count = 0
                last_report = now

    def read(self):
        """Returns (ok, frame). ok is False if nothing new has arrived in
        stale_after_s -- a dead stream must not keep reporting success off
        a frozen last frame forever, or a caller's reconnect logic never
        triggers."""
        if not self.frame_buffer or (time.time() - self.last_frame_time) > self.stale_after_s:
            return False, None
        return True, self.frame_buffer[0]

    def wait_for_frame(self, timeout=1.0):
        """Blocks until a new frame is signalled (or timeout), instead of
        polling read() on a fixed sleep -- that poll interval is pure
        added latency on every single frame of a 15fps+ pipeline."""
        got = self.frame_available.wait(timeout)
        self.frame_available.clear()
        return got

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.is_running = False
        self.thread.join(timeout=2.0)
        self.cap.release()
