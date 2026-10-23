"""RTSP camera capture: a background-threaded reader exposing the single
latest frame, decoupled from however fast the processing loop consumes it.

Three fixes here are carried forward from hard-won v1 operational
experience, not re-derived from scratch -- they are low-level RTSP/
threading correctness, not a settings/UI design decision, so there is
nothing to "start fresh" about:

  1. FFmpeg's RTSP defaults (analyzeduration=5s, probesize=5MB) impose a
     fixed ~5s latency handicap at connection time that a live decoder can
     never claw back. SDP already negotiates codec parameters during the
     RTSP handshake, so this is pure waste for RTSP specifically. Trimmed
     to 100ms/32KB -- found by testing to be the smallest values that
     still reliably parse SPS/PPS under normal network jitter; going all
     the way to near-zero occasionally left the stream unable to open at
     all under any jitter.
  2. rtsp_transport=udp: TCP's retransmit-and-block behaviour means one
     delayed/lost packet stalls every frame queued behind it. UDP just
     drops the bad packet and moves on, which is the right trade for a
     live local-network feed.
  3. release() MUST join the reader thread before releasing the
     VideoCapture object. Doing both at once is a use-after-free race at
     the OpenCV/FFmpeg C++ level -- the reader can be blocked inside
     cap.read() on another thread at the exact moment release() runs,
     and that was a real, process-crashing "double free or corruption"
     bug in v1 until the ordering was fixed. Worth stating explicitly:
     this is a correctness bug, not a style preference, and is exactly
     the kind of thing that looks fine until it crashes under a
     reconnect at the wrong moment.
"""
import os
import time
import threading
import collections

# Must be set before the first cv2.VideoCapture() call -- OpenCV's FFmpeg
# backend reads this at stream-open time, not per-frame.
os.environ.setdefault(
    'OPENCV_FFMPEG_CAPTURE_OPTIONS',
    'rtsp_transport;udp|fflags;nobuffer|flags;low_delay|analyzeduration;100000|probesize;32768'
)

import cv2  # noqa: E402  (must follow the env var above)


class RtspCapture:
    def __init__(self, url, stale_after_s=5.0):
        self.url = url
        self.stale_after_s = stale_after_s
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.frame_buffer = collections.deque(maxlen=1)
        self.last_frame_time = 0.0
        self.frame_available = threading.Event()
        self.is_running = True
        self.fps_measured = 0.0

        # CAP_PROP_BUFFERSIZE is not honoured by OpenCV's FFmpeg backend for
        # network streams -- left here as a harmless no-op in case a future
        # OpenCV build changes that; the real buffering knobs are the env
        # var above.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        frame_count = 0
        last_report = time.time()
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            self.frame_buffer.append(frame)
            self.last_frame_time = time.time()
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
