#!/usr/bin/env python3
"""Records an RTSP camera stream straight to an mp4 on THIS machine.

For the validation corpus docs/VALIDATION.md has been asking for: record
with the camera in its real mounted position, the projector running, and
real tokens on the table, then score the result with tools/replay.py.

Deliberately shells out to ffmpeg with -c copy rather than going through
cv2.VideoCapture + VideoWriter: stream-copy writes the camera's own H.264
bytes straight to the container with no re-encode, so the recording is
the camera's actual signal, not a second lossy compression pass stacked
on top of it -- which matters here, because the whole point of a
validation corpus is to measure against what the camera really delivers.

Also deliberately uses TCP, not the UDP transport tracker/capture.py uses
for LIVE tracking. Those two are optimising for different things: live
tracking wants to drop a bad packet and keep going, because stalling for
a retransmit is worse than losing one frame. A permanent recording wants
the opposite -- a dropped packet here is a corrupted frame baked
permanently into your only copy of this clip, and nothing is waiting on
low latency. TCP's retransmit-and-block behaviour is the correct trade
for a file that has to last.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tracker.settings import redact

# Matches the 7 clip types docs/VALIDATION.md asks for, so --label is
# guided rather than free text that drifts from that table.
CLIP_LABELS = ["still", "slow", "fast", "occluded", "crossing",
              "map-change", "clutter"]


def record(url, out_path, duration=None, transport="tcp", overwrite=False):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not on PATH -- install it (the ffmpeg package) "
            "before recording")
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists -- real footage is hard to "
            f"reproduce exactly, so this refuses to overwrite it silently. "
            f"Pass --force if you mean to.")

    # -timeout bounds socket I/O (connect, and a stalled read) in
    # microseconds. Without it, a wrong stream path or an unresponsive
    # camera leaves ffmpeg blocked indefinitely with NO output at all --
    # measured directly: a bad path here just hangs silently rather than
    # erroring, which is indistinguishable from "still connecting" until
    # you give up waiting. 15s is generous for a local-network RTSP
    # handshake and short enough not to be annoying when it legitimately
    # fails.
    # -use_wallclock_as_timestamps re-stamps every packet with the time it
    # actually arrived HERE, instead of trusting the camera's own RTP
    # timestamps. Found necessary by testing against the real camera: on
    # (re)connecting, it dumped several seconds of buffered backlog in a
    # single near-instant burst, with THOSE frames carrying their
    # original, now-stale timestamps -- which -c copy preserves verbatim.
    # The result was a recording whose declared average framerate was
    # roughly half the camera's real ~20fps and whose timeline visibly
    # "skipped" from one second to a much later one. That is not just a
    # cosmetic playback glitch: tools/replay.py reads this exact
    # container-level average fps to pace every frame it scores, so a
    # corrupted value here would have silently skewed every speed- and
    # timing-dependent metric in the validation corpus this tool exists
    # to produce.
    cmd = ["ffmpeg", "-use_wallclock_as_timestamps", "1",
          "-rtsp_transport", transport, "-timeout", "15000000", "-i", url]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-c", "copy", "-movflags", "+faststart",
           "-y" if overwrite else "-n", out_path]

    print(f"Recording {redact(url)} -> {out_path}")
    print(f"  transport={transport}  "
         f"duration={f'{duration}s' if duration else 'until Ctrl+C'}")

    # Printed command never includes the raw URL -- same discipline as the
    # dashboard's redact(), applied here because a shell history or a
    # terminal scrollback is just as real a leak surface as a log file.
    printable = [redact(a) if a == url else a for a in cmd]
    print(f"  $ {' '.join(printable)}\n")

    t0 = time.time()
    # Run as its own process group so Ctrl+C here can send SIGINT
    # specifically to ffmpeg (which finalises the mp4's container
    # cleanly, writing a valid moov atom) rather than relying on the
    # default SIGINT propagation, which on some shells kills ffmpeg with
    # no chance to flush and leaves a truncated, unplayable file.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        for line in proc.stdout:
            # ffmpeg echoes the exact input argument -- including embedded
            # credentials -- into its OWN error/info lines on a failed or
            # misconfigured connection. The redaction above only covered
            # the command *this script* printed; ffmpeg's output needed
            # the same treatment, or a bad URL leaks the password straight
            # into the terminal/shell history regardless.
            line = line.replace(url, redact(url) or "***")
            if "frame=" in line or "error" in line.lower():
                print("  " + line.strip()[-100:])
        proc.wait()
    except KeyboardInterrupt:
        print("\n  stopping (finalising the recording) ...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Seen in practice: a connection stuck inside the RTSP
            # handshake (e.g. a wrong stream path leaving the camera
            # never responding) can swallow SIGINT without exiting. The
            # earlier version of this let that TimeoutExpired escape
            # uncaught, which crashed with a raw traceback instead of
            # reporting a clean error -- the exact kind of thing that
            # looks fine until a real camera does something the happy
            # path never saw.
            print("  not responding to Ctrl+C -- killing it")
            proc.kill()
            proc.wait(timeout=5)

    elapsed = time.time() - t0
    if proc.returncode not in (0, 255) or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg exited {proc.returncode}; no valid "
                           f"recording was produced")
    return elapsed


def probe(out_path):
    """Reports what was actually recorded, so a bad capture (wrong
    resolution, truncated, corrupt) is caught here rather than discovered
    later as a confusing tools/replay.py failure."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=width,height,avg_frame_rate,codec_name,nb_frames",
         "-show_entries", "format=duration,size",
         "-of", "default=noprint_wrappers=1", out_path],
        capture_output=True, text=True)
    return r.stdout


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("TRACKER_CAMERA_URL"),
                    help="rtsp://user:pass@host:554/... (default: "
                         "$TRACKER_CAMERA_URL)")
    ap.add_argument("--out", default=None,
                    help="output .mp4 path (default: recordings/<label-or-"
                         "clip>_<timestamp>.mp4)")
    ap.add_argument("--label", choices=CLIP_LABELS, default=None,
                    help="which VALIDATION.md clip type this is, used only "
                         "to name the output file")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to record (default: until Ctrl+C)")
    ap.add_argument("--transport", choices=["tcp", "udp"], default="tcp",
                    help="rtsp transport (default tcp -- see module "
                         "docstring for why this differs from live capture)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out if it already exists")
    args = ap.parse_args()

    if not args.url:
        print("ERROR: no --url given and $TRACKER_CAMERA_URL is not set",
              file=sys.stderr)
        return 2

    out_path = args.out
    if out_path is None:
        os.makedirs("recordings", exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = args.label or "clip"
        out_path = os.path.join("recordings", f"{name}_{stamp}.mp4")

    try:
        elapsed = record(args.url, out_path, duration=args.duration,
                         transport=args.transport, overwrite=args.force)
    except (RuntimeError, FileExistsError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nWrote {out_path} ({size_mb:.1f}MB, {elapsed:.1f}s wall time)")
    print("\n--- ffprobe ---")
    print(probe(out_path))
    print("Next: score it against the real pipeline --")
    print(f"  python tools/replay.py {out_path} --calibration data/calibration "
         f"--expect-tokens 6 --annotate annotated.mp4 --json report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
