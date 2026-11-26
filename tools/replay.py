#!/usr/bin/env python3
"""Replays recorded footage through the real pipeline and scores it.

This is the harness for the validation corpus. Every tracking number the
project currently quotes comes from synthetic sequences -- physically
derived blur, lighting and projector models, but not a real table. The
way to close that gap is to record real footage once and replay it
through the exact production pipeline as often as needed, which is what
this does.

The point is that it needs NO ground truth. You cannot label the true
position of a mini in a video without enormous effort, but the failures
that actually hurt are all self-evident from the track log:

  * identity flips   -- a track whose decoded id changes mid-life. There
                        is no legitimate reason for this; a real token
                        does not become a different token.
  * phantoms         -- a reported track that never decoded to any id.
                        v1's visible-red-ghost problem.
  * churn            -- the same id leaving and re-entering as a new
                        track, i.e. the "needs to reidentify constantly"
                        complaint that started v2.
  * coasting share   -- fraction of reports that were extrapolated rather
                        than measured. High means the detector is
                        struggling and the Kalman filter is covering.
  * decode rate      -- fraction of detections that produced an id.
  * real-time margin -- ms/frame against the source frame interval.

All of those are measurable from a recording of tokens simply being
moved around, with no annotation at all. Record footage, run this, and
compare the summary against docs/VALIDATION.md's synthetic numbers.

Usage:
    python tools/replay.py recording.mp4 --calibration data/calibration
    python tools/replay.py recording.mp4 --calibration data/calibration \\
        --annotate annotated.mp4 --json report.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tracker.runtime import _bgr_to_gray
from tracker.vision.calibration.session import CalibrationSession
from tracker.vision.codebook import load_codebook
from tracker.vision.geometry import RingGeometry


def _open(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, n


def homomorphic_view(gray, px_per_mm, geom):
    """Full-frame picture of what the decoder's illumination rejection is
    doing, for eyeballing against the raw frame.

    This is an ANALOGUE, not the real thing. decode.py high-passes the 1D
    strip unwrapped around a single marker's ring, so there is no
    high-passed frame anywhere in the pipeline to dump out. What is
    shared is the operation: take a log, so the projector's multiplicative
    contribution becomes additive, then subtract the slowly-varying part.
    Here that subtraction is a large-sigma blur in 2D rather than dropping
    angular harmonics in 1D.

    Sigma is tied to the token radius: the point is to remove light that
    varies more slowly than one token while keeping the chip detail, which
    is the same scale separation the 1D version relies on.
    """
    import cv2
    lg = np.log(np.maximum(gray.astype(np.float64),
                           max(1e-6, 1e-4 * float(gray.max()))))
    sigma = max(2.0, geom.r_out * px_per_mm)
    # The low-pass is taken on a shrunken copy and scaled back up, rather
    # than blurring a 4K frame with a ~90px sigma directly. That naive
    # version needs a ~500px kernel per frame and measured ~30x slower
    # than the whole rest of the pipeline -- it turned a 5-minute render
    # into one that had not cleared 10% after several minutes. Downscaling
    # by the same factor as the blur radius is exact enough here: the
    # thing being estimated is by definition the part of the image with no
    # detail at this scale.
    k = max(1, int(sigma / 4.0))
    if k > 1:
        small = cv2.resize(lg, (lg.shape[1] // k, lg.shape[0] // k),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / k,
                                 borderType=cv2.BORDER_REPLICATE)
        low = cv2.resize(small, (lg.shape[1], lg.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    else:
        low = cv2.GaussianBlur(lg, (0, 0), sigma,
                               borderType=cv2.BORDER_REPLICATE)
    hp = lg - low
    # Fixed +/-3 sigma window rather than per-frame min/max, so brightness
    # in the output means the same thing from frame to frame -- auto-scaling
    # each frame would hide exactly the illumination changes this is meant
    # to show.
    sd = float(hp.std()) or 1.0
    return np.clip(128.0 + 128.0 * hp / (3.0 * sd), 0, 255).astype(np.uint8)


def replay(video, calibration_dir, n_ids=6, limit=None, annotate=None,
           rescan_hz=0.0, progress=True, annotate_highpass=None):
    import cv2

    geom = RingGeometry()
    geom.validate()
    session = CalibrationSession(calibration_dir, geom=geom,
                                 codebook=load_codebook(n_ids=n_ids))
    st = session.status()
    # status() reports {"calibrated": False} rather than None for an
    # uncalibrated stage, so test the flag, not the dict's truthiness.
    if not st["playarea"].get("calibrated"):
        raise SystemExit(
            "play-area is not calibrated in "
            f"{calibration_dir}. Replay needs it: without a known px/mm "
            "there is no scale to measure positions against. Run the "
            "play-area step in the dashboard against a frame from this "
            "same camera position first.")
    pipe = session.build_pipeline(rescan_hz=rescan_hz)

    cap, src_fps, n_frames = _open(video)
    if limit:
        n_frames = min(n_frames, limit) if n_frames else limit
    frame_dt = 1.0 / src_fps if src_fps > 0 else 1.0 / 15.0

    writer = None
    writer_hp = None
    # Per-track history, keyed by the tracker's own track key, so an
    # identity flip within one track is distinguishable from two separate
    # tracks that happened to hold the same id.
    track_ids = defaultdict(list)
    track_frames = defaultdict(int)
    id_runs = []           # (id, track_key) in first-appearance order
    coasting = total_reports = 0
    decodes_attempted = decodes_ok = 0
    per_frame_ms = []
    ids_per_frame = []

    i = 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (limit and i >= limit):
            break
        gray = _bgr_to_gray(frame.astype(np.float64)) if frame.ndim == 3 \
            else frame.astype(np.float64)
        rect = session.rectify(gray)

        # Virtual clock from the source timestamps, not wall time: the
        # tracker's coasting and rescan cadence are time-based, so
        # replaying faster or slower than real time must not change the
        # result.
        now = i * frame_dt
        t0 = time.perf_counter()
        reported = pipe.process(rect, now=now)
        per_frame_ms.append(1000.0 * (time.perf_counter() - t0))

        seen = set()
        for t in reported:
            total_reports += 1
            track_frames[t.key] += 1
            if t.misses > 0:
                coasting += 1
            if t.marker_id is not None:
                seen.add(t.marker_id)
                if not track_ids[t.key] or track_ids[t.key][-1] != t.marker_id:
                    track_ids[t.key].append(t.marker_id)
                    id_runs.append((t.marker_id, t.key))
            else:
                track_ids[t.key]  # touch, so phantom tracks are counted
        ids_per_frame.append(len(seen))

        if annotate or annotate_highpass:
            vis = cv2.cvtColor(np.clip(rect, 0, 255).astype(np.uint8),
                               cv2.COLOR_GRAY2BGR) if rect.ndim == 2 else rect
            vis_hp = None
            if annotate_highpass:
                vis_hp = cv2.cvtColor(
                    homomorphic_view(rect, pipe.px_per_mm, geom),
                    cv2.COLOR_GRAY2BGR)
            for t in reported:
                # Draw where tokens_mm() would REPORT the token, not where
                # the filter extrapolates it -- otherwise the annotated
                # video shows a confident circle sliding across the table
                # that no consumer of the API would ever have been given.
                px_, py_ = t.report_pos
                x, y = int(px_), int(py_)
                r = int(geom.r_out * pipe.px_per_mm)
                named = t.marker_id is not None
                # green = decoded and measured, amber = coasting,
                # red = reported but never decoded (a phantom)
                col = (0, 200, 0) if named and t.misses == 0 else \
                      (0, 180, 230) if named else (0, 0, 230)
                # Identical overlay on both renders: the two videos differ
                # ONLY in the underlying image, so anything that looks
                # different between them is the filtering, not the tracking.
                for canvas in (vis, vis_hp):
                    if canvas is None:
                        continue
                    cv2.circle(canvas, (x, y), r, col, 2)
                    cv2.putText(canvas, str(t.marker_id) if named else "?",
                                (x - 6, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, col, 2)
            if annotate:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(annotate,
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps or 15.0, (w, h))
                writer.write(vis)
            if annotate_highpass:
                if writer_hp is None:
                    h, w = vis_hp.shape[:2]
                    writer_hp = cv2.VideoWriter(annotate_highpass,
                                                cv2.VideoWriter_fourcc(*"mp4v"),
                                                src_fps or 15.0, (w, h))
                writer_hp.write(vis_hp)

        i += 1
        if progress and i % 50 == 0:
            print(f"  {i}{'/' + str(n_frames) if n_frames else ''} frames",
                  end="\r", flush=True)

    cap.release()
    if writer is not None:
        writer.release()
    if writer_hp is not None:
        writer_hp.release()
    wall = time.perf_counter() - t_start
    if progress:
        print(" " * 40, end="\r")

    flips = [(k, v) for k, v in track_ids.items() if len(v) > 1]
    phantoms = [k for k, v in track_ids.items() if not v]
    appearances = Counter(mid for mid, _ in id_runs)
    # An id that appears in more track_keys than once has been dropped and
    # re-acquired -- the "constantly reidentifying" failure.
    churn = {mid: c - 1 for mid, c in appearances.items() if c > 1}

    ms = np.array(per_frame_ms) if per_frame_ms else np.array([0.0])
    return {
        "video": os.path.basename(video),
        "frames": i,
        "source_fps": round(src_fps, 2),
        "px_per_mm": round(pipe.px_per_mm, 3),
        "calibration": {
            "lens": st["lens"]["calibrated"],
            "detector_tuned": st["detector"]["calibrated"],
            "min_peak": st["detector"]["min_peak"],
            "min_margin": st["detector"]["min_margin"],
            "coverage": st["playarea"].get("verdict"),
        },
        "identity_flips": len(flips),
        "flip_detail": [{"track": k, "ids": v} for k, v in flips],
        "phantom_tracks": len(phantoms),
        "id_churn": churn,
        "distinct_ids_seen": sorted(appearances),
        "median_ids_per_frame": float(np.median(ids_per_frame)) if ids_per_frame else 0.0,
        "max_ids_per_frame": int(max(ids_per_frame)) if ids_per_frame else 0,
        "coasting_fraction": round(coasting / max(1, total_reports), 4),
        "timing_ms": {
            "mean": round(float(ms.mean()), 2),
            "p50": round(float(np.percentile(ms, 50)), 2),
            "p95": round(float(np.percentile(ms, 95)), 2),
            "max": round(float(ms.max()), 2),
        },
        "sustainable_fps": round(1000.0 / max(ms.mean(), 1e-6), 1),
        "realtime_factor": round((i * frame_dt) / wall, 2) if wall else None,
        "pipeline_timing_ms": {k: round(v, 3)
                               for k, v in pipe.timing_report().items()},
    }


def verdict(r, target_fps=15.0, n_expected=None):
    """Turns the summary into pass/attention lines against the project's
    stated requirements, so a recording is scored rather than just
    described."""
    lines = []
    ok = True

    if r["identity_flips"] == 0:
        lines.append("  PASS  no identity flips")
    else:
        ok = False
        lines.append(f"  FAIL  {r['identity_flips']} identity flip(s) -- a track "
                     f"changed id mid-life: {r['flip_detail'][:3]}")

    if r["phantom_tracks"] == 0:
        lines.append("  PASS  no phantom tracks (nothing reported without an id)")
    else:
        ok = False
        lines.append(f"  FAIL  {r['phantom_tracks']} phantom track(s) -- reported "
                     f"but never decoded; this is v1's red-ghost failure")

    if not r["id_churn"]:
        lines.append("  PASS  no id churn (no token dropped and re-acquired)")
    else:
        ok = False
        lines.append(f"  FAIL  id churn {r['id_churn']} -- these ids were lost "
                     f"and re-acquired as new tracks")

    if r["sustainable_fps"] >= target_fps:
        lines.append(f"  PASS  {r['sustainable_fps']}fps sustainable "
                     f"(>= {target_fps} required), p95 "
                     f"{r['timing_ms']['p95']}ms/frame")
    else:
        ok = False
        lines.append(f"  FAIL  only {r['sustainable_fps']}fps sustainable, "
                     f"below the {target_fps}fps requirement")

    cf = r["coasting_fraction"]
    if cf <= 0.10:
        lines.append(f"  PASS  {cf:.1%} of reports were coasted (low)")
    else:
        lines.append(f"  WARN  {cf:.1%} of reports were extrapolated, not "
                     f"measured -- the filter is covering for the detector")

    if n_expected is not None:
        got = r["max_ids_per_frame"]
        if got >= n_expected:
            lines.append(f"  PASS  saw all {n_expected} expected tokens at once")
        else:
            ok = False
            lines.append(f"  FAIL  never saw more than {got} of "
                         f"{n_expected} tokens simultaneously")
    return ok, lines


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--calibration", required=True,
                    help="calibration directory (e.g. data/calibration)")
    ap.add_argument("--n-ids", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N frames")
    ap.add_argument("--annotate", default=None,
                    help="write an overlaid video for eyeballing (green=tracked, "
                         "amber=coasting, red=phantom)")
    ap.add_argument("--annotate-highpass", default=None,
                    help="write a SECOND overlaid video showing the "
                         "illumination-rejected view (see homomorphic_view: "
                         "a 2D analogue of the decoder's per-marker "
                         "high-pass, for comparing against --annotate)")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--target-fps", type=float, default=15.0)
    ap.add_argument("--expect-tokens", type=int, default=None,
                    help="how many tokens are actually in the recording, so "
                         "missing ones are reported as a failure")
    args = ap.parse_args()

    print(f"Replaying {args.video} ...")
    r = replay(args.video, args.calibration, n_ids=args.n_ids,
               limit=args.limit, annotate=args.annotate,
               annotate_highpass=args.annotate_highpass)

    print(f"\n{r['frames']} frames at {r['source_fps']}fps source, "
          f"{r['px_per_mm']}px/mm")
    print(f"coverage: {r['calibration']['coverage']}")
    print(f"ids seen: {r['distinct_ids_seen']}  "
          f"(median {r['median_ids_per_frame']:.0f}, max {r['max_ids_per_frame']} "
          f"per frame)")
    print(f"timing:   mean {r['timing_ms']['mean']}ms, p95 "
          f"{r['timing_ms']['p95']}ms, max {r['timing_ms']['max']}ms "
          f"-> {r['sustainable_fps']}fps sustainable")
    print()
    ok, lines = verdict(r, args.target_fps, args.expect_tokens)
    print("\n".join(lines))
    print(f"\n{'PASS' if ok else 'ATTENTION NEEDED'}")

    if args.annotate:
        print(f"\nAnnotated video: {args.annotate}")
    if args.annotate_highpass:
        print(f"Illumination-rejected view: {args.annotate_highpass}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"Report: {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
