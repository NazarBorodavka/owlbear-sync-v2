"""End-to-end moving-sequence test -- the metric that actually matches the
user's experience, and the one v1 never measured.

Six tokens move over a 150-frame sequence: some stationary, some dragged
slowly, one flicked fast, with hands reaching in and out, a lighting
gradient, clutter on the table and per-token motion blur. What is scored
is not per-frame decode rate but what the operator sees:

  identity stability -- how often a token's reported id changes to the
                        wrong value (v1's "2 steals 3's id")
  phantoms           -- tokens reported that do not exist (v1's red ghosts)
  dropouts           -- fraction of frames a real token is not reported
  position error     -- in millimetres, against ground truth
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tracker.vision.geometry import RingGeometry
from tracker.vision.codebook import load_codebook
from tracker.vision.pipeline import Pipeline
from tracker.vision import simulate as sim

PX_MM = 3.2
GEOM = RingGeometry()
CB = load_codebook(n_ids=8)
W, H = 900, 640
FPS = 15.0


def make_script(n_frames, rng):
    """Six tokens with distinct motion profiles."""
    ids = [0, 1, 2, 3, 4, 5]
    base = [(150, 150), (420, 150), (700, 170), (170, 430), (450, 450), (720, 430)]
    tracks = []
    for i, (bx, by) in enumerate(base):
        tracks.append({"id": ids[i], "x0": float(bx), "y0": float(by),
                       "rot": float(rng.uniform(0, 360))})
    script = []
    for f in range(n_frames):
        t = f / FPS
        frame = []
        for i, tk in enumerate(tracks):
            x, y, speed, heading = tk["x0"], tk["y0"], 0.0, 0.0   # speed in mm/s
            if i == 1:                               # slow drag right
                if 1.0 < t < 5.0:
                    x += (t - 1.0) * 45.0
                    speed, heading = 45.0 / PX_MM, 0.0
                elif t >= 5.0:
                    x += 4.0 * 45.0
            if i == 2:                               # fast flick at t=3
                if 3.0 < t < 3.9:
                    x -= (t - 3.0) * 420.0
                    speed, heading = 420.0 / PX_MM, np.pi
                elif t >= 3.9:
                    x -= 0.9 * 420.0
            if i == 4:                               # slow diagonal
                if t > 2.0:
                    x += (t - 2.0) * 22.0
                    y -= (t - 2.0) * 14.0
                    speed = np.hypot(22.0, 14.0) / PX_MM
                    heading = np.arctan2(-14.0, 22.0)
            occ = 0.0
            if i == 0 and 2.0 < t < 4.0:             # hand over token 0
                occ = 150.0
            if i == 3 and 5.0 < t < 6.0:             # brief grab
                occ = 110.0
            frame.append({"id": tk["id"], "x": x, "y": y, "rot_deg": tk["rot"],
                          "speed_mm_s": speed, "heading": heading,
                          "occl_deg": occ, "occl_start": 20.0 + 25 * i})
        script.append(frame)
    return script


def run(n_frames=150, seed=5, verbose=True, exposure_s=1 / 60):
    rng = np.random.default_rng(seed)
    script = make_script(n_frames, rng)
    clutter = [(300, 560, 18, 0.2), (620, 70, 14, 0.85),
               (820, 560, GEOM.r_out * PX_MM, 0.3),   # token-sized plain disc
               (80, 300, 22, 0.1)]
    pipe = Pipeline(GEOM, CB, PX_MM, rescan_hz=3.0, expected_tokens=6)

    reported_for = {i: 0 for i in range(6)}
    wrong_id_frames = 0
    phantom_frames = 0
    pos_err_mm = []
    id_history = {}
    id_flips = 0
    t_wall = []

    for f, frame in enumerate(script):
        img = sim.render_frame(frame, GEOM, CB, PX_MM, (H, W), rng,
                               clutter=clutter, exposure_s=exposure_s)
        t0 = time.perf_counter()
        out = pipe.process(img, now=f / FPS)
        t_wall.append(time.perf_counter() - t0)

        truth = {tk["id"]: (tk["x"], tk["y"]) for tk in frame}
        seen = {}
        for tok in out:
            mid = tok.marker_id
            x, y = tok.pos
            if mid in truth:
                tx, ty = truth[mid]
                err = np.hypot(x - tx, y - ty) / PX_MM
                if err < 12.0:                      # matched to its own truth
                    seen[mid] = err
                    pos_err_mm.append(err)
                else:
                    wrong_id_frames += 1            # reported far from truth
            else:
                phantom_frames += 1
            prev = id_history.get(tok.key)
            if prev is not None and prev != mid:
                id_flips += 1
            id_history[tok.key] = mid
        for mid in seen:
            reported_for[mid] += 1

    if verbose:
        n = n_frames
        print(f"  frames={n}  tokens=6  wall={1000*np.mean(t_wall):.1f} ms/frame "
              f"({1/np.mean(t_wall):.0f} fps capable)")
        print(f"  rescans={pipe.stats['rescans']}  "
              f"detect={pipe.timing_report()['t_detect']:.1f}ms "
              f"decode={pipe.timing_report()['t_decode']:.1f}ms "
              f"track={pipe.timing_report()['t_track']:.2f}ms  (per frame)")
        for i in range(6):
            print(f"    token {i}: reported in {100*reported_for[i]/n:5.1f}% of frames")
        print(f"  position error: median={np.median(pos_err_mm):.2f}mm "
              f"p95={np.percentile(pos_err_mm, 95):.2f}mm")
        print(f"  WRONG-id reports={wrong_id_frames}  phantoms={phantom_frames}  "
              f"id flips on a track={id_flips}")
    return dict(reported=reported_for, wrong=wrong_id_frames,
                phantom=phantom_frames, flips=id_flips,
                med_err=float(np.median(pos_err_mm)),
                ms=float(1000 * np.mean(t_wall)), n=n_frames)


if __name__ == "__main__":
    print("=== 150-frame moving sequence, 6 tokens, 15fps, exposure 1/60s ===")
    r = run(exposure_s=1 / 60)
    print()
    print("=== same sequence, dim room (camera exposes the full 1/15s frame) ===")
    dim = run(exposure_s=1 / 15)
    print()
    assert r["wrong"] == 0, "reported a token far from its true position"
    assert r["phantom"] == 0, "reported a token that does not exist"
    assert r["flips"] == 0, "a track changed identity mid-life"
    worst = min(r["reported"].values()) / r["n"]
    assert worst > 0.70, f"a token was reported in only {100*worst:.0f}% of frames"
    print("PASS: no wrong ids, no phantoms, no identity flips, "
          f"worst-token coverage {100*worst:.0f}%")
    print(f"      (dim-room run degrades to {100*min(dim['reported'].values())/dim['n']:.0f}% "
          f"worst-token coverage, {dim['wrong']} mislocated, {dim['phantom']} phantoms "
          f"-- argues for capping camera exposure at 1/60s)")
