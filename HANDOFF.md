# Handoff: Physical Token Tracker v2

Written for a fresh agent picking this up in a new session with no prior
context. Read this before touching code. It covers what this project is,
every non-obvious design decision and why it was made that way, what is
verified vs. still assumed, and what to do next.

If you are the user: this is also just a snapshot of where things stand
and what's left — read the "Next steps" section first.

---

## 1. What this is

Tracks physical miniatures on a tabletop with an overhead camera and
mirrors their positions live into an [Owlbear Rodeo](https://owlbear.rodeo)
scene. A web app does the vision and serves an operator dashboard; a
companion Owlbear Rodeo extension reads the detected positions and moves
the matching scene tokens.

**This is v2 — a from-scratch rewrite, not an upgrade of v1.** The user's
original request (paraphrased): *"I think this whole project is built on
top of each other to the point that I just want to restart from scratch.
Forget all the setting types, UI, and other design decisions done
before. Fork this into a version 2. Keep the docker deployment and the
store repository sync, feel free to change the tracker type, research
online and consider all alternatives (assume use of current camera)."*

v1 lives at `../owlbear-sync` (sibling directory) and used CCTag fiducial
markers via a C++ library with a multi-minute cmake/boost/eigen Docker
build. v1 had unfixed field problems: phantom disks when moving tokens,
identity loss requiring constant re-identification, ghost trails. v2
shares **nothing** with v1 except the deployment pattern (Docker + GHCR +
ZimaOS app store sync) — marker design, detector, decoder, tracker,
calibration, UI, and extension are all new. **v1 markers do not decode
under v2.**

### Stated requirements (from the user, verify against these before
calling anything "done")

- Web app + Owlbear Rodeo extension
- Track ≥15fps for 6 active tokens, under variable lighting and partial
  occlusion
- Marker design keeps the **center 70% of the base diameter clear** for
  artwork, assuming **4cm diameter round bases**
- Automatic map zoom so grid size matches token base size
- Web UI names detected tokens, with persistent storage, **shared with**
  the extension
- Extension keeps scene-token ↔ app-token mapping **across sessions**
- Calibration for lens, image correction, and detector tuning — **each
  independently rerunnable, with persistent storage**
- SHOULD HAVE: extension tracks relative to **cast receiver screenspace**,
  not host screenspace
- User offered to provide a dataset for training a custom model if needed
  (not yet taken up — the matched-filter approach didn't need it)

### Hardware context (confirmed with the user)

- Camera: **Reolink E1 Zoom**, RTSP, motorized 3× zoom
- Mounted ~100cm above the table, looking straight down
- Host: Razer Book 13 running **ZimaOS** (a CasaOS-based NAS/app-hub OS)
- Markers printed on **40mm round bases**, with a 1mm margin → **38mm
  printed diameter**
- A **ceiling-mounted** projector projects D&D maps onto the table
  surface, overlaying the tokens — so lighting on the markers is
  variable and content-dependent (whatever the map looks like at that
  spot)
- User ran v1 in **night-vision/IR mode**, which "eliminated some of
  that projected contrast" but not all of it. Was asked to research and
  recommend the best camera mode for v2 — see §4.6.

---

## 2. Repo map

```
owlbear-sync-v2/
├── tracker/
│   ├── vision/              # pure-numpy core — no native deps, unit-testable
│   │   ├── geometry.py      # RingGeometry: marker layout, single source of truth
│   │   ├── codebook.py      # codebook search/quality + load_codebook() (frozen data)
│   │   ├── codebook_data.json  # THE committed codebook — printed markers derive from this
│   │   ├── decode.py        # Matcher: polar-unwrap + matched-filter decode
│   │   ├── detect.py        # Detector: gradient-vote ring finder + subpixel refine
│   │   ├── track.py         # Track + Tracker: Kalman tracking, identity-gated rescue
│   │   ├── pipeline.py       # per-frame cheap refine + slow full rescan orchestration
│   │   ├── marker.py        # printable SVG + numpy test-raster renderer (same geometry)
│   │   ├── grid.py          # projected-grid spacing measurement (for auto-zoom)
│   │   ├── simulate.py      # synthetic frame rendering: blur, occlusion, projector model
│   │   └── calibration/     # lens, play-area, detector-tune, session orchestrator, store
│   ├── web/
│   │   ├── app.py           # Flask app factory, all REST endpoints
│   │   ├── templates/, static/  # calibration wizard dashboard (vanilla JS, no build step)
│   ├── capture.py           # RtspCapture (FFmpeg via cv2, threaded, 3 proven v1 fixes)
│   ├── runtime.py           # TrackerRuntime: owns capture thread + pipeline + latest state
│   ├── names.py             # NameStore: marker_id -> operator name, persisted
│   └── settings.py          # SettingsStore: camera URL persistence (0600, holds a credential)
├── extension/                # Owlbear Rodeo extension (vanilla JS, scene-metadata persisted)
├── tools/
│   ├── generate_codebook.py # mints a NEW codebook (invalidates all printed markers)
│   ├── generate_markers.py  # emits printable SVG from the FROZEN codebook
│   └── replay.py            # scores recorded footage through the real pipeline, no ground truth needed
├── tests/                    # 11 files, plain scripts with __main__ runners (NOT pytest)
├── docs/
│   ├── VALIDATION.md        # what each test checks, known-good numbers, what's unverified
│   └── DEPLOY-ZIMAOS.md     # full ZimaOS deployment walkthrough
├── Dockerfile
├── docker-compose.yml        # standalone/custom-install compose
├── .github/workflows/docker-publish.yml   # test-gated CI -> GHCR -> store dispatch
└── README.md
```

Related repos (siblings, same parent directory):
- `../owlbear-sync` — v1. Still deployed. Do not touch unless asked.
- `../owlbear-sync-store` — the ZimaOS/CasaOS app store. Now serves
  **both** v1 (`com.nazarborodavka.owlbearsync`) and v2
  (`com.nazarborodavka.owlbearsync.v2`) as separate, independently
  installable apps. See `Apps/OwlbearTrackerV2/`.

---

## 3. Architecture

```
Reolink camera ──RTSP──▶ capture ──▶ lens undistort ──▶ play-area rectify
                                                              │
                                      ┌───────────────────────┘
                                      ▼
                         detect rings ──▶ decode ids ──▶ Kalman track
                                                              │
                      ┌───────────────────────────────────────┘
                      ▼                                   ▼
            dashboard (naming, calibration)      /api/tokens  (mm)
                                                          │
                                                          ▼
                                            Owlbear Rodeo extension
```

**Positions cross the API in millimetres relative to the calibrated
play-area origin** — never pixels, never screen coordinates. This is
what satisfies the cast-receiver SHOULD-HAVE: the extension converts mm
to the scene's own grid using `scene.grid.getDpi()` /
`scene.grid.getScale()`, so it works identically on host and cast
receiver screenspace without needing to know anything about either.

### 3.1 Marker design

Each token base has a thin **detection ring** (gradient-vote-found at a
known radius) and, inside it, a **code band** split into 20 angular
chips (black/white). Identity is read by polar-unwrapping the code band
into a 1D strip and cross-correlating it against a set of pre-selected
codewords **at all 20 possible rotations simultaneously** (via FFT
circular cross-correlation) — so identity and rotation are resolved
*jointly*. There is no separate orientation-locking sync mark, which
means nothing can be occluded to break orientation lock (this was a
specific ask: *"do you think the tags can be more resilient to mini dots
occlusion if we remove the orientation locking"* — yes, and this is why).

Default geometry for a 40mm base printed at 38mm (1mm margin):
```
base 38mm | clear <13.3mm | code 13.8-17.0mm (3.2mm, 20 chips) | det ring 17.5-19.0mm
```
The **center 70% of the diameter (26.6mm of 38mm) is clear** — satisfies
the stated requirement. Defined in `tracker/vision/geometry.py`'s
`RingGeometry` dataclass; `clear_frac: float = 0.70`.

**Two independent gates must both pass** before an id is reported
(`tracker/vision/decode.py`):
- `MIN_PEAK = 0.65` — absolute floor: "does this look like any codeword
  at all." v1 never asked this; it always picked the best-fitting
  codeword no matter how bad the fit, which is how bottle caps and lamp
  bases became tracked tokens.
- `MIN_MARGIN = 0.15` — "is the winner clearly ahead of the runner-up."

Anything failing either gate is reported as **no id**, never a guess.
Measured across every stress test: **0% confident-wrong-id rate.**

### 3.2 Codebook is frozen, not generated at runtime

`tracker/vision/codebook_data.json` is a **committed data file** —
`load_codebook()` reads it; nothing regenerates it at import time. This
was a deliberate fix (see commit `fd91bd1`) for two reasons, in order of
importance:

1. **Correctness.** Physical markers are *printed* from this codebook.
   If the decoder re-derived its expected codewords from a random-seed
   search at runtime, any change to that search (numpy RNG behavior
   across versions, a tweaked default, a reordered loop) would silently
   desync the decoder from markers already printed and on the table.
   This is exactly the bug class that cost v1 real time: a print run
   that came out mirrored relative to what the decoder expected, with no
   test or mechanism to catch it.
2. Startup cost: regenerating took 3.89s of every container start (whole
   startup was 5s). Loading the frozen file: 0.049s.

`tools/generate_codebook.py` is the **deliberate regeneration path** —
prompts for confirmation, because running it invalidates every printed
marker. `tools/generate_markers.py` emits printable SVG **from the same
frozen file** the decoder reads, so artwork and decoder cannot disagree.

**`tests/test_print_roundtrip.py` is the test that would have caught
v1's mirrored-print bug**: it rasterises the *actual* printable SVG with
`rsvg-convert` (an external renderer, not our code) and decodes it with
the real detector/matcher. All 6 markers decode to their own id, at 10
rotations including off-chip-grid angles.

**`tests/test_codebook_frozen.py`** guards against drift: asserts the
committed bytes are unchanged, still meet the correlation quality bar,
separate every pair at every rotation exhaustively (15 pairs × 20
rotations), that no codeword is confusable with a lighting gradient,
and that the generator still reproduces the committed bytes exactly. Also guards a real bug (see §3.3) — the codebook must not live
under any mountable data directory.

### 3.3 The codebook's location was a real deployment bug (fixed)

`codebook_data.json` was briefly at `tracker/data/codebook.json`, which
collided with the default `TRACKER_DATA_DIR`. Bind-mounting a host
directory there for calibration persistence — **exactly what
ZimaOS/CasaOS does** — shadowed the file, and the app died at startup
with `FileNotFoundError`. Named Docker volumes hid this completely
(they seed from image contents on first mount); bind mounts do not.
Found by actually building the image and mounting it both ways — **not**
by reading the code.

Fixed: codebook now lives at `tracker/vision/codebook_data.json` (beside
the module that reads it, not in any `data/` tree). Dockerfile sets
`TRACKER_DATA_DIR=/data` and declares it a volume. There's a regression
test (`test_codebook_does_not_live_in_a_mountable_data_directory`) that
asserts this structurally, not just that today's path happens to work.

**Lesson for whoever picks this up: build and run the actual container
before trusting any "it works" claim about deployment.** Three separate
real bugs (§3.3, §3.4, §3.5) were only found this way, not by code
review.

### 3.4 Detection → tracking pipeline

`tracker/vision/detect.py`: single-radius gradient vote (the geometry is
metrically known after calibration, so the ring radius is exactly known
— no need to search a radius range like v1's Hough-circle approach did).
Subpixel circle fit anchored on the ring's **inner edge**
(printed-white-to-printed-ink boundary — background-independent, unlike
anchoring on outer edge which depends on what's outside the marker).
`refine()` uses `trim=0.35` to discard the weakest rays in the fit, so a
partially-occluded marker's circle fit isn't dragged off-center by the
occluded side.

`tracker/vision/track.py`: Kalman filter per track, with:
- **Geometric gate capped at `min(base_px + 3σ, 3×base_px)`** — this was
  a real v1-class bug found during v2 dev: an uncapped quadratic
  acceleration term let a stale track's predicted position drift far
  enough to geometrically "steal" a different real token's detection.
  Long-range recovery after a track is lost now happens via **positive
  decoded-ID match**, not proximity.
- **Coast damping + 0.30s grace period** — undamped ballistic
  extrapolation during an occlusion was measured drifting 118mm before
  the fix.
- ROI re-search fires on the **first** failed frame, not after
  `misses > 0` — a fast flick was measured outrunning its search window
  when the fallback waited one frame too long.

`tracker/vision/pipeline.py`: cheap per-frame `refine_at()` around each
existing track's predicted position; a slow full-frame `detect()` rescan
on a separate cadence (`rescan_hz=3.0` default) to pick up new tokens or
recover ones the cheap path lost.

### 3.5 Camera lifecycle — two deployment-blocking bugs, both fixed

These were found while preparing the actual ZimaOS deployment, by
running the real built image, not by reading code:

1. **First-deploy deadlock.** `TrackerRuntime.start()` used to *require*
   a play-area calibration before starting capture. But play-area
   calibration requires clicking the four corners on a **live camera
   preview**. So: camera can't start until calibrated, can't calibrate
   until camera starts. Fixed — capture and tracking are now separate
   concerns; capture always starts, the pipeline is built if calibration
   exists and left `None` otherwise, and the preview serves the **raw**
   frame (not rectified — the play-area endpoint undistorts the
   *clicked points*, so showing an already-undistorted image would
   double-correct).

2. **Auto-start blocked the web server.** The camera URL now persists
   (`tracker/settings.py`, `SettingsStore`, atomic + `chmod 0600` since
   it holds the camera login) and auto-starts at boot — without this,
   every ZimaOS reboot left the tracker idle until a human retyped the
   RTSP URL. But the *first* implementation ran the connect attempt
   inline during `create_app()`. RTSP connect blocks up to 30s via
   FFmpeg's timeout, so an unreachable camera meant **the dashboard
   never served within 70 seconds** — measured against the real
   container. That's the worst possible failure for an appliance: the
   dashboard is the only place to fix the URL, and `restart:
   unless-stopped` loops on it forever. Fixed: auto-start now runs on a
   background daemon thread. Measured: 229ms to first response with a
   dead camera, vs. 225ms baseline with no camera configured at all.

   A `start()`/`stop()` race was found and fixed while testing this fix:
   `start()` used to publish `self._thread` *before* calling
   `.start()` on it, so a concurrent `stop()` (now reachable, since
   auto-start backgrounds while the dashboard is already usable) could
   call `.join()` on a thread that hadn't started yet and raise.

   **The test that caught this (`test_dashboard_serves_while_an_
   unreachable_camera_connects`) uses a stub capture that deliberately
   blocks on connect**, unlike the earlier stub which connected
   instantly and therefore couldn't have caught the bug. If you add
   capture-related tests, make sure the stub's timing characteristics
   actually resemble the real failure mode you're testing for.

Credential handling: `GET /api/camera` is behind auth and returns a
**redacted** URL (`tracker/settings.py`'s `redact()`, strips embedded
userinfo, fails closed — masks anything it can't parse rather than
echoing it). Verified: the secret never appears on `/api/tokens`,
`/api/version`, or anywhere unauthenticated.

### 3.6 Web app

`tracker/web/app.py`: Flask **application factory** pattern
(`create_app(data_dir=None, users=None)`), not a module-global — this
was also a fix (v1-style globals made the app untestable; see commit
history for the `del sys.modules[...]` trap that doesn't actually give
you a fresh module when re-importing a package attribute).

Key endpoints:
- `/api/tokens` — public, polled by the extension
- `/api/names` (GET/PUT) — shared naming store
- `/api/calibration/{lens,playarea,detector}` (POST/DELETE) — each
  independently persisted and rerunnable, per requirement
- `/api/calibration/playarea/preview` — shows predicted accuracy for a
  candidate corners/scale choice *before* committing
- `/api/camera` (GET/DELETE), `/api/camera/{start,stop}` (POST) — see
  §3.5
- `/api/grid/measure` — projected-grid spacing for auto-zoom
- `/api/version` — public, reports build version/branch/commit (from
  Docker build args) and the running codebook's fingerprint, so "did my
  update deploy, and is it the same codebook my markers were printed
  from" is answerable from the browser

### 3.7 Extension

`extension/main.js` — the scene-token ↔ app-token mapping is persisted
in **OBR scene metadata** (`METADATA_KEY =
"com.physicaltokentracker/mapping"`), not local storage, so it survives
reload and travels with the scene — satisfies the cross-session
requirement properly (local storage would be per-browser, not
per-scene).

mm→scene-px conversion:
```js
const dpi = await OBR.scene.grid.getDpi();
const scale = await OBR.scene.grid.getScale();
const mmPerCell = scale.parsed.multiplier * mmPerUnit(scale.parsed.unit);
const pxPerMm = dpi / mmPerCell;
scenePoint = { x: pt.x_mm * pxPerMm, y: pt.y_mm * pxPerMm };
```
`mmPerUnit()` handles ft/m/in/cm/mi/km with a warn-and-fallback for
anything else.

`runAutoZoom()`: `newScale = currentScale * measured_mm_per_cell /
TOKEN_BASE_MM` (TOKEN_BASE_MM = 40.0) — makes the stated "automatic map
zooming" requirement genuinely automatic (camera-measured, not manual
trial-and-error) via `tracker/vision/grid.py`'s projected-grid spacing
measurement.

### 3.8 Camera settings research (§ for the IR/projector question)

The user asked specifically: v1 ran Reolink in night-vision/IR mode,
which reduced but didn't eliminate projected-map contrast interference;
research and recommend the best mode for v2.

Conclusion reached and encoded in `tests/test_projector.py` +
`tracker/vision/simulate.py`'s `projector_illumination()` model: **force
IR/night mode on**, because the projector's visible-light output doesn't
reach the camera's IR-illuminated exposure — the projector is a
nuisance *visible*-light source, and the camera's own IR LEDs provide
consistent illumination outside that band. Forced IR mode was measured
(synthetically) to outperform plain color mode under projector
interference.

**Hard dependency surfaced by this**: marker ink must be **pigment /
carbon-black**, not dye-based. Dye-based black ink is ~0.8 reflectance
above 800nm (near-transparent to IR) while pigment/carbon black stays
near-zero reflectance (opaque) across that range. A dye-printed marker
under forced IR would show almost no contrast and simply fail to decode.
This is called out in `tools/generate_markers.py`'s printing checklist,
the README, and `docs/DEPLOY-ZIMAOS.md`. **This has NOT been physically
verified** — it's derived from published ink reflectance data, not
measured against an actual printed marker under the actual camera. See
§5, item 2.

Also recommended: exposure capped at 1/60s or faster. Motion blur is
physically `speed × exposure`, so this is derived, not guessed — and
it's the dominant factor in whether fast token drags survive, more than
frame rate. Measured synthetically: 1/60s tracks cleanly to 250mm/s;
1/15s (a plausible dim-room auto-exposure value) already fails at
130mm/s.

---

## 4. What is verified vs. what is still assumed

This distinction matters more than anything else in this document. Be
precise about it when talking to the user — don't let "verified against
synthetic sequences" collapse into "verified."

### 4.1 Verified against synthetic sequences (physically-derived models,
not a real table)

All of `tests/test_track.py`, `test_sequence.py`, `test_projector.py`,
`test_calibration.py`, `test_grid.py`. Known-good numbers are in
`docs/VALIDATION.md` — re-derive them by running the tests; don't trust
a stale copy pasted into chat. Headline numbers as of the last run:
0 wrong ids / 0 phantoms / 0 identity flips across a 150-frame moving
sequence at 3.2px/mm, 0.45mm median position error, 7.7ms/frame.

### 4.2 Verified against the real built Docker container

Image builds and runs; all 11 test suites pass inside the container
against the pinned `cv2 4.13.0` / `numpy 2.5.4` (one test,
`test_print_roundtrip.py`, skips inside the container because
`rsvg-convert` isn't installed there — it's a dev-time cross-check, not
meant to ship). Bind-mount persistence works and survives a restart.
Camera auto-start doesn't block the server. Credentials don't leak.
`/api/version` reports correctly. Published GHCR image
(`ghcr.io/nazarborodavka/owlbear-tracker-v2`) pulls anonymously (no
`docker login`) and boots correctly — confirmed directly, not assumed
from repo visibility settings.

### 4.3 Verified against GitHub Actions (real CI, real GHCR, real store)

CI run #1 passed end to end: tests → build → **smoke-test the built
image** → store dispatch. Store repo (`owlbear-sync-store`) now serves
both apps; `index.json` on the published `gh-pages` branch lists both
`com.nazarborodavka.owlbearsync` and `com.nazarborodavka.owlbearsync.v2`
with the correct pinned image tag.

### 4.4 NOT verified — real camera, real markers, real table

**Nothing listed in §4.1-4.3 has been run against an actual camera
pointed at an actual printed marker on an actual table.** Every
tracking number quoted anywhere in this project's docs comes from
synthetic rendering. This is the single most important caveat to carry
forward. Specifically unverified:

1. Whether pigment ink actually stays opaque under this specific
   camera's IR illumination (§3.8) — derived from published spectral
   data, not measured.
2. Whether the detector/decoder's accuracy numbers hold against real
   sensor noise, real lens characteristics, real projector brightness,
   and real motion blur characteristics (the synthetic blur model is
   physically derived but is still a model).
3. Whether 100cm mount height (current hardware setup) gives usable
   px/mm — `RingGeometry.min_px_per_mm()` and the play-area calibration
   UI will report the *actual* measured px/mm once a real camera is
   calibrated, but no one has done that yet. The README recommends
   ~70cm as more comfortable, computed from the Reolink's sensor specs,
   not measured.
4. Real RTSP stream behavior from the actual Reolink E1 Zoom (capture.py
   carries forward three proven-in-production v1 fixes — FFmpeg
   analyzeduration/probesize tuning, UDP transport, thread-join-before-
   release — but hasn't been exercised against this camera in v2's own
   codebase).

`tools/replay.py` exists specifically to close this gap: record real
footage once, replay it through the exact production pipeline, score it
with **no ground-truth labeling needed** (identity flips, phantom
tracks, id churn, and frame timing are all self-evident from the track
log alone — a real token never becomes a different token, so any id
change mid-track is definitionally a bug). `tests/test_replay.py`
validates the harness itself against a synthetic 6-token recording
(encoded to a real mp4, codec compression included) before trusting it
on real footage — a harness bug pointed at real footage would be
indistinguishable from a tracker regression otherwise.

**This is the highest-priority next step.** See §5.

### 4.5 Deployment-specific verification state

- v2 GHCR image: public, pullable, boots, smoke-tested by CI. ✅
- v2 store entry: live on `gh-pages`, correct pinned tag, distinct app
  id from v1 (`com.nazarborodavka.owlbearsync.v2`), won't collide with
  or replace a running v1 install. ✅
- **`STORE_DISPATCH_TOKEN` secret on the `owlbear-sync-v2` repo: NOT
  YET ADDED.** Without it, future pushes to `owlbear-sync-v2` publish a
  new image fine, but the store never learns about it (the dispatch
  step is `continue-on-error`, so this fails silently from CI's point of
  view) — ZimaOS will stay pinned at whatever tag it currently has and
  never offer an update. This is the single most likely thing to
  surprise the user later ("why isn't it updating") if not addressed.
  Fix: same fine-grained PAT used for v1 (scoped to
  `owlbear-sync-store`, contents: write), added as a repo secret on
  `owlbear-sync-v2`.
- Actual install on the user's ZimaOS box: in progress as of this
  writing — user was walked through store refresh / install steps but
  completion wasn't confirmed in this session.

---

## 5. Next steps, in priority order

1. **Record and replay real footage.** This is the standing commitment
   and the biggest gap between "well-tested" and "known to work."
   `docs/VALIDATION.md` lists 7 clip types worth recording (still, slow,
   fast, occluded, crossing, map-change, clutter) with what each tests.
   Command:
   ```bash
   python tools/replay.py recording.mp4 --calibration data/calibration \
       --expect-tokens 6 --annotate annotated.mp4 --json report.json
   ```
   Needs a play-area calibration to already exist for that camera
   position (replay has no scale without it — refuses rather than
   guessing). Compare the report against §4.1's synthetic numbers; a
   real clip that fails is more valuable than one that passes, because
   it's a concrete test case.

2. **Physically verify the pigment-vs-dye ink assumption.** Print one
   marker (`tools/generate_markers.py`), photograph it under the
   camera's forced-IR mode, confirm the detection ring and code band
   actually show the expected contrast. If this fails, the whole
   printing approach needs a different ink/printer recommendation before
   the user reprints a full marker set.

3. **Add `STORE_DISPATCH_TOKEN` to the `owlbear-sync-v2` repo.** Five
   minutes, prevents a confusing "it's not updating" report later.
   Settings → Secrets and variables → Actions → New repository secret.

4. **Confirm the actual ZimaOS install completes** and walk the user
   through first-run calibration (`docs/DEPLOY-ZIMAOS.md` §3) if not
   already done — in particular the play-area step, since the resulting
   px/mm determines whether the 100cm mount height is actually usable
   or needs to come down to ~70cm.

5. **VAAPI/OpenCL hardware decode** — v1 has this (optional, env-flag
   gated, automatic CPU fallback) for Intel iGPU hosts. v2 doesn't have
   it yet. Not needed unless CPU usage turns out to be a problem on the
   real host (Razer Book 13 — check `docker stats` under real load
   before deciding this is worth doing).

6. **WebSocket push instead of polling** for `/api/tokens` — noted but
   not started. Current polling is simple and tested; only worth doing
   if latency or request volume becomes an actual measured problem.

7. **If replay against real footage (step 1) surfaces failures that
   synthetic testing didn't catch** — this is likely, since §4.4 lists
   several real-world factors the synthetic models can't capture — that
   becomes the real priority list, superseding everything above except
   the ink check.

### Explicitly NOT a next step unless the user asks

- Don't touch `../owlbear-sync` (v1) — it's still deployed and working
  for the user.
- Don't regenerate the codebook (`tools/generate_codebook.py`) — this
  invalidates every printed marker. Only do this if the user explicitly
  wants a larger marker set or has a specific reason; confirm first,
  it's irreversible for anything already printed.
- Don't change the store app ids or ports — they're deliberately chosen
  to let v1 and v2 coexist during migration.

---

## 6. Testing notes for whoever picks this up

- Tests are **plain Python scripts with `if __name__ == "__main__":`
  runners, not pytest.** Run individually (`python tests/test_track.py`)
  or all via `for t in tests/test_*.py; do python "$t" || break; done`.
- The sandboxed dev environment's Python has neither `cv2` nor `flask`
  nor pip access. v1's venv at
  `../owlbear-sync/tracker/venv/bin/python3` has the needed deps
  installed and was used throughout v2 development to run cv2/flask-
  dependent tests — v1's own `requirements.txt` was never modified for
  this, flask/flask-httpauth were just pip-installed into that venv as
  a test runner. Check whether that venv still exists before assuming
  it does; it may need recreating.
- `tests/test_print_roundtrip.py` needs `rsvg-convert` (`librsvg2-bin`)
  on the host; skips itself cleanly if absent. CI installs it.
- CI (`.github/workflows/docker-publish.yml`) runs all tests, builds,
  **smoke-tests the actual built image** (not just unit tests), and
  only then notifies the store. This gate is deliberate — a broken
  tracker reaching the store gets offered as a live update mid-session.
- **Before claiming any deployment-related fix works, build and run the
  real container.** Three separate real bugs in this project (§3.3,
  §3.5 ×2) were only caught this way — unit tests with mocked/stubbed
  components missed all three, specifically because the stubs didn't
  reproduce the real failure's timing or filesystem characteristics.

---

## 7. Key files to read first, in order, if resuming work

1. This file.
2. `docs/VALIDATION.md` — test coverage map and known-good numbers.
3. `tracker/vision/geometry.py` — the marker layout; everything else
   depends on this.
4. `tracker/vision/decode.py` and `detect.py` — the detection/decode
   core, with the "why" for the two-gate acceptance design.
5. `tracker/runtime.py` and `tracker/web/app.py` — how it all wires
   together for deployment; read the docstrings on `start()`,
   `_maybe_autostart_camera()`, and `latest_frame_jpeg()` specifically
   for the first-deploy-deadlock story in §3.5.
6. `docs/DEPLOY-ZIMAOS.md` — the deployment walkthrough, written for the
   user but useful for understanding the full operational picture.
