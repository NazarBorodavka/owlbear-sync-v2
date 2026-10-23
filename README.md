# Physical Token Tracker v2

Tracks physical miniatures on a table with an overhead camera and mirrors
their positions into an [Owlbear Rodeo](https://owlbear.rodeo) scene, so
moving a mini on the table moves its token on the map.

A web app does the vision and serves a dashboard; a companion Owlbear
Rodeo extension consumes its positions. Tokens carry a printed ring code
on the outer edge of their base, leaving the middle clear for artwork.

> **v2 is a from-scratch rewrite, not an upgrade of v1.** It shares only
> the Docker deployment and the app-store sync. The marker design,
> detector, decoder, tracker, calibration, UI and extension are all new.
> v1's markers do **not** work with v2 — see [Printing markers](#printing-markers).
> The v2 image is published under a different name (`owlbear-tracker-v2`)
> so it cannot replace a running v1 install.

---

## How it works

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

Positions cross the API in **millimetres relative to the calibrated
play-area origin**, never in pixels or screen coordinates. That is what
makes the extension work on a cast receiver as well as the host: it
converts mm to the scene's own grid using the scene's DPI and scale, so
nothing depends on anyone's window size or viewport.

### Marker design

Each token base carries a thin outer **detection ring** (found by a
gradient vote at a known radius) and, inside it, a **code band** split
into 20 chips. Identity is read by polar-unwrapping the band and
cross-correlating it against the codeword set — which resolves identity
and rotation *jointly*, so there is no orientation-locking sync mark to
be occluded.

Default geometry for a 40mm base, printed at 38mm (1mm margin each side):

```
base 38mm | clear <13.3mm | code 13.8-17.0mm (3.2mm, 20 chips) | det ring 17.5-19.0mm
```

The centre 70% of the diameter is left clear for artwork, as required.

Two independent gates must both pass before an id is reported: an
absolute **peak floor** ("is this one of our codewords at all" — v1 never
asked this, which is how bottle caps became tokens) and a **margin**
("is the winner clearly ahead of the runner-up"). Anything failing either
gate is reported as *no id*, not as a guess. Across every stress test in
`tests/`, the confidently-wrong rate is 0%.

---

## Hardware

| | |
|---|---|
| Camera | Reolink E1 Zoom (RTSP, motorised 3× zoom) |
| Host | anything that runs Docker; developed against a Razer Book 13 on ZimaOS |
| Bases | 40mm diameter, round |
| Projector | optional, ceiling-mounted, projecting the map onto the table |

### Mounting

The thing that actually matters is **pixels per millimetre on the table**,
and you do not have to compute it from spec sheets — the play-area
calibration step measures it and tells you whether it is enough.

- **2.5 px/mm** is the warned-at minimum (the code band stops being
  reliably resolvable below roughly this scale).
- **3.2 px/mm** is where all the validated numbers in
  [docs/VALIDATION.md](docs/VALIDATION.md) were measured.
- The calibration UI reports your measured px/mm with a verdict, live.

Since play-area width ≈ (horizontal pixels) ÷ (px per mm), a 2560px-wide
sensor at 3.2 px/mm covers about **800mm** of table. Lowering the camera
or zooming in raises px/mm but shrinks the covered area; those trade
directly against each other.

**Lower is easier than zoomed.** At 100cm you need most of the zoom range
to reach a comfortable margin over a reasonable play area, which leaves
nothing in reserve; **around 70cm** gives the same px/mm at a wider
angle. Mount it looking straight down — the play-area calibration
corrects perspective, but a steep angle costs you effective resolution on
the far side of the table.

### Camera settings

Three settings matter, in this order:

1. **Cap exposure at 1/60s or faster.** This is the single biggest factor
   in whether fast token moves survive. Motion blur is `speed ×
   exposure`, so it is exposure, not frame rate, that decides when a
   dragged token stops decoding. Measured: 1/60s tracks to 250mm/s, while
   a dim-room 1/15s already fails at 130mm/s.
2. **Force night/IR mode on.** With a projector running, the map's
   changing brightness is the main nuisance light. IR illumination is
   outside the projector's output, so the markers stay evenly lit while
   the visible-light map washes over them. `tests/test_projector.py`
   measures that forced IR mode beats plain colour mode.
3. **Lock autofocus.** A refocus mid-session changes your px/mm and
   quietly invalidates the play-area calibration.

Reolink exposes these over its CGI API (`SetIsp` for `dayNight` and
exposure, `SetIrLights` for the IR LEDs); they are also all reachable
from the camera's own web UI, which is the easier route for a one-time
setup. Check the parameter names against your firmware version — Reolink
has changed them between releases.

> **Pigment ink is not optional under IR.** Dye-based black is nearly
> transparent above ~800nm, so under IR illumination a dye-printed marker
> shows almost no contrast and will not decode at all. Use a laser
> printer or pigment/carbon-black ink. See `tests/test_projector.py`'s
> docstring.

---

## Printing markers

```bash
python tools/generate_markers.py --outdir markers
# or just the ones you need, plus one file per marker:
python tools/generate_markers.py --n-ids 8 --singles
```

This writes `markers/marker_sheet.svg` from
`tracker/vision/codebook_data.json` — **the same file the decoder reads**, so
the artwork and the decoder cannot disagree about what a marker means.
(v1 kept these in two places and shipped a mirrored print run as a
result; `tests/test_print_roundtrip.py` now rasterises the real SVG with
an external renderer and decodes it, so that failure is a test failure.)

Then:

1. Print at **100% / actual size**. Turn off "fit to page" and "shrink to
   fit" — both silently rescale.
2. **Measure the 50mm bar** on the sheet with a ruler. If it is not 50mm,
   the markers are the wrong size and will not decode at the calibrated
   radius. This check exists because it is the most common failure.
3. Use a **laser printer or pigment ink**, on **matte** paper. Gloss
   retroreflects the IR LEDs into the lens and blows out the code ring.
4. Mount on 40mm round bases, centred.

### Regenerating the codebook

Don't, unless you mean to. `tracker/vision/codebook_data.json` is committed, and
**every marker already printed is invalidated if it changes** — the
decoder would be looking for codewords your markers no longer carry.
`tools/generate_codebook.py` exists for that deliberate act and prompts
before overwriting; CI warns if the file changes in a commit.

---

## Deploying

### Docker

```bash
docker run -d --name owlbear-tracker-v2 \
  -p 5000:5000 \
  -e TRACKER_PASSWORD=choose-something \
  -v owlbear-tracker-v2-data:/data \
  ghcr.io/nazarborodavka/owlbear-tracker-v2:latest
```

Then open `http://<host>:5000`.

| Variable | Default | |
|---|---|---|
| `TRACKER_USER` | `admin` | dashboard login |
| `TRACKER_PASSWORD` | `admin` | **change this** |
| `TRACKER_DATA_DIR` | `/data` | calibration + token names live here |

Mount `TRACKER_DATA_DIR` on a volume or you will recalibrate after every
container restart.

`GET /api/version` is public and reports the running build and the
shipped codebook fingerprint — the quick answer to "did my update
actually deploy".

### ZimaOS / CasaOS

Pushes to `master` publish to GHCR and notify the
`owlbear-sync-store` repo, which is what makes ZimaOS show
*Update available* instead of needing a manual reinstall. v2 dispatches
`new-image-v2` (v1 uses `new-image`) and needs its own store entry; the
store repo must have a handler for that event before the update path
works.

---

## Calibration

Four steps in the dashboard. **Each is independently rerunnable and
persisted separately** — redoing one does not disturb the others, which
`tests/test_calibration.py` verifies.

1. **Lens** — print a chessboard, capture it at several angles. Corrects
   barrel distortion. Only needs redoing if you change camera or zoom.
2. **Play area** — click the four corners of your play surface and enter
   its real width and height in mm. Establishes the mm coordinate system
   and corrects perspective. Reports your measured px/mm and whether it
   is enough. **Redo this if the camera moves, zooms, or refocuses.**
3. **Detector tuning** *(optional)* — measures the actual peak and margin
   distribution your markers produce under your lighting and tightens the
   accept thresholds to a safe band. Improves rejection of non-markers.
4. **Grid** — with a map projected, measures the projected grid's spacing
   so the extension can auto-zoom the map until one grid cell matches a
   40mm base. Removes the manual zoom trial-and-error.

---

## Owlbear Rodeo extension

Add `http://<host>:5000/extension/manifest.json` as a custom extension in
Owlbear Rodeo. (Owlbear must be able to reach that URL — for play outside
your LAN it needs to be exposed, e.g. via a tunnel.)

In the extension panel, map each scene token to a tracked marker id.
Mappings are stored in **scene metadata**, so they survive reloads and
come back next session with the scene — no re-pairing every time. Tokens
whose names match a tracked token's name are matched automatically.

Naming tokens in the web dashboard is what drives that auto-match, and
names persist on the host.

---

## Development

```bash
pip install -r tracker/requirements.txt
python -m tracker.web.app        # http://localhost:5000
```

Tests are plain scripts, no pytest:

```bash
for t in tests/test_*.py; do python "$t" || break; done
```

`test_print_roundtrip.py` needs `rsvg-convert` (`librsvg2-bin`) and skips
itself without it. The core vision code
(`tracker/vision/{geometry,codebook,decode,detect,track,pipeline}.py`) is
pure NumPy with no native dependency; OpenCV is only used by the
calibration package and video I/O.

See [docs/VALIDATION.md](docs/VALIDATION.md) for what each test actually
checks, the known-good numbers, and what still needs verifying on real
hardware.

### Scoring real footage

Record the camera, then replay it through the production pipeline:

```bash
python tools/replay.py recording.mp4 --calibration data/calibration \
    --expect-tokens 6 --annotate annotated.mp4
```

It needs no hand-labelling. The failures that matter are self-evident
from the track log -- a track whose id changes mid-life, a track reported
without ever decoding (v1's red ghosts), an id dropped and re-acquired,
frame times over budget -- so it scores those directly and prints
pass/fail per requirement. `--annotate` writes an overlaid video for
eyeballing: green tracked, amber coasting, red phantom.

docs/VALIDATION.md lists the clips worth recording.

## Status

The vision pipeline, calibration, web app, extension and deployment are
built and passing their test suites. What has **not** happened yet is
validation against real recorded footage — every number quoted here comes
from synthetic sequences with physically-derived blur, lighting and
projector models, which is not the same as a real table. The outstanding
items are listed at the end of
[docs/VALIDATION.md](docs/VALIDATION.md); the real-ink IR check and a
recorded validation corpus are the two that matter most.
