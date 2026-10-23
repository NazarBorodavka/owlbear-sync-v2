# Validation

This file is intentionally short and points at the executable tests, not
at prose claims. The project's own discipline, learned the hard way from
v1: a number only means something if there is a test next to it that a
future change could break.

## Run everything

```bash
for t in tests/test_*.py; do python3 "$t" || break; done
```

Or individually:

```bash
python3 tests/test_track.py            # tracker identity/phantom/collision behaviour
python3 tests/test_sequence.py         # end-to-end moving-sequence simulation
python3 tests/test_projector.py        # projector/IR-mode interference sweep
python3 tests/test_calibration.py      # lens + play-area + detector calibration (cv2)
python3 tests/test_grid.py             # projected-grid spacing measurement
python3 tests/test_codebook_frozen.py  # the printed-marker codebook cannot drift
python3 tests/test_print_roundtrip.py  # the real printable SVG decodes (needs rsvg-convert)
python3 tests/test_runtime.py          # frame processing, pipeline rebuild
python3 tests/test_app.py              # web API surface
python3 tests/test_replay.py           # the recorded-footage replay harness
```

CI runs all of these before publishing an image, so a broken tracker
cannot reach the app store and be offered as an update mid-session.

## What each one is actually checking, and why it exists

**test_track.py** -- five tests, each named after and guarding against a
specific v1 field failure: identity lost across an occlusion gap, an
undecodable blob becoming a visible phantom token, two crossing tokens
swapping identity, two tracks simultaneously holding one identity, and a
fast drag losing its track.

**test_sequence.py** -- the metric that actually matches what an operator
sees: a 150-frame simulated sequence with slow drags, a fast flick,
occlusion, clutter and realistic per-token motion blur (derived physically
from speed x camera exposure, not a free parameter). Also runs the same
sequence at a dim-room exposure to make the exposure/accuracy trade-off
concrete rather than asserted.

**test_projector.py** -- found and fixed a false pass in its own first
version (the IR-mode flag was accepted but unused, and the texture scale
was larger than the whole marker, so nothing in the test could disturb a
single chip). After the fix: 0% wrong-id rate holds up to interference
5-10x stronger than a real projected map would produce, and forced IR/
night mode measurably helps over plain color mode. Documents a hard
dependency this strategy has on pigment (not dye) black ink -- see that
test file's docstring.

**test_codebook_frozen.py** -- the codebook is a committed data file
because the physical markers are PRINTED from it; regenerating it at
runtime from a seed meant any change to the search (or to numpy's RNG
across versions) could silently desync the decoder from markers already
on the table. These tests assert the committed bytes are unchanged, still
meet the correlation quality bar, separate every pair at every rotation,
and are still reproducible by the generator -- plus that the file does
not live under a mountable data directory, which it briefly did, where a
bind mount for persistence shadowed it and killed startup.

**test_print_roundtrip.py** -- closes the gap that let v1's mirrored
print run through. Every other test renders markers with a numpy
reimplementation of the artwork; this one rasterises the actual printable
SVG with an external renderer (rsvg-convert, not our code) and decodes it
with the real detector and matcher. All 6 markers decode to their own
id, at ten rotations including deliberately off-chip-grid angles. Skips
itself if rsvg-convert is absent.

**test_replay.py** -- tests the replay harness itself, because a harness
bug pointed at real footage is indistinguishable from a tracker
regression. Renders a 6-token sequence, encodes it to a real mp4 (codec
compression included deliberately), replays it, and checks the harness
reports what is genuinely in the recording. Separately feeds the scorer
each failure mode one at a time to confirm it actually catches identity
flips, phantoms, churn, slow frames and missing tokens -- a scorer only
tested on passing input is half-tested.

**test_calibration.py** -- lens calibration recovers known-true camera
intrinsics to within 3% from synthetic chessboard captures; the play-area
homography round-trips a known point to 0.04mm; all three calibration
stages (lens, play-area, detector thresholds) persist independently and
survive a fresh process reload, and re-running any ONE stage does not
disturb the others.

## Known-good numbers (re-derive, don't trust stale copies)

As of the commit that added this file, at 3.2px/mm (geometry:
`base 38mm | clear <13.3mm | code 13.8-17.0mm | 20 chips`):

- 150-frame moving sequence: 0 wrong ids, 0 phantoms, 0 identity flips,
  99% worst-token coverage, 0.45mm median position error, 7.7ms/frame
- projector interference: 0% wrong-id rate through 14x realistic strength
- lens calibration: recovers fx/fy within 3%, distortion within 0.02
- play-area homography: 0.04mm round-trip error on a known point

These numbers are a snapshot, not a contract -- the tests are the
contract. Re-run them after any change to geometry, codebook, detect,
decode, track or calibration.

## Still needed before trusting any of this on real hardware

1. A validation CORPUS of real recorded footage (tokens still, slow, fast,
   occluded; real lighting; real projector running) to replay through the
   same pipeline and compare against these synthetic numbers -- this
   project's standing commitment, not optional polish.

   The harness for this exists and is tested: `tools/replay.py`. It needs
   no ground truth, because the failures that actually matter are
   self-evident from the track log -- a track whose id changes mid-life,
   a track reported without ever decoding, an id dropped and re-acquired,
   or frame times above the budget. See its docstring.

   ```bash
   python tools/replay.py recording.mp4 --calibration data/calibration \
       --expect-tokens 6 --annotate annotated.mp4 --json report.json
   ```

   To record a useful corpus, with the projector running and the camera
   in its final mounted position and settings:

   | clip | what to do | what it tests |
   |---|---|---|
   | still | 6 tokens placed, untouched, ~30s | false positives, id stability at rest |
   | slow | drag each token slowly across the area | steady-state tracking accuracy |
   | fast | flick tokens quickly between spots | motion blur limit, the v1 flick failure |
   | occluded | move tokens with a hand over them | identity survival across gaps |
   | crossing | pass two tokens over each other | identity swaps |
   | map change | switch projected maps while tokens sit still | projector interference |
   | clutter | add dice, a mug, a d20 to the table | the peak-floor gate rejecting non-markers |

   Run each clip through the harness and record the output here. A clip
   that fails is more valuable than one that passes -- it is a test case.
2. A real-ink check: print a marker, photograph it under forced IR/night
   mode, confirm the detection ring stays dark (see test_projector.py's
   docstring on pigment vs. dye black ink).
3. Camera settings applied on the real camera: exposure capped at 1/60s
   or faster (see test_sequence.py's dim-room comparison), forced
   night/IR mode, autofocus locked, mount height around 70cm rather than
   100cm for comfortable px/mm margin at a reasonable play-area size.
   The operator-facing version of this is in the README.

Note what is and is not covered: the numbers above come from synthetic
sequences with physically-derived blur, lighting and projector models.
The Docker image, its startup, its API surface and bind-mount
persistence are verified against the real built container. Nothing here
has yet been run against real camera footage.
