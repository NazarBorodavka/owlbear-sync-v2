# Validation

This file is intentionally short and points at the executable tests, not
at prose claims. The project's own discipline, learned the hard way from
v1: a number only means something if there is a test next to it that a
future change could break.

## Run everything

```bash
cd tests
python3 test_track.py          # tracker identity/phantom/collision behaviour
python3 test_sequence.py       # end-to-end moving-sequence simulation
python3 test_projector.py      # projector/IR-mode interference sweep
python3 test_calibration.py    # lens + play-area + detector calibration (needs cv2)
```

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
2. A real-ink check: print a marker, photograph it under forced IR/night
   mode, confirm the detection ring stays dark (see test_projector.py's
   docstring on pigment vs. dye black ink).
3. Camera settings: exposure capped at 1/60s or faster (see
   test_sequence.py's dim-room comparison), forced night/IR mode via the
   Reolink API's `SetIsp`/`SetIrLights`, mount height around 70cm rather
   than 100cm for comfortable px/mm margin at a reasonable play-area size.
