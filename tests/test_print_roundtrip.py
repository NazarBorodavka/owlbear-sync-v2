"""Decodes the ACTUAL printable SVG artwork back to its own marker id.

Every other test renders markers with render_raster(), a numpy
reimplementation of the artwork. But the thing that gets PRINTED is the
SVG from marker_svg(), and those are two parallel implementations of the
same geometry. If they ever disagree -- most dangerously in handedness --
all the raster-based tests would still pass while every physical marker
decoded to the wrong id, or to nothing.

That is precisely the v1 failure: a print run that was mirrored relative
to what the decoder expected, which no test could catch because no test
ever looked at the printed artwork.

So this rasterises the real SVG with an external renderer (rsvg-convert,
i.e. not our code) and pushes the result through the real Detector and
the real Matcher. It is skipped if no rasteriser is installed, because it
is a cross-check on the artwork rather than part of the core maths.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tracker.vision.codebook import codebook_metadata, load_codebook
from tracker.vision.decode import Matcher
from tracker.vision.detect import Detector
from tracker.vision.geometry import RingGeometry
from tracker.vision.marker import marker_svg, render_raster

PX_PER_MM = 12.0
RASTERISER = shutil.which("rsvg-convert")


def _rasterise(svg_text, px_per_mm):
    """SVG -> grayscale float array in [0,1], via an external renderer."""
    with tempfile.TemporaryDirectory() as d:
        svg_path = os.path.join(d, "m.svg")
        png_path = os.path.join(d, "m.png")
        with open(svg_path, "w") as f:
            f.write(svg_text)
        # rsvg-convert's --dpi maps the SVG's mm units to pixels:
        # 1mm = dpi/25.4 px, so dpi = px_per_mm * 25.4.
        dpi = px_per_mm * 25.4
        subprocess.run([RASTERISER, "--dpi-x", str(dpi), "--dpi-y", str(dpi),
                        "-b", "white", "-o", png_path, svg_path],
                       check=True, capture_output=True)
        import cv2
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "rasteriser produced no image"
    return img.astype(np.float64) / 255.0


def test_printed_artwork_decodes_to_its_own_id():
    """The core guarantee: print marker N, the decoder reads N."""
    meta = codebook_metadata()
    geom = RingGeometry(n_chips=meta["n_chips"])
    codebook = load_codebook()
    matcher = Matcher(codebook, geom)
    det = Detector(geom, PX_PER_MM)

    wrong, failed = [], []
    for i, cw in enumerate(codebook):
        img = _rasterise(marker_svg(cw, geom, i, label=False), PX_PER_MM)
        cy, cx = (np.array(img.shape) - 1) / 2.0
        c = det.refine_at(img, cx, cy)
        assert c is not None, f"id{i}: detector found no ring in the printed artwork"
        got, margin = matcher.decode(img, c["x"], c["y"], c["px_per_mm"])
        if got is None:
            failed.append(i)
        elif got != i:
            wrong.append((i, got, margin))

    assert not wrong, (
        f"PRINTED ARTWORK DECODES TO THE WRONG ID: {wrong}. The SVG and the "
        f"decoder disagree -- check handedness (v1 shipped a mirrored print "
        f"run) and chip ordering in marker.py.")
    assert not failed, f"printed artwork failed to decode at all for ids {failed}"
    print(f"  all {len(codebook)} printed markers decode to their own id")


def test_printed_artwork_agrees_with_the_test_raster():
    """marker_svg() and render_raster() must draw the same marker.

    Compared as an angular ink profile around the code band, which is
    exactly the signal the decoder reads -- and the quantity that a
    mirrored or rotated artwork would differ in, while total ink area
    (a weaker check) would look identical.
    """
    meta = codebook_metadata()
    geom = RingGeometry(n_chips=meta["n_chips"])
    cw = load_codebook()[3]

    svg_img = _rasterise(marker_svg(cw, geom, 3, label=False), PX_PER_MM)
    ras_img, _, _ = render_raster(cw, geom, PX_PER_MM)

    def ink_profile(img, n_bins):
        cy, cx = (np.array(img.shape) - 1) / 2.0
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float64)
        dx, dy = xx - cx, yy - cy
        rad = np.hypot(dx, dy) / PX_PER_MM
        ang = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        band = (rad >= geom.code_inner) & (rad < geom.code_outer)
        band &= np.isfinite(img)
        idx = np.minimum((ang / (2 * np.pi) * n_bins).astype(int), n_bins - 1)
        vals = np.where(np.isfinite(img), img, 1.0)
        tot = np.bincount(idx[band], weights=vals[band], minlength=n_bins)
        cnt = np.bincount(idx[band], minlength=n_bins).astype(np.float64)
        return tot / np.maximum(cnt, 1)

    n = meta["n_chips"]
    a, b = ink_profile(svg_img, n), ink_profile(ras_img, n)
    # Binarise: below mid-grey is ink. Antialiasing and the SVG's stroke
    # geometry differ in the exact grey, but which chips are inked must not.
    assert np.array_equal(a < 0.5, b < 0.5), (
        f"SVG and raster disagree on which chips are inked:\n"
        f"  svg={(a < 0.5).astype(int)}\n  ras={(b < 0.5).astype(int)}\n"
        f"  codeword={(np.asarray(cw) <= 0).astype(int)}")
    # And both must match the codeword itself (chip <= 0 is drawn as ink).
    expected = np.asarray(cw) <= 0
    assert np.array_equal(a < 0.5, expected), "SVG ink does not match the codeword"
    print(f"  SVG and test raster ink the same {int(expected.sum())}/{n} chips, "
          f"matching the codeword")


def test_printed_artwork_decodes_at_arbitrary_rotation():
    """Markers land on the table at any angle, so the printed artwork must
    decode at arbitrary rotation -- not just the orientation it was drawn
    in. The matched filter resolves identity and rotation jointly, so a
    rotation-dependent failure here would mean the artwork, not the maths.
    """
    import cv2
    meta = codebook_metadata()
    geom = RingGeometry(n_chips=meta["n_chips"])
    codebook = load_codebook()
    matcher = Matcher(codebook, geom)
    det = Detector(geom, PX_PER_MM)

    # Deliberately includes angles off the chip grid (one chip = 18 deg),
    # since a half-chip offset is the worst case for the filter.
    angles = [0, 7, 9, 18, 45, 90, 137, 180, 263, 351]
    bad = []
    for i in range(len(codebook)):
        img = _rasterise(marker_svg(codebook[i], geom, i, label=False), PX_PER_MM)
        for a in angles:
            h, w = img.shape
            M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), a, 1.0)
            rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
            c = det.refine_at(rot, (w - 1) / 2.0, (h - 1) / 2.0)
            if c is None:
                bad.append((i, a, "no ring"))
                continue
            got, margin = matcher.decode(rot, c["x"], c["y"], c["px_per_mm"])
            if got != i:
                bad.append((i, a, f"got {got} margin {margin:.3f}"))

    assert not bad, f"printed artwork failed at rotation: {bad}"
    print(f"  printed artwork decodes correctly at all "
          f"{len(angles)} rotations x {len(codebook)} ids")


if __name__ == "__main__":
    if RASTERISER is None:
        print("SKIPPED: no rsvg-convert found (install librsvg to run this)")
        sys.exit(0)
    test_printed_artwork_agrees_with_the_test_raster()
    test_printed_artwork_decodes_to_its_own_id()
    test_printed_artwork_decodes_at_arbitrary_rotation()
    print("\nall print round-trip tests passed")
