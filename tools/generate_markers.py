#!/usr/bin/env python3
"""Emits printable marker artwork from the FROZEN codebook.

This reads tracker/vision/codebook_data.json -- the same file the decoder reads
-- so the printed artwork and the decoder's expectations cannot disagree.
That shared-source-of-truth property is the whole point; see the comment
above load_codebook() in tracker/vision/codebook.py.

Printing checklist (a mis-scaled print is the most common failure):
  * Print at 100% / "actual size". Disable "fit to page" and "shrink to
    fit" -- both silently rescale.
  * Measure the 50mm scale bar on the sheet with a ruler. If it is not
    50mm, the markers are the wrong size and will not decode at the
    calibrated radius.
  * Use a PIGMENT / carbon-black ink or a laser printer. Dye-based black
    is near-transparent above ~800nm, so under the camera's IR
    illumination a dye-printed marker shows almost no contrast at all.
  * Matte paper. Gloss retroreflects the IR LEDs straight back into the
    lens and blows out the code ring.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tracker.vision.codebook import codebook_metadata, load_codebook
from tracker.vision.geometry import RingGeometry
from tracker.vision.marker import marker_svg, sheet_svg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-ids", type=int, default=None,
                    help="how many markers to emit (default: all in the codebook)")
    ap.add_argument("--outdir", default="markers",
                    help="directory for the SVG output (default: ./markers)")
    ap.add_argument("--cols", type=int, default=4, help="markers per sheet row")
    ap.add_argument("--singles", action="store_true",
                    help="also write one SVG per marker")
    ap.add_argument("--pitch-extra-mm", type=float, default=4.0,
                    help="gap added to the marker diameter for the sheet grid")
    ap.add_argument("--a4", action="store_true",
                    help="lay the sheet out on a fixed 210x297mm page, so the "
                         "print dialog has nothing to rescale")
    ap.add_argument("--pdf", action="store_true",
                    help="also convert the sheet to PDF (needs rsvg-convert)")
    ap.add_argument("--base-mm", type=float, default=None,
                    help="printed diameter in mm (default: the geometry's "
                         "38mm, which is a 40mm base with a 1mm margin)")
    args = ap.parse_args()

    meta = codebook_metadata()
    geom = RingGeometry(n_chips=meta["n_chips"])
    if args.base_mm is not None:
        geom = RingGeometry(n_chips=meta["n_chips"], base_d_mm=args.base_mm)
    geom.validate()

    codebook = load_codebook(n_ids=args.n_ids)
    os.makedirs(args.outdir, exist_ok=True)

    page_mm = (210.0, 297.0) if args.a4 else None
    sheet_name = "marker_sheet_a4.svg" if args.a4 else "marker_sheet.svg"
    sheet_path = os.path.join(args.outdir, sheet_name)
    with open(sheet_path, "w") as f:
        f.write(sheet_svg(codebook, geom, cols=args.cols,
                          pitch_extra_mm=args.pitch_extra_mm, page_mm=page_mm))

    if args.singles:
        for i, cw in enumerate(codebook):
            with open(os.path.join(args.outdir, f"marker_{i:02d}.svg"), "w") as f:
                f.write(marker_svg(cw, geom, i))

    pdf_path = None
    if args.pdf:
        import shutil
        import subprocess
        if shutil.which("rsvg-convert") is None:
            print("ERROR: --pdf needs rsvg-convert (package librsvg2-bin / librsvg)",
                  file=sys.stderr)
            return 2
        pdf_path = os.path.splitext(sheet_path)[0] + ".pdf"
        # The SVG declares real millimetre units, so no scaling flags are
        # wanted here -- adding any would silently defeat the 50mm bar.
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", pdf_path, sheet_path],
                       check=True)

    print(f"Wrote {len(codebook)} markers to {sheet_path}")
    if pdf_path:
        print(f"  PDF: {pdf_path}")
    print(f"  layout: {geom.describe() if hasattr(geom, 'describe') else ''}")
    print(f"  printed diameter: {geom.base_d_mm:.1f}mm "
          f"(clear centre: {geom.base_d_mm * geom.clear_frac:.1f}mm)")
    print(f"  codebook: n_chips={meta['n_chips']}, "
          f"worst cross-correlation={meta['worst_pairwise_crosscorrelation']}")
    print("\nPrint at 100% scale, then MEASURE the 50mm bar on the sheet.")
    print("Use pigment/carbon-black ink or a laser printer on matte paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
