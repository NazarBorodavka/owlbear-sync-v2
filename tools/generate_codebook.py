#!/usr/bin/env python3
"""Mints a NEW frozen codebook into tracker/vision/codebook_data.json.

WARNING: running this invalidates every physical marker already printed
from the previous codebook. They will all need reprinting. This is
deliberately a separate, manual tool rather than something that happens
automatically at startup -- see the comment above load_codebook() in
tracker/vision/codebook.py for why.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tracker.vision.codebook import build_codebook, quality, to_bits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-chips", type=int, default=20)
    ap.add_argument("--n-ids", type=int, default=6)
    ap.add_argument("--max-cross", type=float, default=0.20,
                    help="worst allowed signed cross-correlation between any "
                         "two markers at any rotation; 0.20 at 20 chips means "
                         "no two markers come within 8 chips of each other")
    ap.add_argument("--low-cap", type=float, default=0.35,
                    help="max share of angular power in harmonics 1-2, the "
                         "shading subspace a lighting gradient lives in")
    ap.add_argument("--top-cap", type=float, default=0.30,
                    help="max share in the Nyquist harmonic, which defocus "
                         "attenuates first")
    ap.add_argument("--harm-cap", type=float, default=0.60,
                    help="max share in any single harmonic")
    ap.add_argument("--auto-cap", type=float, default=0.60,
                    help="max autocorrelation sidelobe, so no marker is "
                         "confusable with a rotation of itself")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tracker", "vision",
        "codebook_data.json"))
    ap.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt")
    args = ap.parse_args()

    if not args.yes:
        print("This invalidates every marker already printed from the current "
              "codebook -- they will all need reprinting.")
        if input("Type 'reprint' to continue: ").strip() != "reprint":
            print("Aborted.")
            return 1

    cb = build_codebook(args.n_chips, args.n_ids, max_cross=args.max_cross,
                        low_cap=args.low_cap, top_cap=args.top_cap,
                        harm_cap=args.harm_cap, auto_cap=args.auto_cap)
    auto, cross = quality(cb)
    payload = {
        "n_chips": args.n_chips,
        "n_ids": args.n_ids,
        "generated_with": {"max_cross": args.max_cross, "low_cap": args.low_cap,
                           "top_cap": args.top_cap, "harm_cap": args.harm_cap,
                           "auto_cap": args.auto_cap},
        "worst_autocorrelation_sidelobe": round(auto, 6),
        "worst_pairwise_crosscorrelation": round(cross, 6),
        "codewords": to_bits(cb),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.n_ids} codewords of {args.n_chips} chips to {args.out}")
    print(f"  worst autocorrelation sidelobe: {auto:.3f}")
    print(f"  worst pairwise crosscorrelation: {cross:.3f}")
    print("\nReprint all markers with tools/generate_markers.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
