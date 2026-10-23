#!/usr/bin/env python3
"""Mints a NEW frozen codebook into tracker/data/codebook.json.

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
    ap.add_argument("--n-ids", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=12000)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tracker", "data", "codebook.json"))
    ap.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt")
    args = ap.parse_args()

    if not args.yes:
        print("This invalidates every marker already printed from the current "
              "codebook -- they will all need reprinting.")
        if input("Type 'reprint' to continue: ").strip() != "reprint":
            print("Aborted.")
            return 1

    cb = build_codebook(args.n_chips, args.n_ids, seed=args.seed,
                        max_attempts=args.max_attempts)
    auto, cross = quality(cb)
    payload = {
        "n_chips": args.n_chips,
        "n_ids": args.n_ids,
        "generated_with": {"seed": args.seed, "max_attempts": args.max_attempts,
                           "auto_thr": 0.45, "cross_thr": 0.40},
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
