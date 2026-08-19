"""Write a leave-one-embryo-out splits file the reference trainer will accept.

`train_unet_transformer.py` and `predict_unet_transformer.py` both take
`--splits <file>`, a JSON list of `{"split": i, "train": [stems], "test":
[stems]}` indexed by `--split`. When the file is absent they fall back to a
`random.Random(0)` 90/10 shuffle over every video, which puts crops of the same
embryo on both sides of the split. The hidden test set comes from embryos that
appear in neither prefix, so that fallback measures in-distribution performance
while the leaderboard measures transfer to an unseen embryo. Do not inherit it.

This writes the same two folds `src/cv.py` already uses, in their format, and
runs the same leak check. Fold 0 holds out 44b6, fold 1 holds out 6bba.

    python scripts/make_embryo_splits.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv import check_no_embryo_leak, embryo_of, leave_one_embryo_out, list_samples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--out", default="data/meta/dataset_splits.json")
    args = ap.parse_args()

    samples = list_samples(args.data)
    if not samples:
        raise SystemExit(f"no samples found under {args.data}")
    folds = leave_one_embryo_out(samples)
    check_no_embryo_leak(folds)

    payload = []
    for i, fold in enumerate(folds):
        payload.append({
            "split": i,
            "train": sorted(fold.train),
            "test": sorted(fold.valid),
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"{len(samples)} samples, {len(payload)} folds")
    for entry in payload:
        held = sorted({embryo_of(s) for s in entry["test"]})
        trained = sorted({embryo_of(s) for s in entry["train"]})
        print(f"  split {entry['split']}: {len(entry['train'])} train "
              f"{trained}, {len(entry['test'])} test {held}")
    print(f"\nwrote {args.out}")
    print("use with:  --splits data/meta/dataset_splits.json --split 0")


if __name__ == "__main__":
    main()
