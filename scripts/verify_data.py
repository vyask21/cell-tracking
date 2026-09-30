"""Check the downloaded dataset against the cached Kaggle file listing.

A stream-unzip that dies partway leaves a plausible-looking tree, so this compares
every expected file against what is on disk, by size. Cheap insurance before a
five-hour CV run that would otherwise fail on sample 140.

    python scripts/verify_data.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

DEFAULT_LISTING = os.path.join("data", "meta", "file_listing.csv")
DEFAULT_ROOT = os.path.join("data", "raw")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default=DEFAULT_LISTING)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    with open(args.listing, newline="", encoding="utf-8") as fh:
        expected = {r["name"]: int(r["total_bytes"]) for r in csv.DictReader(fh)}
    print(f"{len(expected)} files expected")

    missing, wrong_size = [], []
    for name, size in expected.items():
        path = os.path.join(args.root, name)
        if not os.path.exists(path):
            missing.append(name)
        elif os.path.getsize(path) != size:
            wrong_size.append((name, size, os.path.getsize(path)))

    partial = []
    for dirpath, _, files in os.walk(args.root):
        partial.extend(
            os.path.join(dirpath, f) for f in files if f.endswith(".part")
        )

    print(f"missing:    {len(missing)}")
    print(f"wrong size: {len(wrong_size)}")
    print(f"leftover .part files: {len(partial)}")
    for name in missing[: args.show]:
        print(f"  MISSING {name}")
    for name, want, got in wrong_size[: args.show]:
        print(f"  SIZE    {name}: want {want}, got {got}")

    # Per-sample completeness, which is what matters to the CV loop.
    per_sample = collections.Counter()
    for name in missing:
        parts = name.split("/")
        if len(parts) > 1:
            per_sample[f"{parts[0]}/{parts[1]}"] += 1
    if per_sample:
        print("\nincomplete samples:")
        for sample, n in per_sample.most_common(args.show):
            print(f"  {sample}: {n} files missing")

    ok = not missing and not wrong_size and not partial
    print("\nOK" if ok else "\nINCOMPLETE - rerun scripts/download_data.py to fill gaps")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
