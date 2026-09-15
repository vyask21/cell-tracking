"""Compare the same arm across two caches, paired by sample.

`screen_calibration.py` compares arms inside one cache, because that is what
every screen before 2026-09-15 needed: the detections were fixed and the
calibration chain was the variable. Test-time augmentation moves the detections
themselves, so its arm lives in a different cache directory and there is no way
to express the comparison as two arms of one screen.

The comparison is still paired, and that matters. Both caches cover the same 19
videos, so the sample-to-sample spread that dominates an unpaired estimate
cancels here exactly as it does inside a single screen. What does NOT cancel is
anything that differs between the two cache builds besides the variable under
test, which is why `scripts/cache_graphs.py` and the Kaggle cache kernel share
one format module and why the kernel template has a test pinning its settings
against the local cache's.

    .venv/Scripts/python.exe scripts/compare_screens.py ^
        --base data/meta/v11_relink.csv --cand data/meta/v11_tta8.csv --arm v11

Read the caveat `screen_calibration.py` prints. These 19 are video-disjoint from
the support pack's training set and not embryo-disjoint, and the hidden test is.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.screen_calibration import (  # noqa: E402
    division_jaccard,
    paired_bootstrap,
    total_score,
    weighted,
)

INT_COLS = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp",
            "division_fn", "num_pred_nodes")
FLOAT_COLS = ("node_recall", "total_node_ratio", "edge_jaccard",
              "adj_edge_jaccard")


def load(path: str) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = collections.defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for k in INT_COLS:
                row[k] = int(row[k])
            for k in FLOAT_COLS:
                row[k] = float(row[k])
            arms[row["arm"]].append(row)
    return arms


def counts(rows: list[dict]) -> dict:
    out = {k: sum(r[k] for r in rows) for k in INT_COLS}
    out["adj_J"] = weighted(rows)
    out["divJ"] = division_jaccard(rows)
    out["total"] = total_score(rows)
    return out


def align(a: list[dict], b: list[dict]) -> tuple[list[dict], list[dict]]:
    """Order both sides by sample and refuse anything but an exact match.

    A missing sample on one side would silently become an unpaired comparison
    with a different denominator, which is the failure this whole script exists
    to avoid.
    """
    sa = {r["sample"]: r for r in a}
    sb = {r["sample"]: r for r in b}
    if set(sa) != set(sb):
        only_a = sorted(set(sa) - set(sb))
        only_b = sorted(set(sb) - set(sa))
        raise SystemExit(
            f"samples differ: {len(only_a)} only in base {only_a[:3]}, "
            f"{len(only_b)} only in candidate {only_b[:3]}"
        )
    order = sorted(sa)
    return [sa[s] for s in order], [sb[s] for s in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="screen CSV for the reference cache")
    ap.add_argument("--cand", required=True, help="screen CSV for the candidate cache")
    ap.add_argument("--arm", default="", help="arm name; default is every arm in both")
    ap.add_argument("--boot", type=int, default=20000)
    args = ap.parse_args()

    base_arms, cand_arms = load(args.base), load(args.cand)
    if args.arm:
        shared = [args.arm]
        for name, arms in (("base", base_arms), ("candidate", cand_arms)):
            if args.arm not in arms:
                raise SystemExit(f"arm {args.arm!r} not in the {name} file")
    else:
        shared = sorted(set(base_arms) & set(cand_arms))
        if not shared:
            raise SystemExit("no arm appears in both files")

    print(f"base      {args.base}")
    print(f"candidate {args.cand}")
    print()
    header = (f"{'arm':14}{'base':>9}{'cand':>9}{'delta':>10}"
              f"{'95% CI':>22}{'P(>0)':>8}  both embryos")
    print(header)
    for arm in shared:
        a, b = align(base_arms[arm], cand_arms[arm])
        delta, lo, hi, p = paired_bootstrap(a, b, n=args.boot)
        both = []
        for emb in ("44b6", "6bba"):
            ea = [r for r in a if r["embryo"] == emb]
            eb = [r for r in b if r["embryo"] == emb]
            both.append(total_score(eb) - total_score(ea))
        ok = "YES" if all(d > 0 for d in both) else "no"
        print(f"{arm:14}{total_score(a):9.4f}{total_score(b):9.4f}{delta:+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p:8.3f}  {ok}")

    print()
    print("per embryo, and the counts behind it")
    for arm in shared:
        a, b = align(base_arms[arm], cand_arms[arm])
        print()
        print(f"  {arm}")
        for emb in ("44b6", "6bba"):
            ea = [r for r in a if r["embryo"] == emb]
            eb = [r for r in b if r["embryo"] == emb]
            up = sum(1 for x, y in zip(ea, eb)
                     if y["adj_edge_jaccard"] > x["adj_edge_jaccard"])
            d = total_score(eb) - total_score(ea)
            print(f"    {emb}  {d:+.4f}  ({up}/{len(eb)} samples up)")
        ca, cb = counts(a), counts(b)
        for k in ("num_pred_nodes", "edge_tp", "edge_fp", "edge_fn",
                  "division_tp", "division_fp", "division_fn"):
            print(f"    {k:18}{ca[k]:10d}{cb[k]:10d}{cb[k] - ca[k]:+10d}")
        for k in ("adj_J", "divJ", "total"):
            print(f"    {k:18}{ca[k]:10.4f}{cb[k]:10.4f}{cb[k] - ca[k]:+10.4f}")

    print()
    print("These 19 are video-disjoint from the support pack's training set but")
    print("NOT embryo-disjoint, and the hidden test is. Treat any delta as a")
    print("direction and an upper bound.")


if __name__ == "__main__":
    main()
