"""Screen the plateau kernel's twelve arms against this repo's rule, not the notebook's.

The kernel selects an arm when its proxy score beats base by a margin. That is a
point estimate on a single set and it is the rule that let a seven arm sweep
report five arms within 0.00002 of each other as though they were results. This
repo's bar is the one written in CLAUDE.md and used since 2026-08-31:

    a paired bootstrap interval clear of zero, AND both embryos positive.

Both halves matter. The interval is the whole reason exp 8 and exp 11 were logged
as inconclusive rather than as gains, and the embryo split is what caught the
z-shift screen selecting on nine samples in exp 3.

The arms are scored on the 24 held-out train videos the screen kernel builds, not
the notebook's 8. That number is a denominator argument: 8 videos carry about 13
ground truth divisions, and a bootstrap over 13 events cannot resolve a division
change no matter how many resamples it draws.

    .venv/Scripts/python.exe scripts/screen_plateau_arms.py ^
        --results artifacts/plateau_screen/validator_results.csv

Column names differ between the kernel and this repo's own screens, so the reader
maps them once and everything downstream is the code that has already been used
on every screen since 2026-08-20.

Never average a number out of here with a held-out 19 number from exps 4 to 11.
Different sets, different pipeline, different instrument.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.screen_calibration import (  # noqa: E402
    division_jaccard,
    paired_bootstrap,
    total_score,
    weighted,
)

# The kernel writes score_sample()'s keys. This repo's helpers predate it and use
# their own. Mapping here keeps one implementation of the metric rather than two.
COLUMN_MAP = {
    "adjusted_edge_jaccard": "adj_edge_jaccard",
    "div_tp": "division_tp",
    "div_fp": "division_fp",
    "div_fn": "division_fn",
}

# Sample stems are prefixed by embryo. Both must improve for an arm to count.
EMBRYOS = ("44b6", "6bba")


def load_rows(path: str) -> dict[str, dict[str, dict]]:
    """Return {arm: {stem: row}} with columns renamed to this repo's names."""
    by_arm: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            for src, dst in COLUMN_MAP.items():
                if src in row:
                    row[dst] = row[src]
            arm = row["config"]
            stem = row["stem"]
            if stem in by_arm[arm]:
                raise SystemExit(f"duplicate row for arm {arm} sample {stem}")
            by_arm[arm][stem] = row
    return dict(by_arm)


def embryo_of(stem: str) -> str:
    return stem.split("_", 1)[0]


def align(base: dict[str, dict], cand: dict[str, dict]) -> tuple[list, list, list]:
    """Pair the two arms on the samples they share, in one fixed order.

    An unpaired comparison here would be dominated by the sample to sample
    spread, which is larger than every effect being screened.
    """
    stems = sorted(set(base) & set(cand))
    return stems, [base[s] for s in stems], [cand[s] for s in stems]


def div_counts(rows: list[dict]) -> tuple[int, int, int]:
    return (sum(int(r["division_tp"]) for r in rows),
            sum(int(r["division_fp"]) for r in rows),
            sum(int(r["division_fn"]) for r in rows))


def verdict(lo: float, hi: float, per_embryo: dict[str, float]) -> str:
    """The repo's rule, stated once so no caller can soften it.

    An arm that fails either half is reported as inconclusive, never as a gain.
    """
    both_positive = all(v > 0 for v in per_embryo.values())
    clear_of_zero = lo > 0
    if clear_of_zero and both_positive:
        return "REAL"
    if hi < 0:
        return "REJECTED"
    return "inconclusive"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/plateau_screen/validator_results.csv")
    ap.add_argument("--base", default="base")
    ap.add_argument("--out", default="artifacts/plateau_screen/arm_screen.csv")
    ap.add_argument("--n", type=int, default=20000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0, help="fixed and recorded, per CLAUDE.md")
    args = ap.parse_args()

    by_arm = load_rows(args.results)
    if args.base not in by_arm:
        raise SystemExit(f"no arm named {args.base}; found {sorted(by_arm)}")
    base = by_arm[args.base]

    base_rows = list(base.values())
    b_tp, b_fp, b_fn = div_counts(base_rows)
    print(f"{len(by_arm) - 1} arms against {args.base}, "
          f"{len(base)} samples, seed {args.seed}, {args.n} resamples")
    print(f"base total {total_score(base_rows):.6f}  "
          f"adj edge {weighted(base_rows):.6f}  "
          f"divJ {division_jaccard(base_rows):.4f}  "
          f"div {b_tp}tp/{b_fp}fp/{b_fn}fn")
    print(f"ground truth divisions in the set: {b_tp + b_fn}. "
          f"A bootstrap cannot resolve a division change much smaller than "
          f"one over that number.\n")

    out_rows = []
    for arm in sorted(k for k in by_arm if k != args.base):
        stems, a, b = align(base, by_arm[arm])
        if len(stems) != len(base):
            print(f"  {arm}: only {len(stems)} of {len(base)} samples, skipped")
            continue
        delta, lo, hi, p = paired_bootstrap(a, b, n=args.n, seed=args.seed)

        per_embryo = {}
        for e in EMBRYOS:
            ia = [r for r, s in zip(a, stems) if embryo_of(s) == e]
            ib = [r for r, s in zip(b, stems) if embryo_of(s) == e]
            per_embryo[e] = total_score(ib) - total_score(ia) if ia else float("nan")

        tp, fp, fn = div_counts(b)
        out_rows.append({
            "arm": arm,
            "n": len(stems),
            "total": total_score(b),
            "delta": delta,
            "ci_lo": lo,
            "ci_hi": hi,
            "p_gt_0": p,
            "d_44b6": per_embryo["44b6"],
            "d_6bba": per_embryo["6bba"],
            "adj_edge": weighted(b),
            "div_jaccard": division_jaccard(b),
            "div_tp": tp,
            "div_fp": fp,
            "div_fn": fn,
            "verdict": verdict(lo, hi, per_embryo),
        })

    out_rows.sort(key=lambda r: r["delta"], reverse=True)

    head = (f"{'arm':<22}{'delta':>9}{'ci_lo':>9}{'ci_hi':>9}{'P>0':>7}"
            f"{'44b6':>9}{'6bba':>9}{'div':>12}  verdict")
    print(head)
    print("-" * len(head))
    for r in out_rows:
        print(f"{r['arm']:<22}{r['delta']:>+9.5f}{r['ci_lo']:>+9.5f}"
              f"{r['ci_hi']:>+9.5f}{r['p_gt_0']:>7.3f}"
              f"{r['d_44b6']:>+9.5f}{r['d_6bba']:>+9.5f}"
              f"{r['div_tp']:>4}/{r['div_fp']}/{r['div_fn']:<5}  {r['verdict']}")

    real = [r["arm"] for r in out_rows if r["verdict"] == "REAL"]
    print(f"\nclears the repo's rule: {', '.join(real) if real else 'nothing'}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
