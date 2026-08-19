"""Draw a screening set that is representative of all 199 training samples.

The nine-sample set used for four screens this month was drawn by density
quantiles for a detection-grid probe and then reused without ever being checked.
It is not representative: baseline weighted 0.5424 on it against 0.6928 on the
other 190, so it sits at the hard end of the distribution. Exp 3 screened at
+0.061 there and delivered -0.0030 over the full 199.

This draws a replacement by stratified sampling on embryo and on baseline
adjusted-edge-Jaccard quantile, choosing the draw whose weighted mean and score
distribution best match the full 199.

The selection objective uses the **baseline cache only**. Two other configs have
full 199-sample caches (sigma075, zshift2) and both are held out of the
objective, so the delta they recover is a blind test of the resulting set rather
than something it was fitted to. Do not add them to the objective.

    python scripts/build_screen_set.py --n 30
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv import embryo_of  # noqa: E402

SCORES = "artifacts/scores"
BASELINE = "baseline_localmax_4f8a3af0.csv"
# Held out of the objective. Full-199 caches whose true delta is known.
HELDOUT = {
    "sigma075": "detect_sigma075_beb04d9d.csv",
    "zshift2": "detect_zshift2_b7c33eb3.csv",
}

# The set this replaces, kept so the report shows what was wrong with it.
OLD_SET = (
    "44b6_144b256d", "44b6_90724892", "44b6_d5e7d891", "44b6_db3c847b",
    "6bba_268e1230", "6bba_3db54e20", "6bba_55c70843", "6bba_67ebd073",
    "6bba_6ca87370",
)


def load_scores(name: str) -> dict[str, tuple[float, float]]:
    """Return {sample: (adj_edge_jaccard, weight)} from a score cache."""
    out = {}
    with open(os.path.join(SCORES, name), newline="") as fh:
        for row in csv.DictReader(fh):
            w = int(row["edge_tp"]) + int(row["edge_fp"]) + int(row["edge_fn"])
            out[row["sample"]] = (float(row["adj_edge_jaccard"]), float(w))
    return out


def weighted_mean(scores: dict, samples) -> float:
    v = np.array([scores[s][0] for s in samples if s in scores])
    w = np.array([scores[s][1] for s in samples if s in scores])
    return float(np.sum(v * w) / np.sum(w))


def ks(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic, no scipy dependency."""
    grid = np.unique(np.concatenate([a, b]))
    ca = np.searchsorted(a_sorted := np.sort(a), grid, side="right") / a.size
    cb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))


def strata(samples, base) -> dict:
    """Embryo x score-quartile buckets, so a draw spans easy and hard samples."""
    vals = np.array([base[s][0] for s in samples])
    edges = np.quantile(vals, [0.25, 0.5, 0.75])
    out = {}
    for s in samples:
        q = int(np.searchsorted(edges, base[s][0]))
        out.setdefault((embryo_of(s), q), []).append(s)
    return out


def draw(buckets: dict, n: int, total: int, rng) -> list[str]:
    """Proportional allocation across strata, remainder to the largest strata."""
    keys = sorted(buckets)
    exact = np.array([len(buckets[k]) * n / total for k in keys])
    take = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - take))[: n - int(take.sum())]:
        take[i] += 1
    picked = []
    for k, t in zip(keys, take):
        t = min(t, len(buckets[k]))
        picked += list(rng.choice(buckets[k], size=t, replace=False))
    return sorted(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[9, 15, 20, 30, 40, 50])
    ap.add_argument("--out", default="data/meta/screen_set.csv")
    args = ap.parse_args()

    base = load_scores(BASELINE)
    held = {k: load_scores(v) for k, v in HELDOUT.items()}
    samples = sorted(base)
    full = weighted_mean(base, samples)
    full_vals = np.array([base[s][0] for s in samples])
    buckets = strata(samples, base)

    print(f"full 199 baseline weighted adj_edge_jaccard: {full:.4f}")
    for k, sc in held.items():
        print(f"  full 199 delta {k:9s}: {weighted_mean(sc, samples) - full:+.4f}")

    print(f"\nold 9-sample set: baseline {weighted_mean(base, OLD_SET):.4f} "
          f"(off by {weighted_mean(base, OLD_SET) - full:+.4f})")
    for k, sc in held.items():
        d = weighted_mean(sc, OLD_SET) - weighted_mean(base, OLD_SET)
        true = weighted_mean(sc, samples) - full
        print(f"  old-set delta {k:9s}: {d:+.4f} against true {true:+.4f} "
              f"(error {d - true:+.4f})")

    # How much does set size buy? Distribution of recovered delta over many
    # stratified draws. Reported, never selected on.
    print("\nsize sweep, 400 stratified draws each (delta error, held-out configs):")
    print(f"{'n':>4} {'|base err|':>11} {'sigma075 p05..p95':>22} {'zshift2 p05..p95':>22}")
    for n in args.sizes:
        rng = np.random.default_rng(args.seed)
        berr, errs = [], {k: [] for k in held}
        for _ in range(400):
            cand = draw(buckets, n, len(samples), rng)
            b = weighted_mean(base, cand)
            berr.append(abs(b - full))
            for k, sc in held.items():
                errs[k].append((weighted_mean(sc, cand) - b)
                               - (weighted_mean(sc, samples) - full))
        cells = []
        for k in ("sigma075", "zshift2"):
            e = np.array(errs[k])
            cells.append(f"{np.quantile(e, 0.05):+.4f}..{np.quantile(e, 0.95):+.4f}")
        print(f"{n:>4} {np.mean(berr):>11.4f} {cells[0]:>22} {cells[1]:>22}")

    # Pick the best-matching draw at the requested size. Baseline cache only.
    rng = np.random.default_rng(args.seed + 1)
    best, best_cost = None, np.inf
    for _ in range(args.trials):
        cand = draw(buckets, args.n, len(samples), rng)
        vals = np.array([base[s][0] for s in cand])
        cost = (abs(weighted_mean(base, cand) - full) / 0.01
                + ks(vals, full_vals)
                + abs(vals.std() - full_vals.std()) / 0.01)
        if cost < best_cost:
            best, best_cost = cand, cost

    bmean = weighted_mean(base, best)
    bvals = np.array([base[s][0] for s in best])
    print(f"\nchosen set, n={args.n}")
    print(f"  baseline weighted {bmean:.4f} against {full:.4f} "
          f"(off by {bmean - full:+.4f})")
    print(f"  score sd {bvals.std():.4f} against {full_vals.std():.4f}, "
          f"KS {ks(bvals, full_vals):.3f}")
    for e in sorted({embryo_of(s) for s in samples}):
        share = sum(1 for s in best if embryo_of(s) == e) / len(best)
        ref = sum(1 for s in samples if embryo_of(s) == e) / len(samples)
        print(f"  {e}: {share:.2f} of set against {ref:.2f} of full")
    print("  blind test, held out of the objective:")
    for k, sc in held.items():
        d = weighted_mean(sc, best) - bmean
        true = weighted_mean(sc, samples) - full
        print(f"    delta {k:9s}: {d:+.4f} against true {true:+.4f} "
              f"(error {d - true:+.4f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "embryo", "baseline_adj_edge_jaccard", "weight"])
        for s in best:
            w.writerow([s, embryo_of(s), f"{base[s][0]:.6f}", int(base[s][1])])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
