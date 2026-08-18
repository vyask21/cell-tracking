"""Of the GT cells the baseline never matches, are they mislocalised or absent?

`scripts/diagnose_edge_losses.py` established that 87.5% of missing edges trace to
GT nodes with no matched prediction, and that the baseline's node match rate is
0.706. That says detection is the lever but not what is wrong with it, and the two
possibilities call for different work:

- **Absent.** No detection anywhere near the cell. The detector never fired on it,
  so the fix is a detector that responds to nucleus-shaped structure the current
  Otsu-plus-local-maximum does not.
- **Mislocalised.** A detection sits near the cell but outside the 7 um matching
  cap, so it exists and is simply in the wrong place. The fix is centroid accuracy,
  which is a much smaller change.
- **Lost to competition.** A detection sits within 7 um but the optimal assignment
  gave it to a different GT node. Extra detections would not help; these are cases
  where two GT cells contend for one prediction.

Nothing here is fit and nothing is tuned. It reads ground truth for every sample it
touches, so it is a diagnostic only.

    python scripts/diagnose_missed_nodes.py --config conf/baseline.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.data import read_geff, read_scale  # noqa: E402
from src.pipeline import predict_sample  # noqa: E402

MAX_MATCH_UM = 7.0
# Beyond this a detection is too far away to be called "the same cell in the wrong
# place". Chosen as comfortably past the cap, not tuned.
FAR_UM = 12.0

SAMPLES = (
    "44b6_144b256d", "44b6_90724892", "44b6_d5e7d891", "44b6_db3c847b",
    "6bba_268e1230", "6bba_3db54e20", "6bba_55c70843", "6bba_67ebd073",
    "6bba_6ca87370",
)


def analyse(sample, data_dir, cfg):
    zarr_path = os.path.join(data_dir, sample + ".zarr")
    graph, _ = predict_sample(zarr_path, cfg)
    gt = read_geff(os.path.join(data_dir, sample + ".geff"))
    scale = np.asarray(read_scale(zarr_path), dtype=np.float64)

    gt_zyx, gt_t = gt.nodes.zyx(), np.asarray(gt.nodes.t)
    p_zyx, p_t = graph.nodes.zyx(), np.asarray(graph.nodes.t)

    counts = Counter()
    nearest_unmatched = []
    for t in np.unique(gt_t):
        g = gt_zyx[gt_t == t]
        p = p_zyx[p_t == t]
        if g.shape[0] == 0:
            continue
        if p.shape[0] == 0:
            counts["absent"] += g.shape[0]
            continue

        d = np.linalg.norm((g * scale)[:, None, :] - (p * scale)[None, :, :], axis=2)
        cost = np.where(d <= MAX_MATCH_UM, d, 1e6)
        rows, cols = linear_sum_assignment(cost)
        ok = {int(r) for r, c in zip(rows, cols) if d[r, c] <= MAX_MATCH_UM}

        nearest = d.min(axis=1)
        for i in range(g.shape[0]):
            if i in ok:
                counts["matched"] += 1
                continue
            nearest_unmatched.append(float(nearest[i]))
            if nearest[i] <= MAX_MATCH_UM:
                counts["lost_to_competition"] += 1
            elif nearest[i] <= FAR_UM:
                counts["mislocalised"] += 1
            else:
                counts["absent"] += 1
    return counts, nearest_unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/baseline.yaml")
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--out", default="data/meta/missed_nodes.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    total = Counter()
    per_embryo = {}
    near = []
    rows = []
    t0 = time.time()
    for i, s in enumerate(SAMPLES, 1):
        c, n = analyse(s, args.data, cfg)
        total.update(c)
        per_embryo.setdefault(s.split("_")[0], Counter()).update(c)
        near += n
        rows.append({"sample": s, **c})
        print(f"  [{i}/{len(SAMPLES)}] {s} [{(time.time()-t0)/60:.1f} min]", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r if k != "sample"})
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample"] + keys)
        w.writeheader()
        w.writerows(rows)

    order = ("matched", "lost_to_competition", "mislocalised", "absent")
    n_gt = sum(total[k] for k in order)
    print(f"\nconfig {cfg.name}, {len(SAMPLES)} samples, {n_gt} GT nodes\n")
    print(f"{'outcome':>22}{'count':>9}{'% of GT':>10}{'% of unmatched':>17}")
    unm = n_gt - total["matched"]
    for k in order:
        pu = f"{100*total[k]/unm:>16.1f}%" if k != "matched" else " " * 17
        print(f"{k:>22}{total[k]:>9}{100*total[k]/n_gt:>9.1f}%{pu}")

    print(f"\n{'':>22}{'matched':>10}{'lost':>8}{'misloc':>9}{'absent':>9}")
    for e, c in sorted(per_embryo.items()):
        n = sum(c[k] for k in order)
        print(f"{e:>22}" + "".join(f"{100*c[k]/n:>8.1f}%" for k in order))

    if near:
        q = np.percentile(near, [10, 25, 50, 75, 90])
        print(f"\nnearest detection to an unmatched GT node, um:")
        print(f"  p10 {q[0]:.1f}  p25 {q[1]:.1f}  median {q[2]:.1f}  "
              f"p75 {q[3]:.1f}  p90 {q[4]:.1f}  max {max(near):.1f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
