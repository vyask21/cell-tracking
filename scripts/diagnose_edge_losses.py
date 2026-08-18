"""Decompose the baseline's missing and spurious edges into causes.

Why this exists. The baseline scores edge TP 92,702, FP 8,629, FN 36,181 over the
199 training samples, so **false negatives outnumber false positives four to one**
and they, not FPs, are what caps the score. "Raise node recall" was already tried
globally in exp 2 and lost 0.141 on the leaderboard, so before touching anything
again this asks the prior question: of the edges the baseline misses, how many are
missed because a cell was never detected, and how many because both endpoints were
detected and the linker still failed to join them?

Those two numbers point at completely different work. The first says detection, the
second says linking, and the repo has spent two experiments guessing between them.

Method mirrors the scorer rather than approximating it: per timepoint, ground-truth
nodes are matched to predicted nodes by optimal bipartite assignment under the 7 um
cap. A GT edge is then a true positive only when both endpoints matched and the
prediction carries that exact edge.

Detection only feeds this; nothing here is fit, and it reads the ground truth of
every sample it touches, so it is a diagnostic and never a source of a tuned
parameter.

    python scripts/diagnose_edge_losses.py --config conf/baseline.yaml
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

# The same nine samples the detection grid probe used, so findings stay comparable
# across the two diagnostics rather than resting on different data each time.
SAMPLES = (
    "44b6_144b256d", "44b6_90724892", "44b6_d5e7d891", "44b6_db3c847b",
    "6bba_268e1230", "6bba_3db54e20", "6bba_55c70843", "6bba_67ebd073",
    "6bba_6ca87370",
)


def match_frame(gt_zyx, pred_zyx, scale):
    """GT index to pred index under the scorer's rule. Returns a dict."""
    if gt_zyx.shape[0] == 0 or pred_zyx.shape[0] == 0:
        return {}
    s = np.asarray(scale, dtype=np.float64)[None, :]
    d = np.linalg.norm((gt_zyx * s)[:, None, :] - (pred_zyx * s)[None, :, :], axis=2)
    cost = np.where(d <= MAX_MATCH_UM, d, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if d[r, c] <= MAX_MATCH_UM}


def analyse(sample, data_dir, cfg):
    zarr_path = os.path.join(data_dir, sample + ".zarr")
    graph, _ = predict_sample(zarr_path, cfg)
    gt = read_geff(os.path.join(data_dir, sample + ".geff"))
    scale = read_scale(zarr_path)

    gt_zyx, gt_t, gt_ids = gt.nodes.zyx(), np.asarray(gt.nodes.t), np.asarray(gt.nodes.ids)
    p_zyx, p_t = graph.nodes.zyx(), np.asarray(graph.nodes.t)
    gt_pos = {int(v): k for k, v in enumerate(gt_ids)}

    # GT node -> matched pred node id, solved per timepoint the way the scorer does.
    matched: dict[int, int] = {}
    for t in np.unique(gt_t):
        g_idx = np.flatnonzero(gt_t == t)
        p_idx = np.flatnonzero(p_t == t)
        m = match_frame(gt_zyx[g_idx], p_zyx[p_idx], scale)
        for gi, pi in m.items():
            matched[int(g_idx[gi])] = int(p_idx[pi])

    pred_edges = {(int(a), int(b)) for a, b in graph.edges}
    # Where each predicted node's outgoing link went, to tell "linked elsewhere"
    # apart from "not linked at all".
    out_of = {}
    for a, b in graph.edges:
        out_of.setdefault(int(a), []).append(int(b))

    s = np.asarray(scale, dtype=np.float64)
    counts = Counter()
    gap_um = []
    for u, v in gt.edges:
        gu, gv = gt_pos[int(u)], gt_pos[int(v)]
        pu, pv = matched.get(gu), matched.get(gv)
        if pu is None and pv is None:
            counts["fn_both_endpoints_unmatched"] += 1
        elif pu is None or pv is None:
            counts["fn_one_endpoint_unmatched"] += 1
        elif (pu, pv) in pred_edges:
            counts["tp"] += 1
        else:
            # Both cells were found and matched, and the link still was not made.
            d = float(np.linalg.norm((p_zyx[pv] - p_zyx[pu]) * s))
            gap_um.append(d)
            if d > float(cfg.link.get("max_link_um", 7.0)):
                counts["fn_link_beyond_cap"] += 1
            elif out_of.get(pu):
                counts["fn_parent_linked_elsewhere"] += 1
            else:
                counts["fn_parent_unlinked"] += 1

    counts["gt_edges"] = int(gt.edges.shape[0])
    counts["pred_nodes"] = len(graph.nodes)
    counts["pred_edges"] = int(graph.edges.shape[0])
    counts["gt_nodes_matched"] = len(matched)
    counts["gt_nodes"] = int(gt_ids.size)
    return counts, gap_um


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/baseline.yaml")
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--out", default="data/meta/edge_losses.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    total = Counter()
    gaps = []
    rows = []
    t0 = time.time()
    for i, sample in enumerate(SAMPLES, 1):
        c, g = analyse(sample, args.data, cfg)
        total.update(c)
        gaps += g
        rows.append({"sample": sample, **c})
        print(f"  [{i}/{len(SAMPLES)}] {sample} [{(time.time()-t0)/60:.1f} min]", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r if k != "sample"})
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample"] + keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    gt_e = total["gt_edges"]
    print(f"\nconfig {cfg.name}, {len(SAMPLES)} samples, {gt_e} GT edges")
    print(f"node match rate: {total['gt_nodes_matched']}/{total['gt_nodes']} "
          f"= {total['gt_nodes_matched']/total['gt_nodes']:.3f}\n")
    print(f"{'cause':>34}{'count':>9}{'% of GT edges':>15}")
    for k in ("tp", "fn_both_endpoints_unmatched", "fn_one_endpoint_unmatched",
              "fn_link_beyond_cap", "fn_parent_linked_elsewhere", "fn_parent_unlinked"):
        print(f"{k:>34}{total[k]:>9}{100*total[k]/gt_e:>14.1f}%")
    fn = gt_e - total["tp"]
    print(f"\ntotal FN {fn} ({100*fn/gt_e:.1f}% of GT edges)")
    det = total["fn_both_endpoints_unmatched"] + total["fn_one_endpoint_unmatched"]
    lnk = fn - det
    print(f"  attributable to DETECTION (endpoint never matched): {det} ({100*det/fn:.1f}% of FN)")
    print(f"  attributable to LINKING (both matched, no edge)   : {lnk} ({100*lnk/fn:.1f}% of FN)")
    if gaps:
        q = np.percentile(gaps, [50, 90, 99])
        print(f"\nlink-failure gap um: median {q[0]:.2f}, p90 {q[1]:.2f}, p99 {q[2]:.2f}, max {max(gaps):.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
