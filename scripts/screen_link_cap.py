"""Screen `max_link_um` end to end with the organisers' scorer, sharing detections.

Why this is cheap. `max_link_um` affects only linking, so detection can be run once
per sample and reused across every candidate cap. Detection is ~99% of the runtime,
so screening five caps costs barely more than screening one.

Why it scores rather than proxies. The last screen ranked candidates on a
detection-only proxy (`recall x penalty`), promoted a setting that proxy liked, and
lost 0.141 on the leaderboard. This metric scores edges, so a screen has to link
and score. It uses `metrics.score_prediction`, the same call `train.py` makes.

What this is testing. `conf/baseline.yaml` asserts that "the scorer matches nodes
under a 7 um cap, so linking further than that can never produce a true positive
edge". That is wrong. The 7 um cap governs **node-to-node matching**, predicted node
against ground-truth node, not edge length. A predicted edge spanning 12 um is a
true positive whenever each endpoint is independently within 7 um of its own GT
node. `scripts/diagnose_edge_losses.py` measured 170 GT edges, 3.9% of the total,
where both endpoints matched and the link was refused only because of this cap.

This is a screen on 9 samples, not a CV result. Nothing is promoted on it; it
decides whether a full leave-one-embryo-out run is worth 3 to 5 hours.

    python scripts/screen_link_cap.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import DEFAULT_SCALE_ZYX, Image, read_scale  # noqa: E402
from src.detect import detect_sequence  # noqa: E402
from src.link import link_sequence  # noqa: E402

CAPS = (7.0, 9.0, 12.0, 16.0, 25.0)

SAMPLES = (
    "44b6_144b256d", "44b6_90724892", "44b6_d5e7d891", "44b6_db3c847b",
    "6bba_268e1230", "6bba_3db54e20", "6bba_55c70843", "6bba_67ebd073",
    "6bba_6ca87370",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/baseline.yaml")
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--out", default="data/meta/link_cap_screen.csv")
    ap.add_argument("--caps", type=float, nargs="+", default=list(CAPS))
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg.detect or {}
    rows = []
    t0 = time.time()

    for i, sample in enumerate(SAMPLES, 1):
        zarr_path = os.path.join(args.data, sample + ".zarr")
        gt_geff = os.path.join(args.data, sample + ".geff")
        image = Image(zarr_path)
        scale = read_scale(zarr_path) or DEFAULT_SCALE_ZYX

        # Detect once. Every cap below reuses this, which is the whole point.
        det = detect_sequence(
            image,
            sigma_um=float(d.get("sigma_um", 2.0)),
            min_sep_um=float(d.get("min_sep_um", 3.0)),
            threshold_scale=float(d.get("threshold_scale", 0.5)),
            max_detections=int(d.get("max_detections", 20000)),
        )
        for cap in args.caps:
            graph = link_sequence(det, scale_zyx=image.scale, max_link_um=cap,
                                  max_division_um=0.0)
            s = metrics.score_prediction(graph, gt_geff, sample=sample, scale=scale)
            rows.append({
                "sample": sample, "embryo": sample.split("_")[0], "max_link_um": cap,
                "edge_tp": s.edge_tp, "edge_fp": s.edge_fp, "edge_fn": s.edge_fn,
                "edge_jaccard": s.edge_jaccard, "adj_edge_jaccard": s.adj_edge_jaccard,
                "num_pred_nodes": s.num_pred_nodes, "n_pred_edges": int(graph.edges.shape[0]),
            })
        print(f"  [{i}/{len(SAMPLES)}] {sample} [{(time.time()-t0)/60:.1f} min]", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nconfig {cfg.name}, {len(SAMPLES)} samples, scorer-weighted over samples")
    print(f"{'cap um':>8}{'edge_tp':>9}{'edge_fp':>9}{'edge_fn':>9}"
          f"{'pred_edges':>12}{'weighted adj_J':>16}")
    for cap in args.caps:
        rs = [r for r in rows if r["max_link_um"] == cap]
        tp = sum(r["edge_tp"] for r in rs)
        fp = sum(r["edge_fp"] for r in rs)
        fn = sum(r["edge_fn"] for r in rs)
        pe = sum(r["n_pred_edges"] for r in rs)
        # The scorer weight-averages per-sample adjusted Jaccard by TP+FP+FN.
        w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rs]
        wa = sum(a * r["adj_edge_jaccard"] for a, r in zip(w, rs)) / sum(w)
        mark = "  <- baseline" if cap == 7.0 else ""
        print(f"{cap:>8}{tp:>9}{fp:>9}{fn:>9}{pe:>12}{wa:>16.4f}{mark}")

    for emb in ("44b6", "6bba"):
        print(f"\n  {emb}:")
        for cap in args.caps:
            rs = [r for r in rows if r["max_link_um"] == cap and r["embryo"] == emb]
            w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rs]
            wa = sum(a * r["adj_edge_jaccard"] for a, r in zip(w, rs)) / sum(w)
            print(f"    cap {cap:>5}: weighted adj_J {wa:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
