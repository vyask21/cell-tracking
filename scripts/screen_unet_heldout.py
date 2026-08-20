"""The first honest number for the learned detector.

The 50-epoch support pack was trained on the reference seed-0 90/10 split, which
is 180 of the 199 training videos. Scoring it on any of those 180 measures its
training set. A single leaked sample scored 0.9286 against the local-max
baseline's 0.7453, which is exactly the kind of number that starts a bad week.

The 19 videos that split held out are the only honest ones, and they are close to
representative: the local-max baseline scores 0.6813 on them against 0.6876 over
all 199, off by -0.0063.

Only the detector changes. The linker keeps `max_link_um` 7.0 and divisions off,
the scorer is the organisers' own, and the comparison is against the cached
baseline scores for the same 19 samples, so the delta is attributable to the
detector and nothing else.

Nineteen samples is small. The screening-noise table in NOTES.md puts a 20-sample
set at roughly +/- 0.04 on a delta of the size sigma075 produced. That is fine
here only because the effect being measured is an order of magnitude larger; it
would not be fine for a tuning decision.

Launch detached, never as a harness background task:

    Start-Process -FilePath .venv\\Scripts\\python.exe `
      -ArgumentList "scripts/screen_unet_heldout.py","--workers","4" `
      -RedirectStandardOutput artifacts/unet_heldout.out `
      -RedirectStandardError artifacts/unet_heldout.err -NoNewWindow
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASELINE_CACHE = "artifacts/scores/baseline_localmax_4f8a3af0.csv"

# The 19 the reference seed-0 split held out, reproduced by
# random.Random(0).shuffle(sorted(stems))[:19]. These are the only samples the
# 50-epoch pack never saw.
HELDOUT = (
    "44b6_1574802b", "44b6_706092f0", "44b6_d5e7d891", "44b6_d754aa59",
    "44b6_e57ff5c6", "6bba_2312ac41", "6bba_268e1230", "6bba_283bf9f1",
    "6bba_3a1849c2", "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d",
    "6bba_7af54fde", "6bba_7b5d3b2c", "6bba_aeee7805", "6bba_afb141ff",
    "6bba_c27cba08", "6bba_c328f2fd", "6bba_d1acb6ff",
)

FIELDS = [
    "sample", "embryo", "edge_tp", "edge_fp", "edge_fn", "num_pred_nodes",
    "node_recall", "total_node_ratio", "edge_jaccard", "adj_edge_jaccard",
    "seconds",
]


def weighted(rows: list[dict], key: str = "adj_edge_jaccard") -> float:
    w = np.array([int(r["edge_tp"]) + int(r["edge_fp"]) + int(r["edge_fn"])
                  for r in rows], dtype=float)
    v = np.array([float(r[key]) for r in rows], dtype=float)
    return float(np.sum(v * w) / np.sum(w))


def baseline_rows(samples) -> list[dict]:
    keep = set(samples)
    with open(BASELINE_CACHE, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["sample"] in keep]


def run_sample(sample: str, data_dir: str, cache_dir: str, config: str,
               threads: int) -> tuple[str, dict, float]:
    out_path = os.path.join(cache_dir, sample + ".csv")
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as fh:
            return sample, next(iter(csv.DictReader(fh))), 0.0

    import torch
    torch.set_num_threads(max(1, threads))

    from src import metrics
    from src.config import load_config
    from src.data import DEFAULT_SCALE_ZYX, Image, read_scale
    from src.link import link_sequence
    from src.unet import detect_sequence_unet, load_detector

    cfg = load_config(config)
    d = cfg.detect or {}
    link_cfg = cfg.link or {}
    zarr_path = os.path.join(data_dir, sample + ".zarr")
    gt_geff = os.path.join(data_dir, sample + ".geff")

    t0 = time.time()
    detector = load_detector(d.get("weights"), str(d.get("device", "cpu")))
    coords = detect_sequence_unet(
        zarr_path,
        model=detector,
        det_threshold=float(d.get("det_threshold", 0.955)),
        pool_kernel_um=float(d.get("pool_kernel_um", 5.0)),
        det_tta=bool(d.get("det_tta", False)),
        device=str(d.get("device", "cpu")),
    )
    image = Image(zarr_path)
    scale = read_scale(zarr_path) or DEFAULT_SCALE_ZYX
    graph = link_sequence(
        coords,
        scale_zyx=image.scale,
        max_link_um=float(link_cfg.get("max_link_um", 7.0)),
        max_division_um=float(link_cfg.get("max_division_um", 0.0)),
    )
    s = metrics.score_prediction(graph, gt_geff, sample=sample, scale=scale)
    row = {
        "sample": sample, "embryo": sample.split("_")[0],
        "edge_tp": s.edge_tp, "edge_fp": s.edge_fp, "edge_fn": s.edge_fn,
        "num_pred_nodes": s.num_pred_nodes, "node_recall": s.node_recall,
        "total_node_ratio": s.total_node_ratio, "edge_jaccard": s.edge_jaccard,
        "adj_edge_jaccard": s.adj_edge_jaccard,
        "seconds": round(time.time() - t0, 1),
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerow(row)
    os.replace(tmp, out_path)
    return sample, row, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/unet50.yaml")
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--out", default="data/meta/unet_heldout.csv")
    ap.add_argument("--cache", default="data/meta/unet_heldout_cache")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    samples = list(HELDOUT)
    done = sum(1 for s in samples if os.path.exists(os.path.join(args.cache, s + ".csv")))
    print(f"{len(samples)} held-out samples, {args.workers} workers x "
          f"{args.threads} torch threads, config {args.config}")
    print(f"  {done} already cached\n", flush=True)

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_sample, s, args.data, args.cache, args.config,
                        args.threads): s
            for s in samples
        }
        for i, fut in enumerate(as_completed(futures), 1):
            sample, row, secs = fut.result()
            rows.append(row)
            el = (time.time() - t0) / 60
            tag = "cached" if secs == 0.0 else f"{secs / 60:.1f} min"
            print(f"  [{i}/{len(samples)}] {sample} ({tag}) adj_J="
                  f"{float(row['adj_edge_jaccard']):.4f} "
                  f"[{el:.1f} min, eta {el / i * (len(samples) - i):.0f} min]",
                  flush=True)

    rows.sort(key=lambda r: r["sample"])
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    base = baseline_rows(samples)
    bw, uw = weighted(base), weighted(rows)
    print(f"\n{'':>14}{'raw J':>9}{'adj J':>9}{'tp':>8}{'fp':>8}{'fn':>8}"
          f"{'nodes':>10}{'recall':>8}")
    for label, rs in (("local-max", base), ("unet50", rows)):
        print(f"{label:>14}{weighted(rs, 'edge_jaccard'):>9.4f}"
              f"{weighted(rs):>9.4f}"
              f"{sum(int(r['edge_tp']) for r in rs):>8}"
              f"{sum(int(r['edge_fp']) for r in rs):>8}"
              f"{sum(int(r['edge_fn']) for r in rs):>8}"
              f"{sum(int(r['num_pred_nodes']) for r in rs):>10}"
              f"{np.mean([float(r['node_recall']) for r in rs]):>8.3f}")
    print(f"{'delta':>14}{'':>9}{uw - bw:>+9.4f}")

    print("\nper embryo, since a change is believed only when it helps both:")
    for emb in ("44b6", "6bba"):
        b = [r for r in base if r["sample"].startswith(emb)]
        u = [r for r in rows if r["sample"].startswith(emb)]
        if b and u:
            print(f"  {emb} ({len(u)} samples): local-max {weighted(b):.4f}, "
                  f"unet50 {weighted(u):.4f}, delta {weighted(u) - weighted(b):+.4f}")

    print("\nThese 19 are the only samples the pack did not train on. Any number "
          "measured on\nthe other 180 is leakage. 19 samples is a small set, so "
          "read this as a decision\nabout direction, not as a CV result.")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min, wrote {args.out}")


if __name__ == "__main__":
    main()
