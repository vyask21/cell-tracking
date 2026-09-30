"""Score a config on the 19 videos the support pack did not train on.

Generalises `screen_unet_heldout.py`: that one hard-wires the local-max baseline
against `conf/unet50.yaml`, this one runs any config through
`src.pipeline.predict_sample`, which is the path a submission takes, and
compares against every cached arm it can find.

The 19 are the only out-of-sample videos for anything built on these weights, and even
they are out of sample only in a limited sense. They are video-disjoint from the pack's
training set but they are **not embryo-disjoint**: the pack trained on 180 videos
drawn from these same two embryos, while the hidden test set is embryo-disjoint.
Exp 4 measured what that costs. A +0.1270 here arrived as +0.0200 on the
leaderboard. So read a delta here as a direction and an upper bound, never as a
CV number and never as a prediction of the leaderboard.

    .venv\\Scripts\\python.exe scripts/screen_edge_link.py --config conf/unet50_edge.yaml

Launch detached for anything covering all 19, never as a harness background task:

    Start-Process -FilePath .venv\\Scripts\\python.exe `
      -ArgumentList "scripts/screen_edge_link.py","--config","conf/unet50_edge.yaml","--workers","3" `
      -RedirectStandardOutput artifacts/edge_link_screen.out `
      -RedirectStandardError artifacts/edge_link_screen.err -NoNewWindow

Per-sample results cache under `--cache`, written through a temp name, so a kill
resumes for free.
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

# The 19 the reference seed-0 split held out, reproduced by
# random.Random(0).shuffle(sorted(stems))[:19].
HELDOUT = (
    "44b6_1574802b", "44b6_706092f0", "44b6_d5e7d891", "44b6_d754aa59",
    "44b6_e57ff5c6", "6bba_2312ac41", "6bba_268e1230", "6bba_283bf9f1",
    "6bba_3a1849c2", "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d",
    "6bba_7af54fde", "6bba_7b5d3b2c", "6bba_aeee7805", "6bba_afb141ff",
    "6bba_c27cba08", "6bba_c328f2fd", "6bba_d1acb6ff",
)

# Arms already measured on exactly these 19, for the comparison table.
REFERENCE_ARMS = (
    ("local-max", "artifacts/scores/baseline_localmax_4f8a3af0.csv"),
    ("unet50", "data/meta/unet_heldout.csv"),
)

FIELDS = [
    "sample", "embryo", "edge_tp", "edge_fp", "edge_fn", "num_pred_nodes",
    "node_recall", "total_node_ratio", "edge_jaccard", "adj_edge_jaccard",
    "n_candidates", "cand_per_node", "src_covered", "tgt_covered", "seconds",
]


def weighted(rows: list[dict], key: str = "adj_edge_jaccard") -> float:
    """Weight-average by (tp + fp + fn), which is what the scorer does."""
    w = np.array([int(r["edge_tp"]) + int(r["edge_fp"]) + int(r["edge_fn"])
                  for r in rows], dtype=float)
    v = np.array([float(r[key]) for r in rows], dtype=float)
    return float(np.sum(v * w) / np.sum(w))


def load_arm(path: str, samples) -> list[dict]:
    keep = set(samples)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["sample"] in keep]
    return rows if len(rows) == len(keep) else []


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
    from src.data import DEFAULT_SCALE_ZYX, read_scale
    from src.pipeline import predict_sample

    cfg = load_config(config)
    zarr_path = os.path.join(data_dir, sample + ".zarr")
    gt_geff = os.path.join(data_dir, sample + ".geff")

    t0 = time.time()
    graph, stats = predict_sample(zarr_path, cfg)
    scale = read_scale(zarr_path) or DEFAULT_SCALE_ZYX
    s = metrics.score_prediction(graph, gt_geff, sample=sample, scale=scale)

    row = {
        "sample": sample, "embryo": sample.split("_")[0],
        "edge_tp": s.edge_tp, "edge_fp": s.edge_fp, "edge_fn": s.edge_fn,
        "num_pred_nodes": s.num_pred_nodes, "node_recall": s.node_recall,
        "total_node_ratio": s.total_node_ratio, "edge_jaccard": s.edge_jaccard,
        "adj_edge_jaccard": s.adj_edge_jaccard,
        "n_candidates": stats.get("n_candidates", ""),
        "cand_per_node": stats.get("cand_per_node", ""),
        "src_covered": stats.get("src_covered", ""),
        "tgt_covered": stats.get("tgt_covered", ""),
        "seconds": round(time.time() - t0, 1),
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerow(row)
    os.replace(tmp, out_path)
    return sample, row, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/unet50_edge.yaml")
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N samples, for a timing probe")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.config))[0]
    out = args.out or f"data/meta/heldout_{stem}.csv"
    cache = args.cache or f"data/meta/heldout_{stem}_cache"
    os.makedirs(cache, exist_ok=True)

    samples = list(HELDOUT)[:args.limit] if args.limit else list(HELDOUT)
    done = sum(1 for s in samples if os.path.exists(os.path.join(cache, s + ".csv")))
    print(f"{len(samples)} held-out samples, {args.workers} workers x "
          f"{args.threads} torch threads, config {args.config}")
    print(f"  {done} already cached, cache {cache}\n", flush=True)

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_sample, s, args.data, cache, args.config,
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
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    arms = [(label, load_arm(path, [r["sample"] for r in rows]))
            for label, path in REFERENCE_ARMS]
    arms = [(label, rs) for label, rs in arms if rs]
    arms.append((stem, rows))

    print(f"\n{'':>16}{'raw J':>9}{'adj J':>9}{'tp':>8}{'fp':>8}{'fn':>8}"
          f"{'nodes':>10}{'recall':>8}")
    for label, rs in arms:
        print(f"{label:>16}{weighted(rs, 'edge_jaccard'):>9.4f}"
              f"{weighted(rs):>9.4f}"
              f"{sum(int(r['edge_tp']) for r in rs):>8}"
              f"{sum(int(r['edge_fp']) for r in rs):>8}"
              f"{sum(int(r['edge_fn']) for r in rs):>8}"
              f"{sum(int(r['num_pred_nodes']) for r in rs):>10}"
              f"{np.mean([float(r['node_recall']) for r in rs]):>8.3f}")

    if len(arms) > 1:
        prev_label, prev_rows = arms[-2]
        print(f"\ndelta against {prev_label}: "
              f"{weighted(rows) - weighted(prev_rows):+.4f} adjusted, "
              f"{weighted(rows, 'edge_jaccard') - weighted(prev_rows, 'edge_jaccard'):+.4f} raw")
        print("\nper embryo, since a change is believed only when it helps both:")
        for emb in ("44b6", "6bba"):
            a = [r for r in prev_rows if r["sample"].startswith(emb)]
            b = [r for r in rows if r["sample"].startswith(emb)]
            if a and b:
                print(f"  {emb} ({len(b)} samples): {prev_label} {weighted(a):.4f}, "
                      f"{stem} {weighted(b):.4f}, delta {weighted(b) - weighted(a):+.4f}")
        wins = sum(1 for r, q in zip(rows, prev_rows)
                   if float(r["adj_edge_jaccard"]) > float(q["adj_edge_jaccard"]))
        print(f"  per sample: {wins} better, {len(rows) - wins} worse or equal")

    cov = [r for r in rows if r.get("src_covered") not in ("", None)]
    if cov:
        print(f"\ncandidate set: {np.mean([float(r['cand_per_node']) for r in cov]):.2f} "
              f"per node, {np.mean([float(r['src_covered']) for r in cov]):.3f} of "
              f"source nodes and {np.mean([float(r['tgt_covered']) for r in cov]):.3f} "
              f"of target nodes have at least one candidate.")
        print("A source node with no candidate cannot be linked at any cost, so a "
              "low\nnumber here means edge_threshold is deciding the result rather "
              "than the solver.")

    print("\nThe 19 are video-disjoint from the pack's training set but NOT "
          "embryo-disjoint,\nand the hidden test is. Exp 4 saw +0.1270 here arrive "
          "as +0.0200 on the\nleaderboard. Direction and upper bound, not a CV "
          "number.")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min, wrote {out}")


if __name__ == "__main__":
    main()
