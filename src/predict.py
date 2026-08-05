"""Predict every sample in a directory and write the submission CSV.

    python -m src.predict --config conf/baseline.yaml --id 3

This is the half of the code that has to run inside the competition rerun, so it
imports only `src.config`, `src.data`, `src.detect`, `src.link` and `src.pipeline`.
Those depend on tensorstore or zarr, numpy and scipy. It must never import
`src.metrics`, which pulls in tracksdata and is not installed on Kaggle.

Two rules from the submission format that are easy to get wrong:

- **Every dataset in the test directory must appear in the CSV**, even one that
  produced nothing. An empty sample is written with zero rows only if it still
  appears elsewhere; here a sample that yields no detections is reported loudly
  instead, because silently dropping it would fail the whole submission.
- `id` is a single consecutive index across the whole file, not per dataset.
"""

from __future__ import annotations

import argparse
import os
import time

from src.config import load_config
from src.data import list_samples, write_submission
from src.pipeline import predict_sample


def run(
    config_path: str,
    exp_id: int | None = None,
    data_dir: str | None = None,
    out_path: str | None = None,
    limit: int = 0,
) -> str:
    cfg = load_config(config_path)
    test_dir = data_dir or str(cfg.test_dir)

    samples = list_samples(test_dir, require_geff=False)
    if not samples:
        raise SystemExit(f"no .zarr samples under {test_dir}")
    if limit:
        samples = samples[:limit]
    print(f"{len(samples)} samples under {test_dir}")

    graphs = {}
    empty = []
    t0 = time.time()
    for i, sample in enumerate(samples, 1):
        graph, stats = predict_sample(os.path.join(test_dir, sample + ".zarr"), cfg)
        graphs[sample] = graph
        if stats["n_nodes"] == 0:
            empty.append(sample)
        elapsed = time.time() - t0
        eta = elapsed / i * (len(samples) - i)
        print(
            f"  [{i}/{len(samples)}] {sample}: {stats['n_nodes']} nodes, "
            f"{stats['n_edges']} edges, {stats['n_divisions']} divisions "
            f"({stats['detect_s'] + stats['link_s']:.1f}s, eta {eta / 60:.0f} min)",
            flush=True,
        )

    if empty:
        print(f"\nWARNING: {len(empty)} samples produced no detections: {empty[:10]}")

    tag = f"{cfg.name}_{cfg.hash()}"
    if exp_id is not None:
        tag = f"exp{exp_id}_{tag}"
    out = out_path or str(cfg.sub_dir / f"{tag}.csv")
    rows = write_submission(graphs, out)

    print(f"\nwrote {rows} rows for {len(graphs)} datasets to {out}")
    print(f"total {(time.time() - t0) / 60:.1f} min")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--id", type=int, default=None, help="experiment id from experiments.csv")
    ap.add_argument("--data-dir", default=None, help="defaults to the config's test dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a.config, a.id, a.data_dir, a.out, a.limit)
