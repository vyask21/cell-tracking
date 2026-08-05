"""Run one config across the leave-one-embryo-out folds and log it.

    python -m src.train --config conf/baseline.yaml

The name is inherited from the template. The classical baseline fits nothing, so
this is an evaluation loop rather than a training loop, but the interface stays
because a learned detector will slot into the same place and every row in
experiments.csv has to have come from the same path to be comparable.

Two folds, and they are reported separately. Holding out `44b6` trains on dense
annotation and validates on sparse; holding out `6bba` does the reverse on far less
data. Averaging them hides the only thing the split measures. A change is believed
when both folds move the same way, and is inconclusive otherwise.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from src import cv as cv_mod
from src import ledger, metrics
from src.config import REPO_ROOT, load_config
from src.data import DEFAULT_SCALE_ZYX, read_scale
from src.pipeline import predict_sample

# Scoring all 199 samples takes hours, so per-sample results are cached on disk
# and a rerun resumes rather than starting over. The cache is keyed by the config
# hash, so changing any setting invalidates it automatically and there is no way
# to silently mix results from two different configs.
CACHE_FIELDS = [
    "sample", "edge_tp", "edge_fp", "edge_fn",
    "division_tp", "division_fp", "division_fn",
    "num_pred_nodes", "node_recall", "total_node_ratio",
    "edge_jaccard", "adj_edge_jaccard",
]


def cache_path(cfg) -> str:
    d = REPO_ROOT / "artifacts" / "scores"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{cfg.name}_{cfg.hash()}.csv")


def load_cache(path: str) -> dict[str, metrics.SampleScore]:
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            kwargs = {
                k: (row[k] if k == "sample" else float(row[k])) for k in CACHE_FIELDS
            }
            for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp",
                      "division_fp", "division_fn", "num_pred_nodes"):
                kwargs[k] = int(kwargs[k])
            out[row["sample"]] = metrics.SampleScore(**kwargs)
    return out


def append_cache(path: str, score: metrics.SampleScore) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CACHE_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: getattr(score, k) for k in CACHE_FIELDS})


def evaluate_fold(fold, cfg, limit_samples: int = 0, verbose: bool = True):
    """Predict and score every validation sample in one fold."""
    samples = list(fold.valid)
    if limit_samples:
        # Deterministic subsample for quick iteration. Marked in the ledger notes,
        # because a score over 10 samples is not comparable to one over 128.
        rng = np.random.default_rng(cfg.seed)
        idx = rng.permutation(len(samples))[:limit_samples]
        samples = [samples[i] for i in sorted(idx)]

    cache = load_cache(cache_path(cfg))
    scores = []
    t0 = time.time()
    n_new = 0
    for i, sample in enumerate(samples, 1):
        if sample in cache:
            s = cache[sample]
            scores.append(s)
            if verbose:
                print(
                    f"    [{i}/{len(samples)}] {sample}: adj_J={s.adj_edge_jaccard:.4f} "
                    "(cached)",
                    flush=True,
                )
            continue

        zarr_path = str(cfg.train_dir / f"{sample}.zarr")
        gt_geff = str(cfg.train_dir / f"{sample}.geff")
        graph, stats = predict_sample(zarr_path, cfg)
        scale = read_scale(zarr_path) or DEFAULT_SCALE_ZYX
        s = metrics.score_prediction(graph, gt_geff, sample=sample, scale=scale)
        scores.append(s)
        append_cache(cache_path(cfg), s)
        n_new += 1
        if verbose:
            done_left = len(samples) - i
            eta = (time.time() - t0) / max(n_new, 1) * done_left / 60
            print(
                f"    [{i}/{len(samples)}] {sample}: "
                f"adj_J={s.adj_edge_jaccard:.4f} J={s.edge_jaccard:.4f} "
                f"recall={s.node_recall:.3f} "
                f"nodes={s.num_pred_nodes} ratio={s.total_node_ratio:+.2f} "
                f"div={s.division_tp}/{s.division_fp}/{s.division_fn} "
                f"[{stats['detect_s'] + stats['link_s']:.1f}s, eta {eta:.0f} min]",
                flush=True,
            )
    if verbose:
        print(f"    fold {fold.index} took {(time.time() - t0) / 60:.1f} min")
    return scores


def run(config_path: str, notes: str = "", limit_samples: int = 0) -> int:
    cfg = load_config(config_path)
    samples = cv_mod.list_samples(str(cfg.train_dir))
    if not samples:
        raise SystemExit(
            f"no samples with ground truth under {cfg.train_dir}. "
            "Run scripts/download_data.py first."
        )

    folds = cv_mod.make_folds(samples, cfg)
    cv_mod.check_no_embryo_leak(folds)
    print(cv_mod.describe(folds))
    print()

    fold_summaries = []
    for fold in folds:
        print(f"  fold {fold.index}: holding out {fold.held_out_embryo}")
        scores = evaluate_fold(fold, cfg, limit_samples=limit_samples)
        summary = metrics.score_with_interval(scores, seed=cfg.seed)
        summary["held_out"] = fold.held_out_embryo
        summary["n_samples"] = len(scores)
        fold_summaries.append(summary)
        print(
            f"  fold {fold.index} ({fold.held_out_embryo}): score={summary['score']:.4f} "
            f"adj_edge_J={summary['adj_edge_jaccard']:.4f} "
            f"[{summary['adj_edge_jaccard_lo']:.4f}, {summary['adj_edge_jaccard_hi']:.4f}] "
            f"div_J={summary['division_jaccard']:.4f} "
            f"node_recall={summary['node_recall']:.4f}\n"
        )

    fold_scores = [s["score"] for s in fold_summaries]
    cv_mean = float(np.nanmean(fold_scores))
    cv_std = float(np.nanstd(fold_scores))
    detail = ",".join(f"{s['held_out']}={s['score']:.4f}" for s in fold_summaries)
    ci = ";".join(
        f"{s['held_out']}=[{s['adj_edge_jaccard_lo']:.4f},{s['adj_edge_jaccard_hi']:.4f}]"
        for s in fold_summaries
    )

    print("=" * 72)
    print(f"per fold : {detail}")
    print(f"mean     : {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"bootstrap: {ci}")
    spread = abs(fold_scores[0] - fold_scores[-1]) if len(fold_scores) > 1 else 0.0
    if spread > 0.05:
        print(
            f"note: the two folds differ by {spread:.4f}. That is the embryo transfer\n"
            "      gap, not noise. Read them separately before believing the mean."
        )

    if limit_samples:
        notes = (notes + f" [limited to {limit_samples} samples/fold]").strip()

    exp_id = ledger.append(
        name=cfg.name,
        config=str(cfg.path),
        config_hash=cfg.hash(),
        cv_mean=cv_mean,
        cv_std=cv_std,
        folds=len(folds),
        cv_detail=detail,
        cv_ci=ci,
        notes=notes,
    )
    print(f"\nlogged as experiment {exp_id}")
    print(f"predict test with: python -m src.predict --config {config_path} --id {exp_id}")
    return exp_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument(
        "--limit-samples",
        type=int,
        default=0,
        help="score only N samples per fold, for quick iteration",
    )
    a = ap.parse_args()
    run(a.config, a.notes, a.limit_samples)
