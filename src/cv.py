"""Cross-validation.

The scheme has to mirror how the organisers split train from test. Getting this
wrong is the single most expensive mistake available in a competition, so the
choice is explicit here.

The split here is by embryo, because the hidden test set is embryo-disjoint from
train. Train contains exactly two embryos, so leave-one-embryo-out is two folds
and each fold trains on a single embryo. That is coarse, and no rearrangement of
these 199 samples fixes it: with two groups there are two ways to hold one out.

What can be fixed is the *error bar*. The competition metric weight-averages a
per-sample adjusted Jaccard, so a fold score is a weighted mean over samples and
can be bootstrapped across the held-out samples. That gives a usable interval on
each fold's number even though the embryo-to-embryo variance itself rests on two
points. Report both, and never report a bare mean.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


def embryo_of(sample: str) -> str:
    """Sample names are `{embryo}_{crop}`, e.g. `6bba_fe670320`.

    The data page documents a longer `{embryo_id}_{field_of_view}` form with a
    multi-segment example, but every one of the 199 shipped names has exactly two
    segments, so the first is the embryo and the second is an opaque crop id.
    """
    return sample.split("_")[0]


def list_samples(train_dir: str) -> list[str]:
    """Every sample with a ground-truth graph, sorted for reproducibility."""
    return sorted(
        d[: -len(".geff")] for d in os.listdir(train_dir) if d.endswith(".geff")
    )


@dataclass(frozen=True)
class Fold:
    index: int
    held_out_embryo: str
    train: tuple[str, ...]
    valid: tuple[str, ...]


def leave_one_embryo_out(samples: list[str]) -> list[Fold]:
    """The leak-free scheme: hold out a whole embryo, train on the rest.

    This is the only split that asks the question the leaderboard asks. It is
    expensive, since each fold trains on roughly half the data, and it produces
    as many folds as there are embryos, which today is two.
    """
    embryos = sorted({embryo_of(s) for s in samples})
    folds = []
    for i, emb in enumerate(embryos):
        valid = tuple(s for s in samples if embryo_of(s) == emb)
        train = tuple(s for s in samples if embryo_of(s) != emb)
        folds.append(Fold(index=i, held_out_embryo=emb, train=train, valid=valid))
    return folds


def within_embryo_kfold(samples: list[str], n_splits: int, seed: int) -> list[Fold]:
    """A cheap screening split that does NOT mirror the test set.

    Crops of the same embryo land on both sides, so this measures in-distribution
    performance and will read optimistic against the leaderboard. It exists only
    to rank candidate changes quickly when a full leave-one-embryo-out run is too
    expensive. Nothing gets promoted on this number alone, and it never goes in
    the ledger's `cv_mean` column without being labelled.
    """
    rng = np.random.default_rng(seed)
    order = np.array(samples)
    rng.shuffle(order)
    folds = []
    for i in range(n_splits):
        valid = tuple(sorted(order[i::n_splits].tolist()))
        train = tuple(sorted(s for s in samples if s not in set(valid)))
        folds.append(Fold(index=i, held_out_embryo="(mixed)", train=train, valid=valid))
    return folds


def make_folds(samples: list[str], cfg) -> list[Fold]:
    """cv config keys:

      scheme: leave_one_embryo_out | within_embryo_kfold
      n_splits: int, only for within_embryo_kfold
    """
    scheme = cfg.cv.get("scheme", "leave_one_embryo_out")
    if scheme == "leave_one_embryo_out":
        return leave_one_embryo_out(samples)
    if scheme == "within_embryo_kfold":
        return within_embryo_kfold(
            samples, int(cfg.cv.get("n_splits", 5)), int(cfg.seed)
        )
    raise ValueError(f"unknown cv scheme: {scheme!r}")


def check_no_embryo_leak(folds: list[Fold]) -> None:
    """Fail loudly if any embryo appears on both sides of a fold.

    A group split that leaks is silent otherwise, and this is the leak that would
    invalidate every number in the ledger.
    """
    for f in folds:
        train_emb = {embryo_of(s) for s in f.train}
        valid_emb = {embryo_of(s) for s in f.valid}
        overlap = train_emb & valid_emb
        if overlap:
            raise AssertionError(
                f"fold {f.index} leaks embryos across train/valid: {sorted(overlap)}"
            )


def bootstrap_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Weighted mean with a percentile bootstrap interval, resampling samples.

    The competition aggregates a per-sample adjusted Jaccard weighted by that
    sample's `TP + FP + FN`, so this mirrors the scorer. Resampling is over whole
    samples, which is the unit that would differ if a different set of crops had
    been drawn from the same embryo.

    Returns (point estimate, lower bound, upper bound).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(np.average(values, weights=weights))
    if values.size == 1:
        return point, point, point

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_v = values[idx]
    boot_w = weights[idx]
    boot = (boot_v * boot_w).sum(axis=1) / boot_w.sum(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def describe(folds: list[Fold]) -> str:
    check_no_embryo_leak(folds)
    lines = [f"folds={len(folds)}"]
    for f in folds:
        lines.append(
            f"  fold {f.index}: held out {f.held_out_embryo}  "
            f"train={len(f.train)} valid={len(f.valid)}"
        )
    lines.append("  embryo overlap across folds: 0 (checked)")
    return "\n".join(lines)
