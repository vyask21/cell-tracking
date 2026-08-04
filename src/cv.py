"""Cross-validation schemes.

The scheme has to mirror how the organizers split train from test. Getting this
wrong is the single most expensive mistake available in a competition, so the
choice is explicit here and justified in NOTES.md rather than defaulted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


def make_folds(df: pd.DataFrame, cfg) -> np.ndarray:
    """Return an integer fold assignment per row.

    cv config keys:
      scheme: kfold | stratified | group | stratified_group | time
      n_splits: int
      group_col: column name, required for group schemes
      time_col: column name, required for the time scheme
    """
    c = cfg.cv
    scheme = c.get("scheme", "kfold")
    n_splits = int(c.get("n_splits", 5))
    seed = cfg.seed
    y = df[cfg.target].values if cfg.target in df.columns else None
    folds = np.full(len(df), -1, dtype=int)

    if scheme == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(df)

    elif scheme == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(df, y)

    elif scheme == "group":
        groups = df[c["group_col"]]
        splitter = GroupKFold(n_splits=n_splits)
        iterator = splitter.split(df, y, groups)

    elif scheme == "stratified_group":
        groups = df[c["group_col"]]
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        iterator = splitter.split(df, y, groups)

    elif scheme == "time":
        # Rows must already be sorted by time. Later folds train on more history,
        # which is the only honest way to validate when test is in the future.
        df_sorted = df.sort_values(c["time_col"])
        order = df_sorted.index.to_numpy()
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for i, (_, va) in enumerate(splitter.split(order)):
            folds[df.index.get_indexer(order[va])] = i
        return folds

    else:
        raise ValueError(f"unknown cv scheme: {scheme!r}")

    for i, (_, va) in enumerate(iterator):
        folds[va] = i
    return folds


def describe(folds: np.ndarray, df: pd.DataFrame, cfg) -> str:
    lines = [f"scheme={cfg.cv.get('scheme')} n_splits={folds.max() + 1}"]
    for i in range(folds.max() + 1):
        mask = folds == i
        line = f"  fold {i}: n={mask.sum()}"
        if cfg.target in df.columns and df[cfg.target].nunique() < 20:
            line += f" pos_rate={df.loc[mask, cfg.target].mean():.4f}"
        if "group_col" in cfg.cv:
            line += f" groups={df.loc[mask, cfg.cv['group_col']].nunique()}"
        lines.append(line)
    # A group scheme that leaks is silent otherwise, so assert it loudly here.
    if "group_col" in cfg.cv:
        gcol = cfg.cv["group_col"]
        seen: dict = {}
        overlap = 0
        for i in range(folds.max() + 1):
            for g in df.loc[folds == i, gcol].unique():
                if g in seen and seen[g] != i:
                    overlap += 1
                seen[g] = i
        lines.append(f"  group overlap across folds: {overlap} (must be 0)")
    return "\n".join(lines)
