"""Data loading and the feature hook.

Feature engineering that uses the target (target encoding, out-of-fold counts,
anything aggregating y) does NOT belong here. It belongs inside the fold loop
in train.py, fit on the training fold only. Putting it here is the most common
way a competition CV silently becomes fiction.
"""

from __future__ import annotations

import pandas as pd


def load_raw(cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = cfg.data_dir
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    sub_path = d / "sample_submission.csv"
    sample = pd.read_csv(sub_path) if sub_path.exists() else pd.DataFrame()
    return train, test, sample


def build_features(
    train: pd.DataFrame, test: pd.DataFrame, cfg
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Target-independent feature engineering only.

    Safe to fit on train+test together because nothing here touches y.
    Returns (train, test, feature_columns).
    """
    drop = {cfg.target, cfg.id_col} | set(cfg.features.get("drop", []))

    for df in (train, test):
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if col not in drop:
                df[col] = df[col].astype("category")

    # Categories must be aligned across train and test or LightGBM sees different
    # codes for the same string and the test predictions quietly go wrong.
    for col in train.columns:
        if col in drop or col not in test.columns:
            continue
        if str(train[col].dtype) == "category":
            cats = train[col].cat.categories.union(test[col].cat.categories)
            train[col] = train[col].cat.set_categories(cats)
            test[col] = test[col].cat.set_categories(cats)

    features = [c for c in train.columns if c not in drop]
    return train, test, features
