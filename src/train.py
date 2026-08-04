"""Train one config across the CV folds, write OOF predictions and a submission,
and append one row to the experiment ledger.

    python -m src.train --config conf/baseline.yaml

Deliberately boring. The point is that every number in experiments.csv came from
this one path, so two rows are actually comparable.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import cv as cv_mod
from src import data as data_mod
from src import ledger, metrics
from src.config import load_config


def fit_predict_fold(X_tr, y_tr, X_va, y_va, X_te, cfg):
    """Fit one fold, return (validation preds, test preds).

    y_va is used only for early stopping. That makes the CV score very slightly
    optimistic, which is the standard trade and fine as long as every config in
    the ledger pays the same cost. If a competition is tight enough for that to
    matter, switch to a fixed n_estimators and drop early stopping entirely.

    Swap the model here and leave the fold loop, OOF handling and ledger alone.
    """
    kind = cfg.model.get("kind", "lightgbm")
    task = cfg.model.get("task", "binary")  # binary | regression

    if kind == "mean":
        # The anchor baseline. Submit this first, before any modelling, so every
        # later number has something honest to be compared against.
        const = float(np.mean(y_tr))
        return np.full(len(X_va), const), np.full(len(X_te), const)

    if kind == "lightgbm":
        import lightgbm as lgb

        params = dict(cfg.model.get("params", {}))
        params.setdefault("random_state", cfg.seed)
        params.setdefault("verbose", -1)
        rounds = int(cfg.model.get("num_boost_round", 2000))
        early = int(cfg.model.get("early_stopping_rounds", 100))

        Est = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
        model = Est(n_estimators=rounds, **params)
        # LightGBM 4.7 renamed eval_set to eval_X/eval_y. Kaggle notebooks often
        # run an older build than this machine, so pick whichever fit() accepts
        # rather than pinning a version the remote environment may not have.
        import inspect

        sig = inspect.signature(model.fit).parameters
        eval_kw = (
            {"eval_X": X_va, "eval_y": y_va}
            if "eval_X" in sig
            else {"eval_set": [(X_va, y_va)]}
        )
        model.fit(
            X_tr,
            y_tr,
            callbacks=[lgb.early_stopping(early, verbose=False)],
            **eval_kw,
        )
        if task == "regression":
            return model.predict(X_va), model.predict(X_te)
        return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]

    raise ValueError(f"unknown model kind {kind!r}")


def run(config_path: str, notes: str = "") -> int:
    cfg = load_config(config_path)
    train, test, sample = data_mod.load_raw(cfg)
    train, test, features = data_mod.build_features(train, test, cfg)

    folds = cv_mod.make_folds(train, cfg)
    print(cv_mod.describe(folds, train, cfg))

    y = train[cfg.target].values
    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    n_folds = int(folds.max()) + 1
    fold_scores = []

    for f in range(n_folds):
        tr, va = folds != f, folds == f
        p_va, p_te = fit_predict_fold(
            train.loc[tr, features],
            y[tr],
            train.loc[va, features],
            y[va],
            test[features],
            cfg,
        )
        oof[va] = p_va
        test_pred += p_te / n_folds
        s = metrics.score(cfg.metric, y[va], p_va)
        fold_scores.append(s)
        print(f"  fold {f}: {cfg.metric}={s:.6f}")

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    # The pooled OOF score and the mean-of-folds disagree when folds are uneven
    # or the metric is not decomposable (AUC is not). Report both; compare like
    # with like across experiments.
    pooled = metrics.score(cfg.metric, y, oof)
    print(f"\nCV {cfg.metric}: {cv_mean:.6f} +/- {cv_std:.6f}   (pooled OOF: {pooled:.6f})")
    if cv_std > 0 and cv_std * 0.5 > abs(cv_mean) * 0.02:
        print("note: fold spread is wide relative to the mean. Small gains here are noise")

    tag = f"{cfg.name}_{cfg.hash()}"
    np.save(cfg.oof_dir / f"{tag}.npy", oof)

    sub = sample.copy() if len(sample) else pd.DataFrame({cfg.id_col: test[cfg.id_col]})
    sub_target = cfg.raw.get("submission_col", cfg.target)
    sub[sub_target] = test_pred
    sub_path = cfg.sub_dir / f"{tag}.csv"
    sub.to_csv(sub_path, index=False)

    exp_id = ledger.append(
        name=cfg.name,
        config=str(cfg.path),
        config_hash=cfg.hash(),
        cv_mean=cv_mean,
        cv_std=cv_std,
        folds=n_folds,
        notes=notes,
    )
    print(f"\nlogged as experiment {exp_id}")
    print(f"submission: {sub_path}")
    print(f"submit with: python -m src.submit --id {exp_id} --file {sub_path.name}")
    return exp_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    run(a.config, a.notes)
