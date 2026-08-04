"""Competition metrics.

The metric drives the loss, the CV scheme, and whether predictions need
calibrating. Read the evaluation page and set `metric:` in the config to match
exactly. Optimizing a proxy is a silent, expensive error.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

# Whether a higher score is better. Used to sort the ledger and to decide the
# direction of early stopping.
DIRECTION = {
    "auc": "max",
    "accuracy": "max",
    "f1": "max",
    "logloss": "min",
    "rmse": "min",
    "mae": "min",
    "rmsle": "min",
}


def score(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if name == "auc":
        return skm.roc_auc_score(y_true, y_pred)
    if name == "logloss":
        return skm.log_loss(y_true, y_pred)
    if name == "accuracy":
        return skm.accuracy_score(y_true, (y_pred > 0.5).astype(int))
    if name == "f1":
        return skm.f1_score(y_true, (y_pred > 0.5).astype(int))
    if name == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    if name == "mae":
        return skm.mean_absolute_error(y_true, y_pred)
    if name == "rmsle":
        return float(
            np.sqrt(skm.mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0))))
        )
    raise ValueError(f"unknown metric {name!r}. Add it here rather than approximating")


def is_higher_better(name: str) -> bool:
    if name not in DIRECTION:
        raise ValueError(f"unknown metric {name!r}")
    return DIRECTION[name] == "max"
