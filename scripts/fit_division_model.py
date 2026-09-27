"""Fit the learned division model for bet A and choose its threshold.

    python scripts/fit_division_model.py

Inputs are the outputs of cell-tracking-plateau-divcapture1 and 2, downloaded to
artifacts/divcapture/: every division candidate the exp 25 pipeline saw under
loose gates on 119 training videos, with features and ground-truth labels, and
the list of true divisions per video.

Only candidates whose parent matches an annotated cell with an annotated
successor are used, since only those can be scored. The label is y_strict: the
parent is a true division and both the existing child and the candidate are its
two true daughters.

Evaluation replays the whole decision, not a per-candidate AUC. For each video,
candidates are taken in descending score, a parent and a daughter are each used
at most once, and anything under the threshold is dropped. True positives are
accepted candidates with y_strict; false positives are the other accepted ones;
false negatives are true divisions not recovered. The cascade that ships today is
scored the same way from its own accepted flag. The 100 in-sample videos give
out-of-fold predictions by grouped 5-fold; the held-out 19 are scored by a model
trained on all 100, as the check that the in-sample fit transfers to videos the
U-Net never saw. Writes artifacts/divmodel/divmodel.json with the model text, its
dumped trees for a dependency-free fallback, the features and the threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent
IN_DIR = REPO / "artifacts" / "divcapture"
OUT = REPO / "artifacts" / "divmodel" / "divmodel.json"
FEATURES = [
    "child_dist", "parent_dist", "sister_dist", "rank", "n_cands", "mutual_nn", "diverge",
    "dc_cand", "dc_child", "dc_source", "p_child", "p_cand", "fwd_cand", "fwd_child",
    "back_source", "dens_t", "dens_t1", "cos_angle", "mid_dist",
]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=0)
ROUNDS = 300


def select(df: pd.DataFrame, score: np.ndarray, tau: float) -> pd.DataFrame:
    """Greedy one-per-parent, one-per-daughter acceptance, as the kernel does it."""
    d = df.assign(score=score)
    d = d[d.score >= tau].sort_values("score", ascending=False)
    used_s, used_q, keep = set(), set(), []
    for r in d.itertuples():
        a, b = (r.dataset, r.source_id), (r.dataset, r.cand_id)
        if a in used_s or b in used_q:
            continue
        used_s.add(a)
        used_q.add(b)
        keep.append(r.Index)
    return df.loc[keep]


def division_jaccard(accepted: pd.DataFrame, n_true: int) -> tuple[float, int, int, int]:
    tp = int(accepted.y_strict.sum())
    fp = int(len(accepted) - tp)
    fn = max(0, n_true - tp)
    return tp / max(tp + fp + fn, 1), tp, fp, fn


def main() -> int:
    cand = pd.concat([pd.read_csv(p) for p in sorted(IN_DIR.glob("divcand_part*.csv"))], ignore_index=True)
    gt = pd.concat([pd.read_csv(p) for p in sorted(IN_DIR.glob("gtdiv_part*.csv"))], ignore_index=True)
    print(f"candidates {len(cand)}, videos {cand.dataset.nunique()}, true divisions {len(gt)}")
    ann = cand[cand.annotated == 1].reset_index(drop=True)
    print(f"annotated candidates {len(ann)}, strict positives {int(ann.y_strict.sum())}, "
          f"cascade accepted {int(ann.accepted.sum())}")
    held = ann[ann.held_out == 1]
    ins = ann[ann.held_out == 0].reset_index(drop=True)
    n_true_in = int(gt[~gt.dataset.isin(held.dataset.unique())].shape[0])
    n_true_held = int(gt[gt.dataset.isin(held.dataset.unique())].shape[0])

    oof = np.zeros(len(ins))
    for tr, te in GroupKFold(n_splits=5).split(ins, groups=ins.dataset):
        m = lgb.train(PARAMS, lgb.Dataset(ins.loc[tr, FEATURES], ins.loc[tr, "y_strict"]), ROUNDS)
        oof[te] = m.predict(ins.loc[te, FEATURES])

    print("\nin-sample videos, grouped 5-fold out of fold")
    j, tp, fp, fn = division_jaccard(ins[ins.accepted == 1], n_true_in)
    print(f"  cascade today          divJ {j:.4f}  tp {tp} fp {fp} fn {fn}")
    best_tau, best_j = None, -1.0
    for tau in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6):
        j, tp, fp, fn = division_jaccard(select(ins, oof, tau), n_true_in)
        print(f"  model tau {tau:4.2f}         divJ {j:.4f}  tp {tp} fp {fp} fn {fn}")
        if j > best_j:
            best_tau, best_j = tau, j

    final = lgb.train(PARAMS, lgb.Dataset(ins[FEATURES], ins["y_strict"]), ROUNDS)
    ph = final.predict(held[FEATURES])
    print("\nheld-out 19, model trained on the 100 in-sample videos")
    j, tp, fp, fn = division_jaccard(held[held.accepted == 1], n_true_held)
    print(f"  cascade today          divJ {j:.4f}  tp {tp} fp {fp} fn {fn}")
    for tau in sorted({best_tau, 0.2, 0.3, 0.4}):
        j, tp, fp, fn = division_jaccard(select(held, ph, tau), n_true_held)
        print(f"  model tau {tau:4.2f}{' (chosen)' if tau == best_tau else '         '} divJ {j:.4f}  tp {tp} fp {fp} fn {fn}")

    imp = sorted(zip(FEATURES, final.feature_importance("gain")), key=lambda x: -x[1])
    print("\nfeature gain:", ", ".join(f"{k} {v:.0f}" for k, v in imp[:10]))
    probe = held[FEATURES].head(8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "features": FEATURES, "tau": best_tau, "model_str": final.model_to_string(),
        "dump": final.dump_model(), "probe_x": probe.to_numpy().tolist(),
        "probe_p": final.predict(probe).tolist(),
    }))
    print(f"\nchosen tau {best_tau}; wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
