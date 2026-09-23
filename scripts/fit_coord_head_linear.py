"""Fit the linear coordinate head that exp 19 ships.

    python scripts/fit_coord_head_linear.py

Input is artifacts/plateau_headfit/coord_pairs.npz, the detection-to-truth pairs
the cell-tracking-plateau-headfit kernel captured on the support pack's held-out
19 videos: 224 frozen U-Net features per detection and the displacement to its
matched ground truth in um.

Why linear and not the MLP that kernel fitted. Held out by video, the MLP moved
6bba centres 20% closer to truth and every 44b6 video further away, 5 of 5. The
pairs say why: 6bba detections sit about +0.97 um low in z on all 14 of its
videos and 44b6 sits at +0.10, so a flexible model learns the majority embryo's
offset and exports it. A linear map on the same features, fitted through the
module's own bound 2d/(1+|d|), does better on both embryos. The printed table is
the evidence; lambda 10, 100 and 1000 barely differ and 100 is shipped.

The module's head architecture is fixed at Linear-SiLU-Linear. The builder
embeds this linear map in it exactly, by biasing the SiLU far into its identity
regime, so the module itself is untouched.

Normalisation is fitted inside each fold for the held-out table and on all 19
for the shipped head. Writes artifacts/plateau_headfit/coord_head_linear.json.

    python scripts/fit_coord_head_linear.py --wide

reads the 59-video capture instead, artifacts/plateau_headfit_wide/coord_pairs_wide.npz.
Training uses every video; the held-out table leaves out and scores only the
support pack's held-out 19, since the other 40 were seen by the U-Net and the
hidden test was not. Writes coord_head_linear_wide.json beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PAIRS = REPO / "artifacts" / "plateau_headfit" / "coord_pairs.npz"
PAIRS_SHA256 = "eb58db346c10413510907aab7ba9102870ba7e135f2d65d68f9ca008fa5c57d6"
OUT = REPO / "artifacts" / "plateau_headfit" / "coord_head_linear.json"
LAMBDAS = (10.0, 100.0, 1000.0)
HELD_OUT_19 = (
    "44b6_1574802b", "44b6_706092f0", "44b6_d5e7d891", "44b6_d754aa59", "44b6_e57ff5c6",
    "6bba_2312ac41", "6bba_268e1230", "6bba_283bf9f1", "6bba_3a1849c2", "6bba_3abfe10a",
    "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde", "6bba_7b5d3b2c", "6bba_aeee7805",
    "6bba_afb141ff", "6bba_c27cba08", "6bba_c328f2fd", "6bba_d1acb6ff",
)
SHIP = 100.0


def bounded(d: np.ndarray) -> np.ndarray:
    return 2.0 * d / (1.0 + np.linalg.norm(d, axis=1, keepdims=True))


def fit(x: np.ndarray, y: np.ndarray, lam: float, iters: int = 400):
    mean, scale = x.mean(0), x.std(0) + 1e-6
    a = np.c_[(x - mean) / scale, np.ones(len(x))]
    reg = np.eye(a.shape[1]) * lam
    reg[-1, -1] = 0.0
    # For a small displacement the bound is about 2d, so start from ridge / 2.
    w = np.linalg.solve(a.T @ a + 4 * reg, a.T @ y) / 2
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for k in range(1, iters + 1):
        d = a @ w
        n = np.linalg.norm(d, axis=1, keepdims=True)
        err = 2 * d / (1 + n) - y
        g = 2 * err / (1 + n) - 2 * d * (err * d).sum(1, keepdims=True) / ((n + 1e-12) * (1 + n) ** 2)
        grad = a.T @ g * (2 / len(a)) + 2 * (reg @ w) / len(a)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        w -= 0.01 * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
    return mean, scale, w


def predict(mean, scale, w, x):
    return bounded(np.c_[(x - mean) / scale, np.ones(len(x))] @ w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", action="store_true")
    args = ap.parse_args()
    pairs, out = PAIRS, OUT
    if args.wide:
        pairs = REPO / "artifacts" / "plateau_headfit_wide" / "coord_pairs_wide.npz"
        out = pairs.with_name("coord_head_linear_wide.json")
    raw = pairs.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not args.wide and digest != PAIRS_SHA256:
        raise SystemExit("coord_pairs.npz changed")
    d = np.load(pairs)
    x, y, g = d["X"].astype(np.float64), d["Y"].astype(np.float64), d["G"]
    emb = np.array([s[:4] for s in g])
    for e in ("44b6", "6bba"):
        print(f"{e}: {int((emb == e).sum())} pairs, mean offset z,y,x {np.round(y[emb == e].mean(0), 3)} um")

    table = {}
    for lam in LAMBDAS:
        rows = {}
        for stem in (HELD_OUT_19 if args.wide else np.unique(g)):
            te = g == stem
            mean, scale, w = fit(x[~te], y[~te], lam)
            after = np.linalg.norm(y[te] - predict(mean, scale, w, x[te]), axis=1).mean()
            rows[str(stem)] = (float(np.linalg.norm(y[te], axis=1).mean()), float(after), int(te.sum()))
        parts = []
        for e in ("", "44b6", "6bba"):
            r = [v for k, v in rows.items() if k.startswith(e)]
            b = sum(v[0] * v[2] for v in r)
            a = sum(v[1] * v[2] for v in r)
            parts.append(f"{e or 'pooled'} {100 * (a / b - 1):+.1f}% "
                         f"({sum(v[1] < v[0] for v in r)}/{len(r)} improve)")
        print(f"lambda {lam:6.0f}: " + " | ".join(parts))
        table[str(lam)] = rows

    mean, scale, w = fit(x, y, SHIP)
    delta = np.c_[(x - mean) / scale, np.ones(len(x))] @ w
    print(f"shipped lambda {SHIP:.0f}: pre-bound |d| max {np.abs(delta).max():.3f}")
    out.write_text(json.dumps({
        "lambda": SHIP, "pairs_sha256": digest,
        "mean": mean.astype(np.float32).tolist(), "scale": scale.astype(np.float32).tolist(),
        "weight": w[:-1].T.astype(np.float32).tolist(), "bias": w[-1].astype(np.float32).tolist(),
        "max_abs_pre_bound": float(np.abs(delta).max()),
        "lovo": table,
    }))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
