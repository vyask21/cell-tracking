"""Build the head-ensemble training kernel for bet B.

    python scripts/build_headtrain_kernel.py

Exp 25 blended the published head 50/50 with one MLP head fitted on 59 videos and
scored 0.956, against 0.953 for the published head alone and 0.950 for ours
alone. Two heads with different errors beat either. This kernel makes our half
of the blend stronger in the two ways that argument points to: more videos, and
several seeds averaged instead of one.

Inputs are the detection-to-truth pairs already on Kaggle: the 59-video wide
capture and the two combined capture parts, about 160 training videos in all.
Five MLPs, same architecture and bound as the refinement module, 2000 steps as
exp 20's head, seeds 0 to 4, each on every video. Held-out evaluation leaves out
one of the support pack's held-out 19 at a time and scores only those, for one
seed and for the five-seed mean, so the table says whether averaging helps on
data the U-Net never saw. Output: coord_head_s0.pt to coord_head_s4.pt in the
module's format, and a report.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "notebooks" / "plateau_headtrain"
KERNEL_ID = "vyask21/cell-tracking-plateau-headtrain"
SOURCES = [
    "vyask21/cell-tracking-plateau-headfit-wide",
    "vyask21/cell-tracking-plateau-divcapture1",
    "vyask21/cell-tracking-plateau-divcapture2",
]

CODE = r'''
import json
from pathlib import Path

import numpy as np
import torch

HELD_OUT = __HELD_OUT__
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, LR, WD, SEEDS = 2000, 3e-3, 1e-3, (0, 1, 2, 3, 4)


def make_head():
    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def bounded(head, x):
    delta = head(x)
    return 2.0 * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))


files = sorted(p for pat in ("coord_pairs_wide.npz", "coord_pairs_part*.npz")
               for p in Path("/kaggle/input").rglob(pat))
print("pair files:", [str(p) for p in files])
if len(files) != 3:
    raise RuntimeError(("expected three pair files", [str(p) for p in files]))
parts = [np.load(p) for p in files]
X = np.concatenate([d["X"] for d in parts]).astype(np.float32)
Y = np.concatenate([d["Y"] for d in parts]).astype(np.float32)
G = np.concatenate([d["G"] for d in parts])
# The held-out 19 appear in both the wide capture and part 1; keep one copy.
keep = np.ones(len(G), bool)
print(f"pairs {len(X)} over {len(np.unique(G))} videos before de-duplication")
seen = {}
for i, (p, d) in enumerate(zip(files, parts)):
    for stem in np.unique(d["G"]):
        seen.setdefault(stem, i)
offset = 0
for i, d in enumerate(parts):
    g = d["G"]
    owner = np.array([seen[s] for s in g])
    keep[offset:offset + len(g)] = owner == i
    offset += len(g)
X, Y, G = X[keep], Y[keep], G[keep]
print(f"pairs {len(X)} over {len(np.unique(G))} videos after keeping one copy per video")


def fit(Xtr, Ytr, seed):
    mean = Xtr.mean(0)
    scale = Xtr.std(0) + 1e-6
    torch.manual_seed(seed)
    head = make_head().to(DEVICE)
    xt = torch.from_numpy((Xtr - mean) / scale).to(DEVICE)
    yt = torch.from_numpy(Ytr).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = ((bounded(head, xt) - yt) ** 2).sum(1).mean()
        loss.backward()
        opt.step()
    return head.cpu().eval(), mean, scale


def shift(fitted, Xte):
    head, mean, scale = fitted
    with torch.no_grad():
        return bounded(head, torch.from_numpy((Xte - mean) / scale)).numpy()


rows = {"single": {}, "mean5": {}}
for stem in HELD_OUT:
    te = G == stem
    if not te.any():
        continue
    fits = [fit(X[~te], Y[~te], s) for s in SEEDS]
    before = float(np.linalg.norm(Y[te], axis=1).mean())
    single = float(np.linalg.norm(Y[te] - shift(fits[0], X[te]), axis=1).mean())
    mean5 = float(np.linalg.norm(Y[te] - np.mean([shift(f, X[te]) for f in fits], axis=0), axis=1).mean())
    rows["single"][stem] = (before, single, int(te.sum()))
    rows["mean5"][stem] = (before, mean5, int(te.sum()))
for name, r in rows.items():
    line = []
    for e in ("", "44b6", "6bba"):
        v = [x for k, x in r.items() if k.startswith(e)]
        b = sum(x[0] * x[2] for x in v)
        a = sum(x[1] * x[2] for x in v)
        line.append(f"{e or 'pooled'} {100 * (a / b - 1):+.1f}% ({sum(x[1] < x[0] for x in v)}/{len(v)})")
    print(f"{name}: " + " | ".join(line))

for s in SEEDS:
    head, mean, scale = fit(X, Y, s)
    torch.save({"state_dict": head.state_dict(), "mean": torch.from_numpy(mean.astype(np.float32)),
                "scale": torch.from_numpy(scale.astype(np.float32))}, f"/kaggle/working/coord_head_s{s}.pt")
Path("/kaggle/working/coord_headtrain_report.json").write_text(json.dumps(
    {"pairs": int(len(X)), "videos": int(len(np.unique(G))), "steps": STEPS, "seeds": list(SEEDS), "lovo": rows}, indent=2))
print("saved", sorted(p.name for p in Path("/kaggle/working").glob("coord_head_s*.pt")))
'''


def main() -> int:
    spec = importlib.util.spec_from_file_location("headfit", REPO / "scripts" / "build_headfit_kernel.py")
    headfit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(headfit)
    code = CODE.replace("__HELD_OUT__", repr(list(headfit.HELD_OUT_19)))
    compile(code, "headtrain", "exec")
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Coordinate head ensemble training\n", "\n",
                "Built by `scripts/build_headtrain_kernel.py`. Sources and credit are in `NOTES.md`.\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
             "source": code.lstrip().splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "headtrain.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": KERNEL_ID, "title": KERNEL_ID.split("/")[1], "code_file": "headtrain.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [], "kernel_sources": SOURCES,
        "competition_sources": ["biohub-cell-tracking-during-development"], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'headtrain.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
