"""Build the coordinate-head fitting kernel that exp 19 depends on.

    python scripts/build_headfit_kernel.py

The exp 18 stack carries a coordinate-refinement module whose published head sits
in a private dataset. The module is visible and so is how its head was made:
capture frozen U-Net features at each first-seen detection centre, match those
detections to ground truth, and regress the displacement with a 224-32-3 MLP whose
output is bounded to 2 um. This kernel does that on our side.

What differs from the published recipe, on purpose: the capture runs on the 19
videos the support pack held out of its 90/10 split, listed below as HELD_OUT_19. Those are out of sample for the U-Net whose features the head reads,
so the fit does not learn the network's behaviour on its own training data. That
is the defect that cost exp 13.

Built from notebooks/plateau_gapstack/gapstack.ipynb, which is already sanitised.
Cells up to and including prediction are kept, pointed at the 19 in capture mode,
and everything after prediction is replaced by one fitting cell. The fit reports
leave-one-video-out distance to ground truth before and after the head, with the
feature normalisation fitted inside each fold, and saves the head trained on all
19 as coord_head.pt. Nothing here touches a submission.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_gapstack" / "gapstack.ipynb"
OUT_DIR = REPO / "notebooks" / "plateau_headfit"
KERNEL_ID = "vyask21/cell-tracking-plateau-headfit"
PREDICT_CELL = 5  # the cell whose last line prints "Prediction completed"

HELD_OUT_19 = [
    "44b6_1574802b", "44b6_706092f0", "44b6_d5e7d891", "44b6_d754aa59", "44b6_e57ff5c6",
    "6bba_2312ac41", "6bba_268e1230", "6bba_283bf9f1", "6bba_3a1849c2", "6bba_3abfe10a",
    "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde", "6bba_7b5d3b2c", "6bba_aeee7805",
    "6bba_afb141ff", "6bba_c27cba08", "6bba_c328f2fd", "6bba_d1acb6ff",
]

PATCHES: list[tuple[str, str, str]] = [
    ("predict on train, where ground truth exists",
     'TEST_DIR = COMP_DIR / "test"',
     'TEST_DIR = COMP_DIR / "train"'),
    ("predict only the held-out 19",
     "test_stems = list_test_stems()",
     f"test_stems = {HELD_OUT_19!r}"),
    ("capture features instead of refining",
     "os.environ['V1284_MODE']='zero'\n",
     "os.environ['V1284_MODE']='capture'\n"
     "os.environ['V1284_CAPTURE']='/kaggle/working/coord_capture'\n"),
]

FIT_CELL = r'''
# Fit the coordinate head on the captured features. Stand-alone from here on.
import importlib.util
import shutil

import numpy as np
import torch
import tracksdata as td
from scipy.optimize import linear_sum_assignment

_spec = importlib.util.spec_from_file_location(
    "coord_module", str(_ps.parent / "v1284_coordinate_refinement.py"))
coord_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coord_module)

CAPTURE = Path(os.environ["V1284_CAPTURE"])
GT_DIR = COMP_DIR / "train"
GRID_UM = np.array([1.625, 1.625, 1.625])
VOX_UM = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 4.0
STEPS, LR, WD, SEED = 2000, 3e-3, 1e-3, 0

X_parts, Y_parts, G_parts = [], [], []
for stem in test_stems:
    graph = td.graph.IndexedRXGraph.from_geff(GT_DIR / f"{stem}.geff")
    graph = graph[0] if isinstance(graph, tuple) else graph
    gt_by_t = {}
    for row in graph.node_attrs().iter_rows(named=True):
        gt_by_t.setdefault(int(row["t"]), []).append((float(row["z"]), float(row["y"]), float(row["x"])))
    n_pairs = 0
    for npz in sorted((CAPTURE / stem).glob("*.npz")):
        t = int(npz.stem)
        if t not in gt_by_t:
            continue
        with np.load(npz) as d:
            coords, feats = d["coords"], d["features"]
        if len(coords) == 0:
            continue
        det_um = coords[:, 1:].astype(np.float64) * GRID_UM
        gt_um = np.asarray(gt_by_t[t]) * VOX_UM
        cost = np.linalg.norm(det_um[:, None, :] - gt_um[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(np.where(cost <= MATCH_UM, cost, 1e6))
        keep = cost[rows, cols] <= MATCH_UM
        rows, cols = rows[keep], cols[keep]
        X_parts.append(feats[rows].astype(np.float32))
        Y_parts.append((gt_um[cols] - det_um[rows]).astype(np.float32))
        G_parts.append(np.full(len(rows), stem))
        n_pairs += len(rows)
    print(f"  {stem}: {n_pairs} detection-to-truth pairs")

X = np.concatenate(X_parts); Y = np.concatenate(Y_parts); G = np.concatenate(G_parts)
print(f"pairs {len(X)}, feature dim {X.shape[1]}, mean residual {np.linalg.norm(Y, axis=1).mean():.4f} um")

# Save the pairs and drop everything else before training, so a failure below
# still leaves a small output that a CPU kernel can refit from.
np.savez_compressed("/kaggle/working/coord_pairs.npz", X=X, Y=Y, G=G)
KEEP = {"coord_pairs.npz", "coord_head.pt", "coord_head_report.json"}
for entry in Path("/kaggle/working").iterdir():
    if entry.name not in KEEP:
        shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)


def fit(Xtr, Ytr):
    mean = Xtr.mean(0)
    scale = Xtr.std(0) + 1e-6
    torch.manual_seed(SEED)
    head = coord_module.make_head()
    xt = torch.from_numpy((Xtr - mean) / scale)
    yt = torch.from_numpy(Ytr)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = ((coord_module.bounded(head, xt) - yt) ** 2).sum(1).mean()
        loss.backward()
        opt.step()
    return head, mean, scale


def residual(head, mean, scale, Xte, Yte):
    with torch.no_grad():
        shift = coord_module.bounded(head, torch.from_numpy((Xte - mean) / scale)).numpy()
    return np.linalg.norm(Yte - shift, axis=1)


rows = []
for stem in test_stems:
    te = G == stem
    if te.sum() == 0:
        continue
    head, mean, scale = fit(X[~te], Y[~te])
    before = np.linalg.norm(Y[te], axis=1)
    after = residual(head, mean, scale, X[te], Y[te])
    rows.append((stem, int(te.sum()), float(before.mean()), float(after.mean())))
    print(f"  LOVO {stem}: n={te.sum():4d}  before {before.mean():.4f}  after {after.mean():.4f}  "
          f"change {100 * (after.mean() / before.mean() - 1):+.1f}%")

n = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows]); a = np.array([r[3] for r in rows])
pooled = 100 * ((a * n).sum() / (b * n).sum() - 1)
print(f"LOVO pooled change {pooled:+.2f}%   videos improved {(a < b).sum()} of {len(rows)}")

head, mean, scale = fit(X, Y)
torch.save({"state_dict": head.state_dict(),
            "mean": torch.from_numpy(mean.astype(np.float32)),
            "scale": torch.from_numpy(scale.astype(np.float32))},
           "/kaggle/working/coord_head.pt")
Path("/kaggle/working/coord_head_report.json").write_text(json.dumps({
    "videos": test_stems, "pairs": int(len(X)), "match_um": MATCH_UM,
    "steps": STEPS, "lr": LR, "weight_decay": WD, "seed": SEED,
    "lovo_pooled_change_pct": pooled, "videos_improved": int((a < b).sum()),
    "lovo": [{"stem": s, "n": k, "before_um": bb, "after_um": aa} for s, k, bb, aa in rows],
}, indent=2))
print("saved coord_head.pt")
'''

HEADER = """# Coordinate head fit

Built by `scripts/build_headfit_kernel.py`. Captures detector features on the 19
videos the support pack held out, fits the coordinate head, reports
leave-one-video-out distance to ground truth, and saves `coord_head.pt` for the
exp 19 submission kernel.
"""


def main() -> int:
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    tail = "".join(cells[PREDICT_CELL]["source"]).rstrip().splitlines()[-1]
    if "Prediction completed" not in tail:
        raise SystemExit(f"cell {PREDICT_CELL} does not end at prediction: {tail!r}")
    cells = cells[: PREDICT_CELL + 1]

    for label, old, new in PATCHES:
        hits = [i for i, c in enumerate(cells) if old in "".join(c["source"])]
        total = sum("".join(cells[i]["source"]).count(old) for i in hits)
        if total != 1:
            raise SystemExit(f"patch {label!r}: matched {total}, expected 1")
        src = "".join(cells[hits[0]]["source"]).replace(old, new)
        cells[hits[0]]["source"] = src.splitlines(keepends=True)
        print(f"applied: {label}")

    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": FIT_CELL.lstrip().splitlines(keepends=True)})
    cells[0]["source"] = HEADER.splitlines(keepends=True)
    nb["cells"] = cells
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "headfit.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((REPO / "notebooks" / "plateau_gapstack" / "kernel-metadata.json")
                      .read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "headfit.ipynb"
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'headfit.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
