"""Build the wide coordinate-head capture: 59 videos instead of 19.

    python scripts/build_headfit_wide_kernel.py

Exp 19 shipped a linear head fitted on the support pack's held-out 19 and it was
worth about +0.002 on the board, against about +0.007 for the source's private
head. Three things differ between the two heads: model class, whether the fitting
videos were seen by the U-Net, and volume. The thinnest part of the exp 19 fit was
44b6, 5 videos and 1,346 pairs, and it barely improved.

This capture adds 40 videos from the pack's training 180, drawn with a fixed seed
and weighted to 44b6, 25 against 15, so the set is 30 and 29 by embryo. The four
videos that also sit in the example test folder are excluded.

Evaluation stays honest by construction: every head is scored only on the 19
held-out videos, leaving one out at a time, because the hidden test is unseen by
the U-Net and only those 19 match that. The 40 in-sample videos are training data
only, never test data.

In the kernel: capture, match to truth, save the pairs, then fit MLP heads in two
configurations on GPU, report the held-out table for each, and save each head
fitted on all 59. Linear heads are fitted from the saved pairs by
scripts/fit_coord_head_linear.py, so both model classes meet the same test.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("headfit", REPO / "scripts" / "build_headfit_kernel.py")
headfit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(headfit)

OUT_DIR = REPO / "notebooks" / "plateau_headfit_wide"
KERNEL_ID = "vyask21/cell-tracking-plateau-headfit-wide"
SEED = 20260923
EXAMPLE_TEST = {"44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"}


def pick_in_sample() -> list[str]:
    split = json.loads((REPO / "data" / "meta" / "dataset_splits.json").read_text(encoding="utf-8"))[0]
    stems = sorted(split["train"] + split["test"])
    if len(stems) != 199:
        raise SystemExit(f"expected 199 train videos, found {len(stems)}")
    pool = [s for s in stems if s not in set(headfit.HELD_OUT_19) and s not in EXAMPLE_TEST]
    rng = random.Random(SEED)
    a = sorted(rng.sample([s for s in pool if s.startswith("44b6")], 25))
    b = sorted(rng.sample([s for s in pool if s.startswith("6bba")], 15))
    return a + b


FIT_CELL = r'''
# Fit coordinate heads on the wide capture. Held-out 19 are the only test videos.
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

HELD_OUT = __HELD_OUT__
CAPTURE = Path(os.environ["V1284_CAPTURE"])
GT_DIR = COMP_DIR / "train"
GRID_UM = np.array([1.625, 1.625, 1.625])
VOX_UM = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 4.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIGS = {"mlp2000": (2000, 3e-3, 1e-3), "mlp500": (500, 3e-3, 1e-3)}

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
    print(f"  {stem}: {n_pairs} detection-to-truth pairs{'  (held out)' if stem in HELD_OUT else ''}")

X = np.concatenate(X_parts); Y = np.concatenate(Y_parts); G = np.concatenate(G_parts)
print(f"pairs {len(X)}, of which held-out {int(np.isin(G, HELD_OUT).sum())}")
np.savez_compressed("/kaggle/working/coord_pairs_wide.npz", X=X, Y=Y, G=G)
for entry in Path("/kaggle/working").iterdir():
    if entry.name != "coord_pairs_wide.npz":
        shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)


def fit(Xtr, Ytr, steps, lr, wd):
    mean = Xtr.mean(0)
    scale = Xtr.std(0) + 1e-6
    torch.manual_seed(0)
    head = coord_module.make_head().to(DEVICE)
    xt = torch.from_numpy((Xtr - mean) / scale).to(DEVICE)
    yt = torch.from_numpy(Ytr).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((coord_module.bounded(head, xt) - yt) ** 2).sum(1).mean()
        loss.backward()
        opt.step()
    return head.cpu(), mean, scale


def residual(head, mean, scale, Xte, Yte):
    with torch.no_grad():
        shift = coord_module.bounded(head, torch.from_numpy((Xte - mean) / scale)).numpy()
    return np.linalg.norm(Yte - shift, axis=1)


report = {"pairs": int(len(X)), "held_out": HELD_OUT, "videos": list(test_stems), "configs": {}}
for name, (steps, lr, wd) in CONFIGS.items():
    rows = {}
    for stem in HELD_OUT:
        te = G == stem
        head, mean, scale = fit(X[~te], Y[~te], steps, lr, wd)
        rows[stem] = (float(np.linalg.norm(Y[te], axis=1).mean()),
                      float(residual(head, mean, scale, X[te], Y[te]).mean()), int(te.sum()))
    line = []
    for e in ("", "44b6", "6bba"):
        r = [v for k, v in rows.items() if k.startswith(e)]
        b = sum(v[0] * v[2] for v in r)
        a = sum(v[1] * v[2] for v in r)
        line.append(f"{e or 'pooled'} {100 * (a / b - 1):+.1f}% ({sum(v[1] < v[0] for v in r)}/{len(r)})")
    print(f"{name}: " + " | ".join(line))
    report["configs"][name] = {"steps": steps, "lr": lr, "weight_decay": wd, "lovo": rows}
    head, mean, scale = fit(X, Y, steps, lr, wd)
    torch.save({"state_dict": head.state_dict(),
                "mean": torch.from_numpy(mean.astype(np.float32)),
                "scale": torch.from_numpy(scale.astype(np.float32))},
               f"/kaggle/working/coord_head_{name}.pt")
Path("/kaggle/working/coord_head_wide_report.json").write_text(json.dumps(report, indent=2))
print("saved heads:", sorted(p.name for p in Path("/kaggle/working").glob("coord_head_*.pt")))
'''

HEADER = """# Coordinate head fit, wide capture

Built by `scripts/build_headfit_wide_kernel.py`. Captures detector features on 59
training videos, the support pack's held-out 19 plus 40 from its training set,
fits coordinate heads, and scores every head only on the held-out 19, one video
left out at a time. Sources and credit are in `NOTES.md`, 2026-09-23.
"""


def main() -> int:
    stems = list(headfit.HELD_OUT_19) + pick_in_sample()
    if len(stems) != 59 or len(set(stems)) != 59:
        raise SystemExit("expected 59 distinct videos")
    nb = json.loads(headfit.BASE.read_text(encoding="utf-8"))
    cells = nb["cells"][: headfit.PREDICT_CELL + 1]
    patches = [
        headfit.PATCHES[0],
        ("predict the 59", "test_stems = list_test_stems()", f"test_stems = {stems!r}"),
        headfit.PATCHES[2],
    ]
    for label, old, new in patches:
        hits = [i for i, c in enumerate(cells) if old in "".join(c["source"])]
        if sum("".join(cells[i]["source"]).count(old) for i in hits) != 1:
            raise SystemExit(f"patch {label!r} did not match exactly once")
        cells[hits[0]]["source"] = "".join(cells[hits[0]]["source"]).replace(old, new).splitlines(keepends=True)
        print(f"applied: {label}")
    fit_cell = FIT_CELL.replace("__HELD_OUT__", repr(list(headfit.HELD_OUT_19)))
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": fit_cell.lstrip().splitlines(keepends=True)})
    cells[0]["source"] = HEADER.splitlines(keepends=True)
    nb["cells"] = cells
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "headfit_wide.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((REPO / "notebooks" / "plateau_gapstack" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "headfit_wide.ipynb"
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'headfit_wide.ipynb'} with {len(stems)} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
