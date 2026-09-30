"""Build the combined capture kernels for the last two bets, run on training videos.

    python scripts/build_divcapture_kernel.py --part 1
    python scripts/build_divcapture_kernel.py --part 2
    python scripts/build_divcapture_kernel.py --part 3
    python scripts/build_divcapture_kernel.py --part 4

Parts 3 and 4, added 2026-09-28 after exp 30 scored 0.957 with a model trained
on 47 true-division examples, cover every remaining training video, 76 in all.

The exp 25 configuration, which scored 0.956, is run over training videos where
ground truth exists, and two things are recorded on the way:

Division candidates, for a learned division model. The published stack decides
divisions with a hand-set cascade: for each track with one successor, an unlinked
detection within fixed distance gates, then a mutual-nearest-neighbour check, a
divergence check, a DeepCenter veto and a symmetry check, and the nearest
survivor is accepted. Its division Jaccard is about 0.12 to 0.23, so the term
worth 0.1 of the score returns about 0.02. Every candidate under looser gates,
parent within 14 um and sister within 20 um, is logged with the signals the
cascade uses as features plus a flag for whether the cascade accepted it. After
the run, each candidate is labelled against ground truth. A candidate is counted
only when its parent matches an annotated cell with an annotated successor, which
is when the scorer can see it.

Head features, for a larger coordinate-head ensemble. The refinement module is
patched to save the features at every detection while still applying the
exp 25 blend, and the detection-to-truth pairs are extracted as in the wide
capture.

Part 1 is the support pack's held-out 19, kept as the validation set for both
models, plus 41 training videos; part 2 is 60 more. None of the 100 in-sample
videos overlaps the wide capture or the example test videos. The cells after the
per-video loop are dropped, since they only check and describe a test submission.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_gapstack_head_blend" / "gapstack_head_blend.ipynb"
LOOP_CELL = 6
SEED = 20260927


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def video_parts() -> tuple[list[str], list[str]]:
    headfit = load("headfit", REPO / "scripts" / "build_headfit_kernel.py")
    wide = load("wide", REPO / "scripts" / "build_headfit_wide_kernel.py")
    split = json.loads((REPO / "data" / "meta" / "dataset_splits.json").read_text(encoding="utf-8"))[0]
    stems = sorted(split["train"] + split["test"])
    taken = set(headfit.HELD_OUT_19) | set(wide.pick_in_sample()) | wide.EXAMPLE_TEST
    pool = [s for s in stems if s not in taken]
    rng = random.Random(SEED)
    picked = sorted(rng.sample(pool, 100))
    return list(headfit.HELD_OUT_19) + picked[:41], picked[41:]


def rest_parts() -> tuple[list[str], list[str]]:
    """Parts 3 and 4: every training video without a division log after parts 1 and 2.

    That is the 36 videos no capture has touched plus the 40 in-sample videos of
    the wide capture, which ran before division logging existed. The four example
    test videos stay excluded.
    """
    wide = load("wide", REPO / "scripts" / "build_headfit_wide_kernel.py")
    part1, part2 = video_parts()
    split = json.loads((REPO / "data" / "meta" / "dataset_splits.json").read_text(encoding="utf-8"))[0]
    stems = sorted(split["train"] + split["test"])
    done = set(part1) | set(part2) | wide.EXAMPLE_TEST
    rest = sorted(s for s in stems if s not in done)
    return rest[: len(rest) // 2], rest[len(rest) // 2:]


CAPTURE_PATCH = r'''_cap_src = _blend_path.read_text()
_cap_old = "    x = sample_features(feature, arr).float()\n"
if _cap_src.count(_cap_old) != 1:
    raise RuntimeError("capture patch anchor mismatch")
_cap_new = _cap_old + (
    "    if os.environ.get('V1284_CAPTURE_ALSO'):\n"
    "        _cf = Path(os.environ['V1284_CAPTURE_ALSO']) / ds_path.stem\n"
    "        _cf.mkdir(parents=True, exist_ok=True)\n"
    "        np.savez_compressed(_cf / f'{int(t):04d}.npz', coords=arr, features=x.cpu().numpy())\n"
)
_cap_src = _cap_src.replace(_cap_old, _cap_new)
compile(_cap_src, str(_blend_path), "exec")
_blend_path.write_text(_cap_src)
os.environ['V1284_CAPTURE_ALSO'] = '/kaggle/working/coord_capture'
os.environ['DIVLOG_DIR'] = '/kaggle/working/divlog'
print('capture-also and division logging enabled')
'''
CAPTURE_ANCHOR = "_head2 = sorted(Path('/kaggle/input').rglob('coord_head_mlp2000.pt'))\n"

DIVLOG_FN = r'''
DIVLOG_COLUMNS = [
    "dataset", "t", "source_id", "child_id", "cand_id",
    "s_z", "s_y", "s_x", "c_z", "c_y", "c_x", "q_z", "q_y", "q_x",
    "child_dist", "parent_dist", "sister_dist", "rank", "n_cands", "mutual_nn",
    "diverge", "dc_cand", "dc_child", "dc_source", "p_child", "p_cand",
    "fwd_cand", "fwd_child", "back_source", "dens_t", "dens_t1", "cos_angle", "mid_dist",
    "accepted",
]


def _divlog_candidates(nodes_by_id, pre_edges, post_edges, dataset, bundle, frame_cache, dc_cache):
    import csv as _dcsv
    out_by_source, in_by_target, incoming = {}, {}, set()
    for e in pre_edges:
        s, d = int(e["source_id"]), int(e["target_id"])
        out_by_source.setdefault(s, []).append(e)
        in_by_target[d] = s
        incoming.add(d)
    accepted = {(int(e["source_id"]), int(e["target_id"])) for e in post_edges if e.get("safe_division")}
    ids_by_t = {}
    for nid, n in nodes_by_id.items():
        ids_by_t.setdefault(int(n["t"]), []).append(nid)
    pos = {nid: _position_um(n) for nid, n in nodes_by_id.items()}

    def fwd_len(nid):
        n, cur = 0, nid
        while n < 10:
            nxt = out_by_source.get(cur, [])
            if len(nxt) != 1:
                break
            cur, n = int(nxt[0]["target_id"]), n + 1
        return n

    def back_len(nid):
        n, cur = 0, nid
        while n < 10 and cur in in_by_target:
            cur, n = in_by_target[cur], n + 1
        return n

    def prob(e):
        try:
            v = float(e.get("edge_prob"))
            return v if np.isfinite(v) else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    rows = []
    for t in sorted(ids_by_t):
        kids = ids_by_t.get(t + 1, [])
        cands = [c for c in kids if c not in incoming]
        if not kids or not cands:
            continue
        cpos = np.stack([pos[c] for c in cands])
        ctree = cKDTree(cpos)
        ktree = cKDTree(np.stack([pos[k] for k in kids]))
        ttree = cKDTree(np.stack([pos[n] for n in ids_by_t[t]]))
        for s in ids_by_t[t]:
            outs = out_by_source.get(s, [])
            if len(outs) != 1:
                continue
            c1 = int(outs[0]["target_id"])
            n1 = nodes_by_id.get(c1)
            if n1 is None or int(n1["t"]) != t + 1:
                continue
            child_dist = float(np.linalg.norm(pos[s] - pos[c1]))
            if child_dist > 14.0:
                continue
            near = ctree.query_ball_point(pos[s], 14.0)
            if not near:
                continue
            mutual_id = cands[int(ctree.query(pos[c1])[1])]
            dens_t = len(ttree.query_ball_point(pos[s], 10.0))
            dens_t1 = len(ktree.query_ball_point(pos[s], 10.0))
            near = sorted(near, key=lambda i: float(np.linalg.norm(cpos[i] - pos[s])))
            for rank, i in enumerate(near):
                q = cands[i]
                pdist = float(np.linalg.norm(pos[s] - pos[q]))
                sdist = float(np.linalg.norm(pos[c1] - pos[q]))
                if sdist > 20.0:
                    continue
                c1s, qs = out_by_source.get(c1, []), out_by_source.get(q, [])
                diverge = float("nan")
                if len(c1s) == 1 and len(qs) == 1:
                    g1 = nodes_by_id.get(int(c1s[0]["target_id"]))
                    g2 = nodes_by_id.get(int(qs[0]["target_id"]))
                    if g1 is not None and g2 is not None and int(g1["t"]) == t + 2 and int(g2["t"]) == t + 2:
                        diverge = float(np.linalg.norm(_position_um(g1) - _position_um(g2))) - sdist
                dcq = dcc = dcs = None
                if pdist <= 11.0 and sdist <= 16.0:
                    dcq = deepcenter_score_point(dataset, t + 1, node_point(nodes_by_id[q]), bundle, frame_cache, dc_cache)
                    dcc = deepcenter_score_point(dataset, t + 1, node_point(n1), bundle, frame_cache, dc_cache)
                    dcs = deepcenter_score_point(dataset, t, node_point(nodes_by_id[s]), bundle, frame_cache, dc_cache)
                v1, v2 = pos[c1] - pos[s], pos[q] - pos[s]
                cos_angle = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
                mid_dist = float(np.linalg.norm((pos[c1] + pos[q]) / 2 - pos[s]))
                sn, qn = nodes_by_id[s], nodes_by_id[q]
                rows.append([
                    dataset, t, s, c1, q,
                    sn["z"], sn["y"], sn["x"], n1["z"], n1["y"], n1["x"], qn["z"], qn["y"], qn["x"],
                    child_dist, pdist, sdist, rank, len(near), int(q == mutual_id),
                    diverge,
                    float("nan") if dcq is None else dcq, float("nan") if dcc is None else dcc,
                    float("nan") if dcs is None else dcs,
                    prob(outs[0]), prob(qs[0]) if len(qs) == 1 else float("nan"),
                    fwd_len(q), fwd_len(c1), back_len(s), dens_t, dens_t1, cos_angle, mid_dist,
                    int((s, q) in accepted),
                ])
    out_dir = Path(os.environ["DIVLOG_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{dataset}.csv", "w", newline="") as fh:
        w = _dcsv.writer(fh)
        w.writerow(DIVLOG_COLUMNS)
        w.writerows(rows)
    print(f"  [{dataset}] division candidates logged: {len(rows)}, accepted by the cascade: {sum(r[-1] for r in rows)}")


'''
DIVLOG_ANCHOR = "def filter_output_graph("

CALL_OLD = """    edges = add_safe_divisions_postlink(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )
"""
CALL_NEW = ("    _div_pre_edges = list(edges)\n" + CALL_OLD +
            "    if os.environ.get('DIVLOG_DIR'):\n"
            "        try:\n"
            "            _divlog_candidates(nodes_by_id, _div_pre_edges, edges, dataset, deepcenter_bundle,\n"
            "                               repair_frame_cache, deepcenter_heatmap_cache)\n"
            "        except Exception as _divlog_exc:\n"
            "            print(f'  [{dataset}] DIVLOG FAILED: {type(_divlog_exc).__name__}: {_divlog_exc}', flush=True)\n")

LABEL_CELL = r'''
# Label the division candidates against ground truth and extract head pairs.
import csv as _lcsv
import shutil

import tracksdata as td
from scipy.optimize import linear_sum_assignment

PART = __PART__
HELD_OUT = __HELD_OUT__
VOX = np.array([1.625, 0.40625, 0.40625])
GRID = np.array([1.625, 1.625, 1.625])
GT_DIR = COMP_DIR / "train"
DIVLOG = Path(os.environ["DIVLOG_DIR"])
CAPTURE = Path(os.environ["V1284_CAPTURE_ALSO"])

labelled, gt_divisions = [], []
X_parts, Y_parts, G_parts = [], [], []
for stem in test_stems:
    g = td.graph.IndexedRXGraph.from_geff(GT_DIR / f"{stem}.geff")
    g = g[0] if isinstance(g, tuple) else g
    gpos, gt_t = {}, {}
    for r in g.node_attrs().iter_rows(named=True):
        nid = int(r["node_id"])
        gt_t[nid] = int(r["t"])
        gpos[nid] = np.array([r["z"], r["y"], r["x"]], dtype=float) * VOX
    kids, par = {}, {}
    for r in g.edge_attrs().iter_rows(named=True):
        s, d = int(r["source_id"]), int(r["target_id"])
        kids.setdefault(s, []).append(d)
        par[d] = s
    divs = {s for s, k in kids.items() if len(k) >= 2}
    for d in divs:
        gt_divisions.append({"dataset": stem, "t": gt_t[d], "gt_parent": d})
    by_t = {}
    for nid, t in gt_t.items():
        by_t.setdefault(t, []).append(nid)
    trees = {t: (cKDTree(np.stack([gpos[n] for n in ids])), ids) for t, ids in by_t.items()}

    def match(t, p):
        if t not in trees:
            return -1
        tree, ids = trees[t]
        dist, i = tree.query(p)
        return ids[int(i)] if dist <= 7.0 else -1

    f = DIVLOG / f"{stem}.csv"
    if f.exists():
        with open(f) as fh:
            for row in _lcsv.DictReader(fh):
                t = int(row["t"])
                sp = np.array([float(row["s_z"]), float(row["s_y"]), float(row["s_x"])]) * VOX
                cp = np.array([float(row["c_z"]), float(row["c_y"]), float(row["c_x"])]) * VOX
                qp = np.array([float(row["q_z"]), float(row["q_y"]), float(row["q_x"])]) * VOX
                gs, gc, gq = match(t, sp), match(t + 1, cp), match(t + 1, qp)
                window = [gs, par.get(gs, -1)] + kids.get(gs, [])
                row.update({
                    "part": PART, "held_out": int(stem in HELD_OUT),
                    "gt_source": gs, "gt_child": gc, "gt_cand": gq,
                    "annotated": int(gs != -1 and gs in kids),
                    "y_strict": int(gs in divs and gq in kids.get(gs, []) and gc in kids.get(gs, []) and gq != gc),
                    "y_window": int(gq != -1 and any(d in divs for d in window if d != -1)),
                })
                labelled.append(row)

    n_pairs = 0
    for npz in sorted((CAPTURE / stem).glob("*.npz")):
        t = int(npz.stem)
        if t not in by_t:
            continue
        with np.load(npz) as d:
            coords, feats = d["coords"], d["features"]
        if len(coords) == 0:
            continue
        det_um = coords[:, 1:].astype(np.float64) * GRID
        gt_um = np.stack([gpos[n] for n in by_t[t]])
        cost = np.linalg.norm(det_um[:, None, :] - gt_um[None, :, :], axis=-1)
        rr, cc = linear_sum_assignment(np.where(cost <= 4.0, cost, 1e6))
        keep = cost[rr, cc] <= 4.0
        rr, cc = rr[keep], cc[keep]
        X_parts.append(feats[rr].astype(np.float32))
        Y_parts.append((gt_um[cc] - det_um[rr]).astype(np.float32))
        G_parts.append(np.full(len(rr), stem))
        n_pairs += len(rr)
    print(f"  {stem}: {len(divs)} GT divisions, {n_pairs} head pairs")

pd.DataFrame(labelled).to_csv(f"/kaggle/working/divcand_part{PART}.csv", index=False)
pd.DataFrame(gt_divisions).to_csv(f"/kaggle/working/gtdiv_part{PART}.csv", index=False)
np.savez_compressed(f"/kaggle/working/coord_pairs_part{PART}.npz",
                    X=np.concatenate(X_parts), Y=np.concatenate(Y_parts), G=np.concatenate(G_parts))
lab = pd.read_csv(f"/kaggle/working/divcand_part{PART}.csv") if labelled else pd.DataFrame()
if len(lab):
    ann = lab[lab.annotated == 1]
    print(f"candidates {len(lab)}, annotated {len(ann)}, strict positives {int(ann.y_strict.sum())}, "
          f"window positives {int(ann.y_window.sum())}, cascade accepted {int(ann.accepted.sum())}, "
          f"cascade strict TP {int(((ann.accepted == 1) & (ann.y_strict == 1)).sum())}, GT divisions {len(gt_divisions)}")
for entry in Path("/kaggle/working").iterdir():
    if not entry.name.startswith(("divcand_", "gtdiv_", "coord_pairs_", "run_stats")):
        shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)
print("saved part", PART)
'''


def replace_once(cells, old, new):
    hits = [c for c in cells if old in "".join(c["source"])]
    if len(hits) != 1 or "".join(hits[0]["source"]).count(old) != 1:
        raise SystemExit(f"anchor not found exactly once: {old[:60]!r}")
    hits[0]["source"] = "".join(hits[0]["source"]).replace(old, new).splitlines(keepends=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=(1, 2, 3, 4), required=True)
    args = ap.parse_args()
    parts = [*video_parts(), *rest_parts()]
    stems = parts[args.part - 1]
    headfit = load("headfit", REPO / "scripts" / "build_headfit_kernel.py")

    nb = json.loads(BASE.read_text(encoding="utf-8"))
    tail = "".join(nb["cells"][LOOP_CELL]["source"]).rstrip().splitlines()[-1]
    if tail.strip() != 'write_test_submission("base")':
        raise SystemExit(f"cell {LOOP_CELL} does not end at the per-video loop: {tail!r}")
    cells = nb["cells"][: LOOP_CELL + 1]
    replace_once(cells, 'TEST_DIR = COMP_DIR / "test"', 'TEST_DIR = COMP_DIR / "train"')
    replace_once(cells, "test_stems = list_test_stems()", f"test_stems = {stems!r}")
    replace_once(cells, CAPTURE_ANCHOR, CAPTURE_PATCH + CAPTURE_ANCHOR)
    replace_once(cells, DIVLOG_ANCHOR, DIVLOG_FN + DIVLOG_ANCHOR)
    replace_once(cells, CALL_OLD, CALL_NEW)
    label = (LABEL_CELL.replace("__PART__", str(args.part))
             .replace("__HELD_OUT__", repr(list(headfit.HELD_OUT_19))))
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": label.lstrip().splitlines(keepends=True)})
    cells[0]["source"] = [
        f"# Division candidates and head features, part {args.part}\n", "\n",
        "Built by `scripts/build_divcapture_kernel.py`.\n",
    ]
    nb["cells"] = cells
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")

    out_dir = REPO / "notebooks" / f"plateau_divcapture{args.part}"
    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"divcapture{args.part}.ipynb"
    (out_dir / code_file).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((BASE.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = f"vyask21/cell-tracking-plateau-divcapture{args.part}"
    meta["title"] = meta["id"].split("/")[1]
    meta["code_file"] = code_file
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / code_file} with {len(stems)} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
