"""Build the bet A submission kernel: exp 25 with learned divisions.

    python scripts/fit_division_model.py
    python scripts/build_gapstack_learneddiv_kernel.py

One variable against exp 25, which scored 0.956: the post-link division step. The
published cascade (fixed distance gates, mutual nearest neighbour, divergence,
DeepCenter veto, symmetry, nearest survivor) is replaced by the gradient-boosted
model from scripts/fit_division_model.py, scored on the candidates under the same
loose gates the capture kernels logged, with the features computed by the same
code the capture kernels ran, taken from scripts/build_divcapture_kernel.py
rather than rewritten. Candidates are accepted in descending score above the
chosen threshold, each parent and each daughter at most once.

The model is embedded as LightGBM text and as dumped trees. LightGBM is used when
the image has it, otherwise a numpy evaluator walks the dumped trees. Either way
the kernel scores stored probe rows at start-up and raises if the predictions
differ from those recorded at fit time.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_gapstack_head_blend" / "gapstack_head_blend.ipynb"
MODEL = REPO / "artifacts" / "divmodel" / "divmodel.json"
OUT_DIR = REPO / "notebooks" / "plateau_gapstack_head_blend_learneddiv"
KERNEL_ID = "vyask21/cell-tracking-plateau-gapstack-head-blend-learneddiv"

MODEL_CODE = r'''
import json
_DIVMODEL = json.loads(r"""__MODEL__""")


def _tree_value(node, row):
    while "leaf_value" not in node:
        v = row[node["split_feature"]]
        missing = node.get("missing_type", "None")
        if v != v and missing == "None":
            v = 0.0
        if (v != v and missing == "NaN") or (missing == "Zero" and (v != v or v == 0.0)):
            node = node["left_child"] if node.get("default_left", True) else node["right_child"]
        elif v <= node["threshold"]:
            node = node["left_child"]
        else:
            node = node["right_child"]
    return node["leaf_value"]


def _divmodel_predict(X):
    X = np.asarray(X, dtype=np.float64)
    if _DIVMODEL.get("_booster") is not None:
        return _DIVMODEL["_booster"].predict(X)
    raw = np.array([sum(_tree_value(t["tree_structure"], row) for t in _DIVMODEL["dump"]["tree_info"]) for row in X])
    return 1.0 / (1.0 + np.exp(-raw))


try:
    import lightgbm as _lgb
    _DIVMODEL["_booster"] = _lgb.Booster(model_str=_DIVMODEL["model_str"])
    _divmodel_backend = "lightgbm " + _lgb.__version__
except Exception as _lgb_exc:
    _DIVMODEL["_booster"] = None
    _divmodel_backend = f"numpy trees ({type(_lgb_exc).__name__})"
_probe_err = float(np.max(np.abs(_divmodel_predict(_DIVMODEL["probe_x"]) - np.asarray(_DIVMODEL["probe_p"]))))
if _probe_err > 1e-6:
    raise RuntimeError(("division model probe mismatch", _probe_err))
print(f"learned divisions: {_divmodel_backend}, tau {_DIVMODEL['tau']}, probe error {_probe_err:.2e}")


def _learned_divisions_postlink(nodes_by_id, edges, stats, dataset, bundle, frame_cache, dc_cache):
    if not edges or not nodes_by_id:
        return edges
    rows = _div_rows(nodes_by_id, edges, edges, dataset, bundle, frame_cache, dc_cache)
    if not rows:
        return edges
    col = {name: i for i, name in enumerate(DIVLOG_COLUMNS)}
    X = [[float(r[col[f]]) for f in _DIVMODEL["features"]] for r in rows]
    p = _divmodel_predict(X)
    order = np.argsort(-p)
    used_s, used_q, added = set(), set(), []
    for i in order:
        if p[i] < _DIVMODEL["tau"]:
            break
        r = rows[i]
        s, q = int(r[col["source_id"]]), int(r[col["cand_id"]])
        if s in used_s or q in used_q:
            continue
        used_s.add(s)
        used_q.add(q)
        added.append({"source_id": s, "target_id": q, "edge_prob": None,
                      "distance_um": float(r[col["parent_dist"]]), "safe_division": 1})
    stats["learned_division_candidates"] = len(rows)
    stats["learned_divisions_added"] = len(added)
    print(f"  [{dataset}] learned divisions: {len(added)} of {len(rows)} candidates", flush=True)
    return [*edges, *added]


'''


def main() -> int:
    spec = importlib.util.spec_from_file_location("divcap", REPO / "scripts" / "build_divcapture_kernel.py")
    divcap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(divcap)
    fn = divcap.DIVLOG_FN
    cut = fn.index('    out_dir = Path(os.environ["DIVLOG_DIR"])')
    rows_fn = fn[:cut].replace("def _divlog_candidates(", "def _div_rows(") + "    return rows\n\n\n"
    if rows_fn.count("def _div_rows(") != 1:
        raise SystemExit("could not derive the row function")

    model = json.loads(MODEL.read_text(encoding="utf-8"))
    payload = json.dumps({k: model[k] for k in ("features", "tau", "model_str", "dump", "probe_x", "probe_p")})
    if '"""' in payload:
        raise SystemExit("model payload contains a triple quote")
    code = rows_fn + MODEL_CODE.replace("__MODEL__", payload)

    nb = json.loads(BASE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    divcap.replace_once(cells, divcap.DIVLOG_ANCHOR, code + divcap.DIVLOG_ANCHOR)
    divcap.replace_once(cells, divcap.CALL_OLD,
                        "    edges = _learned_divisions_postlink(nodes_by_id, edges, stats, dataset, deepcenter_bundle,\n"
                        "                                        repair_frame_cache, deepcenter_heatmap_cache)\n")
    cells[0]["source"] = [
        "# Plateau, public gap-fill stack, head blend, learned divisions\n", "\n",
        "Built by `scripts/build_gapstack_learneddiv_kernel.py`. Sources and credit are in `NOTES.md`.\n",
    ]
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    code_file = "gapstack_head_blend_learneddiv.ipynb"
    (OUT_DIR / code_file).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((BASE.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = code_file
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / code_file}, tau {model['tau']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
