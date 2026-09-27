"""Build the bet A submission kernel: exp 25 with learned divisions.

    python scripts/fit_division_model.py
    python scripts/build_gapstack_learneddiv_kernel.py --tau 0.01
    python scripts/build_gapstack_learneddiv_kernel.py --tau 0.01 --base ens5

--base ens5 applies the same change on top of the exp 29 five-seed head ensemble
instead of exp 25, for the combination run if both bets pay.

One variable against exp 25, which scored 0.956: the post-link division step. The
published cascade (fixed distance gates, mutual nearest neighbour, divergence,
DeepCenter veto, symmetry, nearest survivor) is replaced by the gradient-boosted
model from scripts/fit_division_model.py, scored on the candidates under the same
loose gates the capture kernels logged, with the features computed by the same
code the capture kernels ran, taken from scripts/build_divcapture_kernel.py
rather than rewritten. Candidates are accepted in descending score above the
chosen threshold, each parent and each daughter at most once.

The model is embedded as LightGBM text, zlib-compressed and base64-encoded, since
Kaggle rejected the 2.6 MB notebook that also carried the dumped trees. LightGBM
is in the Kaggle image; if it were missing the kernel stops at start-up rather
than running without the model. The kernel scores stored probe rows at start-up
and raises if the predictions differ from those recorded at fit time.

Why --tau 0.01 and not the threshold the fit script chose. The model was trained
on the 100 in-sample videos, whose detections the networks have seen, and its
scores shrink on unseen videos: on the held-out 19 it ranks candidates at AUC
0.991 and average precision 0.315 against a 0.006 base rate, but the in-sample
threshold of 0.1 accepts one candidate there, a false one. The test videos are
unseen, so the threshold is set on the held-out 19, where division Jaccard is
0.170, 0.167 and 0.156 at 0.005, 0.01 and 0.02 against 0.071 for the cascade.
0.01 is the middle of that plateau. Chosen on the validation set, so the held-out
number is optimistic; the gain holds across the plateau.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_gapstack_head_blend" / "gapstack_head_blend.ipynb"
MODEL = REPO / "artifacts" / "divmodel" / "divmodel.json"
OUT_DIR = REPO / "notebooks" / "plateau_gapstack_head_blend_learneddiv"
KERNEL_ID = "vyask21/cell-tracking-plateau-learneddiv"

MODEL_CODE = r'''
import json
_DIVMODEL = json.loads(r"""__MODEL__""")


import base64 as _b64
import zlib as _zlib
import lightgbm as _lgb
_divbooster = _lgb.Booster(model_str=_zlib.decompress(_b64.b64decode(_DIVMODEL["model_z"])).decode())


def _divmodel_predict(X):
    return _divbooster.predict(np.asarray(X, dtype=np.float64))


_probe_err = float(np.max(np.abs(_divmodel_predict(_DIVMODEL["probe_x"]) - np.asarray(_DIVMODEL["probe_p"]))))
if _probe_err > 1e-6:
    raise RuntimeError(("division model probe mismatch", _probe_err))
print(f"learned divisions: lightgbm {_lgb.__version__}, tau {_DIVMODEL['tau']}, probe error {_probe_err:.2e}")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=None, help="override the fitted threshold")
    ap.add_argument("--base", choices=("blend", "ens5"), default="blend")
    args = ap.parse_args()
    base, out_dir, kernel_id = BASE, OUT_DIR, KERNEL_ID
    if args.base == "ens5":
        base = REPO / "notebooks" / "plateau_gapstack_head_ens5" / "gapstack_head_ens5.ipynb"
        out_dir = REPO / "notebooks" / "plateau_ens5_learneddiv"
        kernel_id = "vyask21/cell-tracking-plateau-ens5-learneddiv"
    spec = importlib.util.spec_from_file_location("divcap", REPO / "scripts" / "build_divcapture_kernel.py")
    divcap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(divcap)
    fn = divcap.DIVLOG_FN
    cut = fn.index('    out_dir = Path(os.environ["DIVLOG_DIR"])')
    rows_fn = fn[:cut].replace("def _divlog_candidates(", "def _div_rows(") + "    return rows\n\n\n"
    if rows_fn.count("def _div_rows(") != 1:
        raise SystemExit("could not derive the row function")

    model = json.loads(MODEL.read_text(encoding="utf-8"))
    if args.tau is not None:
        model["tau"] = args.tau
    model["model_z"] = base64.b64encode(zlib.compress(model["model_str"].encode(), 9)).decode()
    payload = json.dumps({k: model[k] for k in ("features", "tau", "model_z", "probe_x", "probe_p")})
    if '"""' in payload:
        raise SystemExit("model payload contains a triple quote")
    code = rows_fn + MODEL_CODE.replace("__MODEL__", payload)

    nb = json.loads(base.read_text(encoding="utf-8"))
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
    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = out_dir.name.replace("plateau_", "") + ".ipynb"
    (out_dir / code_file).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((base.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = kernel_id
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = code_file
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / code_file}, tau {model['tau']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
