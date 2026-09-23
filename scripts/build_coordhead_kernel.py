"""Build the exp 19 submission kernel: exp 18 with a coordinate head of our own.

    python scripts/fit_coord_head_linear.py
    python scripts/build_coordhead_kernel.py

One variable against exp 18: the coordinate-refinement module goes from its
passthrough mode to refining detections with a head fitted here on the support
pack's held-out 19 videos. The head is the linear map from
scripts/fit_coord_head_linear.py, whose docstring records why it replaced the MLP
the head-fit kernel produced.

The module's head is fixed at Linear-SiLU-Linear. A linear map d = Wx + b fits
it exactly: the first layer carries W and b plus a constant C = 40 on three
units, which puts SiLU in its identity regime, SiLU(C + d) = C + d to float32
precision while |d| stays well under C, and the second layer subtracts C. The
largest pre-bound |d| on the fitting data is 4.6. The kernel checks the
embedded head against the linear map on random inputs and raises if they differ,
then saves it where the module loads it. The weights are embedded in the
notebook, so no dataset or kernel output has to be mounted.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_gapstack" / "gapstack.ipynb"
OUT_DIR = REPO / "notebooks" / "plateau_coordhead"
KERNEL_ID = "vyask21/cell-tracking-plateau-coordhead"
HEAD_JSON = REPO / "artifacts" / "plateau_headfit" / "coord_head_linear.json"

OLD = "os.environ['V1284_MODE']='zero'\n"
NEW_TEMPLATE = r"""import json as _cjson
import importlib.util as _cilu
import torch as _ctorch
_CH = _cjson.loads(r'''__HEAD_JSON__''')
_cspec = _cilu.spec_from_file_location("coord_head_build", str(_ps.parent / "v1284_coordinate_refinement.py"))
_cmod = _cilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_cmod)
_chead = _cmod.make_head()
_CBIAS = 40.0
with _ctorch.no_grad():
    for _p in _chead.parameters():
        _p.zero_()
    _chead[0].weight[:3] = _ctorch.tensor(_CH["weight"])
    _chead[0].bias[:3] = _ctorch.tensor(_CH["bias"]) + _CBIAS
    _chead[2].weight[:, :3] = _ctorch.eye(3)
    _chead[2].bias[:] = -_CBIAS
    _cx = _ctorch.randn(4096, 224, generator=_ctorch.Generator().manual_seed(0))
    _clin = _cx @ _ctorch.tensor(_CH["weight"]).T + _ctorch.tensor(_CH["bias"])
    _cerr = float((_chead(_cx) - _clin).abs().max())
if _cerr > 1e-3:
    raise RuntimeError(("coordinate head embedding error", _cerr))
_ctorch.save({"state_dict": _chead.state_dict(),
              "mean": _ctorch.tensor(_CH["mean"]), "scale": _ctorch.tensor(_CH["scale"])},
             "/kaggle/working/coord_head.pt")
os.environ['V1284_MODE']='candidate'
os.environ['V1284_HEAD']='/kaggle/working/coord_head.pt'
print('coordinate head: linear, lambda', _CH["lambda"], '| embedding max error', _cerr)
"""

HEADER = """# Plateau, public gap-fill stack, coordinate head fitted here

Built by `scripts/build_coordhead_kernel.py`. The exp 18 notebook with one change:
the coordinate-refinement module runs with a linear head fitted on the support
pack's held-out 19 videos, embedded below. `scripts/fit_coord_head_linear.py`
fits it and records the held-out evidence.
Sources and credit are in `NOTES.md`, 2026-09-23.
"""


def main() -> int:
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    hits = [c for c in nb["cells"] if OLD in "".join(c["source"])]
    if len(hits) != 1 or "".join(hits[0]["source"]).count(OLD) != 1:
        raise SystemExit("passthrough line not found exactly once")
    head = json.loads(HEAD_JSON.read_text(encoding="utf-8"))
    payload = json.dumps({k: head[k] for k in ("lambda", "mean", "scale", "weight", "bias")},
                         separators=(",", ":"))
    new = NEW_TEMPLATE.replace("__HEAD_JSON__", payload)
    hits[0]["source"] = "".join(hits[0]["source"]).replace(OLD, new).splitlines(keepends=True)
    nb["cells"][0]["source"] = HEADER.splitlines(keepends=True)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "coordhead.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((REPO / "notebooks" / "plateau_gapstack" / "kernel-metadata.json")
                      .read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "coordhead.ipynb"
    meta["kernel_sources"] = []
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'coordhead.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
