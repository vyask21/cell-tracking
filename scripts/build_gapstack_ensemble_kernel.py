"""Build the bet B submission kernel: exp 25 with our half of the blend replaced.

    python scripts/build_gapstack_ensemble_kernel.py

One variable against exp 25, which scored 0.956: the head averaged 50/50 with the
published head changes from the single 59-video MLP to the mean of five MLPs,
seeds 0 to 4, each fitted on about 160 training videos by
cell-tracking-plateau-headtrain. The weight stays 0.5. The five heads are mounted
from that kernel's output and the load is guarded to find exactly five.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_gapstack_head" / "gapstack_head.ipynb"
OUT_DIR = REPO / "notebooks" / "plateau_gapstack_head_ens5"
KERNEL_ID = "vyask21/cell-tracking-plateau-gapstack-head-ens5"
HEAD_KERNEL = "vyask21/cell-tracking-plateau-headtrain"

ANCHOR = "_myhead = sorted(Path('/kaggle/input').rglob('v1284_head.pt'))\n"
ENSEMBLE_CODE = r'''_ens_path = _ps.parent / 'v1284_coordinate_refinement.py'
_ens_src = _ens_path.read_text()
_ens_old = "    shift = bounded(head, (x-mean)/scale).cpu().numpy() / SPACING\n"
if _ens_src.count(_ens_old) != 1 or _ens_src.count("_CACHE = None\n") != 1:
    raise RuntimeError("ensemble patch anchor mismatch")
_ens_new = (
    "    shift = bounded(head, (x-mean)/scale)\n"
    "    global _CACHE2\n"
    "    if _CACHE2 is None:\n"
    "        _CACHE2 = []\n"
    "        for _hp in os.environ['V1284_HEADS2'].split(os.pathsep):\n"
    "            s2 = torch.load(_hp, map_location='cpu', weights_only=True)\n"
    "            h2 = make_head().to(feature.device)\n"
    "            h2.load_state_dict(s2['state_dict']); h2.eval()\n"
    "            _CACHE2.append((h2, s2['mean'].to(feature.device), s2['scale'].to(feature.device)))\n"
    "    ours = sum(bounded(h2, (x-m2)/sc2) for h2, m2, sc2 in _CACHE2) / len(_CACHE2)\n"
    "    w2 = float(os.environ['V1284_BLEND'])\n"
    "    shift = ((1.0 - w2) * shift + w2 * ours).cpu().numpy() / SPACING\n"
)
_ens_src = _ens_src.replace(_ens_old, _ens_new).replace("_CACHE = None\n", "_CACHE = None\n_CACHE2 = None\n")
compile(_ens_src, str(_ens_path), "exec")
_ens_path.write_text(_ens_src)
_heads2 = sorted(Path('/kaggle/input').rglob('coord_head_s[0-9].pt'))
if len(_heads2) != 5:
    raise RuntimeError(('ensemble head mount mismatch', [str(p) for p in _heads2]))
os.environ['V1284_HEADS2'] = os.pathsep.join(str(p) for p in _heads2)
os.environ['V1284_BLEND'] = '0.5'
print('head blend: published + mean of', len(_heads2), 'MLP heads at 0.5')
'''


def main() -> int:
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    hits = [c for c in nb["cells"] if ANCHOR in "".join(c["source"])]
    if len(hits) != 1:
        raise SystemExit("anchor not found exactly once")
    hits[0]["source"] = "".join(hits[0]["source"]).replace(ANCHOR, ENSEMBLE_CODE + ANCHOR).splitlines(keepends=True)
    nb["cells"][0]["source"] = [
        "# Plateau, public gap-fill stack, published head blended with a five-seed MLP ensemble\n", "\n",
        "Built by `scripts/build_gapstack_ensemble_kernel.py`. Sources and credit are in `NOTES.md`.\n",
    ]
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gapstack_head_ens5.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((BASE.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "gapstack_head_ens5.ipynb"
    meta["kernel_sources"] = [HEAD_KERNEL]
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'gapstack_head_ens5.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
