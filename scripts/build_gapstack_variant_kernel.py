"""Build a one-variable probe on the published-head kernel, exp 21.

    python scripts/build_gapstack_variant_kernel.py --probe blend
    python scripts/build_gapstack_variant_kernel.py --probe divcombo
    python scripts/build_gapstack_variant_kernel.py --probe blend --weight 0.35

Nothing public beats 0.953: the two most-voted 0.953 notebooks are byte-level
copies of the exp 21 source and the third adds a runtime sweep and still lands at
0.953. Whatever moves past it has to be ours. Two probes, each one change:

blend    The refined shift becomes the mean of the published head's shift and
         the shift from our 59-video MLP, exp 20's head, which reached 0.950 on
         this stack alone. Implemented by patching the written refinement module
         so the second head loads from V1284_HEAD2 and is averaged at weight
         V1284_BLEND = 0.5. The second head is mounted from the output of
         cell-tracking-plateau-headfit-wide. Exp 25 at weight 0.5 scored 0.956
         against 0.953 for the published head alone; --weight sets the share
         of our head for the follow-up probes and names them blend035 and so on.

divcombo The overrides that exp 17's source selected and that moved our anchor
         from 0.947 to 0.948: DeepCenter safe-division veto 0.25 to 0.15, parent
         9.0 to 11.0, sister 14.0 to 16.0, existing child 10.0 to 12.0, relink
         velocity weight 0.5 to 0.25. Leaf pruning at 0.3 was part of that combo
         and cannot be ported, because this stack has no leaf-prune mechanism.
         Tight relink is already 5.5 here. The config guard does not check any
         of these keys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_gapstack_head" / "gapstack_head.ipynb"

BLEND_ANCHOR = "_myhead = sorted(Path('/kaggle/input').rglob('v1284_head.pt'))\n"
BLEND_CODE = r'''_blend_path = _ps.parent / 'v1284_coordinate_refinement.py'
_blend_src = _blend_path.read_text()
_blend_old = "    shift = bounded(head, (x-mean)/scale).cpu().numpy() / SPACING\n"
if _blend_src.count(_blend_old) != 1 or _blend_src.count("_CACHE = None\n") != 1:
    raise RuntimeError("blend patch anchor mismatch")
_blend_new = (
    "    shift = bounded(head, (x-mean)/scale)\n"
    "    global _CACHE2\n"
    "    if _CACHE2 is None:\n"
    "        s2 = torch.load(os.environ['V1284_HEAD2'], map_location='cpu', weights_only=True)\n"
    "        h2 = make_head().to(feature.device)\n"
    "        h2.load_state_dict(s2['state_dict']); h2.eval()\n"
    "        _CACHE2 = (h2, s2['mean'].to(feature.device), s2['scale'].to(feature.device))\n"
    "    h2, m2, sc2 = _CACHE2\n"
    "    w2 = float(os.environ['V1284_BLEND'])\n"
    "    shift = ((1.0 - w2) * shift + w2 * bounded(h2, (x-m2)/sc2)).cpu().numpy() / SPACING\n"
)
_blend_src = _blend_src.replace(_blend_old, _blend_new).replace("_CACHE = None\n", "_CACHE = None\n_CACHE2 = None\n")
compile(_blend_src, str(_blend_path), "exec")
_blend_path.write_text(_blend_src)
_head2 = sorted(Path('/kaggle/input').rglob('coord_head_mlp2000.pt'))
if len(_head2) != 1:
    raise RuntimeError(('second head mount mismatch', [str(p) for p in _head2]))
os.environ['V1284_HEAD2'] = str(_head2[0])
os.environ['V1284_BLEND'] = '0.5'
print('head blend: published + wide MLP at 0.5 |', _head2[0])
'''

DIVCOMBO = [
    ('os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "9.0"', 'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "11.0"'),
    ('os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "14.0"', 'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "16.0"'),
    ('os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "10.0"',
     'os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "12.0"'),
    ('os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"',
     'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.15"'),
    ('os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"',
     'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"\n'
     'os.environ["BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT"] = "0.25"'),
]

TITLES = {
    "blend": "published head blended 50/50 with the wide MLP head",
    "divcombo": "division and relink overrides from the exp 17 selection",
}


def replace_once(cells, old, new):
    hits = [c for c in cells if old in "".join(c["source"])]
    if len(hits) != 1 or "".join(hits[0]["source"]).count(old) != 1:
        raise SystemExit(f"anchor not found exactly once: {old[:60]!r}")
    hits[0]["source"] = "".join(hits[0]["source"]).replace(old, new).splitlines(keepends=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=sorted(TITLES), required=True)
    ap.add_argument("--weight", type=float, default=0.5, help="blend share of the wide MLP head")
    args = ap.parse_args()
    name = args.probe
    title = TITLES[args.probe]
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    meta = json.loads((BASE.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    if args.probe == "blend":
        code = BLEND_CODE
        if args.weight != 0.5:
            code = (code.replace("os.environ['V1284_BLEND'] = '0.5'", f"os.environ['V1284_BLEND'] = '{args.weight}'")
                        .replace("published + wide MLP at 0.5", f"published + wide MLP at {args.weight}"))
            name = f"blend{round(args.weight * 100):03d}"
            title = f"published head blended with the wide MLP head at weight {args.weight}"
        replace_once(cells, BLEND_ANCHOR, code + BLEND_ANCHOR)
        meta["kernel_sources"] = ["vyask21/cell-tracking-plateau-headfit-wide"]
    else:
        for old, new in DIVCOMBO:
            replace_once(cells, old, new)
    cells[0]["source"] = [
        f"# Plateau, public gap-fill stack, {TITLES[args.probe]}\n", "\n",
        "Built by `scripts/build_gapstack_variant_kernel.py`.\n",
    ]
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")

    out_dir = REPO / "notebooks" / f"plateau_gapstack_head_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"gapstack_head_{name}.ipynb"
    (out_dir / code_file).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta["id"] = f"vyask21/cell-tracking-plateau-gapstack-head-{name}"
    meta["title"] = meta["id"].split("/")[1]
    meta["code_file"] = code_file
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / code_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
