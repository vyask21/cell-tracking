"""Build a shift-scale probe on the published-head kernel.

    python scripts/build_gapstack_scaled_kernel.py --scale 0.75

One variable against exp 21: every refined displacement is multiplied by a
constant before it is applied. Exp 21 is the published head at scale 1.0.

Why this axis. Three heads now have board scores: linear on 19 videos 0.948, MLP
on 59 videos 0.950, the published head 0.953. On distance to truth over the
held-out pairs they rank in exactly the opposite order, and the published head
makes the smallest moves, mean 0.85 um against 1.26 for ours. The likely reason
is that refined positions feed the edge model's feature lookup as well as the
output coordinates, and the edge model was trained on native grid positions, so
a larger shift buys position and pays in linking. If that is right, the size of
the shift is the lever, and the board is the only instrument that sees both
sides of the trade.

The module's own 2 um validity check scales with the factor, so a scale above 1
is not rejected by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_gapstack_head" / "gapstack_head.ipynb"

SHIFT_OLD = "shift = bounded(head, (x-mean)/scale).cpu().numpy() / SPACING"
SHIFT_NEW = ("shift = float(os.environ.get(\\'V1284_SCALE\\', \\'1.0\\')) * "
             "bounded(head, (x-mean)/scale).cpu().numpy() / SPACING")
CHECK_OLD = "> 2.00001:"
CHECK_NEW = "> 2.00001 * max(1.0, float(os.environ.get(\\'V1284_SCALE\\', \\'1.0\\'))):"
MODE_OLD = "os.environ['V1284_MODE']='candidate'\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, required=True)
    args = ap.parse_args()
    tag = f"scale{round(args.scale * 100):03d}"
    out_dir = REPO / "notebooks" / f"plateau_gapstack_head_{tag}"
    kernel_id = f"vyask21/cell-tracking-plateau-gapstack-head-{tag}"

    nb = json.loads(BASE.read_text(encoding="utf-8"))
    for old, new in ((SHIFT_OLD, SHIFT_NEW), (CHECK_OLD, CHECK_NEW),
                     (MODE_OLD, MODE_OLD + f"os.environ['V1284_SCALE']='{args.scale}'\n")):
        hits = [c for c in nb["cells"] if old in "".join(c["source"])]
        if len(hits) != 1 or "".join(hits[0]["source"]).count(old) != 1:
            raise SystemExit(f"anchor not found exactly once: {old!r}")
        hits[0]["source"] = "".join(hits[0]["source"]).replace(old, new).splitlines(keepends=True)
    nb["cells"][0]["source"] = [
        f"# Plateau, public gap-fill stack, published head, shift scale {args.scale}\n",
        "\n",
        "Built by `scripts/build_gapstack_scaled_kernel.py`.\n",
    ]
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")

    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"gapstack_head_{tag}.ipynb"
    (out_dir / code_file).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((BASE.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = kernel_id
    meta["title"] = kernel_id.split("/")[1]
    meta["code_file"] = code_file
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / code_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
