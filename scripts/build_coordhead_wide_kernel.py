"""Build the exp 20 submission kernel: exp 18 with the wide MLP coordinate head.

    python scripts/build_coordhead_wide_kernel.py

One variable against exp 19: the head. Exp 19 shipped a linear head fitted on the
support pack's held-out 19. This ships the MLP that cell-tracking-plateau-headfit-wide
fitted on 59 videos, 2000 steps, loaded from that kernel's output.

Chosen on the held-out 19 only, one video left out at a time, which is the only
test that matches a hidden set the U-Net never saw:

    head                        pooled         44b6          6bba
    exp 19 linear, 19 videos    -23.0% 17/19   -2.2% 3/5     -25.3% 14/14
    linear, 59 videos           -23.6% 17/19   -8.0% 3/5     -25.3% 14/14
    MLP 2000 steps, 59 videos   -24.0% 19/19   -6.8% 5/5     -25.8% 14/14
    MLP 500 steps, 59 videos    -24.9% 18/19   -7.7% 4/5     -26.7% 14/14

The 2000-step MLP is the only head that moves every held-out video closer to truth
on both embryos. The pooled differences between the last three are a point or so.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_gapstack" / "gapstack.ipynb"
OUT_DIR = REPO / "notebooks" / "plateau_coordhead_wide"
KERNEL_ID = "vyask21/cell-tracking-plateau-coordhead-wide"
HEAD_KERNEL = "vyask21/cell-tracking-plateau-headfit-wide"
HEAD_FILE = "coord_head_mlp2000.pt"

OLD = "os.environ['V1284_MODE']='zero'\n"
NEW = (
    f"_coord_head = sorted(Path('/kaggle/input').rglob('{HEAD_FILE}'))\n"
    "if len(_coord_head) != 1:\n"
    "    raise RuntimeError(('coordinate head mount mismatch', [str(p) for p in _coord_head]))\n"
    "os.environ['V1284_MODE']='candidate'\n"
    "os.environ['V1284_HEAD']=str(_coord_head[0])\n"
    "print('coordinate head: wide MLP', _coord_head[0])\n"
)

HEADER = """# Plateau, public gap-fill stack, wide MLP coordinate head

Built by `scripts/build_coordhead_wide_kernel.py`. The exp 18 notebook with one
change: the coordinate-refinement module runs with the MLP head fitted on 59
videos by the `cell-tracking-plateau-headfit-wide` kernel.
"""


def main() -> int:
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    hits = [c for c in nb["cells"] if OLD in "".join(c["source"])]
    if len(hits) != 1 or "".join(hits[0]["source"]).count(OLD) != 1:
        raise SystemExit("passthrough line not found exactly once")
    hits[0]["source"] = "".join(hits[0]["source"]).replace(OLD, NEW).splitlines(keepends=True)
    nb["cells"][0]["source"] = HEADER.splitlines(keepends=True)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "coordhead_wide.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((REPO / "notebooks" / "plateau_gapstack" / "kernel-metadata.json")
                      .read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "coordhead_wide.ipynb"
    meta["kernel_sources"] = [HEAD_KERNEL]
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'coordhead_wide.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
