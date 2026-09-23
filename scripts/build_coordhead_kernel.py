"""Build the exp 19 submission kernel: exp 18 with a coordinate head of our own.

    python scripts/build_coordhead_kernel.py

One variable against exp 18: the coordinate-refinement module goes from its
passthrough mode to refining detections with the head that
notebooks/plateau_headfit fits on the support pack's held-out 19 videos. The
head is mounted from that kernel's output, so this kernel lists it in
kernel_sources, and the load is guarded to find exactly one file, as the
published version guarded its own.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_gapstack" / "gapstack.ipynb"
OUT_DIR = REPO / "notebooks" / "plateau_coordhead"
KERNEL_ID = "vyask21/cell-tracking-plateau-coordhead"
HEAD_KERNEL = "vyask21/cell-tracking-plateau-headfit"

OLD = "os.environ['V1284_MODE']='zero'\n"
NEW = (
    "_coord_head = sorted(Path('/kaggle/input').rglob('coord_head.pt'))\n"
    "if len(_coord_head) != 1:\n"
    "    raise RuntimeError(('coordinate head mount mismatch', [str(p) for p in _coord_head]))\n"
    "os.environ['V1284_MODE']='candidate'\n"
    "os.environ['V1284_HEAD']=str(_coord_head[0])\n"
    "print('coordinate head:', _coord_head[0])\n"
)

HEADER = """# Plateau, public gap-fill stack, coordinate head fitted here

Built by `scripts/build_coordhead_kernel.py`. The exp 18 notebook with one change:
the coordinate-refinement module runs with a head fitted by the
`cell-tracking-plateau-headfit` kernel on the support pack's held-out 19 videos.
Sources and credit are in `NOTES.md`, 2026-09-23.
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
    (OUT_DIR / "coordhead.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((REPO / "notebooks" / "plateau_gapstack" / "kernel-metadata.json")
                      .read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "coordhead.ipynb"
    meta["kernel_sources"] = [HEAD_KERNEL]
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'coordhead.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
