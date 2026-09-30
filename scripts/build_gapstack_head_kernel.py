"""Build the exp 21 submission kernel: the public gap-fill stack with its own head.

    python scripts/build_gapstack_head_kernel.py

Exp 18 had to run the source with its coordinate head off, because the head sat
in a private dataset. The author has since made that dataset public, and the
board shows the result: 156 teams moved to exactly 0.953 inside a day. This is
the source as published, head on, with the same name sanitising as exp 18.

The only change beyond sanitising is the head lookup, which finds the file by its
own name rather than by the dataset slug, keeping the exactly-one guard. The
dataset appears in kernel-metadata.json because Kaggle needs it to mount.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("gapstack", REPO / "scripts" / "build_gapstack_kernel.py")
gapstack = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gapstack)

OUT_DIR = REPO / "notebooks" / "plateau_gapstack_head"
KERNEL_ID = "vyask21/cell-tracking-plateau-gapstack-head"
HEAD_DATASET = "srcF/published-coordinate-head"

HEAD_ON = (
    "_myhead = sorted(Path('/kaggle/input').rglob('v1284_head.pt'))\n"
    "if len(_myhead) != 1:\n"
    "    raise RuntimeError(('coordinate head mount mismatch', [str(p) for p in _myhead]))\n"
    "# The published head, now in a public dataset.\n"
    "os.environ['V1284_MODE']='candidate'\n"
    "os.environ['V1284_HEAD']=str(_myhead[0])\n"
)

HEADER = """# Plateau, public gap-fill stack with its published coordinate head

Built by `scripts/build_gapstack_head_kernel.py`. Public work run as published,
with names removed from comments and report strings.
"""


def main() -> int:
    raw = gapstack.SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != gapstack.SOURCE_SHA256:
        raise SystemExit("source changed")
    nb = json.loads(raw.decode("utf-8"))
    joined = "\n#<<CELL>>\n".join("".join(c["source"]) for c in nb["cells"])
    for label, pattern, repl, expected in gapstack.PATCHES:
        if label.startswith("coordinate head off"):
            repl = HEAD_ON
            label = "coordinate head on, found by file name"
        joined, n = re.subn(pattern, lambda _m, r=repl: r, joined)
        if n != expected:
            raise SystemExit(f"patch {label!r}: matched {n}, expected {expected}")
        print(f"applied x{n}: {label}")
    if "V1284_MODE']='candidate'" not in joined or "V1284_MODE']='zero'" in joined:
        raise SystemExit("head not in candidate mode")

    cells = joined.split("\n#<<CELL>>\n")
    cells[0] = gapstack.OWNER_ROOT + cells[0]
    for cell, text in zip(nb["cells"], cells):
        cell["source"] = text.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {},
                           "source": HEADER.splitlines(keepends=True)})
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{i}", "exec")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gapstack_head.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((REPO / "notebooks" / "plateau_gapstack" / "kernel-metadata.json")
                      .read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "gapstack_head.ipynb"
    meta["dataset_sources"] = [HEAD_DATASET] + meta["dataset_sources"]
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'gapstack_head.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
