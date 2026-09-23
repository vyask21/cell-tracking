"""Build the exp 18 submission kernel: the public gap-fill stack with its coordinate head off.

    python scripts/build_gapstack_kernel.py

The source is a public notebook whose own page shows a public score of 0.953, read
2026-09-23. That is a kernel-level score, not an author-level one, and the same
check read the exp 17 source at 0.948 before exp 17 scored exactly that. Source,
authorship and the component lineage are credited in NOTES.md 2026-09-23. This
file and the notebook it writes deliberately carry no competitor names or notebook
titles, per the naming rule in the workspace CLAUDE.md.

The source cannot run as published. It loads a small learned coordinate head from
a private dataset and raises if the file is absent. The module ships its own
passthrough, V1284_MODE='zero', which leaves detections on their native integer
centres, so exp 18 is the published stack minus that one head. It is the control
for any later run that restores a head of our own.

Everything else is sanitising text that names people or notebooks. The mount
paths that spell out the dataset owner are rewritten to find the owner directory
at run time, which resolves to the same path on Kaggle and falls through to the
notebook's own /kaggle/input/<slug> candidates if that layout is absent.

The pristine source lives at external/gapstack_source/source.ipynb, which is
gitignored; its sha256 is checked below so a re-pull that changed is caught.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "external" / "gapstack_source" / "source.ipynb"
SOURCE_SHA256 = "6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d"
OUT_DIR = REPO / "notebooks" / "plateau_gapstack"
KERNEL_ID = "vyask21/cell-tracking-plateau-gapstack"

OWNER_ROOT = (
    "from pathlib import Path as _OwnerPath\n"
    "# Dataset owner directory, found at run time rather than spelled out.\n"
    "_DS_OWNER_ROOT = next((p.parent for p in sorted(_OwnerPath('/kaggle/input/datasets')"
    ".glob('*/biohub-deepcenter-unet3d-center-prior-v1'))), _OwnerPath('/kaggle/input/datasets/_'))\n"
)

# (label, regex, replacement, expected match count)
PATCHES: list[tuple[str, str, str, int]] = [
    ("owner-spelled f-string mount paths",
     r'f"/kaggle/input/datasets/[a-z0-9_-]+/', 'f"{_DS_OWNER_ROOT}/', 2),
    ("owner-spelled plain mount paths",
     r'(?<!f)"/kaggle/input/datasets/[a-z0-9_-]+/', 'f"{_DS_OWNER_ROOT}/', 5),
    ("branch tags in comments",
     r" \(agent/\w+\)", "", 5),
    ("remaining branch tag in a comment",
     r"agent/\w+ applies", "the candidate-cache patch applies", 1),
    ("version tag in a comment",
     r"x1\d\d v1 FAILED SILENTLY: my", "An earlier version FAILED SILENTLY: the", 1),
    ("guard report attribution string",
     r'"method_attribution": "[^"]*"', '"method_attribution": "credited in NOTES.md"', 1),
    ("guard report source string",
     r'"source_kernel": "[^"]*"', '"source_kernel": "credited in NOTES.md"', 1),
    ("coordinate head off: the published head is in a private dataset",
     r"_myhead = sorted\(.*?\n(?:.*\n)*?os\.environ\['V1284_HEAD'\]=str\(_myhead\[0\]\)\n",
     "# Coordinate head OFF. The published run loads a learned head from a private\n"
     "# dataset that is not available. 'zero' is the module's own passthrough, so\n"
     "# detections keep their native centres and the rest of the stack is unchanged.\n"
     "os.environ['V1284_MODE']='zero'\n", 1),
]

HEADER = """# Plateau, public gap-fill stack, coordinate head off

Built by `scripts/build_gapstack_kernel.py`, which holds every change and the
reason for it. This is public work run with one necessary change: the learned
coordinate head it loads is in a private dataset, so the module's own passthrough
mode is used instead. Sources and credit are in `NOTES.md`, 2026-09-23.
"""


def main() -> int:
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(f"source changed: {digest}")
    nb = json.loads(raw.decode("utf-8"))
    if any(c["cell_type"] != "code" for c in nb["cells"]):
        raise SystemExit("expected code cells only in the source")

    joined = "\n#<<CELL>>\n".join("".join(c["source"]) for c in nb["cells"])
    for label, pattern, repl, expected in PATCHES:
        joined, n = re.subn(pattern, lambda _m, r=repl: r, joined)
        if n != expected:
            raise SystemExit(f"patch {label!r}: matched {n}, expected {expected}")
        print(f"applied x{n}: {label}")
    if "V1284_MODE']='candidate'" in joined:
        raise SystemExit("head still in candidate mode")

    cells = joined.split("\n#<<CELL>>\n")
    cells[0] = OWNER_ROOT + cells[0]
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
    (OUT_DIR / "gapstack.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads(
        (REPO / "notebooks" / "plateau_dc025" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "gapstack.ipynb"
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'gapstack.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
