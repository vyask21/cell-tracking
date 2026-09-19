"""Build the second screen: how far down does the DeepCenter division veto go.

    python scripts/build_screen2_kernel.py

A patch on `notebooks/plateau_base/anchor.ipynb`, same as the first screen, so
the anchor stays byte identical and obviously theirs.

The first screen ran twelve arms on 2026-09-19 and eleven of them lost. The one
that moved was the DeepCenter safe-division threshold, and it moved regularly:

    0.20 (base)   8 tp / 14 fp / 26 fn    divJ 0.1667
    0.15          9 tp / 15 fp / 25 fn    divJ 0.1837
    0.10         10 tp / 16 fp / 24 fn    divJ 0.2000

Exactly one more true and one more false positive per step. Pooled Jaccard is
tp / (tp + fp + fn), so admitting one of each moves the numerator by one and the
denominator by one, which raises the ratio while the ratio is below one. That is
why it improves, and it is also why it cannot keep improving: the moment the
newly admitted divisions stop being about half right, the denominator grows
faster than the numerator and the curve turns over.

Nobody knows where it turns over, because 0.10 is the lowest value anyone has
run. This screen finds out. Four arms below 0.10, plus 0.10 itself as a
reproducibility control against the first screen's number, which is worth its
eighteen minutes: if the control does not land on 0.2000 exactly, something in
the cache or the environment moved and nothing else in the run can be trusted.

Cheaper than the first screen by design. Five arms rather than twelve, and the
arms are CPU on graphs that are already predicted, so the cost is roughly the
70 minute prediction stage plus about 18 minutes an arm.

Three patches, all shared with the first screen and all for the same reasons:
tight55 folded into the base so every arm is one variable against what exp 12
shipped; the held-out set at 24 videos rather than 8, because 13 ground truth
divisions cannot support a claim about divisions; and automatic selection off,
because this is a screen and whatever ships later ships because we read the
per-sample table.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_base" / "anchor.ipynb"
OUT_DIR = REPO / "notebooks" / "plateau_screen2"
KERNEL_ID = "vyask21/cell-tracking-plateau-screen2"

# Ordered cheapest first, same discipline as the first screen. Lowering the veto
# widens the set of proposals that survive it, so the lowest threshold does the
# most work in the loop below it and goes last.
ARMS = {
    # The control. Must reproduce 10 tp / 16 fp / 24 fn and divJ 0.2000.
    "div_dc010": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.10},
    "div_dc0075": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.075},
    "div_dc005": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.05},
    "div_dc0025": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.025},
    "div_dc001": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.01},
}

PATCHES: list[tuple[str, str, str]] = [
    (
        "fold in tight55, the one base arm that improved both embryos",
        "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '6.0'",
        "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '5.5'",
    ),
    (
        "held-out set from 8 videos to 24",
        "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
        "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '12'",
    ),
    (
        "disable automatic arm selection, this run is a screen",
        "os.environ['BIOHUB_PPSWEEP_SELECT_MARGIN'] = '0.001'",
        "os.environ['BIOHUB_PPSWEEP_SELECT_MARGIN'] = '9.0'",
    ),
]

HEADER = """# Plateau screen 2, the DeepCenter division veto below 0.10

Built from `notebooks/plateau_base/anchor.ipynb` by `scripts/build_screen2_kernel.py`,
which holds the list of changes and the reason for each.

The base is the public 0.947 chain and it is not ours. Weights, ILP and the repair
chain are srcA's; the 0.938 and 0.940 steps are srcD's; bidirectional
harmonic fusion is srcC's; 0.941 through 0.947 is srcB's; the
three public dataset path substitutions are srcE's. The DeepCenter veto is
theirs too. What is ours is the screen around it.

This run is a screen and not a submission. Automatic arm selection is off, so the
submission it writes is the base, and the thing worth reading is
`validator_results.csv`.
"""


def main() -> int:
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if len(code) != 1:
        raise SystemExit(f"expected one code cell, found {len(code)}")
    src = "".join(code[0]["source"])

    for label, old, new in PATCHES:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"patch {label!r}: anchor matched {n} times, expected 1")
        src = src.replace(old, new)
        print(f"applied: {label}")

    old_line = [ln for ln in src.splitlines()
                if ln.startswith("PP_CANDIDATES: dict[str, dict] = {")]
    if len(old_line) != 1:
        raise SystemExit("could not locate PP_CANDIDATES")
    src = src.replace(old_line[0], "PP_CANDIDATES: dict[str, dict] = " + json.dumps(ARMS))
    print(f"applied: {len(ARMS)} sweep arms")

    code[0]["source"] = src.splitlines(keepends=True)
    header = nb["cells"][0]
    if header["cell_type"] != "markdown":
        raise SystemExit("expected the provenance cell first")
    header["source"] = HEADER.splitlines(keepends=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "screen2.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads(
        (REPO / "notebooks" / "plateau_base" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "screen2.ipynb"
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'screen2.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
