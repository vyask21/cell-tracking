"""Build the exp 13 submission kernel: the validated base with one constant moved.

    python scripts/build_dc010_kernel.py

This is a patch on `notebooks/plateau_base/anchor.ipynb`, the same way
`build_screen_kernel.py` is, so the anchor stays byte identical and obviously
theirs. Credit for everything underneath is in the header cell below.

One variable against exp 12, which scored 0.947. That variable is
`DEEPCENTER_SAFE_DIV_THRESHOLD`, their DeepCenter veto on proposed divisions,
0.20 down to 0.10.

Why this one and nothing else. The 2026-09-19 screen ran twelve arms over 24
held-out videos and this was the only one that moved the score up. On the repo's
own rule it is inconclusive, delta +0.00333 with a 95% interval of
[-0.00098, +0.00946] and P(>0) = 0.869, and it passes the both-embryos half at
+0.0055 and +0.0027. It is submitted anyway on the argument recorded in advance
for exp 8 and again for exp 11: the division term has 34 ground truth events in
the local set against about 58 videos in the public split, so for this one term
the leaderboard is the better instrument. That argument held both times.

The whole of the gain is divisions. Weighted adjusted edge moves by -0.000002,
which is nothing, and the division counts go 8/14/26 to 10/16/24. The mechanism
is regular over three thresholds, 0.20 then 0.15 then 0.10 giving exactly one
more true and one more false positive per step, which is also the warning: the
pooled Jaccard rises only while each newly admitted division is about half right.

Four patches, and the third is the one to read carefully.

1. `MOTION_RELINK_TIGHT_UM` 6.0 to 5.5, which is what the anchor's own sweep
   selected and therefore what exp 12 actually shipped. Folding it in is what
   makes this a one-variable change rather than a two-variable one.
2. The threshold itself.
3. The anchor's configuration guard pins that threshold at 0.20 in
   `_EXPECTED_NUMERIC` and raises RuntimeError if the environment disagrees. The
   guard is moved to 0.10 rather than removed, so it still fails loudly on any
   drift we did not intend. Removing the check would have been the easier patch
   and the wrong one.
4. The held-out validator is switched off. It scores train videos and selects a
   sweep arm, neither of which touches the submission, and at rerun it is charged
   against the same wall clock as the test prediction. Exp 12 took about 8.5 h of
   a 12 h ceiling carrying a validator and seven arms. Dropping it is the cheapest
   available reduction in rerun risk, and it costs nothing we use.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_base" / "anchor.ipynb"
OUT_DIR = REPO / "notebooks" / "plateau_dc010"
KERNEL_ID = "vyask21/cell-tracking-plateau-dc010"

PATCHES: list[tuple[str, str, str]] = [
    (
        "fold in tight55, so this is one variable against exp 12",
        "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '6.0'",
        "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '5.5'",
    ),
    (
        "DeepCenter safe-division veto 0.20 to 0.10, the one variable",
        "os.environ['BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD'] = '0.20'",
        "os.environ['BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD'] = '0.10'",
    ),
    (
        "move the anchor's own config guard to match, rather than removing it",
        "'BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD': 0.20,",
        "'BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD': 0.10,",
    ),
    (
        "validator off, it cannot affect the submission and it costs rerun time",
        "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
        "os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'\n"
        "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
    ),
]

HEADER = """# Plateau, DeepCenter division veto at 0.10

Built from `notebooks/plateau_base/anchor.ipynb` by `scripts/build_dc010_kernel.py`,
which holds the list of changes and the reason for each.

The base is the public 0.947 chain and it is not ours. Weights, ILP and the repair
chain are srcA's; the 0.938 and 0.940 steps are srcD's; bidirectional
harmonic fusion is srcC's; 0.941 through 0.947 is srcB's; the
three public dataset path substitutions are srcE's. The DeepCenter veto and the
threshold being moved here are all theirs. What is ours is the measurement that
said to move it: a twelve-arm screen over 24 held-out videos, scored with a paired
bootstrap rather than a point estimate.

One variable against the run that scored 0.947: `DEEPCENTER_SAFE_DIV_THRESHOLD`
0.20 to 0.10.
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

    # No sweep. There is nothing to select between, and an empty candidate set
    # means the run writes the base submission and stops.
    old_line = [ln for ln in src.splitlines()
                if ln.startswith("PP_CANDIDATES: dict[str, dict] = {")]
    if len(old_line) != 1:
        raise SystemExit("could not locate PP_CANDIDATES")
    src = src.replace(old_line[0], "PP_CANDIDATES: dict[str, dict] = {}")
    print("applied: empty sweep, this run ships its base")

    code[0]["source"] = src.splitlines(keepends=True)
    header = nb["cells"][0]
    if header["cell_type"] != "markdown":
        raise SystemExit("expected the provenance cell first")
    header["source"] = HEADER.splitlines(keepends=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dc010.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads(
        (REPO / "notebooks" / "plateau_base" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "dc010.ipynb"
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'dc010.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
