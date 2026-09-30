"""Build the exp 16 submission kernel: the DeepCenter veto raised, not lowered.

    python scripts/build_dc025_kernel.py

One variable against exp 12, which scored 0.947: DEEPCENTER_SAFE_DIV_THRESHOLD
0.20 to 0.25. A HIGHER threshold means the veto rejects MORE proposed divisions,
so this ships fewer divisions at higher precision. It is the exact opposite of
exp 13, which lowered the same constant to 0.10 and lost 0.002.

The evidence is a leaderboard position rather than a local screen, which is
deliberate after three local readings failed. `srcG/fusion-source`
is a public notebook on the same branch as our anchor, published 2026-09-21 and
carrying 105 votes, and its author sits at rank 247 with a verified 0.948. Its
effective configuration was diffed against our anchor: of eight differing keys,
five are sweep machinery or artifact paths, LEAF_PRUNE_MIN_EDGE_PROB is 0.0 which
disables leaf pruning, and the one substantive difference is this threshold at
0.25 against our 0.20.

Their notebook's own extended sweep then tries lowering it again, to 0.15 and to
0.10, alongside a `divwide` arm at SAFE_DIV_MAX_UM 11.0 and SAFE_DIV_SISTER_MAX_UM
16.0, a sym075 arm and a diverge150 arm. Those are the same four ideas this repo
screened and submitted as exps 13 and 14, and they lost. The author gates them
behind a prefix guard, so whether any ship depends on their validator. What is
not conditional is their base, and their base is 0.25.

This is a leaderboard probe, not a validated gain. There is no local number beside
it and there is not one: the local instrument has now been wrong
about this exact constant once already, reading exp 13's 0.10 as +0.00333 when it
was worth -0.002.

Patches are the same four as the exp 13 kernel, including moving the anchor's own
_EXPECTED_NUMERIC guard to 0.25 rather than removing it, and switching the
held-out validator off since it cannot affect the submission and costs rerun time.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_base" / "anchor.ipynb"
OUT_DIR = REPO / "notebooks" / "plateau_dc025"
KERNEL_ID = "vyask21/cell-tracking-plateau-dc025"

PATCHES: list[tuple[str, str, str]] = [
    (
        "fold in tight55, so this is one variable against exp 12",
        "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '6.0'",
        "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '5.5'",
    ),
    (
        "DeepCenter safe-division veto 0.20 to 0.25, the one variable",
        "os.environ['BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD'] = '0.20'",
        "os.environ['BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD'] = '0.25'",
    ),
    (
        "move the anchor's own config guard to match, rather than removing it",
        "'BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD': 0.20,",
        "'BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD': 0.25,",
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
    (OUT_DIR / "dc025.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads(
        (REPO / "notebooks" / "plateau_base" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "dc025.ipynb"
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'dc025.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
