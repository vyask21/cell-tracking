"""Build the exp 14 submission kernel: the two division gates that bind together.

    python scripts/build_divpair_kernel.py

A patch on `notebooks/plateau_base/anchor.ipynb`, the same way the other kernels
here are, so the anchor stays byte identical and obviously theirs.

Two variables against exp 12, which scored 0.947, and the fact that it is two
rather than one is deliberate and is stated up front rather than buried:

    SAFE_DIV_SISTER_MAX_UM        14.0 to 16.0
    SAFE_DIV_REQUIRE_MUTUAL_NN    on to off

Screened separately on 2026-09-20 in screen 3, each is worth about +0.0013 and
each adds exactly one true division. Together they are worth +0.00412 and add
three. That super-additivity is the reason to ship the pair rather than either
half: the mutual-nearest-neighbour gate allows at most one sister per parent, and
the 14 um sister cap decides which sisters qualify, so relaxing one leaves the
other binding and most of the gain unreachable. Shipping one half would measure
the wrong thing.

    arm                          delta    95% CI                P(>0)   div
    combo(sister16+mutual_off) +0.00412  [-0.00086,+0.01131]    0.875  11/23/33
    sister16                   +0.00133  [-0.00065,+0.00482]    0.640   9/22/35
    mutual_off                 +0.00129  [-0.00064,+0.00491]    0.644   9/22/35
    base                        0                                       8/21/36

INCONCLUSIVE BY THE REPO'S RULE. The interval crosses zero, and the both-embryos
half of the rule cannot run at all because the screen is single-embryo by design.
It is submitted anyway, and the reason is different from the one that failed on
exp 13, which matters more than the fact that both were inconclusive.

Exp 13 moved a threshold on a fitted model and measured it on that model's own
training data. The instrument was invalid for the question. These two constants
are pure geometry with no learned parameters, and they were measured on the 24
`6bba` videos, which is DeepCenter's held-out split entire. Screen 3 also carried
`div_dc010` as a calibration arm precisely to test the instrument against a known
leaderboard outcome: the old mixed validator read that arm at +0.00333 with
P=0.869 and the leaderboard said -0.002, while this validator reads it at +0.00036
with P=0.560. It no longer endorses the arm that lost. That is not proof the
instrument tracks the leaderboard, but it is the first evidence any instrument
here has offered.

What remains is that this is a probe and not a validated
gain, and the leaderboard is now the better instrument for it. Cost of being
wrong is one submission and a rerun; public rank cannot fall, because the board
keeps the best submission and ours is exp 12 at 0.947.

Four patches.

1. `MOTION_RELINK_TIGHT_UM` 6.0 to 5.5, which is what the anchor's own sweep
   selected and therefore what exp 12 shipped. Folding it in is what makes this a
   comparison against 0.947 rather than against something never submitted.
2. The two constants. `BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN` has no assignment in the
   anchor and falls back to its default of on, so the line is added next to the
   sister cap, inside the same environment block and well before either is read.
3. Neither constant appears in the anchor's `_EXPECTED_NUMERIC` guard, so unlike
   the DeepCenter threshold in exp 13 there is nothing to move. The guard is left
   exactly as it is and should still print PASS.
4. The held-out validator is switched off, as in exp 13. It scores train videos
   and selects a sweep arm, neither of which touches the submission, and at rerun
   it is charged against the same wall clock as the test prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_base" / "anchor.ipynb"
OUT_DIR = REPO / "notebooks" / "plateau_divpair"
KERNEL_ID = "vyask21/cell-tracking-plateau-divpair"

PATCHES: list[tuple[str, str, str]] = [
    (
        "fold in tight55, so this is measured against what exp 12 shipped",
        "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '6.0'",
        "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '5.5'",
    ),
    (
        "sister cap 14 to 16 um, and the mutual-NN gate off, the two variables",
        "os.environ['BIOHUB_SAFE_DIV_SISTER_MAX_UM'] = '14.0'",
        "os.environ['BIOHUB_SAFE_DIV_SISTER_MAX_UM'] = '16.0'\n"
        "os.environ['BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN'] = '0'",
    ),
    (
        "validator off, it cannot affect the submission and it costs rerun time",
        "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
        "os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'\n"
        "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
    ),
]

HEADER = """# Plateau, sister cap 16 um with the mutual-NN gate off

Built from `notebooks/plateau_base/anchor.ipynb` by `scripts/build_divpair_kernel.py`,
which holds the list of changes and the reason for each.

The base is the public 0.947 chain and it is not ours. Weights, ILP and the repair
chain are srcA's; the 0.938 and 0.940 steps are srcD's; bidirectional
harmonic fusion is srcC's; 0.941 through 0.947 is srcB's; the
three public dataset path substitutions are srcE's. The safe-division cascade
and both constants changed here are theirs.

What is ours is the measurement that chose them: a screen over the 24 `6bba`
videos, which the split manifest shipped with the public DeepCenter weights shows
to be that model's held-out split entire, scored with a paired bootstrap rather
than a point estimate.

Two variables against the run that scored 0.947, shipped as a pair because they
bind on each other: `SAFE_DIV_SISTER_MAX_UM` 14.0 to 16.0, and
`SAFE_DIV_REQUIRE_MUTUAL_NN` off.
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
    src = src.replace(old_line[0], "PP_CANDIDATES: dict[str, dict] = {}")
    print("applied: empty sweep, this run ships its base")

    code[0]["source"] = src.splitlines(keepends=True)
    header = nb["cells"][0]
    if header["cell_type"] != "markdown":
        raise SystemExit("expected the provenance cell first")
    header["source"] = HEADER.splitlines(keepends=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "divpair.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads(
        (REPO / "notebooks" / "plateau_base" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "divpair.ipynb"
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'divpair.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
