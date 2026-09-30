"""Build screen 3: division geometry, measured on videos DeepCenter never trained on.

    python scripts/build_screen3_kernel.py

A patch on `notebooks/plateau_base/anchor.ipynb`, same as the first two screens,
so the anchor stays byte identical and obviously theirs.

Why this screen differs from the first two, and it is the validator rather than
the arms. Exp 13 moved the DeepCenter division veto from 0.20 to 0.10 on a local
reading of +0.00333 and lost 0.002 on the leaderboard. The manifests shipped with
the public weights say why, and they were downloadable the whole time:

    srcA/biohub-deepcenter-unet3d-center-prior-v1
        weights/full_frame_center/split_manifest.json
        all 199, train 71, val 128, and the split is perfectly by embryo:
        train is every 44b6 video, val is every 6bba video.

    srcA/biohub-temporal-unet3d-seed314159-v1
        weights/unet_transformer/split_0/split_manifest.json
        method unet_transformer_alltrain_seed314159_v1, train 199. All of them.

Two consequences, and the second is the one that cost a submission.

The edge model saw every video, so no local hold-out is out-of-sample for the
edge term. Nothing can fix that short of retraining, which is closed on cost.

DeepCenter saw the whole 44b6 embryo and none of 6bba. The first screen's 24
videos were 12 of each, so half the instrument was reading a model's performance
on its own training data. The gap is not subtle: base division Jaccard is 0.3158
on the 44b6 half and 0.0690 on the 6bba half, a factor of 4.6. This repo's own
rule, that both embryos must improve, made that worse rather than better here,
because it counted the memorised embryo as corroboration.

So the validator is restricted to 6bba. That is the only ground where a division
change can be measured without leakage, and there are 128 such videos to draw on.

`div_dc010` is carried in as a calibration arm and it is the point of the screen
as much as any new idea. Its leaderboard result is known and it is negative. An
instrument that still endorses it is still broken, and we should find that out
here rather than with another submission. On the 6bba half of screen 1 it read
+0.00269 with a 95% interval of [-0.00154, +0.00881] and P(>0) = 0.652, which is
inconclusive rather than correct, so the hope is that more videos turn
that into a verdict rather than a shrug.

The rest of the arms are pure geometry in the division proposal stage, each one
variable, and none of them has ever been screened by us or by the public sweep.

    SAFE_DIV_REQUIRE_MUTUAL_NN   NOTES documents this gate as structurally broken:
                                 it takes the single nearest candidate to the
                                 EXISTING child and rejects every other outright,
                                 so at most one sister can ever be proposed per
                                 parent, chosen by proximity to a sibling that may
                                 itself be wrong. It has only ever been tested
                                 switched off alongside two other changes, in
                                 rank_sym_open. Alone it has never been run.
    SAFE_DIV_SISTER_SYMMETRY_TAU 0.6 in their config and in ours, never varied by
                                 anyone. This repo's 2026-09-01 screen found
                                 symmetry to be the discriminative division
                                 feature, so the threshold is worth a curve.
    SAFE_DIV_MAX_UM              9.0, inherited.
    SAFE_DIV_SISTER_MAX_UM       14.0, inherited.
    SAFE_DIV_FRAME_FRAC_CAP      0.0076, the per-frame fork budget a public
                                 analysis names as the binding constraint. NOTES
                                 records that the per-frame cap breaks out of its
                                 loop without incrementing its counter, so
                                 safe_division_skipped_cap reads 0 whether or not
                                 it binds and nobody can currently tell.

Note what is NOT here. `DEEPCENTER_GAP_THRESHOLD` is a threshold on the same
fitted model and belongs to the same class of question that exp 13 got wrong, so
it stays out until there is an instrument that has earned trust.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_base" / "anchor.ipynb"
OUT_DIR = REPO / "notebooks" / "plateau_screen3"
KERNEL_ID = "vyask21/cell-tracking-plateau-screen3"

# Ordered cheapest first, as with the earlier screens: an arm that widens the
# candidate pool pays for it in the DeepCenter veto loop below it.
ARMS = {
    # Calibration. Known to be worth -0.002 on the leaderboard.
    "div_dc010": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.10},
    # Same pool, different gate threshold.
    "sym05": {"SAFE_DIV_SISTER_SYMMETRY_TAU": 0.5},
    "sym07": {"SAFE_DIV_SISTER_SYMMETRY_TAU": 0.7},
    # The budget. Doubling it is a blunt instrument and that is the point: if the
    # cap is not binding this arm is bit identical to base, which is itself the
    # answer to a question the shipped counter cannot answer.
    "framecap2x": {"SAFE_DIV_FRAME_FRAC_CAP": 0.0152},
    # Geometry, widening the pool.
    "parent11": {"SAFE_DIV_MAX_UM": 11.0},
    "sister16": {"SAFE_DIV_SISTER_MAX_UM": 16.0},
    # The broken gate, alone at last.
    "mutual_off": {"SAFE_DIV_REQUIRE_MUTUAL_NN": 0},
}

PATCHES: list[tuple[str, str, str]] = [
    (
        "fold in tight55, the one base arm that improved both embryos",
        "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '6.0'",
        "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '5.5'",
    ),
    (
        "24 held-out videos, all of them from the one embryo DeepCenter never saw",
        "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
        "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '24'",
    ),
    (
        "restrict the validator to 6bba, which is DeepCenter's val split entire",
        "candidates = [s for s in train_stems_all if s not in test_stem_set]",
        "candidates = [s for s in train_stems_all if s not in test_stem_set "
        "and s.startswith('6bba_')]",
    ),
    (
        "disable automatic arm selection, this run is a screen",
        "os.environ['BIOHUB_PPSWEEP_SELECT_MARGIN'] = '0.001'",
        "os.environ['BIOHUB_PPSWEEP_SELECT_MARGIN'] = '9.0'",
    ),
    (
        "make SAFE_DIV_REQUIRE_MUTUAL_NN sweepable alongside the rest",
        "'GAP_CLOSE_REUSE_UM', 'OUTPUT_EDGE_MAX_UM']",
        "'GAP_CLOSE_REUSE_UM', 'OUTPUT_EDGE_MAX_UM', 'SAFE_DIV_REQUIRE_MUTUAL_NN']",
    ),
]

HEADER = """# Plateau screen 3, division geometry on DeepCenter's own held-out embryo

Built from `notebooks/plateau_base/anchor.ipynb` by `scripts/build_screen3_kernel.py`,
which holds the list of changes and the reason for each.

The base is the public 0.947 chain and it is not ours. Weights, ILP and the repair
chain are srcA's; the 0.938 and 0.940 steps are srcD's; bidirectional
harmonic fusion is srcC's; 0.941 through 0.947 is srcB's; the
three public dataset path substitutions are srcE's. The safe-division cascade
and every constant screened here are theirs.

What is ours is the validator. The split manifests shipped with the public weights
show that DeepCenter trained on all 71 `44b6` videos and none of the 128 `6bba`,
while the secondary edge model trained on all 199. So this run scores only `6bba`
videos, which is the one place a division change can be measured on a model that
did not memorise the answer. Eric raised the same point in the competition
discussion on 2026-09-19 and it matches what our own exp 13 cost us.

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
    (OUT_DIR / "screen3.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads(
        (REPO / "notebooks" / "plateau_base" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "screen3.ipynb"
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'screen3.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
