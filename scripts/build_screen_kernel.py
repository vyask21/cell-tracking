"""Build the screening kernel from the unmodified public base.

    python scripts/build_screen_kernel.py

`notebooks/plateau_base/anchor.ipynb` is someone else's notebook and is kept
exactly as published, so that it stays obviously theirs and so that any later
disagreement about what the anchor scored can be settled by reading it. This
script is the whole of what this repo changes on top of it, written as a patch
rather than as a second copy, so a reader can see our contribution without
diffing two 200 KB notebooks.

What it changes, and why each one:

1. `MOTION_RELINK_TIGHT_UM` 6.0 to 5.5. The base's own sweep selected this on its
   eight held-out videos and it is the only one of its seven arms that improved
   both embryos. Folding it into the base means every arm below is one variable
   against something that has already earned its place.
2. `VALIDATOR_N_PER_TYPE` 4 to 12, so the held-out set goes from 8 videos to 24.
   The eight contain thirteen ground-truth divisions between them, which is the
   same small-denominator trap this repo walked into on exp 8. Nothing about
   divisions can be believed on a denominator of thirteen.
3. `PPSWEEP_SELECT_MARGIN` 0.001 to 9.0, which disables automatic selection. This
   run is a screen, not a submission. Whatever ships later ships because we chose
   it after reading the per-sample table, not because the notebook's own rule
   took the argmax.
4. The divergence rejection counter is split into its three causes. The base
   increments one counter for three different things: a sister with no successor,
   a grandchild at the wrong timepoint, and a separation below the threshold. One
   video logs 989 of them against 97 surviving candidates, and without the split
   there is no way to know which gate is doing the work.
5. A ranking mode for the safe-division proposer. The base ranks candidate forks
   by `parent_dist + 0.15 * sister_dist` ascending and spends a capped budget on
   the tightest pairs, which are duplicate detections rather than divisions. This
   adds a symmetry ranking, which is the one division signal this repo measured
   itself and found discriminative on 2026-09-01. The default stays `geometry`,
   so the base behaviour is untouched.
6. The sweep gains the arms in `ARMS` below, and the three keys they need.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "notebooks" / "plateau_base" / "anchor.ipynb"
OUT_DIR = REPO / "notebooks" / "plateau_screen"
KERNEL_ID = "vyask21/cell-tracking-plateau-screen"

# Twelve arms, each one variable against the base. Budget is about 22 minutes of
# CPU per arm over 24 cached graphs, with no GPU, so the sweep is about 4.5 hours
# on top of a 70 minute prediction stage.
#
# Order matters and it is deliberate. The arms are sorted by how much they widen
# the candidate pool, cheapest first. Every candidate that clears the geometry is
# put through the DeepCenter veto, so an arm that opens a gate pays for it in the
# veto loop, and the two at the bottom could be several times the cost of the
# ones above. Running them last means a slow arm costs only itself rather than
# taking the track length curve down with it.
ARMS = {
    # No effect on the division pool at all, so these cost the base rate. The
    # curve was measured at 10 on this repo's own graph, which sat at a node
    # ratio of +0.148; the base sits at -0.031, so the optimum can move and this
    # is a measurement rather than a transfer.
    "len8": {"OUTPUT_MIN_TRACK_LEN": 8},
    "len10": {"OUTPUT_MIN_TRACK_LEN": 10},
    "len12": {"OUTPUT_MIN_TRACK_LEN": 12},
    # Same pool, later gate. DeepCenter rejects 68 of the 97 candidates that
    # clear the geometry, so this moves the largest single filter.
    "div_dc015": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.15},
    "div_dc010": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.10},
    # Same pool again: the ranker changes which proposals win the budget, not how
    # many are proposed.
    "rank_sym": {"SAFE_DIV_RANK_MODE": "symmetry"},
    # From here the pool widens. The divergence gate rejects roughly ten
    # candidates for every one it passes, so relaxing it sends more through the
    # veto.
    "div_dv15": {"SAFE_DIV_DIVERGE_UM": 1.5},
    "div_dv10": {"SAFE_DIV_DIVERGE_UM": 1.0},
    "div_dv00": {"SAFE_DIV_DIVERGE_UM": 0.0},
    "rank_sym_dv10": {"SAFE_DIV_RANK_MODE": "symmetry", "SAFE_DIV_DIVERGE_UM": 1.0},
    "div_dvoff": {"SAFE_DIV_REQUIRE_DIVERGENCE": 0},
    # The widest arm in the set, and the one the published analysis points at:
    # open the pool and let an evidence ranker spend the budget. Last on purpose.
    "rank_sym_open": {"SAFE_DIV_RANK_MODE": "symmetry", "SAFE_DIV_DIVERGE_UM": 1.0,
                      "SAFE_DIV_REQUIRE_MUTUAL_NN": 0},
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
    (
        "rank mode global",
        "SAFE_DIV_REQUIRE_MUTUAL_NN = os.environ.get('BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN', '1') != '0'",
        "SAFE_DIV_REQUIRE_MUTUAL_NN = os.environ.get('BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN', '1') != '0'\n"
        "SAFE_DIV_RANK_MODE = os.environ.get('BIOHUB_SAFE_DIV_RANK_MODE', 'geometry')",
    ),
    (
        "stats keys for the split divergence counter",
        "'safe_division_symmetry_rejected': 0, 'deepcenter_gap_checked': 0",
        "'safe_division_symmetry_rejected': 0, 'safe_division_div_no_successor': 0, "
        "'safe_division_div_bad_frame': 0, 'safe_division_div_below_threshold': 0, "
        "'deepcenter_gap_checked': 0",
    ),
    (
        "divergence counter, cause one: sister has no single successor",
        "                    if len(c1_succ) != 1 or len(q_succ) != 1:\n"
        "                        stats['safe_division_divergence_rejected'] += 1\n"
        "                        continue",
        "                    if len(c1_succ) != 1 or len(q_succ) != 1:\n"
        "                        stats['safe_division_divergence_rejected'] += 1\n"
        "                        stats['safe_division_div_no_successor'] += 1\n"
        "                        continue",
    ),
    (
        "divergence counter, cause two: grandchild at the wrong timepoint",
        "                        stats['safe_division_divergence_rejected'] += 1\n"
        "                        continue\n"
        "                    grandchild_dist = edge_distance_um(c1_grandchild, q_grandchild)",
        "                        stats['safe_division_divergence_rejected'] += 1\n"
        "                        stats['safe_division_div_bad_frame'] += 1\n"
        "                        continue\n"
        "                    grandchild_dist = edge_distance_um(c1_grandchild, q_grandchild)",
    ),
    (
        "divergence counter, cause three: separation below the threshold",
        "                    if grandchild_dist - sister_dist < SAFE_DIV_DIVERGE_UM:\n"
        "                        stats['safe_division_divergence_rejected'] += 1\n"
        "                        continue",
        "                    if grandchild_dist - sister_dist < SAFE_DIV_DIVERGE_UM:\n"
        "                        stats['safe_division_divergence_rejected'] += 1\n"
        "                        stats['safe_division_div_below_threshold'] += 1\n"
        "                        continue",
    ),
    (
        "the ranker",
        "                score = parent_dist + 0.15 * sister_dist\n"
        "                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))",
        "                _sym = abs(child_dist - parent_dist) / max((child_dist + parent_dist) / 2.0, 1e-6)\n"
        "                if SAFE_DIV_RANK_MODE == 'symmetry':\n"
        "                    score = _sym\n"
        "                else:\n"
        "                    score = parent_dist + 0.15 * sister_dist\n"
        "                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))",
    ),
    (
        "three new sweepable keys",
        "'GAP_CLOSE_REUSE_UM', 'OUTPUT_EDGE_MAX_UM']",
        "'GAP_CLOSE_REUSE_UM', 'OUTPUT_EDGE_MAX_UM', 'SAFE_DIV_REQUIRE_DIVERGENCE', "
        "'SAFE_DIV_REQUIRE_MUTUAL_NN', 'SAFE_DIV_RANK_MODE']",
    ),
]

HEADER = """# Plateau screen

Built from `notebooks/plateau_base/anchor.ipynb` by `scripts/build_screen_kernel.py`,
which is where the list of changes and the reason for each one lives. The provenance
of everything underneath is in the anchor and is unchanged.

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

    # The candidate set is replaced wholesale rather than appended to. Its seven
    # arms were measured on 2026-09-18 and five of them moved nothing; of the two
    # that did, tight55 is now in the base and relaxed9 lost on both terms.
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
    (OUT_DIR / "screen.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads(
        (REPO / "notebooks" / "plateau_base" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[1]
    meta["code_file"] = "screen.ipynb"
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'screen.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
