"""Screen graph-calibration arms against cached detections and affinities.

The expensive half of the pipeline is cached by `cache_graphs.py`, so an arm here
costs an ILP solve plus some graph surgery instead of a U-Net pass. That is what
makes it affordable to obey the one-variable rule across a chain of five or six
steps rather than importing the whole chain and hoping.

Each arm is a named dict of calibration settings. Every arm runs on the same 19
samples from the same cache, so the comparison is paired and the bootstrap below
is the paired one: the sample-to-sample spread that dominates an unpaired
estimate cancels, which is why this can resolve a 0.01 effect on 19 samples when
an unpaired screen needs many more.

    .venv\\Scripts\\python.exe scripts/screen_calibration.py --cache data/meta/graph_cache_t099

Read the caveat printed at the end before quoting anything from here. These 19
are video-disjoint from the support pack's training set but not embryo-disjoint,
and exp 4 measured a local gain of +0.1270 arriving as +0.0200 on the leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.cache_graphs import HELDOUT, cache_path, load_cache  # noqa: E402

# Arms are cumulative on purpose where a step only makes sense on top of another,
# and isolated where it does not. `base` reproduces what exp 5 submitted, so
# every delta below is against a graph we have a leaderboard score for.
# The shipped configuration, conf/unet50_calib_div_gap5.yaml, exp 9, LB 0.878.
# Arms below that start from this are one-variable changes against something the
# leaderboard has scored, which is the only comparison worth making now.
V9: dict = {
    "max_edge_um": 14.0, "gate_um": 14.0,
    "prune_isolated": True,
    "gap_close": True, "gap_close_um": 5.0,
    "min_track_len": 6,
    "linefit_smooth": True, "linefit_weight": 0.8, "linefit_window": 2,
    "safe_divisions": True, "safe_div_divergence_um": 4.0,
}

# V9 plus the one change the 2026-08-31 re-screen settled: the short-track
# filter at 10 rather than the 6 inherited from the public notebooks. The curve
# is 0.9043 at 6, 0.9125 at 10, 0.9128 at 12, 0.9037 at 16 and 0.7713 at 30, so
# 10 sits on the safe side of a cliff rather than on top of it.
V10: dict = V9 | {"min_track_len": 10}

# V10 plus the two division gates that survived the 2026-09-01 screen. Symmetry
# at 0.6 cut false forks 31 to 11 on its own; three-frame daughter persistence
# takes it to 10 and the pair scores 0.9201. Five-frame persistence and adding
# the cosine gate both reach 0.9208 with identical division counts, so the
# cosine gate is subsumed by symmetry and the extra two frames buy one false
# fork. Both of those are inside noise and both are less forgiving of a
# fragmented graph, which the hidden embryo will have more of than this set.
V11: dict = V10 | {"safe_div_min_symmetry": 0.6, "safe_div_child_len": 3}
V11_NOGEO: dict = V11 | {"safe_divisions": False}

ARMS: dict[str, dict] = {
    "base": {"max_edge_um": 7.0},
    "edge14": {"max_edge_um": 14.0},
    "prune": {"max_edge_um": 7.0, "prune_isolated": True},
    "short6": {"max_edge_um": 7.0, "prune_isolated": True, "min_track_len": 6},
    "gap": {"max_edge_um": 7.0, "prune_isolated": True, "gap_close": True},
    "smooth": {"max_edge_um": 7.0, "prune_isolated": True, "linefit_smooth": True},
    "gap_short6": {"max_edge_um": 7.0, "prune_isolated": True, "gap_close": True,
                   "min_track_len": 6},
    "all": {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
            "min_track_len": 6, "linefit_smooth": True},
    # Divisions. `all` forfeits the whole 0.1 term by construction, so this is
    # the first arm in the competition that can score on it at all.
    "all_div": {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
                "min_track_len": 6, "linefit_smooth": True,
                "safe_divisions": True},
    # Gap radius. The public notebook's effective one-frame gap radius is
    # GAP_CLOSE_UM * (gap + 1) = 5.8 * 2 = 11.6 um; ours is a flat 6.0. Its
    # GAP_CLOSE_MAX_GAP of 2 is dead config, clamped by
    # `effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)`, and its separate gap-2
    # recovery path is off. So the radius is the real difference, not the span.
    "all_gap8":   {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
                "min_track_len": 6, "linefit_smooth": True, "gap_close_um": 8.0},
    "all_gap116": {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
                "min_track_len": 6, "linefit_smooth": True, "gap_close_um": 11.6},
    "all_gap14":  {"max_edge_um": 14.0, "prune_isolated": True,
                "min_track_len": 6, "linefit_smooth": True, "gap_close": True, "gap_close_um": 14.0},
    # The radius curve is monotonically decreasing from 6.0 upward, so the
    # optimum may be below where it was set. `all_nogap` is the control that says
    # whether the step is worth having at all inside the finished chain, which
    # the original screen only established on top of `prune`.
    "all_gap5":   {"max_edge_um": 14.0, "prune_isolated": True,
                "min_track_len": 6, "linefit_smooth": True, "gap_close": True, "gap_close_um": 5.0},
    "all_gap4":   {"max_edge_um": 14.0, "prune_isolated": True,
                "min_track_len": 6, "linefit_smooth": True, "gap_close": True, "gap_close_um": 4.0},
    # The two believed changes together, to check they do not interact. Gap
    # closing and safe divisions both compete for the same unlinked detections,
    # so a tighter gap radius leaves more orphans for the division rule and the
    # combination is not guaranteed to be the sum.
    "all_div_gap5": {"max_edge_um": 14.0, "prune_isolated": True,
                     "gap_close": True, "gap_close_um": 5.0,
                     "min_track_len": 6, "linefit_smooth": True,
                     "safe_divisions": True, "safe_div_divergence_um": 4.0},
    "all_nogap":  {"max_edge_um": 14.0, "prune_isolated": True,
                "min_track_len": 6, "linefit_smooth": True, "gap_close": False},
    # The 0.926 notebook runs much tighter geometry than the 0.927 one: parent
    # 4.7 and sister 7.2 against 8.0 and 11.0. With 43 false divisions against 3
    # true at the loose setting, tighter is the obvious direction to test.
    "all_div_tight": {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
                "min_track_len": 6, "linefit_smooth": True,
                "safe_divisions": True,
                      "safe_div_parent_um": 4.7, "safe_div_sister_um": 7.2},
    # Divergence is the discriminator, so raising it is the other lever that
    # attacks false positives without touching how near a sister must be.
    "all_div_div4": {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
                "min_track_len": 6, "linefit_smooth": True,
                "safe_divisions": True, "safe_div_divergence_um": 4.0},
    "all_div_both": {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
                "min_track_len": 6, "linefit_smooth": True,
                "safe_divisions": True,
                     "safe_div_parent_um": 4.7, "safe_div_sister_um": 7.2,
                     "safe_div_divergence_um": 4.0},
    # ---------------------------------------------------------------------
    # Re-screen of the inherited chain against the FINISHED chain [2026-08-31].
    #
    # Every arm in the 2026-08-20 screen was measured on top of a partial chain.
    # Gap closing lost 85% of its measured value once the short-track filter
    # arrived, and NOTES.md records the open worry that `prune`, `short6` and
    # `smooth` may be carrying similarly inflated numbers. These arms move one
    # setting at a time against `v9`, which is exactly conf/unet50_calib_div_gap5
    # and therefore has a leaderboard score of 0.878 behind it.
    #
    # All of them keep `gate_um` at 14 so they reuse the cached solve.
    "v9": V9,
    "v9_short4":    V9 | {"min_track_len": 4},
    "v9_short5":    V9 | {"min_track_len": 5},
    "v9_short7":    V9 | {"min_track_len": 7},
    "v9_short8":    V9 | {"min_track_len": 8},
    "v9_short10":   V9 | {"min_track_len": 10},
    "v9_nosmooth":  V9 | {"linefit_smooth": False},
    "v9_smw05":     V9 | {"linefit_weight": 0.5},
    "v9_smw10":     V9 | {"linefit_weight": 1.0},
    "v9_smwin1":    V9 | {"linefit_window": 1},
    "v9_smwin3":    V9 | {"linefit_window": 3},
    # Output cap alone, candidate gate held at 14 so the solve is shared. The
    # 14 um cap came from the public notebooks and has never been screened as a
    # cap; the 7.0 that `screen_link_cap.py` measured in 2026-08-18 was a gate on
    # the local-max detector and does not carry.
    "v9_cap10":     V9 | {"max_edge_um": 10.0, "gate_um": 14.0},
    "v9_cap12":     V9 | {"max_edge_um": 12.0, "gate_um": 14.0},
    "v9_cap16":     V9 | {"max_edge_um": 16.0, "gate_um": 14.0},
    "v9_noprune":   V9 | {"prune_isolated": False},
    "v9_reuse20":   V9 | {"gap_reuse_um": 2.0},
    "v9_reuse45":   V9 | {"gap_reuse_um": 4.5},
    "v9_gapcap10":  V9 | {"gap_max_added_frac": 0.10},
    "v9_nosingle":  V9 | {"single_parent": False},

    # ---------------------------------------------------------------------
    # How far does the node-count bonus go [2026-08-31]?
    #
    # `adj_J = max(0, J * (1 - 0.1 * (N_pred - N_true) / N_true))` has no upper
    # cap, so predicting FEWER nodes than the organisers' estimate multiplies the
    # Jaccard by more than one. At `all` the weighted ratio is +0.152, meaning we
    # hand back 1.5% of the edge term for over-detection. The short-track filter
    # is the only step that moves the ratio without moving the true positives:
    # 6 to 8 dropped 13,031 nodes and 9 true positives. So the sweep continues
    # until the true positives start to pay for it.
    "v9_short12":   V9 | {"min_track_len": 12},
    "v9_short16":   V9 | {"min_track_len": 16},
    "v9_short20":   V9 | {"min_track_len": 20},
    "v9_short30":   V9 | {"min_track_len": 30},

    # ---------------------------------------------------------------------
    # Division precision [2026-08-31]. The rule returns 3 true forks against 30
    # false ones, and on that ratio precision is worth about three times recall:
    # removing every false positive takes divJ from 0.061 to 0.158, converting
    # one more miss takes it to 0.082. Every gate the rule currently has is a
    # distance, and the dominant failure mode is distance-blind: a neighbouring
    # cell whose own parent link was missed sits near the track, continues, and
    # diverges. These arms test geometry that a missed link cannot fake.
    "v9_cos0":      V9 | {"safe_div_max_cos": 0.0},
    "v9_cosm03":    V9 | {"safe_div_max_cos": -0.3},
    "v9_cos05":     V9 | {"safe_div_max_cos": 0.5},
    "v9_sym04":     V9 | {"safe_div_min_symmetry": 0.4},
    "v9_sym06":     V9 | {"safe_div_min_symmetry": 0.6},
    "v9_rel10":     V9 | {"safe_div_sister_rel": 1.0},
    "v9_rel15":     V9 | {"safe_div_sister_rel": 1.5},
    "v9_len3":      V9 | {"safe_div_child_len": 3},
    "v9_len5":      V9 | {"safe_div_child_len": 5},
    "v9_cos0_sym04": V9 | {"safe_div_max_cos": 0.0, "safe_div_min_symmetry": 0.4},
    "v9_cos0_len3": V9 | {"safe_div_max_cos": 0.0, "safe_div_child_len": 3},
    # Precision gates buy room to widen the distance gates, which is the only
    # way recall moves. Judged on the total, not on the fork count.
    "v9_cos0_wide": V9 | {"safe_div_max_cos": 0.0, "safe_div_parent_um": 10.0,
                          "safe_div_sister_um": 14.0,
                          "safe_div_frame_frac": 0.015},

    # ---------------------------------------------------------------------
    # Division gates, combined, on top of the settled track filter [2026-09-01].
    # Screened singly against V9 the two that work are symmetry at 0.6, which cut
    # false forks 31 to 11 while keeping all three true ones, and the cosine gate
    # with three-frame daughter persistence, which reached the same division
    # Jaccard while also gaining a true fork. These arms ask whether they are the
    # same effect twice or two effects.
    "v10": V10,
    "v10_sym06":        V10 | {"safe_div_min_symmetry": 0.6},
    "v10_cos0len3":     V10 | {"safe_div_max_cos": 0.0, "safe_div_child_len": 3},
    "v10_sym06len3":    V10 | {"safe_div_min_symmetry": 0.6, "safe_div_child_len": 3},
    "v10_sym06len5":    V10 | {"safe_div_min_symmetry": 0.6, "safe_div_child_len": 5},
    "v10_all3":         V10 | {"safe_div_min_symmetry": 0.6, "safe_div_max_cos": 0.0,
                               "safe_div_child_len": 3},
    # Precision bought room; this spends it on recall. `cos0_wide` failed against
    # V9 at -0.0043 with 45 false forks, but cosine alone was the weakest of the
    # three gates. Widened under the strict pair instead.
    "v10_strictwide":   V10 | {"safe_div_min_symmetry": 0.6, "safe_div_child_len": 3,
                               "safe_div_parent_um": 10.0, "safe_div_sister_um": 14.0,
                               "safe_div_frame_frac": 0.015},

    # ---------------------------------------------------------------------
    # The ILP can produce divisions and has never been allowed to [2026-08-31].
    #
    # The solver minimises cost. An edge costs `-1.0 * p`, a track start costs
    # +0.1, and a division costs +1.0, so forking is chosen only when
    # `p2 + 0.1 > 1.0`, that is when the second child's affinity exceeds 0.9.
    # A softmax over a dense candidate matrix essentially never puts 0.9 on a
    # second child, which is why every submission this repo has made carried
    # zero ILP forks and why the division term had to be bolted on afterwards.
    # Probed on 6bba_7b5d3b2c, 6374 nodes: 0 forks at 1.0, 38 at 0.7, 98 at 0.5,
    # 188 at 0.3, 304 at 0.15. The 38 is close to what the public notebook's own
    # division-rate cap implies for that video, and the solve is no slower.
    #
    # This matters because the geometric rule is blind to the affinity model and
    # the solver is not. Each weight needs its own solve, so the paired arms
    # share one: `_geo` keeps the bolt-on rule on top, the bare arm turns it off
    # so the solver's forks are judged alone.
    "v11": V11,
    "divw09":      V11_NOGEO | {"ilp": {"division_weight": 0.9}},
    "divw09_geo":  V11 | {"ilp": {"division_weight": 0.9}},
    "divw08":      V11_NOGEO | {"ilp": {"division_weight": 0.8}},
    "divw08_geo":  V11 | {"ilp": {"division_weight": 0.8}},
    "divw065":     V11_NOGEO | {"ilp": {"division_weight": 0.65}},
    "divw065_geo": V11 | {"ilp": {"division_weight": 0.65}},
    "divw05":      V11_NOGEO | {"ilp": {"division_weight": 0.5}},
    "divw05_geo":  V11 | {"ilp": {"division_weight": 0.5}},
    "divw035":     V11_NOGEO | {"ilp": {"division_weight": 0.35}},
    "divw035_geo": V11 | {"ilp": {"division_weight": 0.35}},
    "divw02":      V11_NOGEO | {"ilp": {"division_weight": 0.2}},
    "divw02_geo":  V11 | {"ilp": {"division_weight": 0.2}},

    # ---------------------------------------------------------------------
    # Dangling branches, which the component filter cannot reach [2026-09-01].
    #
    # `min_track_len` deletes weakly connected components, so a spurious
    # three-node chain hanging off a two-hundred-node track is part of a
    # two-hundred-and-three node component and no threshold will ever remove it.
    # That chain is exactly what the node-count term charges for, and the node
    # count term is now known to be where this pipeline's remaining slack is:
    # `min_track_len` 6 to 10 was worth +0.0082 and almost none of it was the
    # edge term. `filter_short_branches` splits the graph into tracklets and
    # removes the dangling ones, leaving internal tracklets alone however short,
    # because cutting one turns one edge error into two.
    "v11_br3":  V11 | {"min_branch_len": 3},
    "v11_br4":  V11 | {"min_branch_len": 4},
    "v11_br5":  V11 | {"min_branch_len": 5},
    "v11_br6":  V11 | {"min_branch_len": 6},
    "v11_br8":  V11 | {"min_branch_len": 8},
    "v11_br10": V11 | {"min_branch_len": 10},
    # If dangling branches are the junk, the component filter may be able to
    # come back down once they are gone, which would recover the true positives
    # that 10 costs. Two variables on purpose, and only read if the branch
    # filter earns its place on its own first.
    "v11_br5_short6": V11 | {"min_branch_len": 5, "min_track_len": 6},
    "v11_br5_short8": V11 | {"min_branch_len": 5, "min_track_len": 8},

    # Objective arms. SUPERSEDED 2026-09-14 and kept only so the numbers in
    # NOTES.md have their definitions. Every arm below sets min_track_len 6 with
    # no division gates, which was the submission candidate when they were
    # written and has been two chain steps stale since exp 8. Division tp is 0
    # and fn is 19 across all of them, so the division term, a tenth of the
    # metric, contributes nothing to any number they produced. Read their deltas
    # as directional only. The v11_* arms below are the live ones.
    "all_app0": {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
                 "min_track_len": 6, "linefit_smooth": True,
                 "ilp": {"appearance_weight": 0.0}},
    "all_disapp15": {"max_edge_um": 14.0, "prune_isolated": True,
                     "gap_close": True, "min_track_len": 6,
                     "linefit_smooth": True,
                     "ilp": {"disappearance_weight": 1.5}},
    "all_both": {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
                 "min_track_len": 6, "linefit_smooth": True,
                 "ilp": {"appearance_weight": 0.0,
                         "disappearance_weight": 1.5}},
    "all_disapp20": {"max_edge_um": 14.0, "prune_isolated": True,
                     "gap_close": True, "min_track_len": 6,
                     "linefit_smooth": True,
                     "ilp": {"disappearance_weight": 2.0}},
    "all_disapp25": {"max_edge_um": 14.0, "prune_isolated": True,
                     "gap_close": True, "min_track_len": 6,
                     "linefit_smooth": True,
                     "ilp": {"disappearance_weight": 2.5}},
    "all_disapp30": {"max_edge_um": 14.0, "prune_isolated": True,
                     "gap_close": True, "min_track_len": 6,
                     "linefit_smooth": True,
                     "ilp": {"disappearance_weight": 3.0}},

    # Objective arms on V11, which is conf/unet50_divsym.yaml, exp 11, LB 0.892.
    # Use `v11` as the reference arm, never `all`.
    #
    # v11_div12 is the one genuinely unexplored direction. The division weight
    # was screened on 2026-09-01 at 0.9, 0.8, 0.65, 0.5, 0.35 and 0.2 and
    # rejected monotonically, but every one of those is BELOW 1.0. The 0.947
    # public notebook runs BIOHUB_ILP_DIVISION_WEIGHT at 1.2. The axis has only
    # ever been walked downwards from the default.
    # `v11` itself is already defined above and is the reference arm.
    # Motion relink, the last step in the public chain this repo had not built.
    # Screened as a radius curve for the same reason the gap radius was: the
    # public value is 6.0 tight with a 10.0 relaxed tier behind a learned bonus,
    # and the one time this repo took a public radius on faith, 11.6 for gap
    # closing, its own screen put the optimum at 5.0 and the public value cost
    # 0.008. These arms have no learned bonus, so they are the tight tier alone
    # and the curve decides where it sits.
    #
    # Every arm shares V11's objective, so they all reuse the cached ILP solve
    # and the whole curve costs graph surgery only.
    "v11_relink3":  V11 | {"motion_relink": True, "motion_relink_um": 3.0},
    "v11_relink4":  V11 | {"motion_relink": True, "motion_relink_um": 4.0},
    "v11_relink5":  V11 | {"motion_relink": True, "motion_relink_um": 5.0},
    "v11_relink6":  V11 | {"motion_relink": True, "motion_relink_um": 6.0},
    "v11_relink8":  V11 | {"motion_relink": True, "motion_relink_um": 8.0},
    "v11_relink10": V11 | {"motion_relink": True, "motion_relink_um": 10.0},

    # Pricing divisions in the solver, revisited with a filter rather than an
    # additive rule. The 2026-09-01 screen rejected every weight from 0.9 to 0.2
    # because false forks went 10 to 803 while false negatives went 16 to 7. Its
    # arms ran the geometric rule as an ADDITIVE step, so the solver's own forks
    # were never put through the symmetry test that removes two thirds of the
    # geometric rule's false forks. These pair each weight with and without the
    # filter, so the filter's effect is isolated rather than confounded with the
    # weight. All the solves are already cached from that screen, so this costs
    # graph surgery only.
    #
    # `v11_filt` is the control: the shipped chain with the filter on. Its forks
    # all come from the geometric rule and have already passed symmetry 0.6, so
    # anything other than a near-null result here means the filter is reading the
    # geometry differently from the gate and one of the two is wrong.
    "v11_filt":        V11 | {"filter_divisions": True},
    "v11_ilp09":       V11 | {"ilp": {"division_weight": 0.9}},
    "v11_ilp09_filt":  V11 | {"filter_divisions": True,
                              "ilp": {"division_weight": 0.9}},
    "v11_ilp065":      V11 | {"ilp": {"division_weight": 0.65}},
    "v11_ilp065_filt": V11 | {"filter_divisions": True,
                              "ilp": {"division_weight": 0.65}},
    "v11_ilp05":       V11 | {"ilp": {"division_weight": 0.5}},
    "v11_ilp05_filt":  V11 | {"filter_divisions": True,
                              "ilp": {"division_weight": 0.5}},
    "v11_ilp05_filt3": V11 | {"filter_divisions": True, "div_filter_child_len": 3,
                              "ilp": {"division_weight": 0.5}},

    # Chain settings most coupled to the node count, kept ready for a re-screen
    # on a cache whose detections have moved. Eight-view TTA cuts detections by
    # about 9% on the probe, and the 2026-09-01 entry established that the
    # short-track filter pays through the node count term rather than the edge
    # term, so its optimum is a function of how many nodes there are. Running
    # these on the old cache would just reproduce numbers already in this file;
    # they exist for the new one. Gap closing is here for the same reason, since
    # it invents nodes and its radius was chosen against the old density.
    "v11_short8":   V11 | {"min_track_len": 8},
    "v11_short12":  V11 | {"min_track_len": 12},
    "v11_gap4":     V11 | {"gap_close_um": 4.0},
    "v11_gap6":     V11 | {"gap_close_um": 6.0},

    "v11_div12":    V11 | {"ilp": {"division_weight": 1.2}},
    "v11_div15":    V11 | {"ilp": {"division_weight": 1.5}},
    "v11_disapp20": V11 | {"ilp": {"disappearance_weight": 2.0}},
    "v11_app0":     V11 | {"ilp": {"appearance_weight": 0.0}},
}

# The pack's objective, and the value every arm uses unless it says otherwise.
# The public 0.927 notebook runs appearance 0.0 and disappearance 1.5 against
# these, which is the change these arms exist to test.
DEFAULT_ILP: dict[str, float] = {
    "edge_weight": -1.0,
    "appearance_weight": 0.1,
    "disappearance_weight": 0.1,
    "division_weight": 1.0,
}

FIELDS = ["arm", "sample", "embryo", "edge_tp", "edge_fp", "edge_fn",
          "division_tp", "division_fp", "division_fn",
          "num_pred_nodes", "node_recall", "total_node_ratio", "edge_jaccard",
          "adj_edge_jaccard", "seconds"]


def ilp_key(ilp_cfg: dict) -> str:
    """Directory suffix for a solve, covering every weight the solve depends on.

    The first version of this keyed the cached solve on the candidate gate alone.
    That was correct only while every arm solved with the same objective. The
    moment an arm changes a weight, a gate-only key hands back the solve from a
    different objective and the arm reports no change, which reads as a clean
    negative result rather than as a cache collision. The key now covers the
    weights, so a new objective gets a new directory.
    """
    d = DEFAULT_ILP | dict(ilp_cfg)
    parts = [f"{k}{d[k]:g}" for k in sorted(DEFAULT_ILP)]
    return "_".join(parts)


def run_one(arm: str, cfg: dict, sample: str, cache_dir: str, data_dir: str,
            ilp_cfg: dict, threads: int, solve_timeout: float = 1800.0) -> dict:
    import torch
    torch.set_num_threads(max(1, threads))

    from src import metrics
    from src.data import DEFAULT_SCALE_ZYX, read_scale
    from src.link_ilp import link_sequence_ilp
    from src.postprocess import calibrate

    t0 = time.time()
    detections, affinities, scale = load_cache(cache_path(cache_dir, sample))

    # The cache is built with a wide gate so the gate itself can be screened.
    # Narrowing it here gives exactly the candidate set that gating at this value
    # during scoring would have produced, since the gate is a pure distance
    # filter applied after the probabilities were computed.
    # `gate_um` separates two settings that were previously one. `max_edge_um`
    # is the output cap applied by `enforce_edge_rules`; the gate is the radius
    # the candidate set is narrowed to before the solve. Tying them meant every
    # cap value needed its own 20-minute solve, and it also confounded two
    # variables: a cap arm was really testing "narrower candidates AND a
    # narrower output filter". An arm that sets `gate_um` keeps the cached solve
    # and moves the cap alone.
    gate = float(cfg.get("gate_um", cfg.get("max_edge_um", 7.0)))
    scale_a = np.asarray(scale, dtype=np.float64)
    narrowed = []
    for t, aff in enumerate(affinities):
        if aff["i"].size == 0:
            narrowed.append(aff)
            continue
        a, b = detections[t], detections[t + 1]
        d = np.linalg.norm((a[aff["i"]] - b[aff["j"]]) * scale_a[None, :], axis=1)
        keep = d <= gate
        narrowed.append({"i": aff["i"][keep], "j": aff["j"][keep],
                         "p": aff["p"][keep]})

    # The ILP depends only on the candidate set, so every arm sharing a gate
    # shares a solve. Six of the eight arms use 7 um, and the solve is the
    # expensive part at up to ten minutes on the largest video, so caching it by
    # gate turns eight solves into two.
    from src.data import Graph, Nodes

    solve_dir = os.path.join(cache_dir, f"ilp_gate{gate:g}_{ilp_key(ilp_cfg)}")
    legacy_dir = os.path.join(cache_dir, f"ilp_gate{gate:g}")
    if (not os.path.exists(os.path.join(solve_dir, sample + ".npz"))
            and ilp_key(ilp_cfg) == ilp_key({})
            and os.path.exists(os.path.join(legacy_dir, sample + ".npz"))):
        # Solves cached before the key covered the objective were all at the
        # pack defaults, so they are reusable, but only under that exact key.
        solve_dir = legacy_dir
    os.makedirs(solve_dir, exist_ok=True)
    solved_path = os.path.join(solve_dir, sample + ".npz")
    if os.path.exists(solved_path):
        z = np.load(solved_path)
        graph = Graph(
            nodes=Nodes(ids=z["ids"], t=z["t"], z=z["z"], y=z["y"], x=z["x"]),
            edges=z["edges"],
        )
    else:
        w = DEFAULT_ILP | dict(ilp_cfg)
        graph = link_sequence_ilp(
            detections, narrowed, scale,
            edge_weight=float(w["edge_weight"]),
            appearance_weight=float(w["appearance_weight"]),
            disappearance_weight=float(w["disappearance_weight"]),
            division_weight=float(w["division_weight"]),
            num_threads=1, gap=0.0, timeout=solve_timeout,
        )
        tmp = solved_path + ".tmp.npz"
        np.savez_compressed(
            tmp, ids=np.asarray(graph.nodes.ids), t=np.asarray(graph.nodes.t),
            z=np.asarray(graph.nodes.z), y=np.asarray(graph.nodes.y),
            x=np.asarray(graph.nodes.x), edges=graph.edges,
        )
        os.replace(tmp, solved_path)

    graph, _ = calibrate(graph, scale,
                         {k: v for k, v in cfg.items() if k != "gate_um"})

    gt = os.path.join(data_dir, sample + ".geff")
    sc = read_scale(os.path.join(data_dir, sample + ".zarr")) or DEFAULT_SCALE_ZYX
    s = metrics.score_prediction(graph, gt, sample=sample, scale=sc)
    return {
        "arm": arm, "sample": sample, "embryo": sample.split("_")[0],
        "edge_tp": s.edge_tp, "edge_fp": s.edge_fp, "edge_fn": s.edge_fn,
        "division_tp": s.division_tp, "division_fp": s.division_fp,
        "division_fn": s.division_fn,
        "num_pred_nodes": s.num_pred_nodes, "node_recall": s.node_recall,
        "total_node_ratio": s.total_node_ratio, "edge_jaccard": s.edge_jaccard,
        "adj_edge_jaccard": s.adj_edge_jaccard,
        "seconds": round(time.time() - t0, 1),
    }


def weighted(rows: list[dict], key: str = "adj_edge_jaccard") -> float:
    w = np.array([int(r["edge_tp"]) + int(r["edge_fp"]) + int(r["edge_fn"])
                  for r in rows], dtype=float)
    v = np.array([float(r[key]) for r in rows], dtype=float)
    return float(np.sum(v * w) / np.sum(w))


def division_jaccard(rows: list[dict]) -> float:
    """Micro-averaged over pooled counts, which is how the organisers compute it.

    Not a weighted mean of per-sample division Jaccards. Divisions are rare
    enough that most samples have a denominator of a handful or of zero, so
    averaging per sample would let a sample with one division and one hit count
    as much as a sample with thirty. `NOTES.md` records the distinction; this is
    the code that has to honour it.
    """
    tp = sum(int(r["division_tp"]) for r in rows)
    fp = sum(int(r["division_fp"]) for r in rows)
    fn = sum(int(r["division_fn"]) for r in rows)
    den = tp + fp + fn
    return float(tp) / den if den else 0.0


def total_score(rows: list[dict]) -> float:
    """The competition metric: weighted adjusted edge Jaccard plus 0.1 division.

    The screen reported only the edge term until 2026-08-29, which was fine while
    every arm forfeited divisions identically and useless the moment an arm
    started adding them.
    """
    return weighted(rows) + 0.1 * division_jaccard(rows)


def paired_bootstrap(a: list[dict], b: list[dict], n: int = 20000, seed: int = 0):
    """Paired because both arms ran the same samples from the same cache."""
    rng = np.random.default_rng(seed)
    idx = [rng.choice(len(a), len(a), replace=True) for _ in range(n)]
    boot = np.array([total_score([b[i] for i in ix]) - total_score([a[i] for i in ix])
                     for ix in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (total_score(b) - total_score(a), float(lo), float(hi),
            float((boot > 0).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/meta/graph_cache_t099")
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--out", default="data/meta/calibration_screen.csv")
    ap.add_argument("--arms", default="", help="comma separated subset of arms")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--solve-timeout", type=float, default=1800.0,
                    help="seconds before the ILP is refused. Division arms add "
                         "a binary per node and a constraint per fork, so they "
                         "need more head room than the 1800 the pack defaults "
                         "to.")
    args = ap.parse_args()

    arms = ({k: ARMS[k] for k in args.arms.split(",")} if args.arms else ARMS)
    missing = [s for s in HELDOUT if not os.path.exists(cache_path(args.cache, s))]
    if missing:
        raise SystemExit(f"cache incomplete, {len(missing)} samples missing: "
                         f"{missing[:3]}")

    print(f"{len(arms)} arms x {len(HELDOUT)} samples, cache {args.cache}\n",
          flush=True)
    results: dict[str, list[dict]] = {}
    failures: list[tuple[str, str, str]] = []
    t0 = time.time()
    for arm, cfg in arms.items():
        # An arm's "ilp" block is the objective; everything else is calibration.
        ilp_cfg = dict(cfg.get("ilp", {}))
        cfg = {k: v for k, v in cfg.items() if k != "ilp"}
        rows: list[dict] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one, arm, cfg, s, args.cache, args.data,
                                ilp_cfg, args.threads, args.solve_timeout): s
                    for s in HELDOUT}
            for f in as_completed(futs):
                try:
                    rows.append(f.result())
                except Exception as exc:  # noqa: BLE001
                    # A refused solve is the expected failure here: the ILP
                    # raises rather than returning a truncated answer, because a
                    # truncated branch-and-bound scores worse than not solving
                    # at all. Losing one sample is a hole in one arm; losing the
                    # run loses every arm behind it, and objective arms cost
                    # about half an hour each.
                    failures.append((arm, futs[f], repr(exc)))
                    print(f"    FAILED {arm} {futs[f]}: {exc}", flush=True)
        if not rows:
            print(f"  {arm:12} every sample failed, skipped", flush=True)
            continue
        rows.sort(key=lambda r: r["sample"])
        results[arm] = rows
        print(f"  {arm:12} score {total_score(rows):.4f}  "
              f"adj_J {weighted(rows):.4f}  "
              f"divJ {division_jaccard(rows):.4f} "
              f"({sum(int(r['division_tp']) for r in rows)}tp/"
              f"{sum(int(r['division_fp']) for r in rows)}fp/"
              f"{sum(int(r['division_fn']) for r in rows)}fn)  "
              f"nodes {sum(int(r['num_pred_nodes']) for r in rows):8d}  "
              f"tp {sum(int(r['edge_tp']) for r in rows):6d}  "
              f"fp {sum(int(r['edge_fp']) for r in rows):5d}  "
              f"fn {sum(int(r['edge_fn']) for r in rows):5d}  "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for rows in results.values():
            w.writerows(rows)

    # The reference is the first arm run, not an arm that happens to be called
    # "base". Hardcoding the name meant a screen whose arms were named anything
    # else printed its per-arm scores and silently no statistics at all, which is
    # how the 2026-08-28 objective screen finished with no interval on any of its
    # three deltas. A screen that cannot say whether a delta clears the noise has
    # not measured anything.
    if len(results) > 1:
        ref_name = next(iter(results))
        print(f"\n{'arm':14}{'adj J':>9}{'delta':>10}{'95% CI':>22}{'P(>0)':>8}"
              f"{'both embryos':>14}")
        print(f"(reference arm: {ref_name})")
        base = results[ref_name]
        for arm, rows in results.items():
            if arm == ref_name:
                print(f"{arm:14}{total_score(rows):>9.4f}{'':>10}{'':>22}{'':>8}")
                continue
            d, lo, hi, p = paired_bootstrap(base, rows)
            per = []
            for emb in ("44b6", "6bba"):
                a = [r for r in base if r["embryo"] == emb]
                b = [r for r in rows if r["embryo"] == emb]
                per.append(total_score(b) - total_score(a))
            both = "yes" if all(x > 0 for x in per) else "no"
            print(f"{arm:14}{total_score(rows):>9.4f}{d:>+10.4f}"
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p:>8.3f}{both:>14}")

    if failures:
        print(f"\n{len(failures)} sample-arm pairs FAILED and their arms are "
              "scored on fewer samples, so they are NOT comparable:")
        for arm, sample, exc in failures:
            print(f"  {arm} {sample}: {exc}")

    print("\nThe 19 are video-disjoint from the pack's training set but NOT "
          "embryo-disjoint,\nand the hidden test is. Exp 4 saw +0.1270 here "
          "arrive as +0.0200 on the\nleaderboard. Treat any delta as a direction "
          "and an upper bound.")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min, wrote {args.out}")


if __name__ == "__main__":
    main()
