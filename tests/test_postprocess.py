"""Graph calibration, tested on graphs small enough to reason about by hand.

Each step here changes the submitted graph directly, so a silent bug shows up as
a slightly worse score rather than as an error, which is the hardest kind to
notice. These pin the behaviour that the screens will then measure.
"""

from __future__ import annotations

import numpy as np

from src.data import Graph, Nodes
from src.postprocess import (
    calibrate,
    close_single_frame_gaps,
    enforce_edge_rules,
    filter_short_tracks,
    linefit_smooth,
    prune_isolated,
)

SCALE = (1.625, 0.40625, 0.40625)


def make(ts, zyx, edges):
    ts = np.asarray(ts, dtype=np.int64)
    zyx = np.asarray(zyx, dtype=float)
    return Graph(
        nodes=Nodes(ids=np.arange(len(ts), dtype=np.int64), t=ts,
                    z=zyx[:, 0], y=zyx[:, 1], x=zyx[:, 2]),
        edges=(np.asarray(edges, dtype=np.int64) if len(edges)
               else np.empty((0, 2), dtype=np.int64)),
    )


def chain(n, start_t=0, step=(0.0, 1.0, 0.0)):
    ts = [start_t + k for k in range(n)]
    zyx = [[k * step[0], k * step[1], k * step[2]] for k in range(n)]
    edges = [(k, k + 1) for k in range(n - 1)]
    return ts, zyx, edges


def test_backward_and_skipping_edges_are_dropped():
    g = make([0, 1, 3], [[0, 0, 0]] * 3, [(0, 1), (1, 0), (1, 2)])
    out = enforce_edge_rules(g, SCALE, max_edge_um=0)
    assert {tuple(e) for e in out.edges} == {(0, 1)}


def test_edges_longer_than_the_cap_are_dropped():
    # 40 voxels in y is 40 * 0.40625 = 16.25 um, over a 14 um cap.
    g = make([0, 1], [[0, 0, 0], [0, 40, 0]], [(0, 1)])
    assert enforce_edge_rules(g, SCALE, max_edge_um=14.0).edges.shape[0] == 0
    assert enforce_edge_rules(g, SCALE, max_edge_um=20.0).edges.shape[0] == 1


def test_a_node_keeps_only_its_closest_parent():
    g = make([0, 0, 1], [[0, 0, 0], [0, 10, 0], [0, 1, 0]], [(0, 2), (1, 2)])
    out = enforce_edge_rules(g, SCALE, max_edge_um=0, single_parent=True)
    assert {tuple(e) for e in out.edges} == {(0, 2)}


def test_a_one_frame_gap_is_bridged_with_a_new_node():
    # Two tracks, one ending at t=1 and one starting at t=3, close in space.
    g = make([0, 1, 3, 4], [[0, 0, 0], [0, 1, 0], [0, 3, 0], [0, 4, 0]],
             [(0, 1), (2, 3)])
    out = close_single_frame_gaps(g, SCALE, max_gap_um=6.0, reuse_um=0.0)
    assert len(out.nodes) == 5
    assert int(np.asarray(out.nodes.t)[4]) == 2
    assert {tuple(e) for e in out.edges} >= {(1, 4), (4, 2)}


def test_an_existing_detection_is_reused_instead_of_inventing_one():
    # A spare unlinked node sits at t=2 exactly at the midpoint. Reusing it adds
    # no node at all, which is what keeps the node-count penalty flat.
    g = make([0, 1, 2, 3, 4],
             [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0]],
             [(0, 1), (3, 4)])
    out = close_single_frame_gaps(g, SCALE, max_gap_um=6.0, reuse_um=3.2)
    assert len(out.nodes) == 5
    assert {tuple(e) for e in out.edges} >= {(1, 2), (2, 3)}


def test_a_gap_wider_than_the_cap_is_left_alone():
    g = make([0, 1, 3, 4], [[0, 0, 0], [0, 1, 0], [0, 60, 0], [0, 61, 0]],
             [(0, 1), (2, 3)])
    out = close_single_frame_gaps(g, SCALE, max_gap_um=6.0, reuse_um=0.0)
    assert len(out.nodes) == 4
    assert out.edges.shape[0] == 2


def test_gap_closing_respects_its_node_budget():
    ts, zyx, edges = [], [], []
    for k in range(20):
        base = len(ts)
        ts += [0, 1, 3, 4]
        zyx += [[k * 50, 0, 0], [k * 50, 1, 0], [k * 50, 3, 0], [k * 50, 4, 0]]
        edges += [(base, base + 1), (base + 2, base + 3)]
    g = make(ts, zyx, edges)
    # 20 bridgeable gaps available. The fraction would allow 40, the absolute
    # cap allows 3, and the smaller must win.
    out = close_single_frame_gaps(g, SCALE, max_gap_um=6.0, reuse_um=0.0,
                                  max_added_frac=0.5, max_added_abs=3)
    assert len(out.nodes) == len(g.nodes) + 3


def test_the_budget_floor_keeps_small_graphs_working():
    """int(frac * n) is 0 below 20 nodes, which silently disabled the step."""
    g = make([0, 1, 3, 4], [[0, 0, 0], [0, 1, 0], [0, 3, 0], [0, 4, 0]],
             [(0, 1), (2, 3)])
    out = close_single_frame_gaps(g, SCALE, max_gap_um=6.0, reuse_um=0.0,
                                  max_added_frac=0.05)
    assert len(out.nodes) == 5


def test_short_components_are_removed_whole():
    ts_a, zyx_a, e_a = chain(7)
    ts_b, zyx_b, e_b = chain(3)
    n = len(ts_a)
    g = make(ts_a + ts_b, zyx_a + [[100, v[1], v[2]] for v in zyx_b],
             e_a + [(a + n, b + n) for a, b in e_b])
    out = filter_short_tracks(g, min_len=6)
    assert len(out.nodes) == 7
    assert out.edges.shape[0] == 6


def test_a_component_exactly_at_the_minimum_survives():
    ts, zyx, e = chain(6)
    assert len(filter_short_tracks(make(ts, zyx, e), min_len=6).nodes) == 6


def test_isolated_nodes_are_pruned_but_linked_ones_stay():
    ts, zyx, e = chain(3)
    g = make(ts + [9], zyx + [[50, 50, 50]], e)
    out = prune_isolated(g)
    assert len(out.nodes) == 3
    assert out.edges.shape[0] == 2


def test_smoothing_pulls_an_outlier_back_onto_its_track():
    # A straight track in y with one node knocked sideways in z.
    ts = list(range(5))
    zyx = [[0, k, 0] for k in range(5)]
    zyx[2] = [4.0, 2.0, 0.0]
    g = make(ts, zyx, [(k, k + 1) for k in range(4)])
    out = linefit_smooth(g, weight=0.8, window=2)
    assert abs(float(np.asarray(out.nodes.z)[2])) < 4.0


def test_smoothing_leaves_track_ends_alone():
    ts = list(range(5))
    zyx = [[0, k, 0] for k in range(5)]
    zyx[0] = [4.0, 0.0, 0.0]
    g = make(ts, zyx, [(k, k + 1) for k in range(4)])
    out = linefit_smooth(g, weight=0.8, window=2)
    assert float(np.asarray(out.nodes.z)[0]) == 4.0


def test_smoothing_never_changes_topology():
    ts, zyx, e = chain(9)
    g = make(ts, zyx, e)
    out = linefit_smooth(g, weight=0.8, window=2)
    assert np.array_equal(out.edges, g.edges)
    assert len(out.nodes) == len(g.nodes)


def test_calibrate_with_everything_off_is_only_the_edge_rules():
    ts, zyx, e = chain(4)
    g = make(ts, zyx, e)
    out, _ = calibrate(g, SCALE, {"enforce_edge_rules": False})
    assert np.array_equal(out.edges, g.edges)
    assert len(out.nodes) == len(g.nodes)


def test_gaps_close_before_short_tracks_are_filtered():
    """Ordering, and it decides the outcome for this graph.

    Two components of four and three nodes separated by a one-frame gap. Closing
    the gap first makes one component of eight, which survives a minimum of six.
    Filtering first would delete both and leave nothing.
    """
    ts = [0, 1, 2, 3, 5, 6, 7]
    zyx = [[0, k, 0] for k in [0, 1, 2, 3, 5, 6, 7]]
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]
    g = make(ts, zyx, edges)
    out, stats = calibrate(g, SCALE, {
        "gap_close": True, "gap_close_um": 6.0, "gap_reuse_um": 0.0,
        "min_track_len": 6, "max_edge_um": 14.0,
    })
    assert len(out.nodes) == 8
    assert stats["short_track_nodes_removed"] == 0


# --- safe divisions ---------------------------------------------------------
#
# The public notebook that ships this records that its own purely geometric
# version produced hundreds of forks and not one true positive. The three
# structural constraints are what make it work, so each gets a test that shows it
# rejecting on its own, not just a happy path that passes.

from src.postprocess import add_safe_divisions  # noqa: E402

UM = 0.40625  # one voxel in y or x


def division_scene(
    parent_has_predecessor=True,
    sister_gap_vox=4,
    diverge=True,
    orphan_extra=(),
):
    """A parent at t1 with one child, plus an orphan that should become the second.

    Laid out in y only, so every distance is voxels * 0.40625 um and the
    thresholds are easy to reason about. Node order: predecessor, parent, child,
    orphan, child successor, orphan successor.
    """
    ts, zyx, edges = [], [], []

    def add(t, y):
        ts.append(t)
        zyx.append([0.0, float(y), 0.0])
        return len(ts) - 1

    pred = add(0, 0)
    parent = add(1, 0)
    child = add(2, 0)
    orphan = add(2, sister_gap_vox)
    spread = sister_gap_vox + (12 if diverge else 0)
    c_next = add(3, 0)
    o_next = add(3, spread)

    if parent_has_predecessor:
        edges.append((pred, parent))
    edges.append((parent, child))
    edges.append((child, c_next))
    edges.append((orphan, o_next))
    for y in orphan_extra:
        add(2, y)
    return make(ts, zyx, edges), parent, orphan


def test_a_clean_division_is_added():
    g, parent, orphan = division_scene()
    st = {}
    out = add_safe_divisions(g, SCALE, frame_frac_cap=1.0, global_frac_cap=1.0,
                             stats=st)
    assert st["safe_div_added"] == 1
    assert (parent, orphan) in {tuple(e) for e in out.edges}
    assert parent in set(out.divisions().tolist())


def test_a_track_start_is_never_given_a_second_child():
    """Constraint 1. Identical geometry, parent simply has no predecessor."""
    g, _, _ = division_scene(parent_has_predecessor=False)
    st = {}
    add_safe_divisions(g, SCALE, frame_frac_cap=1.0, global_frac_cap=1.0, stats=st)
    assert st["safe_div_added"] == 0


def test_a_pair_that_does_not_diverge_is_rejected():
    """Constraint 3, the discriminator. Sisters stay the same distance apart."""
    g, _, _ = division_scene(diverge=False)
    st = {}
    add_safe_divisions(g, SCALE, frame_frac_cap=1.0, global_frac_cap=1.0, stats=st)
    assert st["safe_div_added"] == 0


def test_an_orphan_nearer_another_orphan_than_the_child_is_rejected():
    """Constraint 2. A closer orphan wins the mutual-nearest test, and it fails
    the divergence check, so nothing is added rather than the wrong thing."""
    g, _, _ = division_scene(sister_gap_vox=8, orphan_extra=(1,))
    st = {}
    add_safe_divisions(g, SCALE, frame_frac_cap=1.0, global_frac_cap=1.0, stats=st)
    assert st["safe_div_added"] == 0


def test_a_sister_beyond_the_cap_is_rejected():
    # 40 voxels is 16.25 um, over the 11 um sister cap.
    g, _, _ = division_scene(sister_gap_vox=40)
    st = {}
    add_safe_divisions(g, SCALE, frame_frac_cap=1.0, global_frac_cap=1.0, stats=st)
    assert st["safe_div_added"] == 0


def test_both_caps_floor_at_one_rather_than_at_zero():
    """A zero fraction still permits exactly one, and that is deliberate.

    Both caps are `max(1, round(frac * size))`, matching the public
    implementation. The floor means a small graph is never silently barred from
    every division by rounding, and it means a fraction cannot be used as an
    off switch. `safe_divisions: false` is the off switch.
    """
    g, _, _ = division_scene()
    st = {}
    add_safe_divisions(g, SCALE, frame_frac_cap=0.0, global_frac_cap=0.0, stats=st)
    assert st["safe_div_added"] == 1


def test_the_global_cap_bounds_a_graph_with_several_candidates():
    """Two independent divisions, a cap that admits one."""
    ts, zyx, edges = [], [], []

    def add(t, y):
        ts.append(t); zyx.append([0.0, float(y), 0.0]); return len(ts) - 1

    for base in (0, 100):
        pred = add(0, base)
        parent = add(1, base)
        child = add(2, base)
        orphan = add(2, base + 4)
        c2 = add(3, base)
        o2 = add(3, base + 16)
        edges += [(pred, parent), (parent, child), (child, c2), (orphan, o2)]
    g = make(ts, zyx, edges)

    both = {}
    add_safe_divisions(g, SCALE, frame_frac_cap=1.0, global_frac_cap=1.0, stats=both)
    assert both["safe_div_added"] == 2

    capped = {}
    add_safe_divisions(g, SCALE, frame_frac_cap=1.0, global_frac_cap=0.13,
                       stats=capped)
    assert capped["safe_div_added"] == 1
    assert capped["safe_div_global_capped"] == 1


def test_an_orphan_is_claimed_by_at_most_one_parent():
    """Two parents, one orphan between them. Only one fork may be created."""
    ts, zyx, edges = [], [], []

    def add(t, y):
        ts.append(t); zyx.append([0.0, float(y), 0.0]); return len(ts) - 1

    pa, pb = add(0, 0), add(0, 8)
    a, b = add(1, 0), add(1, 8)
    ca, cb = add(2, 0), add(2, 8)
    orphan = add(2, 4)
    ca2, cb2 = add(3, 0), add(3, 8)
    o2 = add(3, 20)
    edges += [(pa, a), (pb, b), (a, ca), (b, cb), (ca, ca2), (cb, cb2),
              (orphan, o2)]
    g = make(ts, zyx, edges)
    st = {}
    out = add_safe_divisions(g, SCALE, frame_frac_cap=1.0, global_frac_cap=1.0,
                             stats=st)
    into_orphan = [e for e in out.edges if int(e[1]) == orphan]
    assert len(into_orphan) <= 1


def test_calibrate_leaves_divisions_off_unless_asked():
    g, _, _ = division_scene()
    off, _ = calibrate(g, SCALE, {"max_edge_um": 14.0})
    assert off.divisions().size == 0
    on, st = calibrate(g, SCALE, {"max_edge_um": 14.0, "safe_divisions": True,
                                  "safe_div_frame_frac": 1.0,
                                  "safe_div_global_frac": 1.0})
    assert on.divisions().size == 1
