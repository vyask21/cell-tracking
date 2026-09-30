"""Graph calibration: the steps that run after the linker.

This module exists because of a measurement, not a hunch. Reading the public code
on 2026-08-20 showed that the notebooks at 0.902 to 0.915 use the same weights we
use and the same ILP objective we use, and differ almost entirely in what they do
to the graph afterwards. The author of the 0.902 notebook says it plainly: the
model weights are unchanged and the score comes from calibrating the output graph.
Our pipeline stopped at the ILP and wrote the CSV. This is the missing half.

Every step here is written to be independently switchable, because the workspace
rule is one variable per config and because the point is to find out which of
these pay rather than to import a chain wholesale and hope.

The steps, in the order they must run:

1. `enforce_edge_rules`  drop edges that are not one frame forward, or longer
   than a cap, and reduce any node to a single parent.
2. `filter_asymmetric_divisions`  drop a fork whose two arms are too unequal
   to be a mitosis, by the same symmetry test that gates the geometric rule.
   Off by default, and only meaningful when the solver was allowed to fork.
3. `motion_relink`  look again at every track end, and join it to a track start
   one frame later if the cell's own velocity predicts that position. Adds no
   node and can neither fork a track nor give one a second parent, so the edge
   term is the only thing it moves.
4. `close_single_frame_gaps`  a track that vanishes at t and reappears at t+2
   gets a synthetic node at t+1, which converts one missing edge into two present
   ones.
5. `prune_isolated`  nodes with no edges at all are pure node-count penalty.
6. `filter_short_tracks`  whole components shorter than a minimum are mostly
   false positives, and they cost twice: their edges are FP and their nodes
   inflate the node-count ratio.
7. `add_safe_divisions`  give a node a second child where the geometry and the
   two daughters' subsequent divergence both say a mitosis happened. This is the
   only step that touches the division term, which a one-to-one linker forfeits
   entirely.
8. `linefit_smooth`  move each node toward a line fitted through its temporal
   neighbours. The metric matches predictions to ground truth by distance with a
   7 um cap, so moving a node a micron closer can flip it from unmatched to
   matched, and matching is what edges are scored on.

Divisions were absent until 2026-08-29 and are now behind
`safe_divisions`, off by default. The bound of 0.02 to 0.04 still
stands as the ceiling; what changed is that the rest is settled and the term is
worth 0.1 of the available 1.1, which no submission of ours has ever scored on.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.data import Graph, Nodes


def _physical(zyx: np.ndarray, scale) -> np.ndarray:
    return zyx * np.asarray(scale, dtype=np.float64)[None, :]


def _edge_lengths_um(graph: Graph, scale) -> np.ndarray:
    if graph.edges.shape[0] == 0:
        return np.empty(0)
    zyx = np.stack([graph.nodes.z, graph.nodes.y, graph.nodes.x], axis=1)
    p = _physical(zyx, scale)
    return np.linalg.norm(p[graph.edges[:, 0]] - p[graph.edges[:, 1]], axis=1)


def enforce_edge_rules(
    graph: Graph,
    scale,
    max_edge_um: float = 14.0,
    single_parent: bool = True,
    stats: dict | None = None,
) -> Graph:
    """Keep only legal, plausible edges, and at most one parent per node.

    The length cap is a genuine open question rather than a copied constant.
    `screen_link_cap.py` measured 7.0 as correct on 2026-08-18, but that was the
    local-max detector with distance-based costs, and the public notebooks keep
    edges out to 14 um on top of a learned scorer. The value is a parameter and
    gets screened.

    Single-parent repair keeps the highest-probability incoming edge. The scorer
    caps in-degree implicitly by matching one prediction to one ground-truth node,
    so a second parent can only ever be a false positive.
    """
    st = stats if stats is not None else {}
    if graph.edges.shape[0] == 0:
        return graph

    t = np.asarray(graph.nodes.t)
    src, tgt = graph.edges[:, 0], graph.edges[:, 1]

    keep = t[tgt] == t[src] + 1
    st["dropped_nonconsecutive"] = int((~keep).sum())

    if max_edge_um and max_edge_um > 0:
        lengths = _edge_lengths_um(graph, scale)
        too_long = lengths > max_edge_um
        st["dropped_long"] = int((too_long & keep).sum())
        keep &= ~too_long

    edges = graph.edges[keep]

    if single_parent and edges.shape[0]:
        # Without edge probabilities the tie-break is edge length, shortest
        # first, which is the same ordering the assignment would have preferred.
        zyx = np.stack([graph.nodes.z, graph.nodes.y, graph.nodes.x], axis=1)
        p = _physical(zyx, scale)
        d = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)
        order = np.argsort(d, kind="stable")
        seen: set[int] = set()
        chosen = []
        for k in order:
            tid = int(edges[k, 1])
            if tid in seen:
                continue
            seen.add(tid)
            chosen.append(k)
        st["dropped_multi_parent"] = int(edges.shape[0] - len(chosen))
        edges = edges[np.sort(np.asarray(chosen, dtype=np.int64))]

    return Graph(nodes=graph.nodes, edges=edges)


def filter_asymmetric_divisions(
    graph: Graph,
    scale,
    min_symmetry: float = 0.6,
    min_child_len: int = 0,
    stats: dict | None = None,
) -> Graph:
    """Remove a fork whose two arms are too unequal to be a mitosis.

    The mirror of the gate inside `add_safe_divisions`, pointed the other way.
    That gate decides which forks to CREATE and is the reason the geometric rule
    scores 3 true forks against 10 false ones instead of 3 against 31. This one
    decides which existing forks to KEEP, and it exists because the 2026-09-01
    ILP screen left something untested.

    That screen priced divisions in the solver and rejected every weight from 0.9
    down to 0.2. Its finding was not that the solver cannot find divisions: false
    negatives fell 16 to 7, so the affinity model does know where some of the
    missed ones are. The finding was that precision collapses an order of
    magnitude faster, 10 false forks to 803. Every arm in that screen ran the
    geometric rule as an ADDITIVE step on top, never as a filter, so the solver's
    own forks were never subjected to the symmetry test at all. Whether the test
    that removes two thirds of the geometric rule's false forks also removes the
    solver's is a different question and this is what asks it.

    `min_symmetry` is the ratio of the shorter to the longer parent-to-daughter
    distance, the same quantity `add_safe_divisions` computes, so the two steps
    are testing one criterion and not two. When a fork fails, the longer arm is
    the edge dropped: a parent lies near the midpoint of its own daughters, so
    the child that sits far off is the one more likely to belong elsewhere.

    `min_child_len` additionally requires both daughters to survive that many
    frames, which is V11's other division gate. 0 disables it.

    A node with more than two children keeps its best-symmetry pair and loses the
    rest. Three outgoing edges is not a mitosis under any reading of the metric,
    which pairs one parent with two daughters.

    Nothing here removes nodes. Dropping a fork's arm leaves the child where it
    was, and `prune_isolated` and the short-track filter downstream decide
    whether it still belongs. That ordering is deliberate: the 2026-09-01 screen
    found that a cheap fork rescues an orphan and its whole fragment from the
    track filter, so a step that removes forks has to run before the steps that
    would have cleaned up after them.
    """
    st = stats if stats is not None else {}
    st.setdefault("div_filter_dropped", 0)
    st.setdefault("div_filter_forks_cut", 0)
    n = len(graph.nodes)
    if n == 0 or graph.edges.shape[0] == 0:
        return graph
    if min_symmetry <= 0.0 and min_child_len <= 0:
        return graph

    t = np.asarray(graph.nodes.t)
    zyx = np.stack([graph.nodes.z, graph.nodes.y, graph.nodes.x], axis=1)
    pos = _physical(zyx, scale)

    src = graph.edges[:, 0]
    children: dict[int, list[int]] = {}
    for k in range(graph.edges.shape[0]):
        children.setdefault(int(src[k]), []).append(int(graph.edges[k, 1]))

    succ = np.full(n, -1, dtype=np.int64)
    for s, cs in children.items():
        if len(cs) == 1:
            succ[s] = cs[0]

    def survives(node: int, frames: int) -> bool:
        """Follow single-child successors and report reaching `frames` steps."""
        cur, steps = node, 0
        while steps < frames:
            nxt = int(succ[cur])
            if nxt < 0:
                kids = children.get(cur, [])
                if len(kids) != 1:
                    return len(kids) > 1  # a fork counts as continuing
                nxt = kids[0]
            if int(t[nxt]) != int(t[cur]) + 1:
                return False
            cur, steps = nxt, steps + 1
        return True

    drop: set[tuple[int, int]] = set()
    for s, kids in children.items():
        if len(kids) < 2:
            continue
        best, best_sym = None, -1.0
        for a_i in range(len(kids)):
            for b_i in range(a_i + 1, len(kids)):
                a, b = kids[a_i], kids[b_i]
                da = float(np.linalg.norm(pos[a] - pos[s]))
                db = float(np.linalg.norm(pos[b] - pos[s]))
                if da <= 0.0 or db <= 0.0:
                    continue
                sym = min(da, db) / max(da, db)
                if sym > best_sym:
                    best, best_sym = (a, b, da, db), sym
        if best is None:
            continue
        a, b, da, db = best
        # Everything outside the best pair goes regardless of the threshold.
        for c in kids:
            if c not in (a, b):
                drop.add((s, c))
        fail = best_sym < min_symmetry
        if not fail and min_child_len > 0:
            fail = not (survives(a, min_child_len) and survives(b, min_child_len))
        if fail:
            drop.add((s, a if da > db else b))
            st["div_filter_forks_cut"] += 1

    if not drop:
        return graph
    keep = np.array(
        [(int(e[0]), int(e[1])) not in drop for e in graph.edges], dtype=bool
    )
    st["div_filter_dropped"] += int((~keep).sum())
    return Graph(nodes=graph.nodes, edges=graph.edges[keep])


def motion_relink(
    graph: Graph,
    scale,
    max_um: float = 6.0,
    window: int = 3,
    require_velocity: bool = False,
    max_added_frac: float = 0.05,
    max_added_abs: int = 2000,
    stats: dict | None = None,
) -> Graph:
    """Join a track end to a track start one frame later, at the position motion predicts.

    The last step in the public chain this repo had not built. The ILP decides
    every t to t+1 assignment from the edge head's affinity and a termination
    cost, and where the affinity is weak it prefers to end the track. This looks
    at those ends again with a different kind of evidence: where the cell was
    already going. A track moving steadily through a frame whose affinity dropped
    is the case the solver handles worst and motion handles best.

    Three properties make this the cleanest step in the chain to measure. It adds
    no nodes, so the node-count penalty cannot move. It only ever links a source
    with no outgoing edge, so it cannot invent a division. It only ever links a
    target with no incoming edge, so it cannot create a second parent. The edge
    term is the only thing it can change, in either direction.

    `window` is the frozen-frame guard, and it is the reason velocity is not just
    the previous displacement. 7.47% of adjacent frame pairs in `6bba` are exact
    byte duplicates while the ground truth keeps moving, so a one-step estimate
    reads zero velocity on one transition in thirteen and predicts the cell stays
    put. Estimating over the longest baseline available up to `window` frames
    dilutes a frozen pair to a fraction of the estimate instead of letting it
    become the whole of it.

    Runs after `enforce_edge_rules` and before gap closing, matching the public
    order. After, because the parent lookup that gives velocity its baseline
    assumes one parent per node. Before, because a hole this step fills properly
    at t+1 is a hole gap closing would otherwise bridge across at t+2.
    """
    st = stats if stats is not None else {}
    st.setdefault("relink_added", 0)
    n = len(graph.nodes)
    if n == 0 or max_um <= 0:
        return graph

    t = np.asarray(graph.nodes.t)
    zyx = np.stack([graph.nodes.z, graph.nodes.y, graph.nodes.x], axis=1)
    p = _physical(zyx, scale)

    has_out = np.zeros(n, dtype=bool)
    has_in = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=np.int64)
    if graph.edges.shape[0]:
        has_out[graph.edges[:, 0]] = True
        has_in[graph.edges[:, 1]] = True
        parent[graph.edges[:, 1]] = graph.edges[:, 0]

    def velocity(i: int) -> tuple[np.ndarray, int]:
        """Displacement per frame over the longest baseline up to `window`."""
        j, k = i, 0
        while k < window:
            pj = int(parent[j])
            if pj < 0:
                break
            j = pj
            k += 1
        if k == 0:
            return np.zeros(3), 0
        return (p[i] - p[j]) / float(k), k

    starts_by_t: dict[int, list[int]] = {}
    ends_by_t: dict[int, list[int]] = {}
    for i in range(n):
        if not has_in[i]:
            starts_by_t.setdefault(int(t[i]), []).append(i)
        if not has_out[i]:
            ends_by_t.setdefault(int(t[i]), []).append(i)

    cands: list[tuple[float, int, int]] = []
    for ti, ends in ends_by_t.items():
        nxt = starts_by_t.get(ti + 1)
        if not nxt:
            continue
        preds = []
        keep_ends = []
        for i in ends:
            v, k = velocity(i)
            if require_velocity and k == 0:
                continue
            preds.append(p[i] + v)
            keep_ends.append(i)
        if not keep_ends:
            continue
        nxt_arr = np.asarray(nxt, dtype=np.int64)
        preds_arr = np.stack(preds, axis=0)
        tree = cKDTree(p[nxt_arr])
        for e_idx, hit in enumerate(tree.query_ball_point(preds_arr, r=max_um)):
            for h in hit:
                j = int(nxt_arr[h])
                d = float(np.linalg.norm(p[j] - preds_arr[e_idx]))
                cands.append((d, keep_ends[e_idx], j))

    if not cands:
        return graph

    # Greedy over the whole video by residual, not per node in id order, so the
    # link a cell's own motion predicts best wins over one that merely fits.
    # Same capping discipline as gap closing: a step that can only add edges
    # needs a bound that does not depend on the graph being sensible.
    budget = min(max(1, int(max_added_frac * n)), int(max_added_abs))
    cands.sort(key=lambda c: c[0])
    used_src: set[int] = set()
    used_tgt: set[int] = set()
    new_edges: list[tuple[int, int]] = []
    for d, i, j in cands:
        if len(new_edges) >= budget:
            st["relink_budget_hit"] = True
            break
        if i in used_src or j in used_tgt:
            continue
        new_edges.append((i, j))
        used_src.add(i)
        used_tgt.add(j)

    if not new_edges:
        return graph
    st["relink_added"] += len(new_edges)
    add = np.asarray(new_edges, dtype=np.int64)
    edges = (np.concatenate([graph.edges, add], axis=0)
             if graph.edges.shape[0] else add)
    return Graph(nodes=graph.nodes, edges=edges)


def close_single_frame_gaps(
    graph: Graph,
    scale,
    max_gap_um: float = 6.0,
    reuse_um: float = 3.2,
    max_added_frac: float = 0.05,
    max_added_abs: int = 2000,
    stats: dict | None = None,
) -> Graph:
    """Bridge a one-frame disappearance, reusing a node if one is already there.

    A track that ends at t and resumes at t+2 costs three edges: the two that
    should exist and the one the metric expected in between. Bridging it turns a
    hole into two true positives when the guess is right.

    `reuse_um` matters more than it looks. If an unlinked detection already sits
    near the midpoint, linking through it adds no node at all and therefore costs
    nothing under the node-count penalty. A synthetic node is only invented when
    nothing suitable exists.

    `max_added_frac` caps invention. Every synthetic node is a real node in the
    submission and counts against `estimated_number_of_nodes`, so an unbounded
    version would buy edges with penalty.
    """
    st = stats if stats is not None else {}
    st.setdefault("gap_reused", 0)
    st.setdefault("gap_created", 0)
    n = len(graph.nodes)
    if n == 0:
        return graph

    t = np.asarray(graph.nodes.t)
    zyx = np.stack([graph.nodes.z, graph.nodes.y, graph.nodes.x], axis=1)
    scale_a = np.asarray(scale, dtype=np.float64)

    has_out = np.zeros(n, dtype=bool)
    has_in = np.zeros(n, dtype=bool)
    if graph.edges.shape[0]:
        has_out[graph.edges[:, 0]] = True
        has_in[graph.edges[:, 1]] = True

    # Track ends at t looking for track starts at t+2.
    by_frame: dict[int, list[int]] = {}
    for i in range(n):
        by_frame.setdefault(int(t[i]), []).append(i)

    ends = [i for i in range(n) if not has_out[i]]
    starts_by_t: dict[int, list[int]] = {}
    for i in range(n):
        if not has_in[i]:
            starts_by_t.setdefault(int(t[i]), []).append(i)

    # Floored at 1 and capped absolutely. Without the floor `int(frac * n)` is
    # zero for any graph under 20 nodes, so the step silently does nothing on
    # small inputs: harmless on a 23,000 node video and invisible until a test
    # with seven nodes asks why nothing happened.
    budget = min(max(1, int(max_added_frac * n)), int(max_added_abs))
    new_edges: list[tuple[int, int]] = []
    new_nodes: list[tuple[int, np.ndarray]] = []
    taken_start: set[int] = set()
    next_id = n

    for i in sorted(ends, key=lambda k: int(t[k])):
        ti = int(t[i])
        cands = [j for j in starts_by_t.get(ti + 2, []) if j not in taken_start]
        if not cands:
            continue
        pi = zyx[i] * scale_a
        best, best_d = None, np.inf
        for j in cands:
            d = float(np.linalg.norm(zyx[j] * scale_a - pi))
            if d < best_d:
                best, best_d = j, d
        if best is None or best_d > max_gap_um:
            continue

        mid_vox = 0.5 * (zyx[i] + zyx[best])
        # Prefer an existing unlinked detection near the midpoint over inventing.
        reuse = None
        for k in by_frame.get(ti + 1, []):
            if has_in[k] or has_out[k]:
                continue
            if float(np.linalg.norm((zyx[k] - mid_vox) * scale_a)) <= reuse_um:
                reuse = k
                break

        if reuse is not None:
            new_edges.append((i, reuse))
            new_edges.append((reuse, best))
            has_in[reuse] = has_out[reuse] = True
            st["gap_reused"] += 1
        else:
            if len(new_nodes) >= budget:
                st["gap_skipped_budget"] = st.get("gap_skipped_budget", 0) + 1
                continue
            new_nodes.append((ti + 1, mid_vox))
            new_edges.append((i, next_id))
            new_edges.append((next_id, best))
            next_id += 1
            st["gap_created"] += 1

        has_out[i] = True
        has_in[best] = True
        taken_start.add(best)

    if not new_edges:
        return graph

    if new_nodes:
        add_t = np.array([a for a, _ in new_nodes], dtype=np.int64)
        add_zyx = np.stack([b for _, b in new_nodes], axis=0)
        ids = np.arange(n + len(new_nodes), dtype=np.int64)
        nodes = Nodes(
            ids=ids,
            t=np.concatenate([t, add_t]),
            z=np.concatenate([graph.nodes.z, add_zyx[:, 0]]),
            y=np.concatenate([graph.nodes.y, add_zyx[:, 1]]),
            x=np.concatenate([graph.nodes.x, add_zyx[:, 2]]),
        )
    else:
        nodes = graph.nodes

    edges = np.concatenate(
        [graph.edges, np.asarray(new_edges, dtype=np.int64)], axis=0
    ) if graph.edges.shape[0] else np.asarray(new_edges, dtype=np.int64)
    return Graph(nodes=nodes, edges=edges)


def _components(n: int, edges: np.ndarray) -> np.ndarray:
    """Connected-component label per node, ignoring edge direction."""
    parent = np.arange(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, tt in edges:
        ra, rb = find(int(s)), find(int(tt))
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def filter_short_tracks(
    graph: Graph,
    min_len: int = 6,
    stats: dict | None = None,
) -> Graph:
    """Delete whole connected components with fewer than `min_len` nodes.

    Short fragments are charged twice by this metric: their edges are false
    positives against the edge Jaccard, and their nodes inflate `T_pred` against
    `estimated_number_of_nodes`. Removing them is the one operation that improves
    both terms at once, which is why the public notebooks all run it and why it is
    high on the list to screen.
    """
    st = stats if stats is not None else {}
    n = len(graph.nodes)
    if n == 0 or min_len <= 1:
        return graph

    comp = _components(n, graph.edges)
    sizes = np.bincount(comp, minlength=n)
    keep_node = sizes[comp] >= min_len
    st["short_track_nodes_removed"] = int((~keep_node).sum())
    st["short_track_components_removed"] = int(
        len({int(c) for c in comp[~keep_node]})
    )
    return _subset(graph, keep_node, st)


def filter_short_branches(
    graph: Graph,
    min_len: int = 0,
    stats: dict | None = None,
) -> Graph:
    """Delete short dangling branches, which the component filter cannot reach.

    `filter_short_tracks` works on weakly connected components, and that is a
    blunt instrument once the graph has any forks in it. A spurious three-node
    chain hanging off a two-hundred-node track is part of a two-hundred-and-three
    node component, so no component threshold will ever remove it, and it is
    exactly the junk the node-count term charges for: `adj_J` compares our total
    node count against the organisers' estimate with no upper cap on the reward
    for coming in under it.

    So this splits the graph into tracklets, maximal chains that do not pass
    through a fork or a merge, and removes the ones that dangle. A tracklet is
    removed when it is shorter than `min_len` and at least one of its two ends is
    free, meaning the chain starts with no parent or ends with no child. An
    internal tracklet, one that bridges a fork to a merge, is never removed
    however short it is, because removing it would cut a track in half and turn
    one edge error into two.

    Iterated to a fixed point: deleting a dangling branch can leave its parent
    dangling in turn, and one pass would stop after the first layer.

    Runs before `add_safe_divisions` on purpose. The division rule proposes forks
    out of unclaimed nodes, and a spurious dangling branch is a supply of exactly
    those, so cleaning first both removes nodes and improves what the division
    rule has to choose from.
    """
    st = stats if stats is not None else {}
    st.setdefault("short_branch_nodes_removed", 0)
    st.setdefault("short_branch_passes", 0)
    if min_len <= 1 or len(graph.nodes) == 0 or graph.edges.shape[0] == 0:
        return graph

    g = graph
    for _ in range(20):
        n = len(g.nodes)
        edges = g.edges
        if edges.shape[0] == 0:
            break
        out_deg = np.bincount(edges[:, 0], minlength=n)
        in_deg = np.bincount(edges[:, 1], minlength=n)

        # The single successor of each node that has exactly one, so a chain can
        # be walked without building an adjacency list.
        succ = -np.ones(n, dtype=np.int64)
        one_out = out_deg[edges[:, 0]] == 1
        succ[edges[one_out, 0]] = edges[one_out, 1]

        # A tracklet starts at a node whose parent does not hand it off cleanly:
        # no parent at all, or a parent that forks.
        # A node begins a tracklet when it has no parent or several. Note what
        # is NOT here: a node with one parent and no child is a track END, and
        # listing it as a start makes every track's last node its own one-node
        # tracklet, which is dangling by definition. The first version did that
        # and deleted the entire graph one layer per pass.
        starts = np.flatnonzero((in_deg == 0) | (in_deg > 1))
        parent_forks = np.zeros(n, dtype=bool)
        multi = out_deg[edges[:, 0]] > 1
        parent_forks[edges[multi, 1]] = True
        starts = np.unique(np.concatenate([starts, np.flatnonzero(parent_forks)]))

        drop = np.zeros(n, dtype=bool)
        for s0 in starts:
            chain = [int(s0)]
            cur = int(s0)
            while True:
                nx = int(succ[cur])
                if nx < 0 or in_deg[nx] != 1:
                    break
                chain.append(nx)
                cur = nx
            if len(chain) >= min_len:
                continue
            head_free = in_deg[chain[0]] == 0
            tail_free = out_deg[chain[-1]] == 0
            if head_free or tail_free:
                drop[chain] = True

        if not drop.any():
            break
        st["short_branch_passes"] += 1
        st["short_branch_nodes_removed"] += int(drop.sum())
        g = _subset(g, ~drop, st)

    return g


def prune_isolated(graph: Graph, stats: dict | None = None) -> Graph:
    """Drop nodes that no edge touches.

    Ground truth is sparse, so an isolated node is not scored as an edge error at
    all. It is still charged through the node-count penalty, which is the whole
    reason to remove it.
    """
    st = stats if stats is not None else {}
    n = len(graph.nodes)
    if n == 0:
        return graph
    touched = np.zeros(n, dtype=bool)
    if graph.edges.shape[0]:
        touched[graph.edges[:, 0]] = True
        touched[graph.edges[:, 1]] = True
    st["pruned_isolated"] = int((~touched).sum())
    return _subset(graph, touched, st)


def _subset(graph: Graph, keep: np.ndarray, st: dict) -> Graph:
    """Keep the flagged nodes and reindex, dropping any edge that loses an end."""
    if keep.all():
        return graph
    remap = -np.ones(len(graph.nodes), dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    if graph.edges.shape[0]:
        alive = keep[graph.edges[:, 0]] & keep[graph.edges[:, 1]]
        edges = remap[graph.edges[alive]]
    else:
        edges = np.empty((0, 2), dtype=np.int64)
    nodes = Nodes(
        ids=np.arange(int(keep.sum()), dtype=np.int64),
        t=np.asarray(graph.nodes.t)[keep],
        z=np.asarray(graph.nodes.z)[keep],
        y=np.asarray(graph.nodes.y)[keep],
        x=np.asarray(graph.nodes.x)[keep],
    )
    return Graph(nodes=nodes, edges=edges)


def linefit_smooth(
    graph: Graph,
    weight: float = 0.8,
    window: int = 2,
    stats: dict | None = None,
) -> Graph:
    """Pull each node toward a line fitted through its neighbours in time.

    Why this is worth anything at all: the scorer matches predictions to ground
    truth by optimal assignment on scaled distance with a 7 um cap, and an edge
    only counts when both of its endpoints matched. `diagnose_localisation.py`
    found Z carrying 61% of squared error with a median offset of exactly one Z
    voxel, which is 1.625 um against a 7 um budget. Averaging a node against its
    own trajectory removes the part of that error that is independent per frame.

    Only nodes with a full window on both sides move, so track ends stay put.
    """
    st = stats if stats is not None else {}
    n = len(graph.nodes)
    if n == 0 or graph.edges.shape[0] == 0 or weight <= 0:
        return graph

    nxt: dict[int, list[int]] = {}
    prv: dict[int, list[int]] = {}
    for s, t_ in graph.edges:
        nxt.setdefault(int(s), []).append(int(t_))
        prv.setdefault(int(t_), []).append(int(s))

    zyx = np.stack([graph.nodes.z, graph.nodes.y, graph.nodes.x], axis=1).astype(float)
    out = zyx.copy()
    moved = 0

    for i in range(n):
        chain = [i]
        cur = i
        for _ in range(window):
            back = prv.get(cur, [])
            if len(back) != 1:
                break
            cur = back[0]
            chain.insert(0, cur)
        cur = i
        for _ in range(window):
            fwd = nxt.get(cur, [])
            if len(fwd) != 1:
                break
            cur = fwd[0]
            chain.append(cur)
        if len(chain) < 2 * window + 1:
            continue
        pts = zyx[chain]
        ts = np.arange(len(chain), dtype=float)
        centre = len(chain) // 2
        # Least squares line per axis against index, evaluated at this node.
        fit = np.empty(3)
        for ax in range(3):
            m, b = np.polyfit(ts, pts[:, ax], 1)
            fit[ax] = m * ts[centre] + b
        out[i] = (1.0 - weight) * zyx[i] + weight * fit
        moved += 1

    st["linefit_smoothed"] = moved
    nodes = Nodes(
        ids=np.asarray(graph.nodes.ids), t=np.asarray(graph.nodes.t),
        z=out[:, 0], y=out[:, 1], x=out[:, 2],
    )
    return Graph(nodes=nodes, edges=graph.edges)


def add_safe_divisions(
    graph: Graph,
    scale,
    max_parent_um: float = 8.0,
    max_sister_um: float = 11.0,
    max_existing_child_um: float = 10.0,
    min_divergence_um: float = 2.25,
    sister_score_weight: float = 0.15,
    frame_frac_cap: float = 0.0076,
    global_frac_cap: float = 0.00375,
    max_daughter_cos: float = 1.0,
    min_symmetry: float = 0.0,
    max_sister_rel: float = 0.0,
    min_child_len: int = 1,
    stats: dict | None = None,
) -> Graph:
    """Give a node a second child where the geometry says a division happened.

    Why this exists at all. The division term is 0.1 of the available 1.1 and a
    one-to-one linker scores exactly 0.000 on it by construction, because no node
    can ever have two outgoing edges. Every submission this repo has made forfeits
    the whole term before the tracker sees an image.

    Why it is not simply "fork the nearest orphan". The public notebook that
    ships this records that its own geometric version, using the same distances,
    the same score and the same caps as below, produced hundreds of forks and
    never a single true positive. Distance alone cannot tell a division from a
    detection that happens to sit near a track. Three structural constraints are
    what make it work, and all three are here:

    1. **The parent must be mid-track.** A node with no predecessor is a track
       start, and a track start acquiring two children is far more likely to be a
       detection error than a mitosis.
    2. **The sisters must be mutual nearest orphans.** The candidate has to be the
       nearest unclaimed node to the existing child, not merely near the parent.
       This is what stops one dense region donating orphans to every track in it.
    3. **The pair must diverge.** Both daughters must themselves continue to t+2,
       and the distance between those successors must exceed the sisters' own
       separation by `min_divergence_um`. Post-mitotic sisters move apart; two
       detections of one cell do not. This is the discriminator, and it is the one
       a purely per-frame rule cannot express.

    Caps are fractions rather than counts because videos differ twentyfold in
    cell density. The score `parent_dist + 0.15 * sister_dist` ranks proposals so
    that the cap keeps the most confident ones.

    Runs before the short-track filter, matching the public chain: a division edge
    attaches an orphan to an existing component, which makes that component larger
    and so less likely to be filtered.

    **Four further gates, all off by default [2026-08-31].** The first screen of
    this rule returned 3 true divisions against 30 false ones on the held-out 19,
    and on that ratio precision is worth about three times what recall is: killing
    all 30 false positives takes the division Jaccard from 0.061 to 0.158, while
    converting one more of the 16 misses takes it to 0.082. The gates above are
    all distances, and distance cannot separate the dominant failure mode, which
    is a neighbouring cell whose own link to its own parent was missed. That cell
    sits near the track, continues normally and diverges, so it passes every
    existing test. What it does not do is sit opposite its supposed sister.

    - `max_daughter_cos`  cosine between the two parent-to-daughter directions.
      A real mitosis pushes daughters apart from a shared origin, so the angle at
      the parent is obtuse and the cosine negative. An unlinked neighbour lies off
      to one side of the real child, giving a cosine near +1. 1.0 disables it.
    - `min_symmetry`  ratio of the shorter to the longer parent-to-daughter
      distance. A parent sits near the midpoint of its own daughters; it does not
      sit near the midpoint of its child and an unrelated neighbour. 0.0 disables.
    - `max_sister_rel`  sister separation as a multiple of the median
      nearest-neighbour distance in the target frame. The videos differ
      twentyfold in cell density, so a flat 11 um is tight in one and meaningless
      in another. 0.0 disables it.
    - `min_child_len`  how many frames past the fork both daughters must survive.
      1 is the existing behaviour, both reaching t+2.
    """
    st = stats if stats is not None else {}
    st.setdefault("safe_div_proposals", 0)
    st.setdefault("safe_div_added", 0)
    st.setdefault("safe_div_frame_capped", 0)
    st.setdefault("safe_div_global_capped", 0)

    n = len(graph.nodes)
    if n == 0 or graph.edges.shape[0] == 0:
        return graph

    edges = graph.edges
    t_arr = np.asarray(graph.nodes.t)
    zyx = np.stack([graph.nodes.z, graph.nodes.y, graph.nodes.x], axis=1)
    pos = _physical(zyx, scale)

    out_deg = np.bincount(edges[:, 0], minlength=n)
    in_deg = np.bincount(edges[:, 1], minlength=n)

    # The single successor of every node that has exactly one. -1 elsewhere.
    succ = -np.ones(n, dtype=np.int64)
    single = out_deg[edges[:, 0]] == 1
    succ[edges[single, 0]] = edges[single, 1]

    order = np.argsort(t_arr, kind="stable")
    t_sorted = t_arr[order]
    uniq, starts = np.unique(t_sorted, return_index=True)
    bounds = list(starts) + [len(order)]
    by_t = {int(tv): order[bounds[k]:bounds[k + 1]] for k, tv in enumerate(uniq)}

    global_cap = max(1, int(round(edges.shape[0] * global_frac_cap)))
    claimed: set[int] = set()
    added: list[tuple[int, int]] = []

    def survives(node: int, frames: int) -> bool:
        """Does this node's chain of single successors run `frames` further on?"""
        cur = int(node)
        for _ in range(frames):
            nx = int(succ[cur])
            if nx < 0 or int(t_arr[nx]) != int(t_arr[cur]) + 1:
                return False
            cur = nx
        return True

    for t in sorted(by_t):
        nxt = by_t.get(t + 1)
        if nxt is None:
            continue
        here = by_t[t]

        # Constraint 1: exactly one child, and a predecessor of its own.
        sources = here[(out_deg[here] == 1) & (in_deg[here] >= 1)]
        orphans = nxt[in_deg[nxt] == 0]
        orphans = np.array([o for o in orphans if int(o) not in claimed],
                           dtype=np.int64)
        if sources.size == 0 or orphans.size == 0:
            continue

        tree = cKDTree(pos[orphans])
        frame_cap = max(1, int(round(sources.size * frame_frac_cap)))

        # Local density, used only when `max_sister_rel` is on. The median
        # nearest-neighbour distance over every node in the target frame is the
        # natural unit for "are these two closer than two unrelated cells would
        # be", and it is what makes one setting work across videos that differ
        # twentyfold in cell count.
        sister_cap_rel = np.inf
        if max_sister_rel > 0.0 and nxt.size > 2:
            nn_d, _ = cKDTree(pos[nxt]).query(pos[nxt], k=2)
            sister_cap_rel = max_sister_rel * float(np.median(nn_d[:, 1]))
        proposals: list[tuple[float, int, int]] = []

        for s in sources:
            c1 = int(succ[s])
            if c1 < 0 or int(t_arr[c1]) != t + 1:
                continue
            if np.linalg.norm(pos[s] - pos[c1]) > max_existing_child_um:
                continue

            # Constraint 2: the candidate must be the existing child's nearest
            # orphan, so a crowded region cannot donate one orphan to many tracks.
            d_mn, i_mn = tree.query(pos[c1])
            if d_mn > max_sister_um:
                continue
            cand = int(orphans[int(i_mn)])
            if cand in claimed:
                continue

            parent_dist = float(np.linalg.norm(pos[s] - pos[cand]))
            if parent_dist > max_parent_um:
                continue
            sister_dist = float(np.linalg.norm(pos[c1] - pos[cand]))
            if sister_dist > max_sister_um:
                continue
            if sister_dist > sister_cap_rel:
                continue

            # Symmetry about the parent. Both gates read the same geometry from
            # different directions: the angle subtended at the parent, and how
            # unequal the two arms are. A mitosis is symmetric in both senses and
            # a missed link to a neighbouring cell is symmetric in neither.
            if max_daughter_cos < 1.0 or min_symmetry > 0.0:
                v1, v2 = pos[c1] - pos[s], pos[cand] - pos[s]
                d1, d2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                if d1 <= 0.0 or d2 <= 0.0:
                    continue
                if max_daughter_cos < 1.0:
                    if float(np.dot(v1, v2)) / (d1 * d2) > max_daughter_cos:
                        continue
                if min_symmetry > 0.0:
                    if min(d1, d2) / max(d1, d2) < min_symmetry:
                        continue

            # Constraint 3: both daughters continue and separate.
            s1, s2 = int(succ[c1]), int(succ[cand])
            if s1 < 0 or s2 < 0:
                continue
            if int(t_arr[s1]) != t + 2 or int(t_arr[s2]) != t + 2:
                continue
            if float(np.linalg.norm(pos[s1] - pos[s2])) - sister_dist < min_divergence_um:
                continue
            if min_child_len > 1:
                if not (survives(c1, min_child_len) and survives(cand, min_child_len)):
                    continue

            proposals.append(
                (parent_dist + sister_score_weight * sister_dist, int(s), cand)
            )

        st["safe_div_proposals"] += len(proposals)
        proposals.sort()
        taken = 0
        for _, s, cand in proposals:
            if len(added) >= global_cap:
                st["safe_div_global_capped"] += 1
                break
            if taken >= frame_cap:
                st["safe_div_frame_capped"] += 1
                break
            if cand in claimed:
                continue
            added.append((s, cand))
            claimed.add(cand)
            taken += 1

    st["safe_div_added"] = len(added)
    if not added:
        return graph
    return Graph(
        nodes=graph.nodes,
        edges=np.concatenate([edges, np.array(added, dtype=np.int64)], axis=0),
    )


def calibrate(graph: Graph, scale, cfg: dict | None = None) -> tuple[Graph, dict]:
    """Run the enabled steps in the only order that makes sense.

    Order is not a preference. Edge rules first, because everything after reads
    the edge set. Gaps before pruning and short-track filtering, because closing a
    gap is exactly what rescues a component that would otherwise be too short.
    Smoothing last, because it needs the final topology to fit a line along.
    """
    cfg = cfg or {}
    stats: dict = {}
    g = graph

    if cfg.get("enforce_edge_rules", True):
        g = enforce_edge_rules(
            g, scale,
            max_edge_um=float(cfg.get("max_edge_um", 14.0)),
            single_parent=bool(cfg.get("single_parent", True)),
            stats=stats,
        )
    if cfg.get("filter_divisions", False):
        # Before anything that removes nodes. Cutting a fork's arm orphans the
        # child, and prune_isolated and the short-track filter downstream are
        # what decide whether it survives on its own merits.
        g = filter_asymmetric_divisions(
            g, scale,
            min_symmetry=float(cfg.get("div_filter_min_symmetry", 0.6)),
            min_child_len=int(cfg.get("div_filter_child_len", 0)),
            stats=stats,
        )
    if cfg.get("motion_relink", False):
        g = motion_relink(
            g, scale,
            max_um=float(cfg.get("motion_relink_um", 6.0)),
            window=int(cfg.get("motion_window", 3)),
            require_velocity=bool(cfg.get("motion_require_velocity", False)),
            max_added_frac=float(cfg.get("motion_max_added_frac", 0.05)),
            max_added_abs=int(cfg.get("motion_max_added_abs", 2000)),
            stats=stats,
        )
    if cfg.get("gap_close", False):
        g = close_single_frame_gaps(
            g, scale,
            max_gap_um=float(cfg.get("gap_close_um", 6.0)),
            reuse_um=float(cfg.get("gap_reuse_um", 3.2)),
            max_added_frac=float(cfg.get("gap_max_added_frac", 0.05)),
            max_added_abs=int(cfg.get("gap_max_added_abs", 2000)),
            stats=stats,
        )
    if cfg.get("min_branch_len", 0):
        g = filter_short_branches(g, int(cfg["min_branch_len"]), stats=stats)
    if cfg.get("safe_divisions", False):
        # Before the short-track filter and before smoothing, matching the public
        # chain. A division edge attaches an orphan to an existing component, so
        # running it first makes that component larger and less likely to be cut.
        g = add_safe_divisions(
            g, scale,
            max_parent_um=float(cfg.get("safe_div_parent_um", 8.0)),
            max_sister_um=float(cfg.get("safe_div_sister_um", 11.0)),
            max_existing_child_um=float(cfg.get("safe_div_child_um", 10.0)),
            min_divergence_um=float(cfg.get("safe_div_divergence_um", 2.25)),
            frame_frac_cap=float(cfg.get("safe_div_frame_frac", 0.0076)),
            global_frac_cap=float(cfg.get("safe_div_global_frac", 0.00375)),
            max_daughter_cos=float(cfg.get("safe_div_max_cos", 1.0)),
            min_symmetry=float(cfg.get("safe_div_min_symmetry", 0.0)),
            max_sister_rel=float(cfg.get("safe_div_sister_rel", 0.0)),
            min_child_len=int(cfg.get("safe_div_child_len", 1)),
            stats=stats,
        )
    if cfg.get("prune_isolated", False):
        g = prune_isolated(g, stats=stats)
    if cfg.get("min_track_len", 0):
        g = filter_short_tracks(g, int(cfg["min_track_len"]), stats=stats)
    if cfg.get("linefit_smooth", False):
        g = linefit_smooth(
            g,
            weight=float(cfg.get("linefit_weight", 0.8)),
            window=int(cfg.get("linefit_window", 2)),
            stats=stats,
        )

    stats["n_nodes"] = len(g.nodes)
    stats["n_edges"] = int(g.edges.shape[0])
    return g, stats
