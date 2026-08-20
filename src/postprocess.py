"""Graph calibration: the steps that run after the linker.

This module exists because of a measurement, not a hunch. Reading the public code
on 2026-08-20 showed that the notebooks at 0.902 to 0.915 use the same weights we
use and the same ILP objective we use, and differ almost entirely in what they do
to the graph afterwards. The author of the 0.902 notebook says it plainly: the
model weights are unchanged and the score comes from calibrating the output graph.
Our pipeline stopped at the ILP and wrote the CSV. This is the missing half.

Every step here is written to be independently switchable, because the workspace
rule is one variable per config and because the point is to find out which of
these actually pay rather than to import a chain wholesale and hope.

The steps, in the order they must run:

1. `enforce_edge_rules`  drop edges that are not one frame forward, or longer
   than a cap, and reduce any node to a single parent.
2. `close_single_frame_gaps`  a track that vanishes at t and reappears at t+2
   gets a synthetic node at t+1, which converts one missing edge into two present
   ones.
3. `prune_isolated`  nodes with no edges at all are pure node-count penalty.
4. `filter_short_tracks`  whole components shorter than a minimum are mostly
   false positives, and they cost twice: their edges are FP and their nodes
   inflate the node-count ratio.
5. `linefit_smooth`  move each node toward a line fitted through its temporal
   neighbours. The metric matches predictions to ground truth by distance with a
   7 um cap, so moving a node a micron closer can flip it from unmatched to
   matched, and matching is what edges are scored on.

Divisions are deliberately absent. They are bounded at 0.02 to 0.04 in NOTES.md
and every config here keeps them off; adding them belongs in its own experiment
after the rest is settled.
"""

from __future__ import annotations

import numpy as np

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
    if cfg.get("gap_close", False):
        g = close_single_frame_gaps(
            g, scale,
            max_gap_um=float(cfg.get("gap_close_um", 6.0)),
            reuse_um=float(cfg.get("gap_reuse_um", 3.2)),
            max_added_frac=float(cfg.get("gap_max_added_frac", 0.05)),
            max_added_abs=int(cfg.get("gap_max_added_abs", 2000)),
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
