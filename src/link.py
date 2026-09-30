"""Linking detections across time into a tracking graph.

The scorer discards anything that is not a single-frame step: it keeps only edges
where `t_target - t_source == 1`, so backward edges and gap-closing edges across a
skipped frame contribute nothing and are not emitted here. It also caps out-degree
at two, keeping the two lowest edge ids, so a third child is wasted work.

Matching is an optimal assignment on physical distance, mirroring how the metric
matches predictions to ground truth. Distances are always in microns, never voxels.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from src.data import Graph, Nodes


def _physical(coords: np.ndarray, scale_zyx) -> np.ndarray:
    return coords * np.asarray(scale_zyx, dtype=np.float64)[None, :]


def match_consecutive(
    a: np.ndarray,
    b: np.ndarray,
    scale_zyx,
    max_link_um: float = 7.0,
) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment between two frames' detections.

    Returns (index in a, index in b) pairs closer than `max_link_um`. Pairs beyond
    the cap are dropped after solving rather than being given a large finite cost,
    so a distant pair can never be forced in just to complete the assignment.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return []

    pa, pb = _physical(a, scale_zyx), _physical(b, scale_zyx)

    # A dense cost matrix is fine at this size: frames hold a few hundred to a few
    # thousand detections, so the worst case is a few million entries.
    cost = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    # Anything beyond the cap is made unattractive but still finite, so the solver
    # always has a feasible assignment; the pairs are filtered out below.
    big = float(max_link_um) * 1000.0
    cost_solve = np.where(cost <= max_link_um, cost, big)

    rows, cols = linear_sum_assignment(cost_solve)
    return [
        (int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= max_link_um
    ]


def add_divisions(
    a: np.ndarray,
    b: np.ndarray,
    scale_zyx,
    matched: list[tuple[int, int]],
    max_division_um: float,
) -> list[tuple[int, int]]:
    """Give unmatched t+1 detections a second parent link, forming divisions.

    A daughter that failed the one-to-one assignment is attached to the nearest
    already-matched parent within `max_division_um`, and each parent takes at most
    one extra child, since out-degree above two is discarded by the scorer.

    Worth being cautious here. Only 151 divisions are annotated across all 199
    samples and 112 samples have none, so the division Jaccard pools a tiny number
    of events and spurious forks can swamp it. A fork also counts as a false
    positive when its two child branches trace back to distinct ground-truth
    components, which is exactly what attaching an unrelated detection would do.
    """
    if max_division_um <= 0 or not matched:
        return []

    used_children = {c for _, c in matched}
    free = [i for i in range(b.shape[0]) if i not in used_children]
    if not free:
        return []

    parents = [r for r, _ in matched]
    pa = _physical(a[parents], scale_zyx)
    pb = _physical(b[free], scale_zyx)

    tree = cKDTree(pa)
    dist, idx = tree.query(pb, k=1, distance_upper_bound=max_division_um)

    extra: list[tuple[int, int]] = []
    taken: set[int] = set()
    # Nearest first, so when two orphans compete for one parent the closer wins.
    for order in np.argsort(dist):
        d, j = dist[order], idx[order]
        if not np.isfinite(d) or j >= len(parents):
            continue
        parent = parents[int(j)]
        if parent in taken:
            continue
        taken.add(parent)
        extra.append((parent, free[int(order)]))
    return extra


def link_sequence(
    detections: list[np.ndarray],
    scale_zyx,
    max_link_um: float = 7.0,
    max_division_um: float = 0.0,
) -> Graph:
    """Turn per-timepoint detections into a graph with globally unique node ids."""
    # Node ids are assigned per timepoint in a single running counter, so the
    # mapping from (t, local index) to id is just an offset.
    offsets: list[int] = []
    next_id = 0
    for coords in detections:
        offsets.append(next_id)
        next_id += coords.shape[0]

    total = next_id
    ids = np.arange(total, dtype=np.int64)
    t_arr = np.empty(total, dtype=np.int64)
    zyx = np.empty((total, 3), dtype=np.float64)
    for t, coords in enumerate(detections):
        start, end = offsets[t], offsets[t] + coords.shape[0]
        t_arr[start:end] = t
        zyx[start:end] = coords

    edges: list[tuple[int, int]] = []
    for t in range(len(detections) - 1):
        a, b = detections[t], detections[t + 1]
        matched = match_consecutive(a, b, scale_zyx, max_link_um=max_link_um)
        pairs = list(matched)
        pairs += add_divisions(a, b, scale_zyx, matched, max_division_um)
        for i, j in pairs:
            edges.append((offsets[t] + i, offsets[t + 1] + j))

    edge_arr = (
        np.asarray(edges, dtype=np.int64)
        if edges
        else np.empty((0, 2), dtype=np.int64)
    )
    nodes = Nodes(
        ids=ids, t=t_arr, z=zyx[:, 0], y=zyx[:, 1], x=zyx[:, 2]
    )
    return Graph(nodes=nodes, edges=edge_arr)


def match_consecutive_learned(
    n_a: int,
    n_b: int,
    affinity: dict,
) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment on learned affinity instead of distance.

    The same solver, the same drop-after-solve rule and the same
    candidate set as `match_consecutive`. The only thing that changes is what a
    pair costs: physical distance there, `-log(p)` here. One variable.

    `-log(p)` rather than `-p` because the assignment sums costs, and summing
    negative log probabilities maximises the product over the frame pair, which
    is the quantity the edge head's scores are. Maximising the sum of raw
    probabilities would prefer one near-certain link plus a hopeless one over
    two good ones.

    Pairs the affinity never proposed are unreachable rather than expensive, so
    a detection with no plausible partner ends the track instead of being forced
    into the least-bad link available.
    """
    i, j, p = affinity["i"], affinity["j"], affinity["p"]
    if n_a == 0 or n_b == 0 or i.size == 0:
        return []

    # Clipped so a probability that underflows to zero cannot make the cost
    # matrix non-finite and take the solver with it.
    cost_val = -np.log(np.clip(p.astype(np.float64), 1e-12, 1.0))
    big = float(cost_val.max()) * 1000.0 + 1.0
    cost = np.full((n_a, n_b), big, dtype=np.float64)
    cost[i, j] = cost_val

    rows, cols = linear_sum_assignment(cost)
    return [
        (int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < big
    ]


def link_sequence_learned(
    detections: list[np.ndarray],
    affinities: list[dict],
    scale_zyx,
    max_division_um: float = 0.0,
) -> Graph:
    """`link_sequence` with the learned edge head supplying the costs.

    `max_link_um` is absent on purpose: the distance gate was already applied
    when the candidates were built in `src.unet._gate_candidates`, using the same
    value, because gating there is what keeps the affinity lists small enough to
    hold for a whole video.

    Divisions still go through `add_divisions` on distance. The edge head scores
    pairs, so it has an opinion about which parent a daughter belongs to, but the
    division term is bounded at 0.02 to 0.04 and is off in every
    config, so wiring a second path for it would be untested code in the way.
    """
    offsets: list[int] = []
    next_id = 0
    for coords in detections:
        offsets.append(next_id)
        next_id += coords.shape[0]

    total = next_id
    ids = np.arange(total, dtype=np.int64)
    t_arr = np.empty(total, dtype=np.int64)
    zyx = np.empty((total, 3), dtype=np.float64)
    for t, coords in enumerate(detections):
        start, end = offsets[t], offsets[t] + coords.shape[0]
        t_arr[start:end] = t
        zyx[start:end] = coords

    if len(affinities) != len(detections) - 1:
        raise ValueError(
            f"expected {len(detections) - 1} affinity entries for "
            f"{len(detections)} frames, got {len(affinities)}"
        )

    edges: list[tuple[int, int]] = []
    for t in range(len(detections) - 1):
        a, b = detections[t], detections[t + 1]
        matched = match_consecutive_learned(a.shape[0], b.shape[0], affinities[t])
        pairs = list(matched)
        pairs += add_divisions(a, b, scale_zyx, matched, max_division_um)
        for i, j in pairs:
            edges.append((offsets[t] + i, offsets[t + 1] + j))

    edge_arr = (
        np.asarray(edges, dtype=np.int64)
        if edges
        else np.empty((0, 2), dtype=np.int64)
    )
    nodes = Nodes(ids=ids, t=t_arr, z=zyx[:, 0], y=zyx[:, 1], x=zyx[:, 2])
    return Graph(nodes=nodes, edges=edge_arr)
