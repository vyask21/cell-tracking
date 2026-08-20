"""Global ILP linking over the learned edge probabilities.

The second half of the linker work. `src.link.link_sequence_learned` replaced
physical distance with the pack's learned affinity as the cost and left the solver
alone; this replaces the solver and leaves the cost alone.

**What the ILP buys that an assignment cannot.** `match_consecutive` and
`match_consecutive_learned` both solve one frame pair at a time. Each is optimal
inside its pair and blind across pairs: nothing connects the choice made at t to
the choice made at t+1. The ILP solves the whole video at once under flow
conservation, so a link survives only if the track it belongs to is coherent
before and after it. It also prices appearance and disappearance explicitly,
which gives it a principled reason to leave a detection unlinked rather than
accept the least-bad partner, and it treats division as a constraint instead of
the bolt-on `add_divisions` pass.

This calls the pack's own solver, `td.solvers.ILPSolver`, with the objective the
pack's `predict` builds: `edge_weight * edge_prob` per selected edge plus
appearance, disappearance and division costs. Nothing about the optimisation is
reimplemented here.

**One deviation from the pack, stated rather than buried.** Our candidate set is
gated at `max_link_um` as well as by probability, where the pack gates on
probability alone. The gate is the metric's own node-matching radius, so it is
defensible, but `screen_link_cap.py` measured that cap at the local-max operating
point and it has not been re-measured here. It is a config value for that reason.

**Dependencies.** `tracksdata`, `ilpy` and `pyscipopt`. Install the pack's own
wheels rather than PyPI's: the pack pins `tracksdata 0.1.0rc6.dev3` and PyPI
serves rc7 or later, and the solver API is not guaranteed across them. The pack
ships Linux wheels for the Kaggle rerun; local Windows equivalents exist on PyPI
for `pyscipopt`, and `tracksdata` and `ilpy` are pure Python.
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np

from src.data import Graph, Nodes


class IlpTruncated(RuntimeError):
    """The solve hit its time limit, so its answer must not be used.

    Measured on `6bba_3abfe10a`, the largest video in the held-out set: solved to
    optimality the ILP scores 0.7316 against the assignment's 0.6911, and stopped
    at 600s it scores **0.4196**. A truncated branch-and-bound returns whatever
    feasible solution it happens to be holding, and a feasible solution to this
    model can carry far more edges than the optimum, 70,434 against 65,160 here.

    So a time limit is not a safety net unless the caller refuses the result.
    Accepting a truncated solve is worse than never having run the solver, which
    is the opposite of how a timeout usually behaves and is why this is an
    exception rather than a flag on the return value.
    """


def _require_tracksdata():
    try:
        import polars as pl
        import tracksdata as td
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the ILP linker needs tracksdata, ilpy and pyscipopt. Install the "
            "pack's own wheels so local matches the Kaggle rerun:\n"
            "  pip install --no-deps external/pack50/wheels/tracksdata-*.whl "
            "external/pack50/wheels/ilpy-*.whl\n"
            "  pip install pyscipopt==6.2.1"
        ) from exc
    return td, pl


@contextlib.contextmanager
def _quiet():
    """Swallow the solver's own chatter.

    SCIP prints a presolve banner and a statistics block per solve. At one solve
    per video that is thousands of lines through a screen's log, which buries the
    per-sample progress the log exists for.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def link_sequence_ilp(
    detections: list[np.ndarray],
    affinities: list[dict],
    scale_zyx,
    edge_weight: float = -1.0,
    appearance_weight: float = 0.1,
    disappearance_weight: float = 0.1,
    division_weight: float = 1.0,
    num_threads: int = 1,
    gap: float = 0.0,
    timeout: float | None = None,
) -> Graph:
    """Solve one whole video as a single flow problem over the candidate edges.

    Raises `IlpTruncated` when the solve hits `timeout`, rather than returning
    the partial answer. See that class for why.

    `timeout` and `gap` are exposed because this is the one step in the pipeline
    whose cost is not predictable from the input size. A branch-and-bound solve
    can find its answer immediately or grind, and a video that grinds inside a
    12 hour rerun is a failed submission rather than a slow one.

    On `gap`: 0.01 and 0.0 reached the identical solution on the largest video in
    the held-out set, same 65,160 edges and same score, so a 1% gap costs nothing
    in quality there. It did not save time on that instance either, so it is not
    the runtime lever it usually is. The lever is the time budget plus the
    fallback the caller applies when this raises.
    """
    td, pl = _require_tracksdata()

    offsets: list[int] = []
    next_id = 0
    for coords in detections:
        offsets.append(next_id)
        next_id += coords.shape[0]

    total = next_id
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

    ids = np.arange(total, dtype=np.int64)
    nodes = Nodes(ids=ids, t=t_arr, z=zyx[:, 0], y=zyx[:, 1], x=zyx[:, 2])
    empty = np.empty((0, 2), dtype=np.int64)
    if total == 0:
        return Graph(nodes=nodes, edges=empty)

    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    # Node k in our numbering is td_ids[k] in the solver's. `back` inverts it.
    # Not assumed to be the identity even though it currently is: an id scheme
    # that changed under us would corrupt every edge silently.
    td_ids = graph.bulk_add_nodes([
        {"t": int(t_arr[k]), "z": float(zyx[k, 0]),
         "y": float(zyx[k, 1]), "x": float(zyx[k, 2])}
        for k in range(total)
    ])
    back = {int(v): k for k, v in enumerate(td_ids)}

    scale = np.asarray(scale_zyx, dtype=np.float64)
    payload = []
    for t, aff in enumerate(affinities):
        i, j, p = aff["i"], aff["j"], aff["p"]
        if i.size == 0:
            continue
        gi = offsets[t] + i.astype(np.int64)
        gj = offsets[t + 1] + j.astype(np.int64)
        d = np.linalg.norm((zyx[gi] - zyx[gj]) * scale[None, :], axis=1)
        payload.extend(
            {
                "source_id": td_ids[int(a)],
                "target_id": td_ids[int(b)],
                "edge_prob": float(prob),
                "edge_dist": float(dist),
            }
            for a, b, prob, dist in zip(gi, gj, p, d)
        )

    if not payload:
        return Graph(nodes=nodes, edges=empty)

    graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
    graph.bulk_add_edges(payload)

    solver = td.solvers.ILPSolver(
        edge_weight=edge_weight * td.EdgeAttr("edge_prob"),
        appearance_weight=appearance_weight,
        disappearance_weight=disappearance_weight,
        division_weight=division_weight,
        num_threads=num_threads,
        gap=gap,
        timeout=timeout,
    )
    t0 = time.monotonic()
    with _quiet():
        solved = solver.solve(graph)
    elapsed = time.monotonic() - t0

    # Timing is the detection because the solver does not hand back a status.
    # The 0.98 allows for the solve stopping fractionally under its own limit.
    # A second signal agrees and is worth knowing: a truncated solve returned
    # MORE edges than the assignment it was meant to improve on, where the
    # optimal one returned fewer.
    if timeout is not None and elapsed >= 0.98 * timeout:
        raise IlpTruncated(
            f"the ILP hit its {timeout:.0f}s limit after {elapsed:.0f}s on a "
            f"graph of {total} nodes and {len(payload)} candidate edges. Its "
            "partial answer is worse than not solving at all, so it is refused."
        )

    if solved is None:
        raise RuntimeError(
            "the ILP solver returned no solution. With a timeout set this means "
            "it ran out of time; without one it means the model was infeasible."
        )
    return Graph(nodes=nodes, edges=_edges_from_solved(solved, back))


def _edges_from_solved(solved, back: dict[int, int]) -> np.ndarray:
    """Pull the selected edges back into our own node numbering.

    Defensive on two points, because the solver's return value is the one thing
    here that our own code does not pin.

    The returned view is already filtered to the selected edges in the version
    the pack pins, and it also carries a boolean `solution` column. If a version
    ever returns the full candidate set instead, taking every row would build a
    graph with every candidate edge in it, which would not error and would score
    terribly for no visible reason. So the column is applied when present.

    The id check is the same argument. A silently remapped id would produce edges
    between the wrong cells rather than an exception.
    """
    df = solved.edge_attrs()
    if df.height == 0:
        return np.empty((0, 2), dtype=np.int64)
    if "solution" in df.columns:
        df = df.filter(df["solution"])
        if df.height == 0:
            return np.empty((0, 2), dtype=np.int64)

    src = np.asarray(df["source_id"].to_list(), dtype=np.int64)
    tgt = np.asarray(df["target_id"].to_list(), dtype=np.int64)
    unknown = {int(v) for v in np.concatenate([src, tgt])} - set(back)
    if unknown:
        raise RuntimeError(
            f"the solver returned {len(unknown)} node ids this graph never "
            "added, so the id mapping is wrong and every edge under it would be "
            "wrong too"
        )
    return np.stack(
        [np.fromiter((back[int(v)] for v in src), dtype=np.int64, count=src.size),
         np.fromiter((back[int(v)] for v in tgt), dtype=np.int64, count=tgt.size)],
        axis=1,
    )
