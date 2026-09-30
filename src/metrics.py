"""Scoring, through the organisers' own code.

The competition score is

    adjusted_edge_jaccard + 0.1 * division_jaccard

and it is not reimplemented here. `tracking_cellmot.metrics.evaluate` is called
directly, on graphs loaded the same way the organisers load them. A rewrite would
disagree with the leaderboard in ways that are hard to see and expensive to find,
and a local number that does not track the leaderboard is worse than no number.

Run `python scripts/get_reference_code.py` once to put their repo at the pinned
commit under `external/`. That code is BSD-3-Clause and stays theirs.

Two things about the metric are worth keeping in mind whenever a number from here
is interpreted, both established by reading their source:

- A predicted edge only counts as a false positive when one of its endpoints
  matched an annotated ground-truth node. Predictions in unannotated regions are
  invisible to the edge term and cost only through the node-count penalty.
- The adjusted edge Jaccard is weight-averaged across samples by that sample's
  `TP + FP + FN`, while the division Jaccard is micro-averaged over pooled counts.
  So a single heavily annotated sample can move the edge term a long way.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "external" / "kaggle-cell-tracking-competition"

# Physical voxel size in microns, (Z, Y, X). Matching is capped at 7 um in
# physical space, which is about 4.3 voxels in Z and 17 in Y/X. A distance
# computed in voxels is wrong by a factor of four between the axes.
DEFAULT_SCALE_ZYX: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
MAX_MATCH_DISTANCE_UM = 7.0


def _ensure_reference_on_path() -> None:
    src = REFERENCE_DIR / "src"
    if not src.exists():
        raise RuntimeError(
            "reference code missing. Run:  python scripts/get_reference_code.py\n"
            f"expected it at {src}"
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


@dataclass
class SampleScore:
    """One sample's counts and derived metrics, as the organisers compute them."""

    sample: str
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int
    node_recall: float
    total_node_ratio: float
    edge_jaccard: float
    adj_edge_jaccard: float

    @property
    def weight(self) -> int:
        """The weight this sample carries in the aggregate edge score."""
        return self.edge_tp + self.edge_fp + self.edge_fn


def load_graph(geff_path: str | os.PathLike):
    """Load a .geff into a tracksdata graph, exactly as the organisers do."""
    _ensure_reference_on_path()
    import tracksdata as td

    result = td.graph.IndexedRXGraph.from_geff(Path(geff_path))
    return result[0] if isinstance(result, tuple) else result


def to_tracksdata(graph):
    """Convert a `src.data.Graph` into a tracksdata graph, in memory.

    Avoids a write-to-geff-and-read-back round trip on every scored sample. The
    node ids are preserved so predicted edges keep referring to the right nodes.
    """
    _ensure_reference_on_path()
    import tracksdata as td

    K = td.DEFAULT_ATTR_KEYS
    g = td.graph.IndexedRXGraph()
    # `t` is registered by default; z, y and x are not. Adding an existing key
    # raises, so only register what is missing.
    import polars as pl

    existing = set(g.node_attr_keys())
    for key in (K.Z, K.Y, K.X):
        if key not in existing:
            g.add_node_attr_key(key, pl.Float64, 0.0)
    if K.EDGE_DIST not in set(g.edge_attr_keys()):
        g.add_edge_attr_key(K.EDGE_DIST, pl.Float64, 0.0)

    n = graph.nodes
    g.bulk_add_nodes(
        [
            {K.T: int(t), K.Z: float(z), K.Y: float(y), K.X: float(x)}
            for t, z, y, x in zip(n.t, n.z, n.y, n.x)
        ],
        indices=[int(i) for i in n.ids],
    )
    if graph.edges.size:
        g.bulk_add_edges(
            [
                {K.EDGE_SOURCE: int(s), K.EDGE_TARGET: int(t), K.EDGE_DIST: 0.0}
                for s, t in graph.edges
            ]
        )
    return g


def score_prediction(
    pred_graph,
    gt_geff: str | os.PathLike,
    sample: str,
    scale: tuple[float, float, float] = DEFAULT_SCALE_ZYX,
    max_distance: float = MAX_MATCH_DISTANCE_UM,
) -> SampleScore:
    """Score an in-memory `src.data.Graph` against a ground-truth geff."""
    _ensure_reference_on_path()
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics

    from src.data import read_estimated_nodes

    pred = to_tracksdata(pred_graph)
    gt = load_graph(gt_geff)

    er = evaluate(pred, gt, scale=scale, max_distance=max_distance)
    recall = (
        node_recall(pred, gt) if pred.num_edges() > 0 and pred.num_nodes() > 0 else 0.0
    )
    n_total = read_estimated_nodes(str(gt_geff))
    row = per_sample_metrics(er, float("nan") if n_total is None else n_total, recall)
    return SampleScore(sample=sample, **row)


def score_sample(
    pred_geff: str | os.PathLike,
    gt_geff: str | os.PathLike,
    sample: str,
    scale: tuple[float, float, float] = DEFAULT_SCALE_ZYX,
    max_distance: float = MAX_MATCH_DISTANCE_UM,
) -> SampleScore:
    _ensure_reference_on_path()
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics

    from src.data import read_estimated_nodes

    pred = load_graph(pred_geff)
    gt = load_graph(gt_geff)

    er = evaluate(pred, gt, scale=scale, max_distance=max_distance)
    recall = (
        node_recall(pred, gt)
        if pred.num_edges() > 0 and pred.num_nodes() > 0
        else 0.0
    )
    # The over-detection penalty uses the organisers' node-count estimate, never a
    # count derived from the sparse ground truth.
    n_total = read_estimated_nodes(str(gt_geff))
    row = per_sample_metrics(er, float("nan") if n_total is None else n_total, recall)
    return SampleScore(sample=sample, **row)


def aggregate(scores: list[SampleScore]) -> dict:
    """Run-level summary, using the organisers' `summarise`."""
    _ensure_reference_on_path()
    from tracking_cellmot.metrics import summarise

    rows = [
        {
            "edge_tp": s.edge_tp, "edge_fp": s.edge_fp, "edge_fn": s.edge_fn,
            "division_tp": s.division_tp,
            "division_fp": s.division_fp,
            "division_fn": s.division_fn,
            "num_pred_nodes": s.num_pred_nodes,
            "node_recall": s.node_recall,
            "total_node_ratio": s.total_node_ratio,
            "edge_jaccard": s.edge_jaccard,
            "adj_edge_jaccard": s.adj_edge_jaccard,
        }
        for s in scores
    ]
    return summarise(rows)


def score_with_interval(scores: list[SampleScore], seed: int = 0) -> dict:
    """Aggregate, plus a bootstrap interval on the adjusted edge Jaccard.

    Two embryos means leave-one-embryo-out has two folds, so the fold-to-fold
    spread is a two-point estimate and close to useless on its own. Resampling
    whole samples inside a fold gives a valid interval on that fold's number,
    which is the difference between "this change helped" and "this change moved
    the number by less than the noise".
    """
    from src.cv import bootstrap_weighted_mean

    out = dict(aggregate(scores))
    values = np.array([s.adj_edge_jaccard for s in scores], dtype=float)
    weights = np.array([s.weight for s in scores], dtype=float)
    point, lo, hi = bootstrap_weighted_mean(values, weights, seed=seed)
    out["adj_edge_jaccard_boot"] = point
    out["adj_edge_jaccard_lo"] = lo
    out["adj_edge_jaccard_hi"] = hi
    return out
