"""Reading the images and the ground-truth graphs, and writing a submission.

Deliberately narrow dependencies. The rerun notebook has no internet and a 12 h
budget over roughly 200 unseen samples, so the inference path uses zarr and numpy
and nothing else. `tracksdata`, `geff` and `polars` are needed only for *scoring*,
which happens locally and never inside the submitted notebook.

Images are `(T, Z, Y, X)` uint16 with one chunk per timepoint, so a timepoint is
the natural unit of work and nothing ever needs the whole 400 MB array resident.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import numpy as np

# Two readers, because the two environments do not agree on what exists.
# Locally zarr is installed and convenient. Inside the competition rerun `zarr`
# and `numcodecs` are absent and internet is disabled, so there is no installing
# them; `tensorstore` is present and reads Zarr v3 natively. Verified in
# notebooks/probe_env and notebooks/probe_read. Prefer zarr when available so the
# local path stays simple, and fall back to tensorstore, which is what actually
# runs at submission time.
try:
    import zarr
except ImportError:  # pragma: no cover - exercised on Kaggle, not locally
    zarr = None

try:
    import tensorstore as ts
except ImportError:
    ts = None

# Fallback physical voxel size in microns, (Z, Y, X). Prefer the per-sample value
# parsed from the OME-NGFF metadata; this is only for files that lack it.
DEFAULT_SCALE_ZYX: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

SUBMISSION_COLUMNS = (
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
)


@dataclass
class Nodes:
    """Ground-truth or predicted detections, as flat arrays.

    `ids` are the graph's node ids and are not necessarily contiguous, so anything
    indexing by position has to map through them rather than assume `ids[i] == i`.
    """

    ids: np.ndarray
    t: np.ndarray
    z: np.ndarray
    y: np.ndarray
    x: np.ndarray

    def __len__(self) -> int:
        return int(self.ids.size)

    def zyx(self) -> np.ndarray:
        return np.stack([self.z, self.y, self.x], axis=1).astype(np.float64)


@dataclass
class Graph:
    nodes: Nodes
    edges: np.ndarray  # (N, 2) of (source_id, target_id)
    estimated_number_of_nodes: float | None = None

    def divisions(self) -> np.ndarray:
        """Node ids with two or more outgoing edges."""
        if self.edges.size == 0:
            return np.empty(0, dtype=np.int64)
        src, counts = np.unique(self.edges[:, 0], return_counts=True)
        return src[counts >= 2]


def read_estimated_nodes(geff_path: str) -> float | None:
    """`extra.estimated_number_of_nodes` from the geff metadata.

    This drives the over-detection penalty and is an organiser-supplied count of
    *all* true cells, far larger than the sparse annotation. Never recompute it
    from the ground-truth graph.
    """
    meta_path = os.path.join(geff_path, "zarr.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as fh:
        geff = json.load(fh)["attributes"]["geff"]
    val = (geff.get("extra") or {}).get("estimated_number_of_nodes")
    return float(val) if val is not None else None


def read_geff(path: str) -> Graph:
    """Read a ground-truth graph. Local only: the rerun has no ground truth."""
    if zarr is None:
        raise RuntimeError(
            "read_geff needs zarr, which is only expected to be present locally. "
            "Ground-truth graphs are never read inside the competition rerun."
        )
    grp = zarr.open_group(path, mode="r")
    nodes = Nodes(
        ids=np.asarray(grp["nodes/ids"][:]),
        t=np.asarray(grp["nodes/props/t/values"][:]),
        z=np.asarray(grp["nodes/props/z/values"][:]),
        y=np.asarray(grp["nodes/props/y/values"][:]),
        x=np.asarray(grp["nodes/props/x/values"][:]),
    )
    edges = np.asarray(grp["edges/ids"][:])
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    return Graph(
        nodes=nodes, edges=edges, estimated_number_of_nodes=read_estimated_nodes(path)
    )


def read_scale(zarr_path: str) -> tuple[float, float, float]:
    """Per-sample (Z, Y, X) micron scale from the OME-NGFF metadata."""
    meta_path = os.path.join(zarr_path, "zarr.json")
    if not os.path.exists(meta_path):
        return DEFAULT_SCALE_ZYX
    with open(meta_path, encoding="utf-8") as fh:
        attrs = json.load(fh).get("attributes", {})
    ms = attrs.get("multiscales")
    if not ms:
        return DEFAULT_SCALE_ZYX
    tf = ms[0]["datasets"][0]["coordinateTransformations"][0]
    if tf.get("type") != "scale":
        return DEFAULT_SCALE_ZYX
    return tuple(float(v) for v in tf["scale"][-3:])  # type: ignore[return-value]


def read_quantiles(zarr_path: str) -> dict[float, float]:
    """Precomputed intensity quantiles shipped in the zarr attrs.

    Present for every sample, so normalisation needs no pass over the pixels.
    """
    meta_path = os.path.join(zarr_path, "zarr.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as fh:
        attrs = json.load(fh).get("attributes", {})
    q = (attrs.get("image_statistics") or {}).get("quantiles") or {}
    return {float(k): float(v) for k, v in q.items()}


class Image:
    """Lazy handle on one sample's image. Nothing is read until a frame is asked for."""

    def __init__(self, zarr_path: str, backend: str | None = None):
        self.path = zarr_path
        self.scale = read_scale(zarr_path)
        self.quantiles = read_quantiles(zarr_path)

        if backend is None:
            backend = "zarr" if zarr is not None else "tensorstore"
        if backend == "zarr" and zarr is not None:
            self.backend = "zarr"
            self._arr = zarr.open_group(zarr_path, mode="r")["0"]
        elif ts is not None:
            self.backend = "tensorstore"
            self._arr = ts.open(
                {
                    "driver": "zarr3",
                    "kvstore": {
                        "driver": "file",
                        "path": os.path.join(zarr_path, "0"),
                    },
                },
                read=True,
            ).result()
        else:
            raise RuntimeError(
                "no zarr reader available. Install zarr locally, or tensorstore, "
                "which is what the competition rerun environment provides."
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._arr.shape)

    @property
    def n_timepoints(self) -> int:
        return int(self._arr.shape[0])

    def frame(self, t: int) -> np.ndarray:
        """One timepoint, `(Z, Y, X)`. This is exactly one stored chunk."""
        if self.backend == "tensorstore":
            return np.asarray(self._arr[t].read().result())
        return np.asarray(self._arr[t])

    def normalised_frame(
        self, t: int, q_lo: float = 0.01, q_hi: float = 0.99
    ) -> np.ndarray:
        """Frame rescaled to roughly [0, 1] using the shipped quantiles."""
        frame = self.frame(t).astype(np.float32)
        lo = self.quantiles.get(q_lo)
        hi = self.quantiles.get(q_hi)
        if lo is None or hi is None or hi <= lo:
            lo, hi = float(frame.min()), float(frame.max())
        if hi <= lo:
            return np.zeros_like(frame)
        return np.clip((frame - lo) / (hi - lo), 0.0, None)


def list_samples(directory: str, require_geff: bool = True) -> list[str]:
    """Sample names in a data directory, sorted."""
    names = {d[: -len(".zarr")] for d in os.listdir(directory) if d.endswith(".zarr")}
    if require_geff:
        geffs = {d[: -len(".geff")] for d in os.listdir(directory) if d.endswith(".geff")}
        names &= geffs
    return sorted(names)


def graph_to_submission_rows(graph: Graph, dataset: str) -> list[tuple]:
    """Flatten one graph into submission rows: nodes first, then edges.

    Matches the organisers' `geffs_to_csv.py`. Coordinates are rounded to int,
    fields unused by a row type are -1, and the leading `id` is added by the
    caller so it stays consecutive across the whole file.
    """
    n = graph.nodes
    rows: list[tuple] = [
        (
            dataset, "node", int(nid), int(tt),
            int(round(float(zz))), int(round(float(yy))), int(round(float(xx))),
            -1, -1,
        )
        for nid, tt, zz, yy, xx in zip(n.ids, n.t, n.z, n.y, n.x)
    ]
    rows.extend(
        (dataset, "edge", -1, -1, -1, -1, -1, int(s), int(tg)) for s, tg in graph.edges
    )
    return rows


# A dataset that produced nothing still has to appear in the CSV, so it gets one
# placeholder node. The coordinate is an honest in-volume voxel and deliberately
# not the out-of-volume sentinel used by the metric hack on the public
# leaderboard. An unmatched predicted node is not an edge false positive; it costs
# only its share of the node-count penalty, and one node against a per-sample
# estimate of order 24,000 is nothing. The sample then scores zero on edges, which
# is the truthful score for a sample nothing was detected in.
PLACEHOLDER_ROW = ("node", 0, 0, 0, 0, 0, -1, -1)


def write_submission(
    graphs: dict[str, Graph],
    out_path: str,
    datasets: list[str] | None = None,
) -> tuple[int, list[str]]:
    """Write the submission CSV. Returns (row count, names that were backfilled).

    `datasets` is the authoritative list of names that must appear, normally the
    `.zarr` folders in the test directory. Any name missing from `graphs`, or
    present with an empty graph, is backfilled with a placeholder node row.

    This is the enforcement point for the rule that **every dataset in the test
    set must appear**. A dataset absent from the CSV invalidates the entire
    submission; a dataset scoring zero costs only that one sample. Relying on the
    caller to pass a non-empty graph per sample is not enough, because an empty
    graph flattens to zero rows and the name silently disappears.
    """
    expected = sorted(set(datasets) | set(graphs)) if datasets is not None else sorted(graphs)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    written = 0
    backfilled: list[str] = []
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SUBMISSION_COLUMNS)
        for dataset in expected:
            graph = graphs.get(dataset)
            rows = graph_to_submission_rows(graph, dataset) if graph is not None else []
            if not rows:
                rows = [(dataset,) + PLACEHOLDER_ROW]
                backfilled.append(dataset)
            for row in rows:
                w.writerow((written,) + row)
                written += 1
    return written, backfilled


def verify_submission(out_path: str, datasets: list[str]) -> None:
    """Re-read the written CSV and check every expected dataset is in it.

    Deliberately reads the file back rather than trusting what the writer thinks
    it did. This is the last line of defence before a 12 h rerun produces a
    submission that scores zero for a reason no local number would have shown.
    Raises rather than returning a flag, because there is no sane way to continue.
    """
    seen: set[str] = set()
    with open(out_path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r, None)
        if tuple(header or ()) != SUBMISSION_COLUMNS:
            raise SystemExit(f"submission header is {header}, expected {list(SUBMISSION_COLUMNS)}")
        col = SUBMISSION_COLUMNS.index("dataset")
        for row in r:
            seen.add(row[col])

    missing = sorted(set(datasets) - seen)
    extra = sorted(seen - set(datasets))
    if missing:
        raise SystemExit(
            f"{len(missing)} dataset(s) missing from {out_path}: {missing[:10]}. "
            "Every dataset in the test set must appear or the submission is invalid."
        )
    if extra:
        raise SystemExit(f"{len(extra)} unexpected dataset(s) in {out_path}: {extra[:10]}")
