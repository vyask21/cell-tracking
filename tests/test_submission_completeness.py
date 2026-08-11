"""Every dataset in the test set must appear in the submission CSV.

A dataset missing from the CSV invalidates the whole submission, not just that
sample, and nothing in a local CV number would ever reveal it. The failure mode
these tests guard is specific: a sample that detects nothing flattens to zero
rows, so its name silently vanishes from the file. That is a plausible cause of
the "submission scored 0.0" and "Submission Scored error" reports on the forum.

These tests use synthetic graphs and do not need the competition data on disk.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from src.data import (
    SUBMISSION_COLUMNS,
    Graph,
    Nodes,
    verify_submission,
    write_submission,
)


def _graph(n: int, n_edges: int = 0) -> Graph:
    """A graph with `n` nodes at distinct coordinates and `n_edges` chained edges."""
    ids = np.arange(n, dtype=np.int64)
    return Graph(
        nodes=Nodes(
            ids=ids,
            t=ids.astype(np.int64),
            z=np.full(n, 5.0),
            y=np.full(n, 10.0),
            x=np.full(n, 20.0),
        ),
        edges=np.array([[i, i + 1] for i in range(n_edges)], dtype=np.int64).reshape(-1, 2),
    )


def _empty_graph() -> Graph:
    e = np.empty(0, dtype=np.int64)
    return Graph(
        nodes=Nodes(ids=e, t=e, z=e.astype(float), y=e.astype(float), x=e.astype(float)),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def _read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _datasets_in(path: str) -> set[str]:
    return {r["dataset"] for r in _read(path)}


def test_empty_graph_still_appears_in_the_csv(tmp_path):
    """The core bug: a sample detecting nothing must not vanish from the file."""
    out = str(tmp_path / "sub.csv")
    graphs = {"good": _graph(3, 2), "collapsed": _empty_graph()}

    rows, backfilled = write_submission(graphs, out, datasets=["good", "collapsed"])

    assert _datasets_in(out) == {"good", "collapsed"}
    assert backfilled == ["collapsed"]
    assert rows == 6  # 3 nodes + 2 edges + 1 placeholder


def test_dataset_absent_from_graphs_is_backfilled(tmp_path):
    """A sample that raised never reaches `graphs` at all and must still appear."""
    out = str(tmp_path / "sub.csv")

    rows, backfilled = write_submission(
        {"good": _graph(2)}, out, datasets=["good", "never_ran"]
    )

    assert _datasets_in(out) == {"good", "never_ran"}
    assert backfilled == ["never_ran"]
    assert rows == 3


def test_healthy_submission_is_not_padded(tmp_path):
    """No placeholder may be added when every sample produced real detections."""
    out = str(tmp_path / "sub.csv")

    rows, backfilled = write_submission(
        {"a": _graph(4, 3), "b": _graph(2, 1)}, out, datasets=["a", "b"]
    )

    assert backfilled == []
    assert rows == 10  # (4+3) + (2+1)
    assert sum(r["row_type"] == "node" for r in _read(out)) == 6
    assert sum(r["row_type"] == "edge" for r in _read(out)) == 4


def test_id_column_is_consecutive_across_the_whole_file(tmp_path):
    """`id` is one index across all datasets, not per dataset."""
    out = str(tmp_path / "sub.csv")
    write_submission(
        {"a": _graph(3, 2), "b": _empty_graph(), "c": _graph(2, 1)},
        out,
        datasets=["a", "b", "c"],
    )

    ids = [int(r["id"]) for r in _read(out)]
    assert ids == list(range(len(ids)))


def test_placeholder_is_an_in_volume_coordinate(tmp_path):
    """Guards against anyone turning the placeholder into the metric hack.

    The public-leaderboard hack appends nodes at sentinel coordinates outside any
    real volume (t=-1000, z=y=x=-10000) to inflate the division term. The
    placeholder here exists only to keep a dataset present and must stay an
    honest, in-volume, non-negative voxel.
    """
    out = str(tmp_path / "sub.csv")
    write_submission({}, out, datasets=["collapsed"])

    row = _read(out)[0]
    assert row["row_type"] == "node"
    for field in ("t", "z", "y", "x"):
        assert int(row[field]) >= 0, f"placeholder {field} must be inside the volume"
    assert int(row["source_id"]) == -1 and int(row["target_id"]) == -1


def test_placeholder_contributes_no_edges(tmp_path):
    """A backfilled dataset scores zero honestly rather than inventing structure."""
    out = str(tmp_path / "sub.csv")
    write_submission({}, out, datasets=["collapsed"])

    assert all(r["row_type"] == "node" for r in _read(out))


def test_header_matches_the_required_columns(tmp_path):
    out = str(tmp_path / "sub.csv")
    write_submission({"a": _graph(1)}, out, datasets=["a"])

    with open(out, newline="", encoding="utf-8") as fh:
        assert tuple(next(csv.reader(fh))) == SUBMISSION_COLUMNS


def test_verify_passes_on_a_complete_submission(tmp_path):
    out = str(tmp_path / "sub.csv")
    write_submission({"a": _graph(2), "b": _graph(2)}, out, datasets=["a", "b"])

    verify_submission(out, ["a", "b"])  # must not raise


def test_verify_raises_when_a_dataset_is_missing(tmp_path):
    """The check has to fail loudly locally instead of silently on the rerun."""
    out = str(tmp_path / "sub.csv")
    write_submission({"a": _graph(2)}, out)  # no `datasets`, so nothing is backfilled

    with pytest.raises(SystemExit, match="missing"):
        verify_submission(out, ["a", "b"])


def test_verify_raises_on_an_unexpected_dataset(tmp_path):
    out = str(tmp_path / "sub.csv")
    write_submission({"a": _graph(2), "typo": _graph(2)}, out, datasets=["a", "typo"])

    with pytest.raises(SystemExit, match="unexpected"):
        verify_submission(out, ["a"])


def test_verify_raises_on_a_wrong_header(tmp_path):
    out = str(tmp_path / "sub.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "dataset", "wrong"])
        w.writerow([0, "a", "x"])

    with pytest.raises(SystemExit, match="header"):
        verify_submission(out, ["a"])


def test_predict_run_survives_a_sample_that_raises(tmp_path, monkeypatch, capsys):
    """End to end: one exploding sample must not cost the other 199.

    This is the whole point of the guard. The run continues, the failure is
    printed with its traceback for the Kaggle log, and the dead sample still
    reaches the CSV so the submission stays valid.
    """
    from src import predict

    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for name in ("s1", "s2", "s3"):
        (test_dir / f"{name}.zarr").mkdir()

    def fake_predict_sample(zarr_path, cfg, **kw):
        if os.path.basename(zarr_path).startswith("s2"):
            raise RuntimeError("corrupt chunk")
        g = _graph(3, 2)
        return g, {
            "n_nodes": 3, "n_edges": 2, "n_divisions": 0,
            "detect_s": 0.1, "link_s": 0.1,
        }

    monkeypatch.setattr(predict, "predict_sample", fake_predict_sample)

    out = str(tmp_path / "sub.csv")
    predict.run("conf/baseline.yaml", data_dir=str(test_dir), out_path=out)

    assert _datasets_in(out) == {"s1", "s2", "s3"}
    captured = capsys.readouterr().out
    assert "FAILED" in captured
    assert "corrupt chunk" in captured
    verify_submission(out, ["s1", "s2", "s3"])


def test_predict_run_survives_every_sample_raising(tmp_path, monkeypatch):
    """Even a total wipeout produces a valid, honest, zero-scoring submission."""
    from src import predict

    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for name in ("s1", "s2"):
        (test_dir / f"{name}.zarr").mkdir()

    def always_raises(zarr_path, cfg, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(predict, "predict_sample", always_raises)

    out = str(tmp_path / "sub.csv")
    predict.run("conf/baseline.yaml", data_dir=str(test_dir), out_path=out)

    assert _datasets_in(out) == {"s1", "s2"}
    verify_submission(out, ["s1", "s2"])
