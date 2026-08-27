"""The calibration chain has to reach the submitted graph, not just the screen.

`scripts/screen_calibration.py` calls `src.postprocess.calibrate` directly on a
cache, so the chain measured +0.0581 there while the production path still wrote
the uncalibrated graph. These pin the seam: a config either asks for calibration
and gets it, or does not ask and is byte-identical to the run before the chain
existed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config, load_config
from src.data import Graph, Nodes


def straight_track(n=8):
    ts = np.arange(n, dtype=np.int64)
    return Graph(
        nodes=Nodes(ids=np.arange(n, dtype=np.int64), t=ts,
                    z=np.zeros(n), y=np.arange(n, dtype=float), x=np.zeros(n)),
        edges=np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int64),
    )


class FakeImage:
    scale = (1.625, 0.40625, 0.40625)

    def __init__(self, *a, **k):
        pass


def run_pipeline(monkeypatch, cfg, graph):
    """Drive predict_sample with detection and linking stubbed out.

    Only the seam is under test, so the expensive halves are replaced. The graph
    the linker returns is the one calibration must act on.
    """
    import src.pipeline as pipeline

    monkeypatch.setattr(pipeline, "Image", FakeImage)
    monkeypatch.setattr(
        pipeline, "detect_sequence",
        lambda *a, **k: [np.zeros((1, 3)) for _ in range(8)],
    )
    monkeypatch.setattr(pipeline, "link_sequence", lambda *a, **k: graph)
    return pipeline.predict_sample("unused.zarr", cfg)


def base_cfg(**calib):
    return Config(
        name="t", detect={"backend": "localmax"}, link={"backend": "distance"},
        calibrate=calib,
    )


def test_calibration_is_off_unless_a_config_asks(monkeypatch):
    g = straight_track()
    out, stats = run_pipeline(monkeypatch, base_cfg(), g)
    assert out is g
    assert "calibrate_s" not in stats


def test_an_enabled_config_reaches_the_graph(monkeypatch):
    # A short component and an isolated node, both of which the chain removes.
    g = straight_track(3)
    out, stats = run_pipeline(
        monkeypatch,
        base_cfg(enabled=True, max_edge_um=14.0, prune_isolated=True,
                 min_track_len=6),
        g,
    )
    assert len(out.nodes) == 0
    assert "calibrate_s" in stats
    assert stats["calib_n_nodes"] == 0


def test_a_track_at_the_minimum_length_survives_the_chain(monkeypatch):
    g = straight_track(6)
    out, _ = run_pipeline(
        monkeypatch,
        base_cfg(enabled=True, max_edge_um=14.0, prune_isolated=True,
                 min_track_len=6),
        g,
    )
    assert len(out.nodes) == 6


@pytest.mark.parametrize("path", ["conf/unet50_calib.yaml", "conf/unet50_t099.yaml"])
def test_the_shipped_configs_load(path):
    load_config(path)


def test_the_calibrated_config_gates_candidates_and_edges_alike():
    """The candidate gate and the output cap must agree.

    `link.max_link_um` decides what the solver may pick; `calibrate.max_edge_um`
    decides what survives. The screen ran them at one value, so a config that
    splits them is reporting a number no screen produced.
    """
    cfg = load_config("conf/unet50_calib.yaml")
    assert cfg.link["max_link_um"] == cfg.calibrate["max_edge_um"]


def test_the_calibrated_config_differs_from_its_parent_in_one_block():
    """One variable per config, checked rather than asserted in a comment."""
    child = load_config("conf/unet50_calib.yaml")
    parent = load_config("conf/unet50_t099.yaml")
    assert child.detect == parent.detect
    assert not parent.calibrate and child.calibrate["enabled"]
    # The gate moves with the chain because the screened arm moved it.
    differing = {k for k in set(child.link) | set(parent.link)
                 if child.link.get(k) != parent.link.get(k)}
    assert differing == {"max_link_um"}
