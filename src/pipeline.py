"""One sample in, one predicted tracking graph out.

This is the seam between the parts that must run inside the competition rerun
(detection and linking, which touch only tensorstore, numpy and scipy) and the
parts that only ever run locally (scoring, which needs tracksdata). Nothing here
may import anything from `src.metrics`.
"""

from __future__ import annotations

import os
import time

from src.data import Graph, Image
from src.detect import detect_sequence
from src.link import link_sequence


def predict_sample(
    zarr_path: str,
    cfg,
    timepoints=None,
    verbose: bool = False,
) -> tuple[Graph, dict]:
    """Detect then link one sample. Returns the graph and a small stats dict."""
    image = Image(zarr_path)
    detect_cfg = cfg.detect or {}
    link_cfg = cfg.link or {}

    t0 = time.time()
    detections = detect_sequence(
        image,
        sigma_um=float(detect_cfg.get("sigma_um", 2.0)),
        min_sep_um=float(detect_cfg.get("min_sep_um", 3.0)),
        threshold_scale=float(detect_cfg.get("threshold_scale", 0.5)),
        max_detections=int(detect_cfg.get("max_detections", 20000)),
        timepoints=timepoints,
        progress_every=25 if verbose else 0,
    )
    t_detect = time.time() - t0

    t0 = time.time()
    graph = link_sequence(
        detections,
        scale_zyx=image.scale,
        max_link_um=float(link_cfg.get("max_link_um", 7.0)),
        max_division_um=float(link_cfg.get("max_division_um", 0.0)),
    )
    t_link = time.time() - t0

    stats = {
        "n_nodes": len(graph.nodes),
        "n_edges": int(graph.edges.shape[0]),
        "n_divisions": int(graph.divisions().size),
        "n_frames": len(detections),
        "nodes_per_frame": round(len(graph.nodes) / max(1, len(detections)), 1),
        "detect_s": round(t_detect, 2),
        "link_s": round(t_link, 2),
    }
    return graph, stats


def predict_dir(data_dir: str, samples: list[str], cfg, verbose: bool = True) -> dict:
    """Predict every named sample in a directory. Returns {sample: Graph}."""
    out: dict[str, Graph] = {}
    for i, sample in enumerate(samples, 1):
        zarr_path = os.path.join(data_dir, sample + ".zarr")
        graph, stats = predict_sample(zarr_path, cfg)
        out[sample] = graph
        if verbose:
            print(
                f"  [{i}/{len(samples)}] {sample}: {stats['n_nodes']} nodes "
                f"({stats['nodes_per_frame']}/frame), {stats['n_edges']} edges, "
                f"{stats['n_divisions']} divisions, "
                f"{stats['detect_s']}s detect + {stats['link_s']}s link",
                flush=True,
            )
    return out
