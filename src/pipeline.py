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
from src.link import link_sequence, link_sequence_learned


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
    backend = str(detect_cfg.get("backend", "localmax"))
    link_backend = str(link_cfg.get("backend", "distance"))
    if link_backend not in ("distance", "learned"):
        raise ValueError(f"unknown link.backend {link_backend!r}")
    if link_backend == "learned" and backend != "unet":
        raise ValueError(
            "link.backend 'learned' needs detect.backend 'unet'. The edge scores "
            "come from the same network pass as the detections, so there is "
            "nothing to score local-max peaks with."
        )

    affinities = None
    if link_backend == "learned":
        # Detection and edge scoring share one U-Net pass. Running them
        # separately would double the expensive half for no gain, so this branch
        # returns both and the detections it returns are the same detections the
        # detect-only branch below would produce for the same settings.
        from src.unet import detect_and_score_sequence

        if timepoints is not None:
            raise ValueError(
                "the learned linker needs the whole sequence: windows stride by "
                "window_size - 1 so every consecutive pair is scored exactly "
                "once, and a subset would silently drop pairs."
            )
        detections, affinities = detect_and_score_sequence(
            zarr_path,
            det_threshold=float(detect_cfg.get("det_threshold", 0.955)),
            pool_kernel_um=float(detect_cfg.get("pool_kernel_um", 5.0)),
            det_tta=bool(detect_cfg.get("det_tta", False)),
            device=str(detect_cfg.get("device", "cpu")),
            weights=detect_cfg.get("weights"),
            edge_activation=str(link_cfg.get("edge_activation", "softmax")),
            edge_threshold=float(link_cfg.get("edge_threshold", 0.05)),
            max_link_um=float(link_cfg.get("max_link_um", 7.0)),
            progress_every=25 if verbose else 0,
        )
    elif backend == "unet":
        # The learned detector returns the same contract, a list indexed by t of
        # (N, 3) full-resolution voxel coordinates, so everything downstream is
        # untouched and the detector is the only variable.
        from src.unet import detect_sequence_unet

        detections = detect_sequence_unet(
            zarr_path,
            det_threshold=float(detect_cfg.get("det_threshold", 0.955)),
            pool_kernel_um=float(detect_cfg.get("pool_kernel_um", 5.0)),
            det_tta=bool(detect_cfg.get("det_tta", False)),
            device=str(detect_cfg.get("device", "cpu")),
            weights=detect_cfg.get("weights"),
            timepoints=timepoints,
            progress_every=25 if verbose else 0,
        )
    elif backend != "localmax":
        raise ValueError(f"unknown detect.backend {backend!r}")
    else:
        detections = detect_sequence(
            image,
            sigma_um=float(detect_cfg.get("sigma_um", 2.0)),
            min_sep_um=float(detect_cfg.get("min_sep_um", 3.0)),
            threshold_scale=float(detect_cfg.get("threshold_scale", 0.5)),
            max_detections=int(detect_cfg.get("max_detections", 20000)),
            suppress_radii_um=(tuple(float(v) for v in detect_cfg["suppress_radii_um"])
                               if detect_cfg.get("suppress_radii_um") else None),
            z_shift_vox=float(detect_cfg.get("z_shift_vox", 0.0)),
            timepoints=timepoints,
            progress_every=25 if verbose else 0,
        )
    t_detect = time.time() - t0

    t0 = time.time()
    if affinities is not None:
        graph = link_sequence_learned(
            detections,
            affinities,
            scale_zyx=image.scale,
            max_division_um=float(link_cfg.get("max_division_um", 0.0)),
        )
    else:
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
        # With the learned linker, detect_s covers the shared U-Net pass that
        # produced both the detections and the edge scores, and link_s is only
        # the assignment. The two are not comparable across link backends.
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
