"""One sample in, one predicted tracking graph out.

This is the seam between the parts that must run inside the competition rerun
(detection and linking, which touch only tensorstore, numpy and scipy) and the
parts that only ever run locally (scoring, which needs tracksdata). Nothing here
may import anything from `src.metrics`.
"""

from __future__ import annotations

import os
import time

import numpy as np

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
    if link_backend not in ("distance", "learned", "ilp"):
        raise ValueError(f"unknown link.backend {link_backend!r}")
    if link_backend in ("learned", "ilp") and backend != "unet":
        raise ValueError(
            "link.backend 'learned' needs detect.backend 'unet'. The edge scores "
            "come from the same network pass as the detections, so there is "
            "nothing to score local-max peaks with."
        )

    affinities = None
    if link_backend in ("learned", "ilp"):
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
    ilp_fell_back = False
    if link_backend == "ilp":
        # Imported here rather than at module scope so the rerun only needs
        # tracksdata, ilpy and pyscipopt when a config actually asks for the ILP.
        from src.link_ilp import IlpTruncated, link_sequence_ilp

        # A time limit is normally a safe degradation. Here it is the opposite.
        # On 6bba_3abfe10a the optimal solve scores 0.7316 and the assignment
        # scores 0.6911, but a solve stopped at 600s scores 0.4196: a truncated
        # branch-and-bound hands back whichever feasible solution it happens to
        # hold, and a feasible solution here can carry thousands of spurious
        # edges. The budget bounds runtime, which the 12 hour rerun needs, and
        # the fallback bounds damage, which the score needs. Without the second
        # half the first half is a liability.
        try:
            graph = link_sequence_ilp(
                detections,
                affinities,
                scale_zyx=image.scale,
                edge_weight=float(link_cfg.get("ilp_edge_weight", -1.0)),
                appearance_weight=float(link_cfg.get("ilp_appearance_weight", 0.1)),
                disappearance_weight=float(link_cfg.get("ilp_disappearance_weight", 0.1)),
                division_weight=float(link_cfg.get("ilp_division_weight", 1.0)),
                num_threads=int(link_cfg.get("ilp_num_threads", 1)),
                gap=float(link_cfg.get("ilp_gap", 0.0)),
                timeout=(float(link_cfg["ilp_timeout_s"])
                         if link_cfg.get("ilp_timeout_s") else None),
            )
        except IlpTruncated as exc:
            print(f"    ILP refused: {exc}", flush=True)
            print("    falling back to the per-frame assignment.", flush=True)
            ilp_fell_back = True
            graph = link_sequence_learned(
                detections,
                affinities,
                scale_zyx=image.scale,
                max_division_um=float(link_cfg.get("max_division_um", 0.0)),
            )
    elif affinities is not None:
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

    if affinities is not None:
        # Coverage of the candidate set, because `edge_threshold` silently
        # decides what the assignment is even allowed to consider. A source node
        # with no candidate cannot be linked at any cost, so this is the number
        # that says whether the floor is doing harm rather than filtering.
        n_cand = sum(int(a["i"].size) for a in affinities)
        src_cov = [
            float(np.unique(a["i"]).size) / d.shape[0]
            for a, d in zip(affinities, detections[:-1]) if d.shape[0]
        ]
        tgt_cov = [
            float(np.unique(a["j"]).size) / d.shape[0]
            for a, d in zip(affinities, detections[1:]) if d.shape[0]
        ]
        extra = {
            "n_candidates": n_cand,
            "cand_per_node": round(n_cand / max(1, len(graph.nodes)), 2),
            "src_covered": round(float(np.mean(src_cov)) if src_cov else 0.0, 4),
            "tgt_covered": round(float(np.mean(tgt_cov)) if tgt_cov else 0.0, 4),
        }
    else:
        extra = {}

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
        **({"ilp_fell_back": ilp_fell_back} if link_backend == "ilp" else {}),
        **extra,
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
