"""Summarise every ground-truth graph, one row per sample.

Answers the questions the CV scheme depends on: how many timepoints each sample
covers, how many cells are annotated, how many divisions there are, and how the
organisers' node-count estimate compares to the sparse annotation actually shipped.

Writes data/meta/sample_profile.csv.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
import zarr

DEFAULT_ROOT = os.path.join("data", "raw")
DEFAULT_OUT = os.path.join("data", "meta", "sample_profile.csv")

# Microns per voxel, from the competition data description. The metric's 7 um
# matching cap is physical, so anything measuring distance has to scale first.
SCALE_ZYX = (1.625, 0.40625, 0.40625)


def read_geff(path: str) -> dict:
    grp = zarr.open_group(path, mode="r")
    node_ids = np.asarray(grp["nodes/ids"][:])
    props = {}
    for axis in ("t", "z", "y", "x"):
        props[axis] = np.asarray(grp[f"nodes/props/{axis}/values"][:])
    edges = np.asarray(grp["edges/ids"][:])
    return {"node_ids": node_ids, "props": props, "edges": edges}


def read_estimated_nodes(path: str) -> object:
    """`estimated_number_of_nodes` drives the over-detection penalty.

    It is an organiser-supplied count of all true nodes, which is much larger than
    the sparse annotation. Never recompute it from the GT.
    """
    meta_path = os.path.join(path, "zarr.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    stack = [meta]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "estimated_number_of_nodes" in node:
                return node["estimated_number_of_nodes"]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


def profile(path: str) -> dict:
    g = read_geff(path)
    t = g["props"]["t"]
    edges = g["edges"]
    n_nodes = int(len(g["node_ids"]))

    if edges.size:
        src = edges[:, 0]
        out_degree = Counter(src.tolist())
        n_div = sum(1 for _, c in out_degree.items() if c >= 2)
        n_edges = int(edges.shape[0])
    else:
        n_div = 0
        n_edges = 0

    # Track count is the number of nodes with no incoming edge, i.e. lineage roots.
    if edges.size:
        has_parent = set(edges[:, 1].tolist())
    else:
        has_parent = set()
    n_roots = int(sum(1 for nid in g["node_ids"].tolist() if nid not in has_parent))

    est = read_estimated_nodes(path)

    row = {
        "sample": os.path.basename(path).replace(".geff", ""),
        "embryo": os.path.basename(path).split("_")[0],
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_divisions": n_div,
        "n_roots": n_roots,
        "estimated_number_of_nodes": est,
        "t_min": int(t.min()) if n_nodes else -1,
        "t_max": int(t.max()) if n_nodes else -1,
        "n_timepoints_annotated": int(len(np.unique(t))) if n_nodes else 0,
        "nodes_per_t_mean": round(n_nodes / max(1, len(np.unique(t))), 2) if n_nodes else 0,
    }
    for axis in ("z", "y", "x"):
        v = g["props"][axis]
        row[f"{axis}_min"] = float(v.min()) if n_nodes else -1
        row[f"{axis}_max"] = float(v.max()) if n_nodes else -1
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    train_dir = os.path.join(args.root, "train")
    geffs = sorted(
        os.path.join(train_dir, d) for d in os.listdir(train_dir) if d.endswith(".geff")
    )
    print(f"{len(geffs)} geff directories under {train_dir}")

    rows = []
    for i, path in enumerate(geffs, 1):
        try:
            rows.append(profile(path))
        except Exception as exc:  # noqa: BLE001 - one bad sample should not stop the sweep
            print(f"  FAIL {os.path.basename(path)}: {exc!r}")
        if i % 25 == 0:
            print(f"  {i}/{len(geffs)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
