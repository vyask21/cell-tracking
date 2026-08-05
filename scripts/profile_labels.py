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


def read_geff_meta(path: str) -> dict:
    """Read the geff metadata block.

    `extra.estimated_number_of_nodes` is the organiser-supplied count of *all*
    true nodes, which is far larger than the sparse annotation, and it drives the
    over-detection penalty. Never recompute it from the GT.

    `axes` carries the per-axis min/max of the annotated coordinates, which is a
    free read and says how much of each volume is actually annotated.
    """
    meta_path = os.path.join(path, "zarr.json")
    out: dict = {"estimated_number_of_nodes": None}
    if not os.path.exists(meta_path):
        return out
    with open(meta_path, encoding="utf-8") as fh:
        geff = json.load(fh)["attributes"]["geff"]
    out["estimated_number_of_nodes"] = (geff.get("extra") or {}).get(
        "estimated_number_of_nodes"
    )
    for ax in geff.get("axes") or []:
        out[f"gt_{ax['name']}_min"] = ax.get("min")
        out[f"gt_{ax['name']}_max"] = ax.get("max")
    return out


def read_image_meta(root: str, sample: str) -> dict:
    """Image shape and the precomputed intensity quantiles from the OME-Zarr.

    The quantiles are shipped in the zarr attrs, so normalisation costs nothing at
    inference time. There is no `translation` transform in these files, so the
    metadata does not say where a crop sits inside its parent acquisition.
    """
    out: dict = {}
    zpath = os.path.join(root, f"{sample}.zarr")
    grp_meta = os.path.join(zpath, "zarr.json")
    arr_meta = os.path.join(zpath, "0", "zarr.json")
    if os.path.exists(arr_meta):
        with open(arr_meta, encoding="utf-8") as fh:
            arr = json.load(fh)
        out["image_shape"] = "x".join(str(s) for s in arr.get("shape", []))
        out["image_dtype"] = arr.get("data_type")
    if os.path.exists(grp_meta):
        with open(grp_meta, encoding="utf-8") as fh:
            attrs = json.load(fh).get("attributes", {})
        q = (attrs.get("image_statistics") or {}).get("quantiles") or {}
        for key in ("0.01", "0.1", "0.9", "0.99"):
            out[f"q{key}"] = q.get(key)
        ms = attrs.get("multiscales")
        if ms:
            tf = ms[0]["datasets"][0]["coordinateTransformations"][0]
            out["scale"] = ",".join(str(v) for v in tf.get("scale", []))
    return out


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

    sample = os.path.basename(path).replace(".geff", "")
    row = {
        "sample": sample,
        "embryo": sample.split("_")[0],
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_divisions": n_div,
        "n_roots": n_roots,
        "t_min": int(t.min()) if n_nodes else -1,
        "t_max": int(t.max()) if n_nodes else -1,
        "n_timepoints_annotated": int(len(np.unique(t))) if n_nodes else 0,
        "nodes_per_t_mean": round(n_nodes / max(1, len(np.unique(t))), 2) if n_nodes else 0,
    }
    for axis in ("z", "y", "x"):
        v = g["props"][axis]
        row[f"{axis}_min"] = float(v.min()) if n_nodes else -1
        row[f"{axis}_max"] = float(v.max()) if n_nodes else -1
    row.update(read_geff_meta(path))
    row.update(read_image_meta(os.path.dirname(path), sample))
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

    # Union of keys, in first-seen order: a sample missing its zarr metadata would
    # otherwise blow up the writer or silently drop columns.
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
