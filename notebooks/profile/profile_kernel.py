"""Profile every ground-truth graph in the competition, on the Kaggle mount.

Runs as a private CPU kernel. Kaggle rate limits per-file downloads at a few
hundred files, and the labels alone are 4,585 files, so pulling them locally is
not practical. The data is already mounted here, so the profile is computed in
place and only the resulting CSV comes back.

Writes /kaggle/working/sample_profile.csv, one row per train sample.

The Kaggle base image does not ship zarr, so this kernel installs it and runs with
internet enabled. That is fine here because this is a utility kernel, but it is
*not* available to the competition rerun, where internet is disabled. How the
submission notebook gets zarr is a separate open problem.
"""

import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

try:
    import zarr
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "zarr>=3"]
    )
    import zarr

print("zarr", zarr.__version__)

DATA = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
if not os.path.exists(DATA):  # local dry run
    DATA = os.path.join("data", "raw")

OUT = "/kaggle/working/sample_profile.csv"
if not os.path.exists("/kaggle/working"):
    OUT = os.path.join("data", "meta", "sample_profile_full.csv")


def read_geff(path):
    grp = zarr.open_group(path, mode="r")
    return {
        "ids": np.asarray(grp["nodes/ids"][:]),
        "t": np.asarray(grp["nodes/props/t/values"][:]),
        "z": np.asarray(grp["nodes/props/z/values"][:]),
        "y": np.asarray(grp["nodes/props/y/values"][:]),
        "x": np.asarray(grp["nodes/props/x/values"][:]),
        "edges": np.asarray(grp["edges/ids"][:]),
    }


def read_geff_meta(path):
    out = {"estimated_number_of_nodes": None}
    meta_path = os.path.join(path, "zarr.json")
    if not os.path.exists(meta_path):
        return out
    with open(meta_path, encoding="utf-8") as fh:
        geff = json.load(fh)["attributes"]["geff"]
    out["estimated_number_of_nodes"] = (geff.get("extra") or {}).get(
        "estimated_number_of_nodes"
    )
    return out


def read_image_meta(zarr_path):
    """Shape, dtype, scale and the precomputed intensity quantiles.

    Metadata only. Reading pixels for 199 samples would dominate the runtime and
    none of it is needed to characterise the annotation.
    """
    out = {}
    arr_meta = os.path.join(zarr_path, "0", "zarr.json")
    grp_meta = os.path.join(zarr_path, "zarr.json")
    if os.path.exists(arr_meta):
        with open(arr_meta, encoding="utf-8") as fh:
            arr = json.load(fh)
        out["image_shape"] = "x".join(str(s) for s in arr.get("shape", []))
        out["image_dtype"] = str(arr.get("data_type"))
    if os.path.exists(grp_meta):
        with open(grp_meta, encoding="utf-8") as fh:
            attrs = json.load(fh).get("attributes", {})
        q = (attrs.get("image_statistics") or {}).get("quantiles") or {}
        for key in ("0.0", "0.01", "0.1", "0.9", "0.99", "1.0"):
            out["q" + key] = q.get(key)
        ms = attrs.get("multiscales")
        if ms:
            tf = ms[0]["datasets"][0]["coordinateTransformations"][0]
            out["scale"] = ",".join(str(v) for v in tf.get("scale", []))
    return out


def profile(train_dir, sample):
    g = read_geff(os.path.join(train_dir, sample + ".geff"))
    t = g["t"]
    edges = g["edges"]
    n_nodes = int(g["ids"].size)

    if edges.size:
        n_edges = int(edges.shape[0])
        src, counts = np.unique(edges[:, 0], return_counts=True)
        n_div = int((counts >= 2).sum())
        n_div3 = int((counts >= 3).sum())
        has_parent = set(edges[:, 1].tolist())
        # Frame gap: a GT edge should join t and t+1. Anything else would mean the
        # annotation itself contains gap-closing links, which the scorer discards.
        id_to_t = dict(zip(g["ids"].tolist(), t.tolist()))
        gaps = np.array(
            [id_to_t[int(b)] - id_to_t[int(a)] for a, b in edges.tolist()]
        )
        n_gap_not_one = int((gaps != 1).sum())
    else:
        n_edges = n_div = n_div3 = n_gap_not_one = 0
        has_parent = set()

    n_roots = int(sum(1 for nid in g["ids"].tolist() if nid not in has_parent))
    uniq_t = np.unique(t) if n_nodes else np.array([])

    # Nodes per timepoint says how many lineages are traced concurrently.
    per_t = np.bincount(t.astype(np.int64)) if n_nodes else np.array([0])
    per_t = per_t[per_t > 0]

    row = {
        "sample": sample,
        "embryo": sample.split("_")[0],
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_divisions": n_div,
        "n_divisions_3plus": n_div3,
        "n_roots": n_roots,
        "n_edges_gap_not_one": n_gap_not_one,
        "t_min": int(t.min()) if n_nodes else -1,
        "t_max": int(t.max()) if n_nodes else -1,
        "n_timepoints_annotated": int(uniq_t.size),
        "nodes_per_t_mean": round(float(per_t.mean()), 3) if per_t.size else 0.0,
        "nodes_per_t_max": int(per_t.max()) if per_t.size else 0,
    }
    for axis in ("z", "y", "x"):
        v = g[axis]
        row[axis + "_min"] = float(v.min()) if n_nodes else -1
        row[axis + "_max"] = float(v.max()) if n_nodes else -1
    row.update(read_geff_meta(os.path.join(train_dir, sample + ".geff")))
    row.update(read_image_meta(os.path.join(train_dir, sample + ".zarr")))
    return row


def main():
    train_dir = os.path.join(DATA, "train")
    samples = sorted(
        d[: -len(".geff")] for d in os.listdir(train_dir) if d.endswith(".geff")
    )
    print("%d train samples under %s" % (len(samples), train_dir))

    rows = []
    t0 = time.time()
    for i, sample in enumerate(samples, 1):
        try:
            rows.append(profile(train_dir, sample))
        except Exception as exc:
            print("  FAIL %s: %r" % (sample, exc))
        if i % 25 == 0:
            print("  %d/%d  %.1fs" % (i, len(samples), time.time() - t0), flush=True)

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    out_dir = os.path.dirname(os.path.abspath(OUT))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print("wrote %d rows to %s in %.1fs" % (len(rows), OUT, time.time() - t0))

    # Print the headline numbers so they are readable in the kernel log even if
    # the output file is awkward to fetch.
    n = np.array([r["n_nodes"] for r in rows], dtype=float)
    d = np.array([r["n_divisions"] for r in rows], dtype=float)
    est = np.array(
        [float(r["estimated_number_of_nodes"] or "nan") for r in rows], dtype=float
    )
    gap = np.array([r["n_edges_gap_not_one"] for r in rows], dtype=float)
    print("\n=== headline ===")
    print("samples: %d" % len(rows))
    for emb in sorted({r["embryo"] for r in rows}):
        sel = [r for r in rows if r["embryo"] == emb]
        print(
            "  embryo %s: %d samples, %d annotated nodes, %d annotated divisions"
            % (
                emb,
                len(sel),
                sum(r["n_nodes"] for r in sel),
                sum(r["n_divisions"] for r in sel),
            )
        )
    print("total annotated nodes: %d" % n.sum())
    print("total annotated divisions: %d" % d.sum())
    print("total estimated true nodes: %.0f" % np.nansum(est))
    print("annotated coverage: %.3f%%" % (100 * n.sum() / np.nansum(est)))
    print("samples with zero annotated divisions: %d" % int((d == 0).sum()))
    print("GT edges not spanning exactly one frame: %d" % gap.sum())


main()
