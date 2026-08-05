"""Can tensorstore read the competition zarr, and how fast, and what does a
trivial local-maximum detector find?

Three questions in one CPU kernel, because each kernel run costs a session:

1. tensorstore is the only zarr v3 reader present at rerun time. Prove it opens
   these files and returns the right dtype and shape.
2. Time a per-timepoint read. The rerun has 12 h for roughly 200 unseen samples of
   100 timepoints each, so the per-frame budget is about 2 seconds all-in. If
   reading alone eats that, the whole design has to change.
3. Calibrate detection. `estimated_number_of_nodes` spans 3,783 to 78,644 across
   samples, so cell density varies by a factor of 20 and a fixed threshold cannot
   be right. Measure what a simple detector returns against that target.

Writes /kaggle/working/detect_probe.csv.
"""

import csv
import json
import os
import time

import numpy as np
import scipy.ndimage as ndi
import tensorstore as ts

DATA = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
TRAIN = os.path.join(DATA, "train")
OUT = "/kaggle/working/detect_probe.csv"

# Microns per voxel. Matching is capped at 7 um in physical space, so every radius
# below is specified in microns and converted, never in voxels.
SCALE_ZYX = np.array([1.625, 0.40625, 0.40625])


def open_image(sample):
    """Open one sample's image array with tensorstore. No data is read yet."""
    return ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": os.path.join(TRAIN, sample + ".zarr", "0")},
        },
        read=True,
    ).result()


def estimated_nodes(sample):
    path = os.path.join(TRAIN, sample + ".geff", "zarr.json")
    with open(path, encoding="utf-8") as fh:
        geff = json.load(fh)["attributes"]["geff"]
    return (geff.get("extra") or {}).get("estimated_number_of_nodes")


def quantiles(sample):
    path = os.path.join(TRAIN, sample + ".zarr", "zarr.json")
    with open(path, encoding="utf-8") as fh:
        attrs = json.load(fh).get("attributes", {})
    q = (attrs.get("image_statistics") or {}).get("quantiles") or {}
    return {float(k): float(v) for k, v in q.items()}


def detect(frame, q, sigma_um=2.0, min_sep_um=4.0, thresholds=(0.2, 0.4, 0.6, 0.8)):
    """Smooth, then take local maxima above a threshold, at several thresholds.

    Returns {threshold: n_detections}. Deliberately the dumbest thing that could
    work: the point is to see how detection count moves with threshold relative to
    the true cell count, not to be good yet.
    """
    lo = q.get(0.01, float(frame.min()))
    hi = q.get(0.99, float(frame.max()))
    img = np.clip((frame.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, None)

    sigma_vox = sigma_um / SCALE_ZYX
    sm = ndi.gaussian_filter(img, sigma=sigma_vox)

    # Local-maximum footprint sized in physical space, so it is anisotropic in
    # voxels: about 2.5 voxels in Z against 10 in Y/X for a 4 um separation.
    size = np.maximum(1, np.round(min_sep_um / SCALE_ZYX).astype(int))
    size = size + (size % 2 == 0)  # odd, so the window is centred
    mx = ndi.maximum_filter(sm, size=tuple(size))
    is_peak = (sm == mx) & (sm > 0)

    out = {}
    for thr in thresholds:
        out[thr] = int(np.count_nonzero(is_peak & (sm > thr)))
    return out, sm


def main():
    samples = sorted(
        d[: -len(".geff")] for d in os.listdir(TRAIN) if d.endswith(".geff")
    )
    # A spread over both embryos and over the density range, not the first N.
    by_emb = {}
    for s in samples:
        by_emb.setdefault(s.split("_")[0], []).append(s)
    picked = []
    for emb, group in sorted(by_emb.items()):
        group = sorted(group, key=lambda s: estimated_nodes(s) or 0)
        idx = [0, len(group) // 4, len(group) // 2, 3 * len(group) // 4, len(group) - 1]
        picked.extend(group[i] for i in sorted(set(idx)))
    print("probing %d samples: %s" % (len(picked), picked))

    thresholds = (0.2, 0.4, 0.6, 0.8)
    rows = []
    for sample in picked:
        est = estimated_nodes(sample)
        q = quantiles(sample)
        arr = open_image(sample)
        print("\n%s  shape=%s dtype=%s  est_N=%s (%.0f cells/frame)"
              % (sample, arr.shape, arr.dtype, est, (est or 0) / 100.0))

        t0 = time.time()
        frame = np.asarray(arr[10].read().result())
        t_read = time.time() - t0

        t0 = time.time()
        counts, _ = detect(frame, q, thresholds=thresholds)
        t_detect = time.time() - t0

        target = (est or 0) / 100.0
        print("  read %.2fs  detect %.2fs" % (t_read, t_detect))
        for thr in thresholds:
            ratio = counts[thr] / target if target else float("nan")
            print("    thr=%.1f -> %6d detections  (%.2f x true count)"
                  % (thr, counts[thr], ratio))

        row = {
            "sample": sample,
            "embryo": sample.split("_")[0],
            "estimated_number_of_nodes": est,
            "cells_per_frame": target,
            "read_s": round(t_read, 3),
            "detect_s": round(t_detect, 3),
            "frame_dtype": str(frame.dtype),
            "frame_shape": "x".join(str(s) for s in frame.shape),
        }
        for thr in thresholds:
            row["n_thr_%.1f" % thr] = counts[thr]
        rows.append(row)

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s" % OUT)

    read_times = np.array([r["read_s"] for r in rows])
    detect_times = np.array([r["detect_s"] for r in rows])
    print("\n=== runtime budget ===")
    print("median read   %.3fs/frame" % np.median(read_times))
    print("median detect %.3fs/frame" % np.median(detect_times))
    per_frame = np.median(read_times) + np.median(detect_times)
    print("=> %.2fs/frame, %.1f min per 100-frame sample, %.1f h for 200 samples"
          % (per_frame, per_frame * 100 / 60, per_frame * 100 * 200 / 3600))


main()
