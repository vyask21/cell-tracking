"""Does suppression stated in physical space beat the voxel box?

`maximum_filter(size=...)` takes a box in voxels. At `min_sep_um` 3.0 and this
data's anisotropy that box is 3 x 7 x 7 voxels, which is a suppression radius of
1.625 um in Z against 1.219 um in Y and X, and it suppresses a neighbour one Z
step away however far that neighbour sits in Y and X.
`scripts/diagnose_localisation.py` measured Z as carrying 61% of the squared
localisation error, and `diagnose_missed_nodes.py` found 86.7% of unmatched
ground-truth cells have a detection 7 to 12 um away, so the misses are
mislocalised rather than absent. `z_shift_vox` attacked that as a symptom and
failed. This attacks the footprint that causes it.

The ladder holds in-plane suppression roughly fixed and sweeps only Z. The
ellipsoid radii are 1.605 um in Y and X, which turns on 45 voxels in the central
plane against the box's 49, so in-plane strength is matched and the Z profile is
the variable:

    box       3 x 7 x 7, 147 voxels on, the current setting and the control
    rz 1.200  no Z suppression at all, 45 on
    rz 1.625  the box's Z radius, near column only at one Z step, 47 on
    rz 2.400  one Z step, 25 of 49 in-plane, 95 on
    rz 3.300  two Z steps, 121 on

Runs on the 30-sample set from `scripts/build_screen_set.py`, which matches the
full 199 to +0.0010 on baseline weighted score. That set carries about 0.010 of
screening noise at the 90% level, so anything smaller than that says nothing.
Nothing is promoted from here; a winner earns a full 199-sample run and no more.

The `box` arm is not redundant. It reruns the current setting through this
harness, so if it does not reproduce the cached baseline weighted score on these
same 30 samples the harness is wrong and every other row is meaningless.

Per-sample results cache to `--cache`, so a killed run resumes for free. Samples
are spread over worker processes rather than arms, so each sample's zarr is read
once by one worker instead of once per arm.

Launch it detached, never as a harness background task, and never tail it with a
monitor. Both have silently reaped this repo's long jobs before:

    Start-Process -FilePath .venv\\Scripts\\python.exe `
      -ArgumentList "scripts/screen_ellipsoid_nms.py","--workers","6" `
      -RedirectStandardOutput artifacts/ellipsoid_screen.out `
      -RedirectStandardError artifacts/ellipsoid_screen.err -NoNewWindow
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import DEFAULT_SCALE_ZYX, Image, read_scale  # noqa: E402
from src.detect import detect_sequence, ellipsoid_footprint  # noqa: E402
from src.link import link_sequence  # noqa: E402

YX_UM = 1.605  # in-plane radius matching the box's 49-voxel central plane
Z_RADII = (1.200, 1.625, 2.400, 3.300)
BASELINE_CACHE = "artifacts/scores/baseline_localmax_4f8a3af0.csv"

FIELDS = [
    "sample", "embryo", "arm", "rz_um", "edge_tp", "edge_fp", "edge_fn",
    "num_pred_nodes", "node_recall", "total_node_ratio", "edge_jaccard",
    "adj_edge_jaccard", "seconds",
]


def build_arms(z_radii, yx_um) -> list[tuple[str, tuple | None]]:
    arms = [("box", None)]
    for rz in z_radii:
        arms.append((f"rz{rz:.3f}", (rz, yx_um, yx_um)))
    return arms


def load_set(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [r["sample"] for r in csv.DictReader(fh)]


def cached_baseline(samples: list[str]) -> float | None:
    """Weighted baseline score over these samples, read from the run cache."""
    if not os.path.exists(BASELINE_CACHE):
        return None
    keep = set(samples)
    v, w = [], []
    with open(BASELINE_CACHE, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["sample"] in keep:
                v.append(float(r["adj_edge_jaccard"]))
                w.append(int(r["edge_tp"]) + int(r["edge_fp"]) + int(r["edge_fn"]))
    if not v:
        return None
    return float(np.average(v, weights=w))


def weighted(rows: list[dict]) -> float:
    """The scorer weight-averages per-sample adjusted Jaccard by TP+FP+FN."""
    w = np.array([int(r["edge_tp"]) + int(r["edge_fp"]) + int(r["edge_fn"])
                  for r in rows], dtype=float)
    v = np.array([float(r["adj_edge_jaccard"]) for r in rows], dtype=float)
    return float(np.sum(v * w) / np.sum(w))


def run_sample(sample: str, data_dir: str, cache_dir: str, config: str,
               z_radii: list[float], yx_um: float) -> tuple[str, list[dict], float]:
    """Score every arm on one sample. Cached, so a rerun costs nothing."""
    out_path = os.path.join(cache_dir, sample + ".csv")
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as fh:
            return sample, list(csv.DictReader(fh)), 0.0

    cfg = load_config(config)
    d = cfg.detect or {}
    link_cfg = cfg.link or {}
    zarr_path = os.path.join(data_dir, sample + ".zarr")
    gt_geff = os.path.join(data_dir, sample + ".geff")
    image = Image(zarr_path)
    scale = read_scale(zarr_path) or DEFAULT_SCALE_ZYX

    rows = []
    t_sample = time.time()
    for name, radii in build_arms(z_radii, yx_um):
        t0 = time.time()
        det = detect_sequence(
            image,
            sigma_um=float(d.get("sigma_um", 2.0)),
            min_sep_um=float(d.get("min_sep_um", 3.0)),
            threshold_scale=float(d.get("threshold_scale", 0.5)),
            max_detections=int(d.get("max_detections", 20000)),
            suppress_radii_um=radii,
            z_shift_vox=float(d.get("z_shift_vox", 0.0)),
        )
        graph = link_sequence(
            det,
            scale_zyx=image.scale,
            max_link_um=float(link_cfg.get("max_link_um", 7.0)),
            max_division_um=float(link_cfg.get("max_division_um", 0.0)),
        )
        s = metrics.score_prediction(graph, gt_geff, sample=sample, scale=scale)
        rows.append({
            "sample": sample, "embryo": sample.split("_")[0], "arm": name,
            "rz_um": "" if radii is None else radii[0],
            "edge_tp": s.edge_tp, "edge_fp": s.edge_fp, "edge_fn": s.edge_fn,
            "num_pred_nodes": s.num_pred_nodes, "node_recall": s.node_recall,
            "total_node_ratio": s.total_node_ratio,
            "edge_jaccard": s.edge_jaccard, "adj_edge_jaccard": s.adj_edge_jaccard,
            "seconds": round(time.time() - t0, 1),
        })

    # Write through a temp name so a kill mid-write cannot leave a half row set
    # that a resume would then trust.
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_path)
    return sample, rows, time.time() - t_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/baseline.yaml")
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--set", default="data/meta/screen_set.csv")
    ap.add_argument("--out", default="data/meta/ellipsoid_screen.csv")
    ap.add_argument("--cache", default="data/meta/ellipsoid_cache")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--z-radii", type=float, nargs="+", default=list(Z_RADII))
    ap.add_argument("--yx-um", type=float, default=YX_UM)
    args = ap.parse_args()

    samples = load_set(args.set)
    arms = build_arms(args.z_radii, args.yx_um)
    os.makedirs(args.cache, exist_ok=True)

    print(f"{len(samples)} samples, {len(arms)} arms, {args.workers} workers, "
          f"config {args.config}", flush=True)
    for name, radii in arms:
        if radii is None:
            print(f"  {name:>10}: voxel box, 3 x 7 x 7, 147 on")
        else:
            m = ellipsoid_footprint(radii, DEFAULT_SCALE_ZYX)
            print(f"  {name:>10}: ellipsoid {radii}, shape {m.shape}, "
                  f"{int(m.sum())} on")
    done = sum(1 for s in samples if os.path.exists(os.path.join(args.cache, s + ".csv")))
    print(f"  {done} of {len(samples)} already cached\n", flush=True)

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_sample, s, args.data, args.cache, args.config,
                        args.z_radii, args.yx_um): s
            for s in samples
        }
        for i, fut in enumerate(as_completed(futures), 1):
            sample, sample_rows, secs = fut.result()
            rows += sample_rows
            el = (time.time() - t0) / 60
            tag = "cached" if secs == 0.0 else f"{secs / 60:.1f} min"
            print(f"  [{i}/{len(samples)}] {sample} ({tag}) "
                  f"[{el:.1f} min elapsed, eta {el / i * (len(samples) - i):.0f} min]",
                  flush=True)

    rows.sort(key=lambda r: (r["sample"], r["arm"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    cached = cached_baseline(samples)
    box = weighted([r for r in rows if r["arm"] == "box"])
    shown = "n/a" if cached is None else f"{cached:.4f}"
    print(f"\nharness check: box arm {box:.4f} against cached baseline {shown}")
    if cached is not None:
        print(f"  difference {box - cached:+.5f}")
        if abs(box - cached) > 1e-4:
            print("  WARNING: the box arm does not reproduce the cache. Treat "
                  "every row below as unverified until this is explained.")

    print(f"\n{'arm':>10}{'edge_tp':>9}{'edge_fp':>9}{'edge_fn':>9}{'nodes':>10}"
          f"{'node_ratio':>12}{'weighted adj_J':>16}{'vs box':>9}")
    for name, _ in arms:
        rs = [r for r in rows if r["arm"] == name]
        wa = weighted(rs)
        nr = float(np.mean([float(r["total_node_ratio"]) for r in rs]))
        print(f"{name:>10}{sum(int(r['edge_tp']) for r in rs):>9}"
              f"{sum(int(r['edge_fp']) for r in rs):>9}"
              f"{sum(int(r['edge_fn']) for r in rs):>9}"
              f"{sum(int(r['num_pred_nodes']) for r in rs):>10}{nr:>12.3f}"
              f"{wa:>16.4f}{wa - box:>+9.4f}")

    print("\nper embryo, since a change is believed only when it helps both:")
    for emb in ("44b6", "6bba"):
        base = weighted([r for r in rows if r["arm"] == "box" and r["embryo"] == emb])
        cells = []
        for name, _ in arms:
            rs = [r for r in rows if r["arm"] == name and r["embryo"] == emb]
            cells.append(f"{name} {weighted(rs) - base:+.4f}")
        print(f"  {emb} (box {base:.4f}): " + "  ".join(cells))

    print("\nScreening noise on this 30-sample set is about 0.010 at the 90% level.")
    print("Anything smaller is not a result, and nothing is promoted without a")
    print("full 199-sample run.")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min, wrote {args.out}")


if __name__ == "__main__":
    main()
