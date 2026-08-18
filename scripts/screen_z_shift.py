"""What is the detector's systematic Z bias worth, if anything?

`scripts/diagnose_localisation.py` found that predictions sit a median 1.62 um
below the annotated centre in Z, which is exactly one Z voxel, and that Z carries
61% of the squared localisation error on cells the detector already matches.

The first thing checked was whether this is a coordinate convention error. It is
not: ground-truth z spans the full 0 to 63 voxel range, is entirely integer, and
uses the same 0-based indexing as the image. So the bias is a real property of the
detector, most plausibly the light-sheet PSF being asymmetric in Z combined with
Z being smoothed far less than Y and X in voxel terms (sigma 1.23 voxels against
4.92).

This screens a constant Z offset applied to every detection. A constant shift
changes no node counts, so the node-count penalty is untouched and the whole effect
is localisation. That makes it about the cleanest single-variable test available
here.

**Read the per-embryo columns before believing the pooled number.** A shift chosen
because it helps one acquisition is fitting that acquisition, and the hidden test
is a different embryo. This is a screen on 9 samples, not a CV result, and it is
also a parameter that would have to be selected on training folds rather than on
everything, so nothing is promoted straight from here.

    python scripts/screen_z_shift.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import DEFAULT_SCALE_ZYX, Image, read_scale  # noqa: E402
from src.detect import detect_sequence  # noqa: E402
from src.link import link_sequence  # noqa: E402

# In Z voxels. 1 voxel is 1.625 um, so this brackets the measured -1 voxel median.
SHIFTS = (0.0, 0.5, 1.0, 1.5, 2.0)

SAMPLES = (
    "44b6_144b256d", "44b6_90724892", "44b6_d5e7d891", "44b6_db3c847b",
    "6bba_268e1230", "6bba_3db54e20", "6bba_55c70843", "6bba_67ebd073",
    "6bba_6ca87370",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/baseline.yaml")
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--out", default="data/meta/z_shift_screen.csv")
    ap.add_argument("--shifts", type=float, nargs="+", default=list(SHIFTS))
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg.detect or {}
    link_cfg = cfg.link or {}
    rows = []
    t0 = time.time()

    for i, sample in enumerate(SAMPLES, 1):
        zarr_path = os.path.join(args.data, sample + ".zarr")
        gt_geff = os.path.join(args.data, sample + ".geff")
        image = Image(zarr_path)
        scale = read_scale(zarr_path) or DEFAULT_SCALE_ZYX
        depth = image.shape[1]

        det = detect_sequence(
            image,
            sigma_um=float(d.get("sigma_um", 2.0)),
            min_sep_um=float(d.get("min_sep_um", 3.0)),
            threshold_scale=float(d.get("threshold_scale", 0.5)),
            max_detections=int(d.get("max_detections", 20000)),
        )
        for sh in args.shifts:
            # Shift in Z only, clipped to the volume so a shifted detection cannot
            # leave the image and become unmatchable for the wrong reason.
            shifted = []
            for c in det:
                if c.shape[0] == 0:
                    shifted.append(c)
                    continue
                c2 = c.copy()
                c2[:, 0] = np.clip(c2[:, 0] + sh, 0, depth - 1)
                shifted.append(c2)
            graph = link_sequence(
                shifted, scale_zyx=image.scale,
                max_link_um=float(link_cfg.get("max_link_um", 7.0)),
                max_division_um=float(link_cfg.get("max_division_um", 0.0)),
            )
            s = metrics.score_prediction(graph, gt_geff, sample=sample, scale=scale)
            rows.append({
                "sample": sample, "embryo": sample.split("_")[0], "z_shift_vox": sh,
                "edge_tp": s.edge_tp, "edge_fp": s.edge_fp, "edge_fn": s.edge_fn,
                "node_recall": s.node_recall, "num_pred_nodes": s.num_pred_nodes,
                "edge_jaccard": s.edge_jaccard, "adj_edge_jaccard": s.adj_edge_jaccard,
            })
        print(f"  [{i}/{len(SAMPLES)}] {sample} [{(time.time()-t0)/60:.1f} min]", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def weighted(rs):
        w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rs]
        return sum(a * r["adj_edge_jaccard"] for a, r in zip(w, rs)) / sum(w)

    print(f"\nconfig {cfg.name}, {len(SAMPLES)} samples. node counts are identical "
          f"across shifts, so this is pure localisation.\n")
    print(f"{'z shift':>9}{'um':>8}{'edge_tp':>9}{'edge_fp':>9}{'edge_fn':>9}"
          f"{'recall':>9}{'pooled':>9}{'44b6':>9}{'6bba':>9}")
    for sh in args.shifts:
        rs = [r for r in rows if r["z_shift_vox"] == sh]
        tp = sum(r["edge_tp"] for r in rs)
        fp = sum(r["edge_fp"] for r in rs)
        fn = sum(r["edge_fn"] for r in rs)
        rec = np.mean([r["node_recall"] for r in rs])
        a = weighted([r for r in rs if r["embryo"] == "44b6"])
        b = weighted([r for r in rs if r["embryo"] == "6bba"])
        mark = "  <- baseline" if sh == 0.0 else ""
        print(f"{sh:>9}{sh*1.625:>8.2f}{tp:>9}{fp:>9}{fn:>9}{rec:>9.3f}"
              f"{weighted(rs):>9.4f}{a:>9.4f}{b:>9.4f}{mark}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
