"""Measure how detection responds to `min_sep_um` and `sigma_um`, without scoring.

Why this exists. The thr030 sweep cut `threshold_scale` by 40% and moved node
counts by 0.5% and 1.2%. A detector that were threshold-limited would respond far
more than that, so NOTES.md names `detect.min_sep_um: 3.0` as the suspected binding
constraint: non-maximum suppression at a 3 um radius caps how many peaks a frame
can emit no matter how low the acceptance threshold goes. That is a hypothesis
consistent with node counts, not a measured result, and a full leave-one-embryo-out
run costs about 5.5 hours. This probe costs minutes and either confirms it or kills
it first.

`sigma_um` is on the grid alongside it because the two are coupled and only one of
them can be the culprit. The Gaussian runs *before* the maximum filter, so a 2.0 um
smoothing kernel can merge two adjacent nuclei into a single peak that NMS then
never has to separate. Probing separation alone could exonerate the wrong
parameter: if node counts move under sigma but not under min_sep, the recall
shortfall is a smoothing artefact and the whole "cannot place two cells closer than
3 um" reading is wrong.

Two quantities come out, and the second is the one that decides anything:

- **detections per frame**, against the organisers' `estimated_number_of_nodes`.
  Tells us whether the parameter moves the detector at all.
- **node recall**, the fraction of annotated GT nodes with a detection inside the
  metric's 7 um cap. Tells us whether the extra detections are *real cells*. More
  nodes at flat recall is over-detection buying nothing, and the node-count penalty
  would charge for it. Only recall moving is evidence of a lever.

Detection only. No linking, no scoring, no ledger row. This does not produce a CV
number and nothing gets promoted on it; it chooses which single config is worth a
5.5 hour run.

    python scripts/probe_detect_grid.py --out data/meta/detect_grid.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import Image, read_geff  # noqa: E402
from src.detect import _odd  # noqa: E402

# The grid. `threshold_scale` is held at the baseline 0.5 throughout, so this is
# one variable family against conf/baseline.yaml rather than a free-for-all.
SIGMA_UM = (2.0, 1.25, 0.75)
# These four are chosen because they land on four *distinct* suppression
# footprints. `min_sep_um` is converted to a voxel footprint and forced odd, and
# the data is anisotropic enough that the mapping is badly conditioned in Z: 3.0
# and 2.5 both give (3, 7, 7), and 2.0 and 1.5 both give (1, 5, 5). A first pass
# at 3.0/2.0/1.5/1.0 therefore measured three settings while appearing to measure
# four. Read the footprint, not the micron value.
MIN_SEP_UM = (3.5, 3.0, 2.0, 1.0)
THRESHOLD_SCALE = 0.5

# The metric matches nodes by optimal assignment under this cap, in microns.
MAX_MATCH_UM = 7.0

N_TIMEPOINTS = 5

# The canonical failure case: 0.05x the true node count while samples of similar
# density sit near 0.7x. Undiagnosed, and forced onto the sample list because if
# the collapse is a smoothing or suppression artefact this grid will show it.
FORCED = ("6bba_3db54e20",)

# Quantiles of per-embryo cell density to sample at. Density varies twentyfold
# across the 199 crops and a parameter that helps the sparse ones can easily hurt
# the dense ones, so the probe has to span the range rather than sit in its middle.
DENSITY_QUANTILES = (0.10, 0.35, 0.65, 0.90)


def load_profile(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def choose_samples(profile: list[dict]) -> list[dict]:
    """Pick samples spanning both embryos and the full density range, deterministically."""
    chosen: dict[str, dict] = {}
    by_embryo: dict[str, list[dict]] = {}
    for row in profile:
        by_embryo.setdefault(row["embryo"], []).append(row)

    for embryo, rows in sorted(by_embryo.items()):
        rows = sorted(rows, key=lambda r: float(r["estimated_number_of_nodes"]))
        for q in DENSITY_QUANTILES:
            idx = min(len(rows) - 1, int(round(q * (len(rows) - 1))))
            chosen[rows[idx]["sample"]] = rows[idx]

    by_name = {r["sample"]: r for r in profile}
    for name in FORCED:
        if name not in by_name:
            # Loudly, because a silently skipped sample is how a probe ends up
            # covering one embryo while its output looks complete.
            raise SystemExit(f"{name} is not in the profile; wrong profile file?")
        chosen[name] = by_name[name]
    return [chosen[k] for k in sorted(chosen)]


def choose_timepoints(row: dict, n: int) -> list[int]:
    """Timepoints spread over the sample's *annotated* range.

    43 of the 199 samples start late, one not until t=46. Probing fixed timepoints
    would compute recall against an empty ground truth on those, so the window
    follows the annotation rather than the video.
    """
    lo, hi = int(row["t_min"]), int(row["t_max"])
    if hi <= lo:
        return [lo]
    return sorted({int(round(v)) for v in np.linspace(lo, hi, n)})


def _distances(
    det_zyx: np.ndarray, gt_zyx: np.ndarray, scale: tuple[float, float, float]
) -> np.ndarray:
    """GT-by-detection distance matrix in microns.

    Physical space, not voxel space. The data is anisotropic by a factor of four,
    so a voxel-space distance would not be the quantity the 7 um cap is about.
    GT nodes per frame are single digits to low tens, so the full matrix is small
    and a KD-tree would cost more than it saves.
    """
    s = np.asarray(scale, dtype=np.float64)[None, :]
    return np.linalg.norm((gt_zyx * s)[:, None, :] - (det_zyx * s)[None, :, :], axis=2)


def node_hits(dist: np.ndarray) -> int:
    """GT nodes with *any* detection within the 7 um cap.

    Deliberately not the scorer's number, and it degenerates as detections get
    dense: a 104 um cube holding 2,000 detections has a mean spacing near 3 um, so
    almost every GT node has something inside 7 um whether or not the detector
    found that cell. Kept only as the optimistic bound to compare `node_matched`
    against; a setting where the two diverge is over-detecting, not detecting.
    """
    if dist.size == 0:
        return 0
    return int((dist.min(axis=1) <= MAX_MATCH_UM).sum())


def node_matched(dist: np.ndarray) -> int:
    """GT nodes matched under optimal bipartite assignment, capped at 7 um.

    This mirrors what the scorer does, and it is the number that means something:
    each detection serves at most one GT node, so stacking duplicate detections on
    one cell cannot inflate it.
    """
    if dist.size == 0:
        return 0
    from scipy.optimize import linear_sum_assignment

    # Forbid over-cap pairs outright rather than filtering after assignment, so the
    # optimiser cannot spend a detection on a pair that would be thrown away.
    cost = np.where(dist <= MAX_MATCH_UM, dist, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return int((dist[rows, cols] <= MAX_MATCH_UM).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/train")
    # sample_profile.csv is the complete 199-row profile. Note that
    # sample_profile_full.csv, despite the name, holds only 22 rows and only 44b6.
    ap.add_argument("--profile", default="data/meta/sample_profile.csv")
    ap.add_argument("--out", default="data/meta/detect_grid.csv")
    ap.add_argument("--n-timepoints", type=int, default=N_TIMEPOINTS)
    ap.add_argument("--sigma", type=float, nargs="+", default=list(SIGMA_UM))
    ap.add_argument("--min-sep", type=float, nargs="+", default=list(MIN_SEP_UM))
    args = ap.parse_args()

    sigmas, min_seps = tuple(args.sigma), tuple(args.min_sep)

    from src.detect import detect_frame

    samples = choose_samples(load_profile(args.profile))
    n_settings = len(sigmas) * len(min_seps)
    print(
        f"{len(samples)} samples x {n_settings} settings x {args.n_timepoints} "
        f"timepoints = {len(samples) * n_settings * args.n_timepoints} detections",
        flush=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fh = open(args.out, "w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(
        (
            "sample", "embryo", "est_nodes", "cells_per_frame",
            "sigma_um", "min_sep_um", "fp_z", "fp_y", "fp_x", "threshold_scale",
            "t", "n_det", "n_gt", "n_gt_hit", "n_gt_matched", "detect_s",
        )
    )

    t_start = time.time()
    for i, row in enumerate(samples, 1):
        name = row["sample"]
        est_nodes = float(row["estimated_number_of_nodes"])
        cells_per_frame = est_nodes / 100.0

        image = Image(os.path.join(args.data, name + ".zarr"))
        graph = read_geff(os.path.join(args.data, name + ".geff"))
        ts = choose_timepoints(row, args.n_timepoints)

        # Read each timepoint once and reuse it across all 12 settings. One frame
        # is one stored chunk at ~0.086 s, so re-reading per setting would add
        # nothing but wall clock.
        frames = {t: image.frame(t) for t in ts}
        gt_at = {
            t: graph.nodes.zyx()[graph.nodes.t == t] for t in ts
        }

        for sigma in sigmas:
            for min_sep in min_seps:
                for t in ts:
                    t0 = time.time()
                    det = detect_frame(
                        frames[t],
                        scale_zyx=image.scale,
                        quantiles=image.quantiles,
                        sigma_um=sigma,
                        min_sep_um=min_sep,
                        threshold_scale=THRESHOLD_SCALE,
                    )
                    dt = time.time() - t0
                    gt = gt_at[t]
                    if det.shape[0] and gt.shape[0]:
                        dist = _distances(det, gt, image.scale)
                    else:
                        dist = np.empty((0, 0))
                    # Record the footprint actually used, because the micron value
                    # alone does not identify the setting.
                    fp = _odd(np.asarray(min_sep, dtype=np.float64) / np.asarray(image.scale))
                    writer.writerow(
                        (
                            name, row["embryo"], int(est_nodes), round(cells_per_frame, 2),
                            sigma, min_sep, int(fp[0]), int(fp[1]), int(fp[2]),
                            THRESHOLD_SCALE,
                            t, det.shape[0], gt.shape[0],
                            node_hits(dist), node_matched(dist), round(dt, 3),
                        )
                    )
                fh.flush()
        elapsed = (time.time() - t_start) / 60.0
        print(
            f"  [{i}/{len(samples)}] {name} "
            f"({cells_per_frame:.0f} cells/frame) done [{elapsed:.1f} min]",
            flush=True,
        )

    fh.close()
    print(f"wrote {args.out} in {(time.time() - t_start) / 60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
