"""Cache the expensive half of the pipeline so post-processing arms are cheap.

Screening the graph-calibration chain the way the workspace rules demand, one
variable per config, costs about 70 minutes an arm if each arm re-runs the U-Net.
That is nearly all waste: detection and edge scoring do not depend on anything
the chain does. This runs them once per sample and writes the result to disk, so
every downstream arm reduces to an ILP solve plus some graph surgery.

The same trick paid for the `max_link_um` screen on 2026-08-18, where detection
ran once and five caps shared it. It is worth more here, because the chain has
six or seven steps and each needs its own arm.

**What is cached and what is deliberately not.** Detections and edge affinities
are cached. The ILP is not, because several planned arms change its input: the
edge-length gate is one of the settings under test. The ILP costs seconds on most
videos and up to ten minutes on the largest, which is acceptable per arm; the
U-Net pass is ten minutes on every video, which is not.

**Why the candidate gate is wide here.** `conf/unet50_ilp.yaml` gates candidates
at 7 um, and that gate is one of the things being questioned: the public 0.902
notebook keeps edges out to 14 um. Caching at 7 um would bake in the setting under
test, so the cache is built at `--gate-um 20` and every arm narrows from there.
Narrowing a cached candidate set gives exactly the set that gating at that value
would have produced, because the gate is a pure distance filter applied after
scoring.

**Why each detection threshold needs its own cache.** The peak test itself is
threshold-independent, so it is tempting to detect once at a low threshold and
filter. That would be wrong: the edge head's softmax normalises over the source
nodes present in the frame, so admitting extra low-confidence nodes changes the
probabilities assigned to the good ones. Threshold is therefore a cache key.

    .venv\\Scripts\\python.exe scripts/cache_graphs.py --det-threshold 0.99

Launch detached; about 70 minutes for the 19 on three workers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HELDOUT = (
    "44b6_1574802b", "44b6_706092f0", "44b6_d5e7d891", "44b6_d754aa59",
    "44b6_e57ff5c6", "6bba_2312ac41", "6bba_268e1230", "6bba_283bf9f1",
    "6bba_3a1849c2", "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d",
    "6bba_7af54fde", "6bba_7b5d3b2c", "6bba_aeee7805", "6bba_afb141ff",
    "6bba_c27cba08", "6bba_c328f2fd", "6bba_d1acb6ff",
)


def cache_path(cache_dir: str, sample: str) -> str:
    return os.path.join(cache_dir, sample + ".npz")


def save_cache(path: str, detections, affinities, scale) -> None:
    """Flatten the ragged per-frame arrays into one npz.

    Stored as concatenated arrays plus offsets rather than an object array, so
    the file loads without pickle and stays readable from a Kaggle kernel.
    """
    det_counts = np.array([d.shape[0] for d in detections], dtype=np.int64)
    det_all = (np.concatenate(detections, axis=0) if len(detections)
               else np.empty((0, 3)))
    aff_counts = np.array([a["i"].size for a in affinities], dtype=np.int64)
    empty32 = np.empty(0, dtype=np.int32)
    np.savez_compressed(
        path,
        det_counts=det_counts,
        det=det_all.astype(np.float32),
        aff_counts=aff_counts,
        aff_i=(np.concatenate([a["i"] for a in affinities]) if affinities else empty32),
        aff_j=(np.concatenate([a["j"] for a in affinities]) if affinities else empty32),
        aff_p=(np.concatenate([a["p"] for a in affinities]) if affinities
               else np.empty(0, dtype=np.float32)),
        scale=np.asarray(scale, dtype=np.float64),
    )


def load_cache(path: str):
    """Inverse of `save_cache`. Returns (detections, affinities, scale)."""
    z = np.load(path)
    det = z["det"].astype(np.float64)
    bounds = np.concatenate([[0], np.cumsum(z["det_counts"])])
    detections = [det[bounds[k]:bounds[k + 1]] for k in range(len(z["det_counts"]))]
    ab = np.concatenate([[0], np.cumsum(z["aff_counts"])])
    affinities = [
        {"i": z["aff_i"][ab[k]:ab[k + 1]],
         "j": z["aff_j"][ab[k]:ab[k + 1]],
         "p": z["aff_p"][ab[k]:ab[k + 1]]}
        for k in range(len(z["aff_counts"]))
    ]
    return detections, affinities, tuple(z["scale"])


def run_sample(sample: str, data_dir: str, cache_dir: str, det_threshold: float,
               pool_kernel_um: float, det_tta: bool, det_tta_views: int,
               gate_um: float, edge_threshold: float, threads: int):
    out = cache_path(cache_dir, sample)
    if os.path.exists(out):
        return sample, 0.0, -1, -1

    import torch
    torch.set_num_threads(max(1, threads))

    from src.data import Image
    from src.unet import detect_and_score_sequence, load_detector

    zarr_path = os.path.join(data_dir, sample + ".zarr")
    t0 = time.time()
    bundle = load_detector(None, "cpu")
    dets, affs = detect_and_score_sequence(
        zarr_path, model=bundle,
        det_threshold=det_threshold,
        pool_kernel_um=pool_kernel_um,
        det_tta=det_tta,
        det_tta_views=det_tta_views,
        device="cpu",
        edge_activation="softmax",
        edge_threshold=edge_threshold,
        max_link_um=gate_um,
    )
    scale = Image(zarr_path).scale
    tmp = out + ".tmp.npz"
    save_cache(tmp, dets, affs, scale)
    os.replace(tmp, out)
    n_nodes = sum(d.shape[0] for d in dets)
    n_cand = sum(a["i"].size for a in affs)
    return sample, time.time() - t0, n_nodes, n_cand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--det-threshold", type=float, required=True)
    ap.add_argument("--pool-kernel-um", type=float, default=5.0)
    ap.add_argument("--det-tta", action="store_true")
    ap.add_argument("--det-tta-views", type=int, default=4, choices=(4, 8),
                    help="size of the dihedral group averaged over in the YX "
                         "plane. 4 is the flips, 8 adds the quarter turns. "
                         "Measured on one sample at 6 frames: 4 views cost "
                         "3.4x a plain pass and 8 cost 6.8x.")
    ap.add_argument("--gate-um", type=float, default=20.0)
    ap.add_argument("--edge-threshold", type=float, default=0.05)
    ap.add_argument("--cache", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    tag = f"t{args.det_threshold:g}".replace(".", "")
    if args.det_tta:
        tag += f"_tta{args.det_tta_views}"
    cache = args.cache or f"data/meta/graph_cache_{tag}"
    os.makedirs(cache, exist_ok=True)
    done = sum(1 for s in HELDOUT if os.path.exists(cache_path(cache, s)))
    print(f"caching {len(HELDOUT)} samples to {cache}")
    print(f"  det_threshold {args.det_threshold}, tta {args.det_tta}"
          f"{args.det_tta_views if args.det_tta else ''}, "
          f"gate {args.gate_um} um, edge_threshold {args.edge_threshold}")
    print(f"  {done} already cached\n", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(run_sample, s, args.data, cache, args.det_threshold,
                        args.pool_kernel_um, args.det_tta, args.det_tta_views,
                        args.gate_um, args.edge_threshold, args.threads): s
            for s in HELDOUT
        }
        for i, f in enumerate(as_completed(futs), 1):
            s, secs, n_nodes, n_cand = f.result()
            el = (time.time() - t0) / 60
            tag2 = "cached" if secs == 0.0 else f"{secs / 60:.1f} min"
            extra = "" if n_nodes < 0 else f", {n_nodes} nodes, {n_cand} candidates"
            print(f"  [{i}/{len(HELDOUT)}] {s} ({tag2}){extra} "
                  f"[{el:.1f} min, eta {el / i * (len(HELDOUT) - i):.0f} min]",
                  flush=True)

    print(f"\ntotal {(time.time() - t0) / 60:.1f} min, cache at {cache}")


if __name__ == "__main__":
    main()
