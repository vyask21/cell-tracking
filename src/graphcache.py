"""The on-disk format for cached detections and edge affinities.

This lives in `src/` rather than next to the screening script for one reason: the
cache is now built in two places. Locally `scripts/cache_graphs.py` writes it on
CPU, and on Kaggle a GPU kernel writes it for the same 19 samples, because
eight-view TTA costs 6.8 times a plain pass and that is an overnight job on this
machine against a couple of hours on a T4. Two writers of one binary format is
exactly the situation where a silent divergence produces a screen that compares
one arm's detections against another arm's, so both import these functions rather
than each carrying a copy.

The format is plain: concatenated arrays plus per-frame counts, no
object arrays and no pickle, so it loads under `np.load` with default settings on
any numpy the Kaggle image happens to ship.
"""

from __future__ import annotations

import os

import numpy as np

# The 19 videos the 50-epoch support pack held out of its own training. They are
# the only samples in the 199 on which a number from this pipeline means
# anything, since the pack memorised the other 180. Video-disjoint from its
# training set but NOT embryo-disjoint, which is the standing caveat on every
# screen that uses them.
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
