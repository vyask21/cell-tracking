"""Do fields of view from the same embryo overlap in space?

Open since session 1 and it matters for validation, not for score. Every fold
score, every bootstrap interval and the 30-sample screening draw treat samples
within an embryo as independent. If two crops cover overlapping tissue they share
physical cells, and then they are not independent: a bootstrap over samples would
understate its own spread, and a screening set could double count a region.

The obvious route is closed. `zarr.json` carries `multiscales` with a `scale`
transform and `image_statistics`, and **no translation or offset field**, so a
crop's position in the embryo is not recorded anywhere in the data. That also
settles the naming question from session 1: the trailing 8 hex characters are an
opaque id, not coordinates.

So this uses labels only, and a translation-invariant signature. If two crops
overlap, a cell in the shared region appears in both, and its frame-to-frame
displacement is the same in both because an unknown constant crop offset cancels
in a difference. A track is therefore summarised by its sequence of displacement
vectors, and two crops that share cells should share signatures.

What this can and cannot say. Ground truth is sparse, so two crops can overlap
while annotating different cells, and this would see nothing. A positive result is
strong evidence of overlap; a negative result is weak evidence of no overlap, and
is reported as such rather than as a clean bill of health.

    .venv\\Scripts\\python.exe scripts/measure_crop_overlap.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_data_dir
from src.data import list_samples, read_geff

# A signature is kept only if the track is at least this long. Short tracks
# produce short signatures, and short signatures collide by chance.
MIN_TRACK_LEN = 6
# Displacements are quantised to this many voxels before hashing, so two
# observations of the same cell survive rounding differences in the annotation.
QUANT = 1.0


def track_signatures(graph, min_len: int = MIN_TRACK_LEN) -> set[tuple]:
    """Displacement-vector signatures for every long enough track in one sample.

    Translation invariant by construction: the signature is built from
    differences between consecutive positions, so an unknown crop offset cancels.
    """
    if graph.edges.shape[0] == 0:
        return set()

    pos = {
        int(i): (float(z), float(y), float(x))
        for i, z, y, x in zip(graph.nodes.ids, graph.nodes.z, graph.nodes.y,
                              graph.nodes.x)
    }
    nxt = defaultdict(list)
    has_parent = set()
    for s, t in graph.edges:
        nxt[int(s)].append(int(t))
        has_parent.add(int(t))

    roots = [n for n in pos if n not in has_parent]
    sigs: set[tuple] = set()
    for root in roots:
        # Follow single-child steps only. A division ends the chain, since after
        # it the two daughters are separate tracks anyway.
        chain = [root]
        cur = root
        while len(nxt.get(cur, ())) == 1:
            cur = nxt[cur][0]
            chain.append(cur)
        if len(chain) < min_len:
            continue
        p = np.array([pos[n] for n in chain], dtype=np.float64)
        d = np.round(np.diff(p, axis=0) / QUANT).astype(np.int64)
        # A track that never moves has the all-zero signature, which identifies
        # nothing and is shared by every sample that contains a stationary cell.
        # At min_len 4 that single degenerate signature was the entire apparent
        # overlap signal: three 6bba samples "shared" a track, and it was this
        # one. Drop signatures with fewer than two non-zero steps.
        if int(np.count_nonzero(d.any(axis=1))) < 2:
            continue
        sigs.add(tuple(map(tuple, d)))
    return sigs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--min-len", type=int, default=MIN_TRACK_LEN)
    args = ap.parse_args()

    data_dir = args.data or str(resolve_data_dir() / "train")
    samples = list_samples(data_dir, require_geff=True)
    print(f"{len(samples)} samples with ground truth in {data_dir}")
    print(f"signature = displacement vectors of a track of at least "
          f"{args.min_len} nodes, quantised to {QUANT} voxel\n")

    sigs: dict[str, set[tuple]] = {}
    for i, s in enumerate(samples, 1):
        g = read_geff(os.path.join(data_dir, s + ".geff"))
        sigs[s] = track_signatures(g, args.min_len)
        if i % 50 == 0:
            print(f"  read {i}/{len(samples)}", flush=True)

    counts = np.array([len(v) for v in sigs.values()])
    print(f"\nsignatures per sample: median {np.median(counts):.0f}, "
          f"min {counts.min()}, max {counts.max()}, "
          f"{int((counts == 0).sum())} samples with none")

    by_embryo: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        by_embryo[s.split("_")[0]].append(s)

    total_hits = 0
    for emb, members in sorted(by_embryo.items()):
        members = [m for m in members if sigs[m]]
        pairs = 0
        hits: list[tuple[str, str, int]] = []
        for a_i in range(len(members)):
            for b_i in range(a_i + 1, len(members)):
                a, b = members[a_i], members[b_i]
                pairs += 1
                shared = sigs[a] & sigs[b]
                if shared:
                    hits.append((a, b, len(shared)))
        hits.sort(key=lambda h: -h[2])
        total_hits += len(hits)
        print(f"\n{emb}: {len(members)} samples with signatures, {pairs} pairs, "
              f"{len(hits)} pairs sharing at least one")
        for a, b, n in hits[:10]:
            print(f"    {a}  {b}  {n} shared")

    # Cross-embryo pairs are the control. The two embryos are different animals,
    # so any hit there is the signature space colliding by chance, and it sets
    # the floor for reading the within-embryo numbers above.
    embs = sorted(by_embryo)
    control = 0
    control_pairs = 0
    if len(embs) == 2:
        left = [m for m in by_embryo[embs[0]] if sigs[m]]
        right = [m for m in by_embryo[embs[1]] if sigs[m]]
        for a in left:
            for b in right:
                control_pairs += 1
                if sigs[a] & sigs[b]:
                    control += 1
        print(f"\ncontrol, {embs[0]} against {embs[1]}: {control} of "
              f"{control_pairs} cross-embryo pairs share a signature")

    print("\nReading this: a shared signature means two crops contain a cell "
          "whose annotated\nmotion is identical over at least "
          f"{args.min_len - 1} frames. Compare the within-embryo rate against "
          "the\ncross-embryo control, which is pure chance collision. Ground "
          "truth is sparse, so\ncrops can overlap without sharing annotated "
          "cells: a positive here is strong, a\nnegative is weak.")


if __name__ == "__main__":
    main()
