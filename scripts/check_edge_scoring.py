"""Check our edge scoring against the pack's own `predict_video`.

`src.unet.detect_and_score_sequence` re-implements the loop that runs the edge
head, because the pack's version is welded to its own graph builder and its own
greedy selection. Re-implementing a loop is exactly where a silent divergence
gets in: the feature map is indexed with downsampled coordinates, the
transformer's distance term takes full resolution ones, and the positional
embedding uses window-relative time. Any of those being wrong produces
plausible probabilities that are quietly not the pack's.

So this runs both on the same frames of the same video and compares the
detections and the edge probabilities directly. It is a correctness check, not a
score, and it is cheap: four frames.

    .venv\\Scripts\\python.exe scripts/check_edge_scoring.py

The sample is one of the pack's 19 held-out videos, so nothing here touches its
training data even though no score is produced.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import resolve_data_dir
from src.unet import _ensure_pack_on_path, load_detector, detect_and_score_sequence

SAMPLE = "44b6_d754aa59"
MAX_FRAMES = 4
DET_THRESHOLD = 0.9550
POOL_KERNEL_UM = 5.0
EDGE_THRESHOLD = 0.05


def main() -> int:
    _ensure_pack_on_path()
    import torch
    from predict_unet_transformer import PredictConfig, predict_video

    zarr_path = resolve_data_dir() / "train" / f"{SAMPLE}.zarr"
    if not zarr_path.exists():
        raise SystemExit(f"no sample at {zarr_path}")

    bundle = load_detector(device="cpu")
    model, window_size, downsample, _ = bundle

    print(f"sample {SAMPLE}, first {MAX_FRAMES} frames, window {window_size}, "
          f"downsample {downsample}")

    # use_ilp=True only to stop __post_init__ imposing the 1/2 greedy caps, so
    # every candidate above the threshold is emitted and the two candidate sets
    # are comparable. predict_video itself never looks at use_ilp.
    cfg = PredictConfig(
        det_threshold=DET_THRESHOLD,
        det_tta=False,
        pool_kernel_um=POOL_KERNEL_UM,
        edge_activation="softmax",
        threshold=EDGE_THRESHOLD,
        use_ilp=True,
    )
    with torch.no_grad():
        ref_coords, ref_edges = predict_video(
            model, zarr_path, torch.device("cpu"), cfg,
            window_size=window_size, max_frames=MAX_FRAMES,
            downsample=downsample,
        )
    print(f"  pack: {len(ref_coords)} nodes, {len(ref_edges)} candidate edges")

    # max_link_um wide open so the comparison is against the pack's full
    # candidate set rather than against our distance gate.
    dets, affs = detect_and_score_sequence(
        zarr_path, model=bundle,
        det_threshold=DET_THRESHOLD,
        pool_kernel_um=POOL_KERNEL_UM,
        det_tta=False,
        device="cpu",
        edge_activation="softmax",
        edge_threshold=EDGE_THRESHOLD,
        max_link_um=float("inf"),
        max_frames=MAX_FRAMES,
    )
    n_ours = sum(d.shape[0] for d in dets)
    n_edges_ours = sum(a["i"].size for a in affs)
    print(f"  ours: {n_ours} nodes, {n_edges_ours} candidate edges")

    ok = True

    # --- detections -------------------------------------------------------
    ref_t = ref_coords[:, 0].astype(int)
    for t, d in enumerate(dets):
        ref_frame = ref_coords[ref_t == t][:, 1:].astype(np.float64)
        ours = np.round(d).astype(np.float64)
        if ref_frame.shape != ours.shape or not np.array_equal(
            np.sort(ref_frame, axis=0), np.sort(ours, axis=0)
        ):
            print(f"  MISMATCH detections at t={t}: "
                  f"pack {ref_frame.shape}, ours {ours.shape}")
            ok = False
    if ok:
        print(f"  detections identical across {len(dets)} frames")

    # --- edge probabilities ----------------------------------------------
    # Pack edges carry global node indices; coords are appended frame by frame
    # in ascending t, so the local index is the global one minus the frame's
    # first index.
    frame_start = {t: int(np.argmax(ref_t == t)) for t in np.unique(ref_t)}
    ref_map: dict[tuple[int, int, int], float] = {}
    for gi, gj, prob, _dist in ref_edges:
        ts, tt = int(ref_t[gi]), int(ref_t[gj])
        if tt != ts + 1:
            continue
        ref_map[(ts, gi - frame_start[ts], gj - frame_start[tt])] = float(prob)

    ours_map: dict[tuple[int, int, int], float] = {}
    for t, a in enumerate(affs):
        for i, j, p in zip(a["i"], a["j"], a["p"]):
            ours_map[(t, int(i), int(j))] = float(p)

    only_pack = set(ref_map) - set(ours_map)
    only_ours = set(ours_map) - set(ref_map)
    both = set(ref_map) & set(ours_map)
    if both:
        diffs = np.array([abs(ref_map[k] - ours_map[k]) for k in both])
        print(f"  shared candidates {len(both)}, max |dp| {diffs.max():.3e}, "
              f"mean |dp| {diffs.mean():.3e}")
        if diffs.max() > 1e-5:
            print("  MISMATCH probabilities differ by more than 1e-5")
            ok = False
    else:
        print("  MISMATCH no shared candidates at all")
        ok = False

    if only_pack or only_ours:
        print(f"  MISMATCH candidate sets differ: pack only {len(only_pack)}, "
              f"ours only {len(only_ours)}")
        ok = False
    else:
        print(f"  candidate sets identical, {len(both)} pairs")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
