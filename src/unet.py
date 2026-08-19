"""The learned detector, wrapped to return exactly what `src.detect` returns.

The point of this module is that swapping the detector changes one variable.
`detect_sequence_unet` returns the same thing `src.detect.detect_sequence`
returns, a list indexed by t of `(N, 3)` float voxel coordinates in full
resolution, so the existing linker, the organisers' scorer, the leave-one-embryo-
out folds and the bootstrap all keep working untouched. That makes the first
U-Net number directly comparable to the 0.6876 the local-max detector gets,
rather than a second pipeline whose difference could be anything.

Nothing here reimplements the reference preprocessing. Quantile normalisation,
strided loading, the max-pool local-max, the pool kernel in microns and the
flip TTA are imported from the support pack, because a silent divergence in
normalisation would be invisible and would invalidate every comparison. The
0.1%/99.9% quantiles the pack uses are not the 1%/99% pair `src.detect.normalise`
uses, which is exactly the kind of difference this avoids re-deriving by hand.

The weights are `srcA/biohub-tracking-support-pack-50ep-v1`, the same
architecture the organisers released trained to 50 epochs rather than 3. Public
and freely available, so within the competition rules on pretrained models.

No GPU on this machine, so a local run is a correctness check on a few frames.
Anything covering the folds runs in a Kaggle notebook against this same module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = REPO_ROOT / "external" / "pack50"
DEFAULT_WEIGHTS = PACK_DIR / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"


def _ensure_pack_on_path() -> None:
    """Put the support pack's package and scripts on `sys.path`.

    `scripts/predict_unet_transformer.py` is not part of the installed package
    but holds the loader, the frame reader and the peak extractor, so both
    directories are needed.
    """
    src = PACK_DIR / "repo" / "src"
    scripts = PACK_DIR / "repo" / "scripts"
    if not src.exists() or not scripts.exists():
        raise RuntimeError(
            "support pack missing. Run:\n"
            "  kaggle datasets download srcA/biohub-tracking-support-pack-50ep-v1 "
            "-p external/pack50 --unzip\n"
            f"expected it at {PACK_DIR}"
        )
    for path in (str(src), str(scripts)):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_detector(weights: str | os.PathLike | None = None, device: str = "cpu"):
    """Load the model once. Returns `(model, window_size, downsample, torch)`.

    Loading is the expensive part on CPU, so callers scoring many samples should
    hold on to the result rather than reloading per sample.
    """
    _ensure_pack_on_path()
    import torch
    from predict_unet_transformer import load_model

    weights_path = Path(weights) if weights else DEFAULT_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(f"no weights at {weights_path}")
    model, window_size, downsample = load_model(weights_path, torch.device(device))
    return model, window_size, downsample, torch


def detect_sequence_unet(
    zarr_path: str | os.PathLike,
    model=None,
    det_threshold: float = 0.9550,
    pool_kernel_um: float = 5.0,
    det_tta: bool = False,
    device: str = "cpu",
    weights: str | os.PathLike | None = None,
    timepoints: range | None = None,
    progress_every: int = 0,
) -> list[np.ndarray]:
    """Detect over every timepoint of one sample, returning full-resolution voxels.

    `det_threshold` is the sigmoid probability a local-max peak must clear. The
    organisers released 0.99; the public notebooks land near 0.955 to 0.969 and
    report it as one of the two or three settings that actually move the score.
    It is a config value here for that reason, and it gets chosen on the folds
    rather than copied.

    `det_tta` averages logits over Y and X flips. Z is deliberately not flipped:
    the data is four times coarser in Z, so a Z-flipped volume is out of
    distribution.

    Windows slide with stride `window_size - 1` and each timepoint is detected
    the first time it appears, which mirrors the reference inference exactly.
    The U-Net carries temporal attention across the window, so a frame's features
    depend on its window partner and a cheaper non-overlapping pass would not
    give the same detections.
    """
    _ensure_pack_on_path()
    import torch
    from biohub_tracking.io import open_dataset
    from predict_unet_transformer import (
        _detect_cells_pooled,
        _load_frame,
        pool_kernel_from_um,
    )
    import zarr

    if model is None:
        model, window_size, downsample, torch = load_detector(weights, device)
    else:
        model, window_size, downsample, torch = model

    ds = open_dataset(Path(zarr_path), normalize=False, load_image=False,
                      downsample=downsample)
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"zarr attrs missing image_statistics.quantiles for {zarr_path}")
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]

    n_t = ds.image_shape[0]
    ts = list(timepoints) if timepoints is not None else list(range(n_t))
    target_shape = list(ds.image_shape[1:])
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = pool_kernel_from_um(pool_kernel_um, voxel_size)
    scale_back = np.asarray(downsample, dtype=np.float64)

    w = int(window_size)
    stride = max(w - 1, 1)
    last_t = max(ts)
    starts = list(range(min(ts), max(last_t - w + 2, min(ts) + 1), stride))
    if starts and starts[-1] + w - 1 < last_t:
        starts.append(max(last_t - w + 1, 0))

    wanted = set(ts)
    found: dict[int, np.ndarray] = {}
    dev = torch.device(device)

    with torch.no_grad():
        for i, ws in enumerate(starts):
            frame_indices = [min(ws + k, n_t - 1) for k in range(w)]
            if not any(t in wanted and t not in found for t in frame_indices):
                continue
            imgs = torch.stack([
                _load_frame(zarr_arr, t, target_shape, downsample)
                for t in frame_indices
            ])
            # Quantile normalisation at 0.1% and 99.9%, matching the pack's
            # training pipeline. This is not the same pair src.detect uses.
            imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
            imgs = imgs.unsqueeze(0).to(dev)

            _, det_logits = model.encode(imgs)

            if det_tta:
                for dims in [(-1,), (-2,), (-2, -1)]:
                    _, det_flip = model.encode(imgs.flip(dims))
                    for f in range(w):
                        det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                    del det_flip
                for f in range(w):
                    det_logits[f] = det_logits[f] / 4
            del imgs

            for f_idx, t in enumerate(frame_indices):
                if t in wanted and t not in found:
                    arr = _detect_cells_pooled(
                        det_logits[f_idx][0], t, det_threshold, pool_k,
                    )
                    # (N, 4) as [t, z, y, x] in the downsampled grid. Drop t and
                    # scale back, since the caller indexes by t already.
                    coords = arr[:, 1:].astype(np.float64) * scale_back
                    found[t] = coords
            if progress_every and (i + 1) % progress_every == 0:
                print(f"    window {i + 1}/{len(starts)}, {len(found)} frames done",
                      flush=True)

    return [found.get(t, np.empty((0, 3), dtype=np.float64)) for t in ts]
