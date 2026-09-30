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
WEIGHTS_REL = Path("weights") / "unet_transformer" / "split_0" / "edge_predictor_best.pth"


def pack_dir() -> Path:
    """Where the support pack lives.

    Locally it is `external/pack50`. In a Kaggle notebook it is an attached
    dataset whose mount path Kaggle has moved between conventions, so the kernel
    template sets `CELLMOT_PACK_DIR` after locating it. Never hard-code a mount.
    """
    env = os.environ.get("CELLMOT_PACK_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "external" / "pack50"


def default_weights() -> Path:
    return pack_dir() / WEIGHTS_REL


def _ensure_pack_on_path() -> None:
    """Put the support pack's package and scripts on `sys.path`.

    `scripts/predict_unet_transformer.py` is not part of the installed package
    but holds the loader, the frame reader and the peak extractor, so both
    directories are needed.
    """
    pack = pack_dir()
    src = pack / "repo" / "src"
    scripts = pack / "repo" / "scripts"
    if not src.exists() or not scripts.exists():
        raise RuntimeError(
            "support pack missing. Run:\n"
            "  kaggle datasets download srcA/biohub-tracking-support-pack-50ep-v1 "
            "-p external/pack50 --unzip\n"
            f"expected it at {pack}"
        )
    for path in (str(src), str(scripts)):
        if path not in sys.path:
            sys.path.insert(0, path)


def resolve_device(device: str = "auto") -> str:
    """Resolve `auto` to cuda when a GPU is present.

    The config says `auto` so one config runs unchanged on this CPU machine and on
    a Kaggle GPU kernel. Hard-coding `cuda` would break local runs and hard-coding
    `cpu` would silently waste the GPU we asked for.
    """
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_detector(weights: str | os.PathLike | None = None, device: str = "auto"):
    """Load the model once. Returns `(model, window_size, downsample, torch)`.

    Loading is the expensive part on CPU, so callers scoring many samples should
    hold on to the result rather than reloading per sample.
    """
    _ensure_pack_on_path()
    import torch
    from predict_unet_transformer import load_model

    weights_path = Path(weights) if weights else default_weights()
    if not weights_path.is_absolute() and not weights_path.exists():
        weights_path = default_weights()
    if not weights_path.exists():
        raise FileNotFoundError(f"no weights at {weights_path}")
    model, window_size, downsample = load_model(
        weights_path, torch.device(resolve_device(device)))
    return model, window_size, downsample, torch


# The dihedral group of the square, acting in the YX plane only. Elements are
# numbered g = 2k + f, where k is the number of quarter turns and f says whether
# an X flip follows. Numbering it this way makes the four-element subgroup a
# literal subset, (0, 1, 4, 5) = {identity, flip X, rotate 180, flip Y}, so the
# four-view path stays exactly what it was before the rotations were added.
_D4_KLEIN = (0, 1, 4, 5)
_D4_FULL = (0, 1, 2, 3, 4, 5, 6, 7)


def _d4_apply(x, g, torch):
    k, f = divmod(g, 2)
    y = torch.rot90(x, k, dims=(-2, -1)) if k else x
    return y.flip(-1) if f else y


def _d4_invert(x, g, torch):
    """Undo `_d4_apply`, which means undoing the flip before the rotation."""
    k, f = divmod(g, 2)
    y = x.flip(-1) if f else x
    return torch.rot90(y, -k, dims=(-2, -1)) if k else y


def _tta_views(imgs, views: int) -> tuple:
    """Which group elements to average over, given the volume's own shape.

    Z is never touched. The data is four times coarser in Z, so a Z flipped
    volume is out of distribution. Y and X are 256 voxels each at 0.40625 um in
    every sample profiled, so a quarter turn in that plane is both shape
    preserving and in distribution, which is what makes the full group available.

    The shape check is not decoration. A quarter turn transposes Y and X, so on a
    volume that is not square in those axes it would hand the network a tensor of
    a shape it never saw. Rather than fail on such a sample, fall back to the
    four flips, which are shape preserving for any volume.
    """
    if int(views) < 8:
        return _D4_KLEIN
    if imgs.shape[-1] != imgs.shape[-2]:
        return _D4_KLEIN
    return _D4_FULL


def _tta_average_det(model, imgs, det_logits, w: int, views: int, torch):
    """Average detection logits over the chosen group, undoing each transform.

    Only the detection head is averaged. The association features from the same
    forward pass are taken from the untransformed view alone, so the edge
    probabilities change here only through the detections they are computed on.
    Averaging the association features as well is the step the public 0.947
    notebook reports as its single largest gain, and it is a separate variable.
    """
    group = _tta_views(imgs, views)
    for g in group[1:]:
        _, det_g = model.encode(_d4_apply(imgs, g, torch))
        for f in range(w):
            det_logits[f] = det_logits[f] + _d4_invert(det_g[f], g, torch)
        del det_g
    n = float(len(group))
    for f in range(w):
        det_logits[f] = det_logits[f] / n
    return det_logits


def detect_sequence_unet(
    zarr_path: str | os.PathLike,
    model=None,
    det_threshold: float = 0.9550,
    pool_kernel_um: float = 5.0,
    det_tta: bool = False,
    det_tta_views: int = 4,
    device: str = "auto",
    weights: str | os.PathLike | None = None,
    timepoints: range | None = None,
    progress_every: int = 0,
) -> list[np.ndarray]:
    """Detect over every timepoint of one sample, returning full-resolution voxels.

    `det_threshold` is the sigmoid probability a local-max peak must clear. The
    organisers released 0.99; the public notebooks land near 0.955 to 0.969 and
    report it as one of the two or three settings that move the score.
    It is a config value here for that reason, and it gets chosen on the folds
    rather than copied.

    `det_tta` averages detection logits over a dihedral group acting in the YX
    plane, `det_tta_views` of them: 4 for the flips alone, 8 for the full group
    including quarter turns. Z is never touched, because the data is
    four times coarser in Z and a Z-flipped volume is out of distribution.

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
    dev = torch.device(resolve_device(device))

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
                det_logits = _tta_average_det(
                    model, imgs, det_logits, w, det_tta_views, torch)
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


def detect_and_score_sequence(
    zarr_path: str | os.PathLike,
    model=None,
    det_threshold: float = 0.9550,
    pool_kernel_um: float = 5.0,
    det_tta: bool = False,
    det_tta_views: int = 4,
    device: str = "auto",
    weights: str | os.PathLike | None = None,
    edge_activation: str = "softmax",
    edge_threshold: float = 0.05,
    max_link_um: float = 7.0,
    max_frames: int | None = None,
    progress_every: int = 0,
) -> tuple[list[np.ndarray], list[dict]]:
    """Detect and score candidate links in one pass, returning both.

    Why one function rather than detect then link. The edge head consumes the
    U-Net feature map for the window the frame was detected in, so scoring a
    link after the fact would mean running the U-Net a second time, and that is
    the expensive half. `detect_sequence_unet` throws `unet_out` away; this
    keeps it and pays for edge scoring with what is already in memory.

    Returns `(detections, affinities)`.

    `detections` is exactly what `detect_sequence_unet` returns, a list indexed
    by t of `(N, 3)` full resolution voxel coordinates, and for the same
    detection settings it is the same list. The detector is unchanged here.

    `affinities[t]` describes the links from frame t to frame t+1 as
    `{"i": (K,) int32, "j": (K,) int32, "p": (K,) float32}`, indices into
    `detections[t]` and `detections[t+1]`. Only pairs that clear
    `edge_threshold` and sit within `max_link_um` are kept. The gate matters: a
    dense score matrix per frame pair is a few megabytes and there are about 99
    of them per video, and the linker discards everything beyond `max_link_um`
    anyway, so carrying the rest would cost memory to reach the same answer.

    `max_frames` truncates the video, for checking this against the pack's own
    `predict_video` cheaply. Leave it None for real runs.

    `edge_activation` follows the pack: `softmax` normalises each target node's
    scores over the source nodes, `sigmoid` scores each pair independently.
    Softmax is the pack's default and what its reported numbers use.
    """
    _ensure_pack_on_path()
    import torch
    from biohub_tracking.io import open_dataset
    from predict_unet_transformer import (
        _detect_cells_pooled,
        _load_frame,
        pool_kernel_from_um,
    )
    from train_unet_transformer import extract_pos_features
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

    # `max_frames` exists so this can be checked against the pack's own
    # `predict_video`, which takes the same argument, without paying for a whole
    # video. It is a test hook, not a tuning knob.
    n_t = int(ds.image_shape[0])
    if max_frames is not None:
        n_t = min(n_t, int(max_frames))
    target_shape = list(ds.image_shape[1:])
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = pool_kernel_from_um(pool_kernel_um, voxel_size)
    scale_back = np.asarray(downsample, dtype=np.float64)
    # Physical size of a full resolution voxel, so the distance gate is in
    # microns like everything else in this repo. `ds.scale` is already the
    # downsampled spacing, so divide the downsample factor back out.
    full_scale = np.asarray(
        [s / d for s, d in zip(ds.scale, downsample)], dtype=np.float64)

    dev = torch.device(resolve_device(device))
    ds_arr_t = torch.from_numpy(
        np.asarray(downsample, dtype=np.float32)).to(dev)

    w = int(window_size)
    stride = max(w - 1, 1)
    # Stride w-1 so every consecutive pair falls inside exactly one window.
    # This is the pack's own scheme, kept identical: a pair scored from two
    # different windows would get two different U-Net contexts.
    starts = list(range(0, max(n_t - w + 1, 1), stride))
    if not starts or starts[-1] + w < n_t:
        last = max(n_t - w, 0)
        if not starts or last != starts[-1]:
            starts.append(last)

    found: dict[int, np.ndarray] = {}
    # Detections in the downsampled grid, kept because the edge head indexes the
    # feature map with them. The returned coordinates are scaled back up.
    found_ds: dict[int, np.ndarray] = {}
    affinities: dict[int, dict] = {}
    seen_pairs: set[tuple[int, int]] = set()

    with torch.no_grad():
        for wi, ws in enumerate(starts):
            frame_indices = [min(ws + k, n_t - 1) for k in range(w)]
            imgs = torch.stack([
                _load_frame(zarr_arr, t, target_shape, downsample)
                for t in frame_indices
            ])
            imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
            imgs = imgs.unsqueeze(0).to(dev)

            unet_out, det_logits = model.encode(imgs)

            if det_tta:
                det_logits = _tta_average_det(
                    model, imgs, det_logits, w, det_tta_views, torch)
            del imgs

            for f_idx, t in enumerate(frame_indices):
                if t not in found:
                    arr = _detect_cells_pooled(
                        det_logits[f_idx][0], t, det_threshold, pool_k,
                    )
                    coords_ds = arr[:, 1:].astype(np.float64)
                    found_ds[t] = coords_ds
                    found[t] = coords_ds * scale_back

            for f_idx in range(w - 1):
                t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]
                if t_src == t_tgt or (t_src, t_tgt) in seen_pairs:
                    continue
                seen_pairs.add((t_src, t_tgt))
                c_src, c_tgt = found_ds[t_src], found_ds[t_tgt]
                if c_src.shape[0] == 0 or c_tgt.shape[0] == 0:
                    affinities[t_src] = _empty_affinity()
                    continue

                probs = _edge_probs(
                    model, torch, dev, unet_out, f_idx, c_src, c_tgt,
                    ds_arr_t, w, target_shape, edge_activation,
                    extract_pos_features,
                )
                affinities[t_src] = _gate_candidates(
                    probs, found[t_src], found[t_tgt], full_scale,
                    edge_threshold, max_link_um,
                )

            del unet_out, det_logits
            if progress_every and (wi + 1) % progress_every == 0:
                print(f"    window {wi + 1}/{len(starts)}, {len(found)} frames done",
                      flush=True)

    detections = [found.get(t, np.empty((0, 3), dtype=np.float64))
                  for t in range(n_t)]
    aff = [affinities.get(t, _empty_affinity()) for t in range(n_t - 1)]
    return detections, aff


def _empty_affinity() -> dict:
    return {
        "i": np.empty(0, dtype=np.int32),
        "j": np.empty(0, dtype=np.int32),
        "p": np.empty(0, dtype=np.float32),
    }


def _edge_probs(
    model, torch, dev, unet_out, f_idx, c_src, c_tgt, ds_arr_t, w,
    target_shape, edge_activation, extract_pos_features,
) -> np.ndarray:
    """The pack's edge head on one consecutive pair. Returns (n_src, n_tgt).

    Every convention here is the pack's and none of it is re-derived. The
    feature map is indexed with downsampled coordinates, the transformer's own
    distance term takes full resolution ones, and the positional embedding uses
    window-relative time normalised by the window rather than the absolute frame
    index. Getting any of those wrong produces plausible numbers that are
    silently wrong, which is the failure mode this module exists to avoid.
    """
    n_src, n_tgt = c_src.shape[0], c_tgt.shape[0]
    p_coords_src = torch.from_numpy(c_src.astype(np.float32)).unsqueeze(0).to(dev)
    p_coords_tgt = torch.from_numpy(c_tgt.astype(np.float32)).unsqueeze(0).to(dev)

    window_shape = (w,) + tuple(target_shape)
    src_rel = np.concatenate(
        [np.full((n_src, 1), f_idx, dtype=np.float64), c_src], axis=1)
    tgt_rel = np.concatenate(
        [np.full((n_tgt, 1), f_idx + 1, dtype=np.float64), c_tgt], axis=1)
    p_pos_src = torch.from_numpy(
        extract_pos_features(src_rel, window_shape)).unsqueeze(0).to(dev)
    p_pos_tgt = torch.from_numpy(
        extract_pos_features(tgt_rel, window_shape)).unsqueeze(0).to(dev)

    m_src = torch.ones(1, n_src, dtype=torch.bool, device=dev)
    m_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=dev)

    feat_src = model._index_features(unet_out[:, f_idx], p_coords_src, m_src)
    feat_tgt = model._index_features(unet_out[:, f_idx + 1], p_coords_tgt, m_tgt)
    logits = model.predict_edges(
        feat_src, feat_tgt,
        p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
        p_pos_src, p_pos_tgt, m_src, m_tgt,
    )[0]

    if edge_activation == "softmax":
        probs = torch.softmax(logits, dim=0)
    elif edge_activation == "sigmoid":
        probs = torch.sigmoid(logits)
    else:
        raise ValueError(f"unknown edge_activation {edge_activation!r}")
    return probs.float().cpu().numpy()


def _gate_candidates(
    probs: np.ndarray,
    a_full: np.ndarray,
    b_full: np.ndarray,
    full_scale: np.ndarray,
    edge_threshold: float,
    max_link_um: float,
) -> dict:
    """Keep pairs above the probability floor and inside the distance gate.

    The distance gate is the same `max_link_um` the distance linker uses, so both
    linkers choose from the same candidate set and the only thing that differs is
    how a candidate is scored. Without that the comparison would confound the
    cost function with the search space.
    """
    keep = probs > edge_threshold
    if not keep.any():
        return _empty_affinity()
    ii, jj = np.nonzero(keep)
    pa = a_full[ii] * full_scale[None, :]
    pb = b_full[jj] * full_scale[None, :]
    d = np.linalg.norm(pa - pb, axis=1)
    near = d <= max_link_um
    return {
        "i": ii[near].astype(np.int32),
        "j": jj[near].astype(np.int32),
        "p": probs[ii[near], jj[near]].astype(np.float32),
    }
