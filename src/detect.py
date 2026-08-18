"""Cell detection: one frame in, centroids out.

The probe in `notebooks/probe_read/` showed that a fixed intensity threshold is not
usable here. The same setting returned between 0.05x and 1.46x the true cell count
across ten samples, because cell density varies twentyfold and the two embryos
differ in brightness by about 2.5x. `estimated_number_of_nodes` would tell us the
answer but it lives in the ground-truth metadata, so it is not available at
prediction time. Detection therefore has to calibrate itself from the image.

The scoring asymmetry sets the operating point. A predicted node that matches
nothing costs nothing on the edge term, because an edge only counts as a false
positive when one of its endpoints matched an annotated ground-truth node. Surplus
nodes are charged only through `1 - 0.1 * (N_pred - N_true) / N_true`, so doubling
the node count costs 10%. Missing a real cell costs edge recall outright. So this
deliberately errs toward over-detection.

Everything spatial is specified in microns and converted to voxels here. The data
is strongly anisotropic, 1.625 um in Z against 0.40625 in Y and X, so a radius that
is symmetric in voxels is four times wider in Z in physical space, which is the
space the 7 um matching cap lives in.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def _odd(n: np.ndarray) -> np.ndarray:
    """Force each footprint dimension odd and at least 1, so windows stay centred."""
    n = np.maximum(1, np.round(n).astype(int))
    return n + (n % 2 == 0)


def normalise(frame: np.ndarray, quantiles: dict[float, float] | None,
              q_lo: float = 0.01, q_hi: float = 0.99) -> np.ndarray:
    """Scale a frame to roughly [0, inf) using the quantiles shipped in the zarr.

    The competition ships `image_statistics.quantiles` per sample, so this needs no
    pass over the pixels. Falls back to per-frame min/max when they are absent.
    """
    frame = frame.astype(np.float32)
    lo = hi = None
    if quantiles:
        lo, hi = quantiles.get(q_lo), quantiles.get(q_hi)
    if lo is None or hi is None or hi <= lo:
        lo, hi = float(frame.min()), float(frame.max())
    if hi <= lo:
        return np.zeros_like(frame)
    return np.clip((frame - lo) / (hi - lo), 0.0, None)


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu's threshold on a 1-D array of intensities.

    Written out rather than imported from skimage so that this module depends only
    on numpy and scipy. The rerun environment does have skimage, but keeping the
    inference path's dependency list short is worth more than the few lines saved.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(finite, bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0

    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return lo
    csum = np.cumsum(hist * centres)
    m0 = np.divide(csum, w0, out=np.zeros_like(csum), where=w0 > 0)
    m1 = np.divide(csum[-1] - csum, w1, out=np.zeros_like(csum), where=w1 > 0)
    between = w0 * w1 * (m0 - m1) ** 2
    between[~valid] = -1.0
    return float(centres[int(np.argmax(between))])


def detect_frame(
    frame: np.ndarray,
    scale_zyx: tuple[float, float, float],
    quantiles: dict[float, float] | None = None,
    sigma_um: float = 2.0,
    min_sep_um: float = 3.0,
    threshold_scale: float = 0.5,
    min_threshold: float = 0.02,
    max_detections: int = 20000,
) -> np.ndarray:
    """Detect cell centres in one `(Z, Y, X)` frame. Returns `(N, 3)` of (z, y, x).

    `threshold_scale` multiplies the per-frame Otsu threshold. Below 1.0 it accepts
    dimmer peaks, which is the intended direction given over-detection is cheap.
    Set it from a config and tune it against the fold scores, not against per-frame
    detection F1, because those two disagree under this metric.
    """
    scale = np.asarray(scale_zyx, dtype=np.float64)
    img = normalise(frame, quantiles)

    sigma_vox = np.asarray(sigma_um, dtype=np.float64) / scale
    smooth = ndi.gaussian_filter(img, sigma=sigma_vox)

    # Otsu on the frame itself, so brightness differences between embryos and
    # density differences between samples do not need a hand-set threshold.
    thr = max(otsu_threshold(smooth.ravel()) * threshold_scale, min_threshold)

    footprint = _odd(np.asarray(min_sep_um, dtype=np.float64) / scale)
    peak = smooth == ndi.maximum_filter(smooth, size=tuple(footprint))
    peak &= smooth > thr

    coords = np.argwhere(peak)
    if coords.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    # A plateau of equal values produces several adjacent maxima. Collapse each
    # connected group to its centre of mass so one cell yields one detection.
    labels, n_labels = ndi.label(peak)
    if n_labels and n_labels < coords.shape[0]:
        centres = ndi.center_of_mass(smooth, labels, np.arange(1, n_labels + 1))
        coords = np.asarray(centres, dtype=np.float64)
    else:
        coords = coords.astype(np.float64)

    if coords.shape[0] > max_detections:
        # Keep the brightest. Hitting this cap means the threshold is wrong, so it
        # is a guard rail rather than a feature.
        idx = np.round(coords).astype(int)
        idx = np.clip(idx, 0, np.array(smooth.shape) - 1)
        vals = smooth[idx[:, 0], idx[:, 1], idx[:, 2]]
        coords = coords[np.argsort(vals)[::-1][:max_detections]]

    return coords


def detect_sequence(
    image,
    sigma_um: float = 2.0,
    min_sep_um: float = 3.0,
    threshold_scale: float = 0.5,
    max_detections: int = 20000,
    z_shift_vox: float = 0.0,
    timepoints: range | None = None,
    progress_every: int = 0,
) -> list[np.ndarray]:
    """Detect over every timepoint of one sample. `image` is a `src.data.Image`.

    Returns a list indexed by t, each `(N_t, 3)` in voxel coordinates.

    `z_shift_vox` adds a constant offset along Z, in voxels, after detection. It
    corrects a measured systematic bias: predictions sit below the annotated centre
    in Z, and Z carries 61% of the squared localisation error even on cells that do
    match. This is not a coordinate convention error, which was checked separately
    (ground-truth z spans the full 0..63 voxel range, is integer, and is 0-based
    like the image). The most plausible cause is the light-sheet PSF being
    asymmetric in Z combined with Z being smoothed far less than Y and X in voxel
    terms, sigma 1.23 voxels against 4.92.

    A constant shift changes no node counts, so the node-count penalty is untouched
    and the entire effect is localisation. Detections are clipped to the volume so
    a shifted point cannot leave the image.
    """
    ts = timepoints if timepoints is not None else range(image.n_timepoints)
    out = []
    for i, t in enumerate(ts):
        coords = detect_frame(
            image.frame(t),
            scale_zyx=image.scale,
            quantiles=image.quantiles,
            sigma_um=sigma_um,
            min_sep_um=min_sep_um,
            threshold_scale=threshold_scale,
            max_detections=max_detections,
        )
        if z_shift_vox and coords.shape[0]:
            coords = coords.copy()
            coords[:, 0] = np.clip(coords[:, 0] + z_shift_vox, 0, image.shape[-3] - 1)
        out.append(coords)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    t={t + 1}/{len(ts)}  {coords.shape[0]} detections", flush=True)
    return out
