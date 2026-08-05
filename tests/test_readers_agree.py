"""The two zarr backends must return identical pixels.

Local development uses zarr; the competition rerun has only tensorstore. If those
two ever disagree, every local CV number silently stops describing what the
leaderboard measures, which is the exact failure the prime directive is about.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.config import resolve_data_dir
from src.data import Image, list_samples

TRAIN = os.path.join(str(resolve_data_dir()), "train")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(TRAIN), reason="competition data not downloaded"
)


def _first_sample() -> str:
    samples = list_samples(TRAIN, require_geff=False)
    if not samples:
        pytest.skip("no samples on disk")
    return samples[0]


def test_backends_agree_on_shape_and_pixels():
    ts = pytest.importorskip("tensorstore")  # noqa: F841
    pytest.importorskip("zarr")

    sample = _first_sample()
    path = os.path.join(TRAIN, sample + ".zarr")

    a = Image(path, backend="zarr")
    b = Image(path, backend="tensorstore")

    assert a.shape == b.shape
    assert a.n_timepoints == b.n_timepoints
    assert a.scale == b.scale

    for t in (0, a.n_timepoints // 2, a.n_timepoints - 1):
        fa, fb = a.frame(t), b.frame(t)
        assert fa.shape == fb.shape
        assert fa.dtype == fb.dtype
        assert np.array_equal(fa, fb), f"backends disagree at t={t}"


def test_scale_is_physical_and_anisotropic():
    """Guards against anyone 'simplifying' the scale to isotropic voxels.

    Matching is capped at 7 um in physical space. In voxels that is about 4.3 in Z
    and 17 in Y/X, so treating the axes as equal would be wrong by a factor of four.
    """
    sample = _first_sample()
    scale = Image(os.path.join(TRAIN, sample + ".zarr")).scale
    assert len(scale) == 3
    assert scale[0] > scale[1], "Z spacing should be coarser than Y"
    assert scale[1] == pytest.approx(scale[2]), "Y and X should match"
