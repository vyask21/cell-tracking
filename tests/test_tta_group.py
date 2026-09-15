"""The D4 averaging has to put every view back where it came from.

A transform that is not exactly undone averages a shifted copy of the detection
map into the original, which blurs peaks rather than sharpening them. That fails
silently: the logits stay the right shape and the score just gets slightly
worse. These pin the group algebra directly, with no network involved.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.unet import _D4_FULL, _D4_KLEIN, _d4_apply, _d4_invert, _tta_views


def test_every_element_is_undone_exactly():
    x = torch.arange(2 * 3 * 5 * 5, dtype=torch.float32).reshape(2, 3, 5, 5)
    for g in _D4_FULL:
        back = _d4_invert(_d4_apply(x, g, torch), g, torch)
        assert torch.equal(back, x), f"element {g} does not round trip"


def test_the_eight_elements_are_distinct_transforms():
    x = torch.arange(4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
    seen = {_d4_apply(x, g, torch).numpy().tobytes() for g in _D4_FULL}
    assert len(seen) == 8


def test_the_four_view_subgroup_is_the_flips_it_always_was():
    x = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 1, 4, 4)
    got = [_d4_apply(x, g, torch) for g in _D4_KLEIN]
    want = [x, x.flip(-1), x.flip(-2, -1), x.flip(-2)]
    for a, b in zip(got, want):
        assert torch.equal(a, b)


def test_z_is_never_touched():
    # A volume that differs only along Z must still differ only along Z after
    # any element, since the group acts in the YX plane alone.
    x = torch.zeros(1, 4, 5, 5)
    for z in range(4):
        x[0, z] = z
    for g in _D4_FULL:
        out = _d4_apply(x, g, torch)
        for z in range(4):
            assert torch.equal(out[0, z], torch.full((5, 5), float(z)))


def test_a_non_square_volume_falls_back_to_the_flips():
    square = torch.zeros(1, 2, 8, 8)
    oblong = torch.zeros(1, 2, 8, 16)
    assert _tta_views(square, 8) == _D4_FULL
    assert _tta_views(oblong, 8) == _D4_KLEIN
    assert _tta_views(square, 4) == _D4_KLEIN


def test_averaging_a_symmetric_map_is_a_no_op():
    # A map already invariant under the group must come back unchanged, which is
    # the sharpest check that the weights sum to one and nothing is shifted.
    w = 2
    base = torch.zeros(1, 1, 5, 5)
    base[0, 0, 2, 2] = 1.0  # centre, invariant under every element

    class FakeModel:
        def encode(self, imgs):
            return None, [imgs[0, 0].clone() for _ in range(w)]

    imgs = base.unsqueeze(0)
    logits = [imgs[0, 0].clone() for _ in range(w)]
    out = __import__("src.unet", fromlist=["_tta_average_det"])._tta_average_det(
        FakeModel(), imgs, logits, w, 8, torch)
    for f in range(w):
        assert torch.allclose(out[f], base[0, 0], atol=1e-6)
