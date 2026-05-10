from __future__ import annotations

import torch

from glr.data import SyntheticFactorDataset, make_compositional_splits
from glr.data.masking import apply_image_mask, random_token_mask


def test_synthetic_dataset_shapes():
    ds = SyntheticFactorDataset(n_samples=16, image_size=32, two_view=True)
    assert len(ds) == 16
    item = ds[0]
    assert item["image"].shape == (1, 32, 32)
    assert item["view1"].shape == (1, 32, 32)
    assert item["view2"].shape == (1, 32, 32)
    assert item["latents"].shape == (6,)


def test_compositional_splits():
    pairs = make_compositional_splits(held_out_count=4, seed=0)
    assert len(pairs) == 4
    assert all(0 <= a < 3 and 0 <= b < 6 for (a, b) in pairs)


def test_random_token_mask():
    mask = random_token_mask(2, 100, mask_ratio=0.3)
    assert mask.shape == (2, 100)
    assert mask.dtype == torch.bool
    assert mask.sum(dim=-1).tolist() == [30, 30]


def test_apply_image_mask():
    img = torch.ones(2, 1, 16, 16)
    # 16x16 image with patch_size=4 -> 4x4 = 16 patches
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[0, 0] = True  # top-left patch of first image
    out = apply_image_mask(img, mask, patch_size=4)
    assert out.shape == img.shape
    assert (out[0, 0, :4, :4] == 0).all()
    assert (out[0, 0, 4:, :] == 1).all()
    assert (out[1] == 1).all()  # second image untouched
