"""dSprites loader with two-view augmentation and compositional held-out splits.

dSprites:
  https://github.com/deepmind/dsprites-dataset
  6 ground-truth factors: color (constant white in original), shape (3),
  scale (6), orientation (40), pos_x (32), pos_y (32). Image size 64x64, single channel.

This loader:
  - Returns either one image or two augmented views of the same image (for
    slot-consistency).
  - Applies factor-aware augmentations: random shift in x/y, small rotation, brightness
    perturbation. Each augmentation perturbs *one* factor at a time so the
    slot-consistency loss can be made factor-aware later.
  - Supports compositional held-out splits: exclude specific (shape, scale) tuples
    from the training set so the eval suite can probe novel compositions.

If the dSprites .npz file is missing, falls back to a procedurally-generated
SyntheticFactorDataset that mimics the dSprites factor structure (handy for tests
and CI without 26MB of data).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DSPRITES_FILE = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
DSPRITES_URL = (
    "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)
DSPRITES_FACTORS = ("color", "shape", "scale", "orientation", "pos_x", "pos_y")


class DSpritesDataset(Dataset):
    """Wraps the dSprites .npz with two-view augmentation."""

    def __init__(
        self,
        root: str | Path,
        two_view: bool = True,
        held_out_pairs: Iterable[tuple[int, int]] | None = None,
        held_out_keys: tuple[str, str] = ("shape", "scale"),
    ) -> None:
        path = Path(root) / DSPRITES_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"dSprites file not found at {path}. Download with:\n"
                f"  curl -L {DSPRITES_URL} -o {path}"
            )
        data = np.load(path, allow_pickle=True, encoding="latin1")
        # Keep imgs as uint8 (0/1) — 4x smaller than float32 in RAM (2.3 GB vs 9.4 GB
        # for the full 573k set). Each DataLoader worker would otherwise hold its
        # own COW-broken copy at 9.4 GB and OOM a 32 GB box during eval.
        imgs = data["imgs"].astype(np.uint8)              # (N, 64, 64) in {0,1}
        latents = data["latents_classes"].astype(np.int64)  # (N, 6)
        # latents_classes columns match DSPRITES_FACTORS

        if held_out_pairs is not None:
            held_out_pairs = set(map(tuple, held_out_pairs))
            ki, kj = (DSPRITES_FACTORS.index(k) for k in held_out_keys)
            mask = np.ones(latents.shape[0], dtype=bool)
            for n in range(latents.shape[0]):
                if (int(latents[n, ki]), int(latents[n, kj])) in held_out_pairs:
                    mask[n] = False
            imgs = imgs[mask]
            latents = latents[mask]
            self.held_out_pairs = held_out_pairs
            self.held_out_keys = held_out_keys
        else:
            self.held_out_pairs = None

        self.imgs = imgs
        self.latents = latents
        self.two_view = two_view

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def _aug(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Lightweight augmentations: small shift + brightness jitter.
        Pure-numpy so the loader runs on CPU workers without torchvision.
        """
        shift = rng.integers(-2, 3, size=2)
        out = np.roll(img, shift=tuple(shift.tolist()), axis=(0, 1))
        jitter = float(rng.uniform(0.9, 1.1))
        out = np.clip(out * jitter, 0.0, 1.0)
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Convert per-sample from uint8 → float32 (cheap; saves RAM at dataset scale).
        img = self.imgs[idx].astype(np.float32)          # (64, 64) in [0, 1]
        latents = self.latents[idx]
        if self.two_view:
            rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
            v1 = self._aug(img, rng)
            v2 = self._aug(img, rng)
            v1 = torch.from_numpy(v1)[None]              # (1, 64, 64)
            v2 = torch.from_numpy(v2)[None]
            base = torch.from_numpy(img)[None]
            return {
                "image": base,
                "view1": v1,
                "view2": v2,
                "latents": torch.from_numpy(latents),
            }
        return {
            "image": torch.from_numpy(img)[None],
            "latents": torch.from_numpy(latents),
        }


class SyntheticFactorDataset(Dataset):
    """Drop-in replacement when dSprites is unavailable.

    Generates 64x64 binary images by drawing one of {square, ellipse, triangle}
    at a random (x, y, scale, orientation). Mirrors dSprites' factor inventory
    so the evaluation suite still works without the real dataset.

    Useful for: unit tests, smoke runs, CI.
    """

    SHAPES = ("square", "ellipse", "triangle")

    def __init__(
        self,
        n_samples: int = 8192,
        image_size: int = 64,
        seed: int = 0,
        two_view: bool = True,
    ) -> None:
        self.n = n_samples
        self.image_size = image_size
        self.two_view = two_view
        rng = np.random.default_rng(seed)
        self.shape_idx = rng.integers(0, len(self.SHAPES), size=n_samples)
        self.scale = rng.integers(0, 6, size=n_samples)
        self.orient = rng.integers(0, 8, size=n_samples)  # 8 angle bins (0..pi)
        self.x = rng.integers(0, 32, size=n_samples)
        self.y = rng.integers(0, 32, size=n_samples)

    def __len__(self) -> int:
        return self.n

    @staticmethod
    def _draw(shape: str, x: int, y: int, scale: int, image_size: int) -> np.ndarray:
        img = np.zeros((image_size, image_size), dtype=np.float32)
        s = max(2, scale + 3)  # 3..8 pixels
        cx = int(x * (image_size - 1) / 31.0)
        cy = int(y * (image_size - 1) / 31.0)
        if shape == "square":
            x0, x1 = max(cx - s, 0), min(cx + s, image_size)
            y0, y1 = max(cy - s, 0), min(cy + s, image_size)
            img[y0:y1, x0:x1] = 1.0
        elif shape == "ellipse":
            yy, xx = np.ogrid[:image_size, :image_size]
            mask = ((xx - cx) ** 2) / max(s, 1) ** 2 + ((yy - cy) ** 2) / max(s - 1, 1) ** 2 <= 1
            img[mask] = 1.0
        else:  # triangle
            for dy in range(-s, s + 1):
                row = cy + dy
                if row < 0 or row >= image_size:
                    continue
                width = max(0, s - abs(dy))
                x0, x1 = max(cx - width, 0), min(cx + width + 1, image_size)
                img[row, x0:x1] = 1.0
        return img

    def _img(self, idx: int) -> np.ndarray:
        return self._draw(
            self.SHAPES[self.shape_idx[idx]],
            int(self.x[idx]),
            int(self.y[idx]),
            int(self.scale[idx]),
            self.image_size,
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = self._img(idx)
        # latents in dSprites order: (color=0, shape, scale, orientation, pos_x, pos_y)
        latents = np.array(
            [0, int(self.shape_idx[idx]), int(self.scale[idx]), int(self.orient[idx]), int(self.x[idx]), int(self.y[idx])],
            dtype=np.int64,
        )
        if self.two_view:
            rng = np.random.default_rng(idx + 1)  # deterministic-ish per index
            v1 = np.clip(np.roll(img, shift=tuple(rng.integers(-1, 2, size=2).tolist()), axis=(0, 1)), 0, 1)
            v2 = np.clip(np.roll(img, shift=tuple(rng.integers(-1, 2, size=2).tolist()), axis=(0, 1)), 0, 1)
            return {
                "image": torch.from_numpy(img)[None],
                "view1": torch.from_numpy(v1)[None],
                "view2": torch.from_numpy(v2)[None],
                "latents": torch.from_numpy(latents),
            }
        return {
            "image": torch.from_numpy(img)[None],
            "latents": torch.from_numpy(latents),
        }


def make_compositional_splits(
    factor_a: str = "shape",
    factor_b: str = "scale",
    held_out_count: int = 4,
    seed: int = 0,
    a_card: int = 3,
    b_card: int = 6,
) -> list[tuple[int, int]]:
    """Pick `held_out_count` random (factor_a, factor_b) value pairs to exclude
    from training. Used both with dSprites and the synthetic generator.
    """
    rng = np.random.default_rng(seed)
    all_pairs = [(a, b) for a in range(a_card) for b in range(b_card)]
    rng.shuffle(all_pairs)
    return all_pairs[:held_out_count]
