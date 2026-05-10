"""Random patch-mask generator for the masked-prediction loss."""

from __future__ import annotations

import torch


def random_token_mask(
    batch_size: int,
    n_tokens: int,
    mask_ratio: float = 0.4,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean mask of shape (B, N), True at masked positions."""
    n_mask = int(round(mask_ratio * n_tokens))
    mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device)
    if n_mask <= 0:
        return mask
    for i in range(batch_size):
        idx = torch.randperm(n_tokens, device=device)[:n_mask]
        mask[i, idx] = True
    return mask


def apply_image_mask(image: torch.Tensor, mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Zero out masked patches in an image.

    image: (B, C, H, W). mask: (B, N) where N = (H/patch) * (W/patch).
    Returns image with masked patches set to 0 (caller can also use other strategies
    like learnable mask token).
    """
    b, c, h, w = image.shape
    ph, pw = h // patch_size, w // patch_size
    if mask.size(1) != ph * pw:
        raise ValueError(f"mask has {mask.size(1)} tokens but expected {ph * pw} for {h}x{w}")
    out = image.clone()
    mask_grid = mask.view(b, ph, pw)
    # expand to (B, 1, H, W)
    mask_full = mask_grid.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    out = out * (~mask_full).float().unsqueeze(1)
    return out
