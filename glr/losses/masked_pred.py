"""Masked-feature prediction loss (JEPA-flavored).

Given a target feature map (e.g., the encoder's output on the un-masked image)
and a predicted feature map produced by routing slot states back to spatial
positions, demand the prediction matches the target on masked positions only.

This is the loss term that pushes slots to encode information sufficient to
reconstruct the latent representation of any masked region — the JEPA insight,
applied at the slot level instead of the patch level.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """Smooth-L1 over masked positions only.

    Args:
        pred:   (B, N, D) predicted features.
        target: (B, N, D) target features (typically detached).
        mask:   (B, N) boolean — True where the position was masked in the input.
                Loss is computed only on these positions.
        huber_delta: smooth-L1 transition point.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    target = target.detach()
    diff = pred - target
    # smooth-L1 / Huber per element
    abs_diff = diff.abs()
    quadratic = 0.5 * (diff**2) / huber_delta
    linear = abs_diff - 0.5 * huber_delta
    elem = torch.where(abs_diff <= huber_delta, quadratic, linear)  # (B, N, D)
    elem = elem.mean(dim=-1)  # (B, N)

    if mask.any():
        return elem[mask].mean()
    return elem.mean() * 0.0
