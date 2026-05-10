"""VICReg loss (Bardes et al. 2021), adapted for slot-content distributions.

Three terms:
  - invariance: not used here (we apply VICReg to slots within a single batch as a
    decorrelation regularizer, not across two views; slot-consistency handles two-view).
  - variance:   each slot dimension's std across the batch should be at least 1.
  - covariance: off-diagonal entries of the slot-content cov matrix should be ~0.

We apply this *per slot index*: slot k's contents across the batch should have
unit std and decorrelated dimensions. This is what defends against representation
collapse across slots.
"""

from __future__ import annotations

import torch


def variance_loss(x: torch.Tensor, eps: float = 1e-4, target_std: float = 1.0) -> torch.Tensor:
    """Hinge on the standard deviation of each feature dim. x: (..., D)."""
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return torch.mean(torch.relu(target_std - std))


def covariance_loss(x: torch.Tensor) -> torch.Tensor:
    """Off-diagonal mean-squared of the feature covariance matrix. x: (B, D)."""
    n, d = x.shape
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off**2).sum() / d


def vicreg_loss(
    slots: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
    target_std: float = 1.0,
) -> dict[str, torch.Tensor]:
    """VICReg-style regularization on slot contents.

    Args:
        slots: (B, K, D). Each slot index is treated as a separate "channel" — variance
               and covariance are computed across the batch dimension within each slot.

    Returns dict of (var, cov, total).
    """
    b, k, d = slots.shape
    # treat (B, D) per slot index, then average across K
    var_terms = []
    cov_terms = []
    for i in range(k):
        si = slots[:, i, :]  # (B, D)
        var_terms.append(variance_loss(si, target_std=target_std))
        cov_terms.append(covariance_loss(si))
    var_l = torch.stack(var_terms).mean()
    cov_l = torch.stack(cov_terms).mean()
    total = var_weight * var_l + cov_weight * cov_l
    return {"var": var_l, "cov": cov_l, "total": total}
