"""Tensor Product Representations (Smolensky 1990) for slot binding.

Each slot's `slot_dim`-dimensional content is decomposed into a (role, filler)
outer-product binding: slot = sum_r role_r ⊗ filler_r. Roles are a small learnable
basis; the binding gives compositional structure that pure slot vectors lack
(answers to "who did what to whom" — Smolensky's original motivation).

This module provides:
  bind:    fillers (B, K, d_filler) + role weights (B, K, R) -> bound (B, K, R, d_filler)
  unbind:  bound (B, K, R, d_filler) + role probe (R,)       -> filler (B, K, d_filler)
  flatten: (B, K, R, d_filler) <-> (B, K, R*d_filler)         -> compatible with downstream layers
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TPRBinder(nn.Module):
    """Bind a per-slot filler vector to a learned role basis via outer product.

    Args:
        slot_dim:   total slot width (must equal num_roles * filler_dim).
        num_roles:  R, size of the learnable role basis.
        filler_dim: per-role filler dimensionality.
    """

    def __init__(self, slot_dim: int, num_roles: int, filler_dim: int) -> None:
        super().__init__()
        if slot_dim != num_roles * filler_dim:
            raise ValueError(
                f"slot_dim ({slot_dim}) must equal num_roles ({num_roles}) * filler_dim ({filler_dim})"
            )
        self.slot_dim = slot_dim
        self.num_roles = num_roles
        self.filler_dim = filler_dim

        # role basis vectors used both for binding and unbinding
        self.role_basis = nn.Parameter(torch.empty(num_roles, num_roles))
        nn.init.orthogonal_(self.role_basis)

        # learned per-slot router: from slot_dim -> (num_roles, filler_dim) factorization
        self.role_router = nn.Linear(slot_dim, num_roles)
        self.filler_proj = nn.Linear(slot_dim, filler_dim)

    def factor(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Factor slot vectors into (role weights, filler).

        slots: (B, K, slot_dim)
        Returns:
            role_weights: (B, K, R) — softmax over roles
            filler:       (B, K, filler_dim)
        """
        role_logits = self.role_router(slots)
        role_weights = F.softmax(role_logits, dim=-1)
        filler = self.filler_proj(slots)
        return role_weights, filler

    def bind(self, role_weights: torch.Tensor, filler: torch.Tensor) -> torch.Tensor:
        """Outer-product bind a filler with a soft role distribution.

        role_weights: (B, K, R)
        filler:       (B, K, filler_dim)
        Returns: (B, K, R, filler_dim)
        """
        # role_basis (R, R) lets the network rotate role identities away from one-hot.
        roles = role_weights @ self.role_basis  # (B, K, R)
        return torch.einsum("b k r, b k f -> b k r f", roles, filler)

    def unbind(self, bound: torch.Tensor, role_probe: torch.Tensor) -> torch.Tensor:
        """Recover a filler conditioned on a role probe.

        bound:      (B, K, R, filler_dim)
        role_probe: (R,) or (B, K, R) — which role to read out.
        Returns:    (B, K, filler_dim)
        """
        if role_probe.dim() == 1:
            role_probe = role_probe.view(1, 1, -1).expand(bound.size(0), bound.size(1), -1)
        rotated = role_probe @ self.role_basis.T
        return torch.einsum("b k r, b k r f -> b k f", rotated, bound)

    def flatten(self, bound: torch.Tensor) -> torch.Tensor:
        """(B, K, R, filler_dim) -> (B, K, R*filler_dim)."""
        return bound.flatten(start_dim=-2)

    def unflatten(self, flat: torch.Tensor) -> torch.Tensor:
        """(B, K, R*filler_dim) -> (B, K, R, filler_dim)."""
        b, k, _ = flat.shape
        return flat.view(b, k, self.num_roles, self.filler_dim)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """End-to-end: slots -> bound TPR -> flattened back to slot_dim."""
        role_weights, filler = self.factor(slots)
        bound = self.bind(role_weights, filler)
        return self.flatten(bound)
