"""Bound Concept States — Stage A only uses the Absorb sublayer.

Absorb: image patches -> slot-attention -> TPR-bound slot states.

Resonate and Emit are added in later curriculum stages (B+, D+) and live elsewhere.
This file is intentionally narrow to Stage A's scope.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from glr.models.slot_attention import SlotAttention, soft_position_embedding_2d
from glr.models.tpr import TPRBinder


class ConvEncoder(nn.Module):
    """Tiny CNN backbone that lifts an image to a per-pixel feature grid."""

    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AbsorbBlock(nn.Module):
    """Absorb sublayer for Stage A.

    Image -> CNN features -> position embedding -> slot attention -> TPR-bound slots.

    Args:
        image_size:  H = W of the input image.
        in_channels: image channels.
        feat_dim:    CNN output feature dimensionality.
        slot_dim:    slot vector width (must equal num_roles * filler_dim).
        num_slots:   K.
        num_roles:   R.
        filler_dim:  per-role filler dimensionality.
        slot_iters:  iterations of slot attention.
        sinkhorn_iters: 0 to disable balanced routing, else number of Sinkhorn passes.
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 1,
        feat_dim: int = 64,
        slot_dim: int = 128,
        num_slots: int = 8,
        num_roles: int = 8,
        filler_dim: int = 16,
        slot_iters: int = 3,
        sinkhorn_iters: int = 3,
    ) -> None:
        super().__init__()
        if slot_dim != num_roles * filler_dim:
            raise ValueError("slot_dim must equal num_roles * filler_dim")

        self.image_size = image_size
        self.encoder = ConvEncoder(in_channels, feat_dim)
        self.pos_embed = soft_position_embedding_2d(image_size, image_size, feat_dim)
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.slot_attn = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=feat_dim,
            n_iters=slot_iters,
            sinkhorn_iters=sinkhorn_iters,
        )
        self.tpr = TPRBinder(slot_dim, num_roles, filler_dim)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode an image into TPR-bound slot states.

        Args:
            image: (B, C, H, W).
        Returns:
            dict with:
              slots:       (B, K, slot_dim)        — raw slot vectors
              slots_bound: (B, K, R, filler_dim)   — TPR-bound representation
              slots_flat:  (B, K, slot_dim)        — flattened bound (for downstream layers)
              attn:        (B, N, K)               — input-token-to-slot attention (last iter)
              feats:       (B, N, feat_dim)        — pre-slot encoder features
        """
        b = image.size(0)
        feat_map = self.encoder(image)                               # (B, feat_dim, H, W)
        feat_map = rearrange(feat_map, "b d h w -> b h w d")
        feat_map = self.pos_embed(feat_map)                          # (B, H, W, feat_dim)
        feats = rearrange(feat_map, "b h w d -> b (h w) d")
        feats = self.feat_mlp(self.feat_norm(feats))

        slots, attn = self.slot_attn(feats)                          # (B, K, slot_dim), (B, N, K)
        role_weights, filler = self.tpr.factor(slots)
        bound = self.tpr.bind(role_weights, filler)
        return {
            "slots": slots,
            "slots_bound": bound,
            "slots_flat": self.tpr.flatten(bound),
            "attn": attn,
            "feats": feats,
            "role_weights": role_weights,
            "filler": filler,
        }
