"""Slot Attention (Locatello et al., NeurIPS 2020).

Iterative attention from a fixed set of slots to a variable set of input features.
The competitive softmax-over-slots normalization is what makes slots specialize on
distinct input regions / factors instead of collapsing onto the same content.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat


class SlotAttention(nn.Module):
    """Slot Attention with optional Sinkhorn-balanced routing.

    Args:
        num_slots: K, number of slot vectors.
        slot_dim:  d, slot vector dimensionality.
        input_dim: dimensionality of the per-token input features (e.g. patch embeddings).
        n_iters:   number of attention iterations per forward call.
        hidden_mlp_dim: width of the per-slot MLP that follows the GRU update.
        eps: numerical stabilizer for attention normalization.
        sinkhorn_iters: if > 0, use Sinkhorn iterations to balance slot usage; 0 disables.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        input_dim: int,
        n_iters: int = 3,
        hidden_mlp_dim: int = 512,
        eps: float = 1e-8,
        sinkhorn_iters: int = 3,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.eps = eps
        self.sinkhorn_iters = sinkhorn_iters
        self.scale = slot_dim**-0.5

        # Slot init: parameterized mean + log-std (Gaussian prior, sampled per forward).
        self.slots_mu = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)

        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_mlp_dim),
            nn.GELU(),
            nn.Linear(hidden_mlp_dim, slot_dim),
        )

    def _init_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        return mu + sigma * torch.randn_like(mu)

    @staticmethod
    def _sinkhorn(log_attn: torch.Tensor, n_iters: int) -> torch.Tensor:
        """Doubly-stochastic-ish normalization in log space.

        log_attn: (B, N, K) — pre-softmax logits over slots for each input token.
        Alternates row (over-slots) and column (over-tokens) log-softmax to balance
        slot usage. Returns log-attention; caller exponentiates.
        """
        for _ in range(n_iters):
            log_attn = log_attn - log_attn.logsumexp(dim=-1, keepdim=True)  # row-stochastic over K
            log_attn = log_attn - log_attn.logsumexp(dim=-2, keepdim=True)  # col-balanced over N
        return log_attn - log_attn.logsumexp(dim=-1, keepdim=True)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update slots from inputs.

        Args:
            inputs: (B, N, input_dim).
        Returns:
            slots:     (B, K, slot_dim) — final slot states.
            attn_last: (B, N, K)        — attention from inputs to slots in the last iter.
        """
        b, n, _ = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        slots = self._init_slots(b, inputs.device)
        attn = inputs.new_zeros(b, n, self.num_slots)

        for _ in range(self.n_iters):
            slots_prev = slots
            slots_n = self.norm_slots(slots)
            q = self.to_q(slots_n) * self.scale  # (B, K, d)

            # logits: (B, N, K) — each input token attends over slots.
            log_attn = torch.einsum("b n d, b k d -> b n k", k, q)

            if self.sinkhorn_iters > 0:
                log_attn = self._sinkhorn(log_attn, self.sinkhorn_iters)
                attn = log_attn.exp()
            else:
                attn = log_attn.softmax(dim=-1)

            # weighted-mean update: each slot pulls a value-weighted sum of inputs assigned to it.
            attn_norm = attn + self.eps
            attn_norm = attn_norm / attn_norm.sum(dim=-2, keepdim=True)
            updates = torch.einsum("b n k, b n d -> b k d", attn_norm, v)

            slots = self.gru(
                rearrange(updates, "b k d -> (b k) d"),
                rearrange(slots_prev, "b k d -> (b k) d"),
            )
            slots = rearrange(slots, "(b k) d -> b k d", k=self.num_slots)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def soft_position_embedding_2d(h: int, w: int, dim: int) -> nn.Module:
    """Linear soft positional embedding for 2D feature grids (Locatello et al.).

    The 4-channel grid (x, 1-x, y, 1-y) is projected into the feature dim and added
    to the encoder output so slot attention can break the spatial symmetry.
    """

    class _SoftPos(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            xs = torch.linspace(0.0, 1.0, w)
            ys = torch.linspace(0.0, 1.0, h)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack([xx, 1.0 - xx, yy, 1.0 - yy], dim=-1)  # (H, W, 4)
            self.register_buffer("grid", grid.unsqueeze(0))  # (1, H, W, 4)
            self.proj = nn.Linear(4, dim)

        def forward(self, feats: torch.Tensor) -> torch.Tensor:
            # feats: (B, H, W, dim)
            return feats + self.proj(self.grid)

    return _SoftPos()
