"""Slot-consistency loss across two augmented views of the same input.

Same-image, different-augmentation pairs should land in the same concept-slot
configuration. We do this by Hungarian-matching slots across the two views
(via greedy permutation, which is good enough at K ~ 64 and avoids importing
scipy at runtime), then minimize cosine distance between matched pairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _greedy_match(cost: torch.Tensor) -> torch.Tensor:
    """Greedy K x K matching by sorting cost ascending. cost: (K, K). Returns perm of size K."""
    k = cost.size(0)
    perm = cost.new_full((k,), -1, dtype=torch.long)
    used_cols = cost.new_zeros(k, dtype=torch.bool)
    flat = cost.flatten()
    order = flat.argsort()
    for idx in order.tolist():
        r, c = divmod(idx, k)
        if perm[r] == -1 and not used_cols[c]:
            perm[r] = c
            used_cols[c] = True
            if (perm >= 0).all():
                break
    return perm


def slot_consistency_loss(slots_a: torch.Tensor, slots_b: torch.Tensor) -> torch.Tensor:
    """Cosine-distance loss between matched slots across two views.

    slots_a, slots_b: (B, K, D). Returns scalar mean over the batch.
    """
    b, k, d = slots_a.shape
    a = F.normalize(slots_a, dim=-1)
    bb = F.normalize(slots_b, dim=-1)
    losses = []
    for i in range(b):
        # cost = 1 - cosine sim
        sim = a[i] @ bb[i].T  # (K, K)
        cost = 1.0 - sim
        perm = _greedy_match(cost.detach())
        matched = cost[torch.arange(k, device=cost.device), perm]
        losses.append(matched.mean())
    return torch.stack(losses).mean()


def slot_usage_balance_loss(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Penalize low-entropy slot-usage distributions (anti-slot-deadness).

    attn: (B, N, K). We compute the average over (B, N) of slot probability,
    then take negative entropy of that distribution (lower entropy => imbalanced
    usage). Loss is `log K - H(p)`, so 0 when uniformly used.
    """
    p = attn.mean(dim=(0, 1))  # (K,)
    p = p + eps
    p = p / p.sum()
    ent = -(p * p.log()).sum()
    return torch.log(torch.tensor(p.numel(), dtype=p.dtype, device=p.device)) - ent
