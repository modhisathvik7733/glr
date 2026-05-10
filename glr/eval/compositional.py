"""Compositional held-out probe: does the slot space generalize to novel
(factor_a, factor_b) combinations seen at probe-test time but excluded during
slot training?

The probe is trained on training-set examples (with the held-out compositions removed)
and evaluated on examples whose (factor_a, factor_b) tuple was never seen during
training. If slots compose, probe accuracy on held-out should be within 5pt of
training-distribution accuracy. Stage A gate target.
"""

from __future__ import annotations

from typing import Iterable

import torch

from glr.eval.probe import LinearProbe, _flatten


def compositional_probe(
    train_feat: torch.Tensor,
    train_factors: torch.Tensor,
    in_dist_feat: torch.Tensor,
    in_dist_factors: torch.Tensor,
    held_out_feat: torch.Tensor,
    held_out_factors: torch.Tensor,
    factor_idx: int,
) -> dict[str, float]:
    """Train probe on (train_*) for one factor, evaluate on both in-dist and held-out.

    Args:
        train_feat:  (B_train, K, D) slot tensors (no held-out compositions)
        train_factors: (B_train, F) integer labels
        in_dist_feat / in_dist_factors: examples drawn from the same training distribution
            but unseen during slot training (a clean test split).
        held_out_feat / held_out_factors: examples whose (factor_a, factor_b) was excluded.
        factor_idx:  which factor to predict (column in factors arrays).

    Returns:
        {"in_dist_acc": float, "held_out_acc": float, "gap": float}
    """
    probe = LinearProbe()
    probe.fit(_flatten(train_feat), train_factors[:, factor_idx].cpu().numpy())
    id_acc = probe.score(_flatten(in_dist_feat), in_dist_factors[:, factor_idx].cpu().numpy())
    ho_acc = probe.score(_flatten(held_out_feat), held_out_factors[:, factor_idx].cpu().numpy())
    return {"in_dist_acc": id_acc, "held_out_acc": ho_acc, "gap": id_acc - ho_acc}
