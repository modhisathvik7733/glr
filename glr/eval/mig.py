"""Mutual Information Gap (Chen et al. 2018) for slot disentanglement.

For each ground-truth factor v_j, estimate I(slot_k ; v_j) for every slot k
(via slot quantization), find the top-2 slots by MI, and report the gap
(top1 - top2) divided by H(v_j). Average across factors.

Higher MIG = each factor is dominantly captured by a single slot.
Stage A gate target: MIG > 0.4.
"""

from __future__ import annotations

import numpy as np
import torch


def _discrete_entropy(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def _discrete_mi(disc_x: np.ndarray, disc_y: np.ndarray) -> float:
    """Mutual information between two discrete arrays."""
    bins_x = np.unique(disc_x)
    bins_y = np.unique(disc_y)
    joint = np.zeros((len(bins_x), len(bins_y)))
    for i, vx in enumerate(bins_x):
        mask_x = disc_x == vx
        for j, vy in enumerate(bins_y):
            joint[i, j] = (mask_x & (disc_y == vy)).sum()
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = joint * (np.log(joint + 1e-12) - np.log(px + 1e-12) - np.log(py + 1e-12))
    return float(mi.sum())


def _quantize(arr: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Equal-frequency binning to make a continuous array discrete."""
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(arr, quantiles)
    return np.digitize(arr, edges)


def mutual_information_gap(
    slot_features: torch.Tensor,
    factors: torch.Tensor,
    n_bins: int = 20,
    factor_names: list[str] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Compute per-factor MIG and the average.

    slot_features: (B, K, D) — per-slot vectors. We summarize each slot by its L2 norm
                   *and* its mean across D (cheap two-feature summary), giving 2K total
                   features. This is a stand-in for richer summaries; for the published
                   MIG number you'd ideally probe per-dimension.
    factors:       (B, num_factors) integer labels.

    Returns:
        {"per_factor": {name: mig_j}, "mig": float}
    """
    b, k, d = slot_features.shape
    feats = torch.cat([slot_features.norm(dim=-1), slot_features.mean(dim=-1)], dim=-1)
    feats_np = feats.detach().cpu().numpy()
    factors_np = factors.detach().cpu().numpy()

    # quantize each slot summary
    disc_feats = np.stack([_quantize(feats_np[:, s], n_bins=n_bins) for s in range(feats_np.shape[1])], axis=1)
    n_factors = factors_np.shape[1]
    if factor_names is None:
        factor_names = [f"f{j}" for j in range(n_factors)]

    per_factor: dict[str, float] = {}
    for j, name in enumerate(factor_names):
        v = factors_np[:, j]
        if len(np.unique(v)) <= 1:
            continue
        h_v = _discrete_entropy(v)
        if h_v <= 1e-6:
            continue
        mis = np.array([_discrete_mi(disc_feats[:, s], v) for s in range(disc_feats.shape[1])])
        order = np.argsort(mis)[::-1]
        gap = (mis[order[0]] - mis[order[1]]) / h_v
        per_factor[name] = float(gap)

    mig = float(np.mean(list(per_factor.values()))) if per_factor else 0.0
    return {"per_factor": per_factor, "mig": mig}
