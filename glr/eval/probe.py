"""Linear probe on slot contents: do slots linearly encode ground-truth factors?

For each ground-truth factor we train a logistic-regression probe on the
flattened slot tensor (B, K*D) with frozen features. Test accuracy is the
gate metric. Stage A gate target: >= 80% averaged across factors.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


class LinearProbe:
    """Wrapper around sklearn's LogisticRegression for slot-feature probing."""

    def __init__(self, max_iter: int = 1000, C: float = 1.0) -> None:
        self.max_iter = max_iter
        self.C = C
        self.clf: LogisticRegression | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        clf = LogisticRegression(max_iter=self.max_iter, C=self.C)
        clf.fit(features, labels)
        self.clf = clf

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        if self.clf is None:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        return float(self.clf.score(features, labels))


def _flatten(feat: torch.Tensor) -> np.ndarray:
    """(B, K, D) -> (B, K*D) numpy array."""
    return feat.detach().cpu().reshape(feat.size(0), -1).numpy()


def train_linear_probe(
    train_feat: torch.Tensor,
    train_labels: torch.Tensor,
    test_feat: torch.Tensor,
    test_labels: torch.Tensor,
    max_iter: int = 1000,
) -> float:
    """Train a single-factor linear probe and return test accuracy.

    train_feat / test_feat: (B, K, D) slot tensors.
    train_labels / test_labels: (B,) integer labels for one factor.
    """
    probe = LinearProbe(max_iter=max_iter)
    probe.fit(_flatten(train_feat), train_labels.cpu().numpy())
    return probe.score(_flatten(test_feat), test_labels.cpu().numpy())


def probe_all_factors(
    train_feat: torch.Tensor,
    train_factors: torch.Tensor,
    test_feat: torch.Tensor,
    test_factors: torch.Tensor,
    factor_names: list[str],
    skip_constant: bool = True,
) -> dict[str, float]:
    """Probe each factor independently. Returns dict of {factor_name: test_accuracy}.

    train_factors / test_factors: (B, num_factors) integer labels.
    """
    out: dict[str, float] = {}
    for j, name in enumerate(factor_names):
        train_lbl = train_factors[:, j]
        test_lbl = test_factors[:, j]
        if skip_constant and len(torch.unique(train_lbl)) <= 1:
            continue
        out[name] = train_linear_probe(train_feat, train_lbl, test_feat, test_lbl)
    return out
