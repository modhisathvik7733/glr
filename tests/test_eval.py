"""End-to-end test of the eval metrics on synthetic features."""

from __future__ import annotations

import numpy as np
import torch

from glr.eval.causal_swap import causal_slot_swap
from glr.eval.compositional import compositional_probe
from glr.eval.mig import mutual_information_gap
from glr.eval.probe import LinearProbe, _flatten, train_linear_probe


def _toy_factor_features(n: int = 200, k: int = 4, d: int = 16, n_factors: int = 3, seed: int = 0):
    """Generate features that linearly encode each factor in a single slot.

    Slot 0 carries factor 0, slot 1 carries factor 1, etc. with random noise.
    """
    rng = np.random.default_rng(seed)
    factors = rng.integers(0, 4, size=(n, n_factors))
    feats = rng.normal(0, 0.1, size=(n, k, d)).astype(np.float32)
    for j in range(min(n_factors, k)):
        # encode factor j into slot j's first dim
        feats[:, j, 0] = factors[:, j].astype(np.float32) + 0.05 * rng.normal(size=n)
    return torch.from_numpy(feats), torch.from_numpy(factors)


def test_linear_probe_recovers_factor():
    feats, factors = _toy_factor_features(n=400)
    cut = 200
    acc = train_linear_probe(feats[:cut], factors[:cut, 0], feats[cut:], factors[cut:, 0])
    assert acc > 0.7


def test_mig_runs():
    feats, factors = _toy_factor_features(n=400)
    out = mutual_information_gap(feats, factors, n_bins=10, factor_names=["f0", "f1", "f2"])
    assert "mig" in out
    assert isinstance(out["mig"], float)


def test_compositional_probe():
    feats, factors = _toy_factor_features(n=600)
    res = compositional_probe(
        train_feat=feats[:300],
        train_factors=factors[:300],
        in_dist_feat=feats[300:450],
        in_dist_factors=factors[300:450],
        held_out_feat=feats[450:],
        held_out_factors=factors[450:],
        factor_idx=0,
    )
    assert "in_dist_acc" in res and "held_out_acc" in res and "gap" in res


def test_causal_swap_with_dummy_model():
    """Use the real Stage A model on tiny synthetic data; verify swap eval runs."""
    from glr.models import StageAModel

    m = StageAModel(image_size=16, in_channels=1, feat_dim=16, slot_dim=32, num_slots=4, num_roles=4, filler_dim=8, decoder_hidden=16)
    m.eval()
    a = torch.randn(8, 1, 16, 16)
    b = torch.randn(8, 1, 16, 16)
    fa = torch.randint(0, 3, (8, 6))
    fb = torch.randint(0, 3, (8, 6))
    # build a probe on slots from random images
    with torch.no_grad():
        out = m(torch.randn(40, 1, 16, 16))
    feats = out["slots"]
    labels = torch.randint(0, 3, (40,))
    probe = LinearProbe()
    probe.fit(_flatten(feats), labels.numpy())

    res = causal_slot_swap(m, a, b, fa, fb, factor_idx=1, probe=probe)
    assert "swap_success_rate" in res
