"""Loss-function unit tests."""

from __future__ import annotations

import torch

from glr.losses import masked_prediction_loss, slot_consistency_loss, vicreg_loss
from glr.losses.consistency import slot_usage_balance_loss


def test_vicreg_loss():
    slots = torch.randn(32, 4, 16)
    out = vicreg_loss(slots)
    assert out["total"].dim() == 0
    assert out["var"] >= 0
    assert out["cov"] >= 0


def test_masked_prediction_loss():
    pred = torch.randn(2, 10, 8, requires_grad=True)
    target = torch.randn(2, 10, 8)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, :3] = True
    loss = masked_prediction_loss(pred, target, mask)
    assert loss.dim() == 0
    loss.backward()
    assert pred.grad is not None


def test_masked_prediction_no_mask():
    pred = torch.randn(2, 10, 8)
    target = torch.randn(2, 10, 8)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    # all-zero mask should still produce a finite loss (zero)
    loss = masked_prediction_loss(pred, target, mask)
    assert torch.isfinite(loss)


def test_slot_consistency_loss():
    a = torch.randn(4, 6, 16, requires_grad=True)
    b = torch.randn(4, 6, 16)
    loss = slot_consistency_loss(a, b)
    assert loss.dim() == 0
    loss.backward()
    assert a.grad is not None


def test_slot_consistency_identity():
    """Same slots in both views -> loss should be small.

    Soft Sinkhorn matching at low temperature gives near-identity weights
    on identical inputs, so residual loss is tiny but not exactly 0.
    """
    torch.manual_seed(0)
    a = torch.randn(2, 8, 64)
    loss = slot_consistency_loss(a, a)
    assert loss.abs() < 0.05


def test_slot_usage_balance():
    # uniform attention -> 0
    attn = torch.full((2, 10, 4), 0.25)
    loss = slot_usage_balance_loss(attn)
    assert loss.abs() < 1e-5
    # imbalanced attention -> positive
    attn_imb = torch.zeros(2, 10, 4)
    attn_imb[..., 0] = 1.0
    loss_imb = slot_usage_balance_loss(attn_imb)
    assert loss_imb > 0.5
