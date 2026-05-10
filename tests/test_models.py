"""Shape and basic-correctness tests for model components."""

from __future__ import annotations

import torch

from glr.models import AbsorbBlock, SlotAttention, StageAModel, TPRBinder


def test_slot_attention_shapes():
    sa = SlotAttention(num_slots=4, slot_dim=32, input_dim=16, n_iters=2, sinkhorn_iters=1)
    feats = torch.randn(2, 25, 16)
    slots, attn = sa(feats)
    assert slots.shape == (2, 4, 32)
    assert attn.shape == (2, 25, 4)
    # Sinkhorn: each input token's attention over slots should sum to ~1
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 25), atol=1e-3)


def test_tpr_round_trip():
    tpr = TPRBinder(slot_dim=32, num_roles=4, filler_dim=8)
    slots = torch.randn(3, 5, 32)
    role_w, filler = tpr.factor(slots)
    bound = tpr.bind(role_w, filler)
    assert bound.shape == (3, 5, 4, 8)
    flat = tpr.flatten(bound)
    assert flat.shape == (3, 5, 32)
    unflat = tpr.unflatten(flat)
    assert unflat.shape == (3, 5, 4, 8)
    # forward
    out = tpr(slots)
    assert out.shape == (3, 5, 32)


def test_absorb_block_shapes():
    block = AbsorbBlock(image_size=16, in_channels=1, feat_dim=16, slot_dim=32, num_slots=4, num_roles=4, filler_dim=8)
    img = torch.randn(2, 1, 16, 16)
    out = block(img)
    assert out["slots"].shape == (2, 4, 32)
    assert out["slots_bound"].shape == (2, 4, 4, 8)
    assert out["slots_flat"].shape == (2, 4, 32)
    assert out["attn"].shape == (2, 16 * 16, 4)
    assert out["role_weights"].shape == (2, 4, 4)
    assert out["filler"].shape == (2, 4, 8)


def test_stage_a_model_forward():
    m = StageAModel(image_size=16, in_channels=1, feat_dim=16, slot_dim=32, num_slots=4, num_roles=4, filler_dim=8, decoder_hidden=16)
    img = torch.randn(2, 1, 16, 16)
    out = m(img)
    assert out["recon"].shape == (2, 1, 16, 16)
    assert out["masks"].shape == (2, 4, 1, 16, 16)
    assert out["per_slot_recon"].shape == (2, 4, 1, 16, 16)
    assert out["pred_feats"].shape == (2, 16 * 16, 16)
    # per-pixel masks should sum to 1 across slots
    mask_sum = out["masks"].sum(dim=1)
    assert torch.allclose(mask_sum, torch.ones_like(mask_sum), atol=1e-4)


def test_stage_a_backward():
    m = StageAModel(image_size=16, in_channels=1, feat_dim=16, slot_dim=32, num_slots=4, num_roles=4, filler_dim=8, decoder_hidden=16)
    img = torch.randn(2, 1, 16, 16)
    out = m(img)
    loss = out["recon"].pow(2).mean() + out["pred_feats"].pow(2).mean()
    loss.backward()
    # at least the encoder's first conv should have non-zero gradient
    grad = m.absorb.encoder.net[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_stage_a_backward_with_checkpointing():
    """Gradient checkpointing path produces non-zero gradients on the same params."""
    m = StageAModel(
        image_size=16, in_channels=1, feat_dim=16, slot_dim=32, num_slots=4,
        num_roles=4, filler_dim=8, decoder_hidden=16, use_checkpointing=True,
    )
    m.train()
    img = torch.randn(2, 1, 16, 16, requires_grad=False)
    out = m(img)
    loss = out["recon"].pow(2).mean() + out["pred_feats"].pow(2).mean()
    loss.backward()
    enc_grad = m.absorb.encoder.net[0].weight.grad
    dec_grad = m.decoder.net[0].weight.grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0
    assert dec_grad is not None and dec_grad.abs().sum() > 0
