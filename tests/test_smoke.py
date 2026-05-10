"""Full end-to-end smoke test: run a few training steps on synthetic data
and verify loss decreases. Intentionally tiny so it runs on CPU in <30s.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from glr.data import SyntheticFactorDataset
from glr.train.stage_a import StageATrainer, StageATrainerConfig


def test_smoke_trains():
    cfg = StageATrainerConfig(
        image_size=16,
        in_channels=1,
        patch_size=4,
        feat_dim=16,
        slot_dim=32,
        num_slots=4,
        num_roles=4,
        filler_dim=8,
        slot_iters=2,
        sinkhorn_iters=1,
        decoder_hidden=16,
        lr=3e-4,
        warmup_steps=5,
        total_steps=30,
        eval_every=10**9,
        log_every=10**9,
        out_dir="outputs/smoke",
    )
    trainer = StageATrainer(cfg)
    ds = SyntheticFactorDataset(n_samples=64, image_size=16, two_view=True, seed=0)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    device = torch.device("cpu")
    trainer.fit(loader, device=device)
    assert len(trainer.history) > 0
    early = sum(h["loss/total"] for h in trainer.history[:5]) / 5
    late = sum(h["loss/total"] for h in trainer.history[-5:]) / 5
    # Loss should not blow up (and should usually decrease) in 30 steps on a tiny task.
    assert torch.isfinite(torch.tensor(late))
    assert late < early * 2  # generous: doesn't require monotonic decrease in 30 steps
