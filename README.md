# GLR — Grounded Latent Reasoner

A research program toward a data-efficient, grounded-concept reasoning model.
This repo implements **Stage A** of the curriculum described in [ARCHITECTURE.md](ARCHITECTURE.md):
*does a slot-attention + TPR + VICReg model develop stable, disentangled, compositional concept attractors on synthetic visual factor datasets?*

If Stage A's gate passes, the rest of the curriculum (Stages B–H) becomes worth attempting.
If it fails, we have a falsifiable answer in 5 days of A100 time and learn what to fix.

The full plan, motivation, and downstream stages are in [ARCHITECTURE.md](ARCHITECTURE.md).

## What's here

```
glr/
├── models/        # slot attention, TPR binding, BCS layers
├── losses/        # VICReg, Sinkhorn routing, masked prediction, slot consistency
├── data/          # dSprites / 3D Shapes / CLEVR loaders
├── eval/          # linear probe, MIG, compositional split, causal slot-swap
├── train/         # Stage A training loop
└── utils/         # logging, config, seeding
scripts/           # entry points (train_stage_a.py, eval_stage_a.py)
tests/             # smoke tests for each module
configs/           # stage_a.yaml
```

## Stage A gate (must pass before moving to Stage B)

| Metric | Target |
|---|---|
| Linear probe accuracy on held-out factors | ≥ 80% |
| Mutual Information Gap (MIG) across slots | > 0.4 |
| Compositional held-out probe within | 5pt of training |
| Causal slot-swap intervention success | ≥ 70% |

Run all four with three different seeds before declaring the gate passed.

## Quickstart (work in progress)

```bash
# install
pip install -e ".[data,dev]"

# smoke test (CPU, tiny synthetic data)
pytest tests/test_smoke.py -v

# Stage A training (requires GPU, dSprites in data/raw/)
python scripts/train_stage_a.py --config configs/stage_a.yaml
```

## Status

- [x] Project skeleton
- [x] Slot attention encoder
- [x] TPR binding
- [x] VICReg / Sinkhorn / masked prediction losses
- [x] dSprites loader
- [x] Stage A training loop (smoke level)
- [x] Linear probe, MIG, compositional, causal-swap eval
- [ ] Run on dSprites end-to-end
- [ ] Add 3D Shapes loader
- [ ] Add CLEVR / CLEVRTex / MOVi-C loaders
- [ ] Stage A gate evaluation suite (3 seeds)

## License

MIT
