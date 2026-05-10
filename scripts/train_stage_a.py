"""Stage A training entry point.

Usage:
  python scripts/train_stage_a.py --config configs/stage_a.yaml [--device cpu|cuda|mps]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from glr.data import DSpritesDataset, SyntheticFactorDataset, make_compositional_splits
from glr.data.dsprites import DSPRITES_FACTORS
from glr.eval.mig import mutual_information_gap
from glr.eval.probe import probe_all_factors
from glr.train.stage_a import StageATrainer, StageATrainerConfig
from glr.utils.config import load_config
from glr.utils.seed import seed_all


def build_dataset(cfg: dict) -> tuple[torch.utils.data.Dataset, list[tuple[int, int]]]:
    held_out = make_compositional_splits(
        factor_a=cfg["data"]["held_out_keys"][0],
        factor_b=cfg["data"]["held_out_keys"][1],
        held_out_count=cfg["data"]["held_out_count"],
        seed=cfg["seed"],
    )
    if cfg["data"]["source"] == "dsprites":
        try:
            ds = DSpritesDataset(
                root=cfg["data"]["root"],
                two_view=cfg["data"]["two_view"],
                held_out_pairs=held_out,
                held_out_keys=tuple(cfg["data"]["held_out_keys"]),
            )
            print(f"[data] dSprites: {len(ds)} training images, {len(held_out)} held-out compositions")
            return ds, held_out
        except FileNotFoundError as e:
            print(f"[data] dSprites unavailable ({e}); falling back to synthetic.")
    ds = SyntheticFactorDataset(n_samples=8192, image_size=cfg["data"]["image_size"], seed=cfg["seed"])
    print(f"[data] synthetic: {len(ds)} images")
    return ds, held_out


def make_eval_fn(eval_loader: DataLoader, device: torch.device):
    factor_names = list(DSPRITES_FACTORS)

    def eval_fn(model) -> dict[str, float]:
        model.eval()
        feats, latents = [], []
        with torch.no_grad():
            for batch in eval_loader:
                out = model(batch["image"].to(device))
                feats.append(out["slots"].detach().cpu())
                latents.append(batch["latents"])
        feats = torch.cat(feats)
        latents = torch.cat(latents)
        n = feats.size(0)
        cut = n // 2
        train_feat, test_feat = feats[:cut], feats[cut:]
        train_lat, test_lat = latents[:cut], latents[cut:]
        probe_acc = probe_all_factors(train_feat, train_lat, test_feat, test_lat, factor_names)
        mig = mutual_information_gap(feats, latents, factor_names=factor_names)
        return {
            "probe": {k: round(v, 3) for k, v in probe_acc.items()},
            "mig": round(mig["mig"], 3),
        }

    return eval_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_a.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None, help="Override total_steps.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    device = torch.device(args.device or cfg.get("device", "cpu"))
    print(f"[device] {device}")

    # TF32 on Ampere+ silences the torch._inductor float32-matmul warning and
    # gives ~10-20% TFLOPS on float32 matmuls without affecting bf16 autocast paths.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    train_ds, held_out_pairs = build_dataset(cfg)
    n_workers = cfg["data"]["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        # persistent_workers keeps the workers alive across epochs (saves
        # ~3-5 sec per epoch boundary, big deal on small datasets).
        persistent_workers=(n_workers > 0),
        # prefetch_factor=2: each worker buffers 2 batches ahead. Down from 4
        # to bound peak RAM usage on 32 GB boxes (each buffered batch is a
        # full COW-broken dataset copy in worker memory).
        prefetch_factor=2 if n_workers > 0 else None,
    )

    # eval split: a separate (small) subset for periodic gate checks
    eval_ds = SyntheticFactorDataset(
        n_samples=cfg["eval"]["n_eval_samples"],
        image_size=cfg["data"]["image_size"],
        seed=cfg["seed"] + 999,
        two_view=False,
    ) if cfg["data"]["source"] == "synthetic" else train_ds
    # Match eval batch_size to train batch_size (and drop_last) so the compiled
    # model graph hits the cache on eval too. With mismatched shapes torch.compile
    # silently recompiles the eval-mode forward, producing the multi-minute pause
    # observed at step 2000.
    # num_workers=0: eval runs in main process. Fork-copying the dataset to extra
    # workers at step `eval_every` was the OOM trigger on the 32 GB box.
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    train_cfg = StageATrainerConfig(
        image_size=cfg["data"]["image_size"],
        in_channels=cfg["data"]["in_channels"],
        feat_dim=cfg["model"]["feat_dim"],
        slot_dim=cfg["model"]["slot_dim"],
        num_slots=cfg["model"]["num_slots"],
        num_roles=cfg["model"]["num_roles"],
        filler_dim=cfg["model"]["filler_dim"],
        slot_iters=cfg["model"]["slot_iters"],
        sinkhorn_iters=cfg["model"]["sinkhorn_iters"],
        decoder_hidden=cfg["model"]["decoder_hidden"],
        decoder_type=cfg["model"].get("decoder_type", "token"),
        decoder_layers=cfg["model"].get("decoder_layers", 4),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=tuple(cfg["training"]["betas"]),
        grad_clip=cfg["training"]["grad_clip"],
        warmup_steps=cfg["training"]["warmup_steps"],
        total_steps=args.steps or cfg["training"]["total_steps"],
        aux_warmup_frac=cfg["training"]["aux_warmup_frac"],
        patch_size=cfg["training"]["patch_size"],
        mask_ratio=cfg["training"]["mask_ratio"],
        pred_weight=cfg["training"]["pred_weight"],
        vicreg_var_weight=cfg["training"]["vicreg_var_weight"],
        vicreg_cov_weight=cfg["training"]["vicreg_cov_weight"],
        vicreg_total_weight=cfg["training"]["vicreg_total_weight"],
        consistency_weight=cfg["training"]["consistency_weight"],
        usage_balance_weight=cfg["training"]["usage_balance_weight"],
        log_every=cfg["training"]["log_every"],
        eval_every=cfg["training"]["eval_every"],
        use_amp=cfg["training"].get("use_amp", True),
        use_checkpointing=cfg["training"].get("use_checkpointing", False),
        use_compile=cfg["training"].get("use_compile", True),
        compile_mode=cfg["training"].get("compile_mode", "default"),
        use_wandb=cfg.get("wandb", {}).get("enabled", False),
        wandb_project=cfg.get("wandb", {}).get("project", "glr-stage-a"),
        wandb_run_name=cfg.get("wandb", {}).get("run_name") or f"seed{cfg['seed']}",
        wandb_tags=tuple(cfg.get("wandb", {}).get("tags", [])),
        out_dir=cfg["out_dir"],
        seed=cfg["seed"],
    )

    trainer = StageATrainer(train_cfg)
    print(f"[model] params: {trainer.model.n_params() / 1e6:.2f}M")
    eval_fn = make_eval_fn(eval_loader, device)

    trainer.fit(train_loader, device=device, eval_fn=eval_fn)

    # save final
    out = Path(train_cfg.out_dir) / "final.pt"
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "config": train_cfg.__dict__,
            "history": trainer.history[-200:],
            "held_out_pairs": held_out_pairs,
        },
        out,
    )
    print(f"[done] checkpoint saved to {out}")


if __name__ == "__main__":
    main()
