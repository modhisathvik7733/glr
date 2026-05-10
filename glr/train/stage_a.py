"""Stage A trainer: Absorb-only model with reconstruction + masked prediction +
VICReg + slot consistency + slot-usage balance.

Loss schedule:
  - reconstruction loss: on from step 0 (the model needs a base task to anchor on)
  - masked prediction:   on from step 0 at low weight, ramp up
  - VICReg + balance:    auxiliary loss warmup over the first 5% of steps
  - slot consistency:    same warmup; requires two-view dataloader

The eval cadence runs every `eval_every` steps and computes the four Stage A
gates on a small held-out batch. Full gate evaluation is run separately via
scripts/eval_stage_a.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from glr.data.masking import apply_image_mask, random_token_mask
from glr.losses import masked_prediction_loss, slot_consistency_loss, vicreg_loss
from glr.losses.consistency import slot_usage_balance_loss
from glr.models import StageAModel
from glr.utils.memory import env_debug_enabled, mem_snapshot, mem_track, reset_peak


@dataclass
class StageATrainerConfig:
    # data
    image_size: int = 64
    in_channels: int = 1
    patch_size: int = 8
    # model
    feat_dim: int = 64
    slot_dim: int = 128
    num_slots: int = 8
    num_roles: int = 8
    filler_dim: int = 16
    slot_iters: int = 3
    sinkhorn_iters: int = 3
    decoder_hidden: int = 64
    decoder_type: str = "token"   # "token" (DINOSAUR-style, fast) or "spatial" (legacy)
    decoder_layers: int = 4       # only used by token decoder
    # optim
    lr: float = 5e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 500
    total_steps: int = 20_000
    aux_warmup_frac: float = 0.05  # auxiliary losses ramp in over the first 5%
    # mixed precision — bfloat16 autocast on CUDA. bf16 needs no GradScaler.
    # Roughly halves activation memory and uses tensor cores; recommended on
    # any Ampere+ / RTX 30+ / RTX 40+ / RTX 50+ card.
    use_amp: bool = True
    # gradient checkpointing on encoder + spatial-broadcast decoder. Required
    # for plan-spec K=64, d=512 on a 32GB GPU. Costs ~1.3x compute, saves
    # ~3-4x activation memory on the heaviest path.
    use_checkpointing: bool = False
    # torch.compile for the model. Adds ~1-2 min compile warmup on first step,
    # then typically 1.3-1.8x speedup on A100 for our shapes. Falls back to
    # eager mode automatically if compilation fails.
    use_compile: bool = True
    compile_mode: str = "default"  # "default" | "reduce-overhead" | "max-autotune"
    # masked prediction
    mask_ratio: float = 0.4
    pred_weight: float = 1.0
    # VICReg
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 0.04
    vicreg_total_weight: float = 0.5
    # consistency
    consistency_weight: float = 0.3
    usage_balance_weight: float = 0.1
    # logging
    eval_every: int = 1000
    log_every: int = 50
    out_dir: str = "outputs/stage_a"
    seed: int = 0
    # Weights & Biases. Activates when use_wandb=True AND WANDB_API_KEY is set
    # in the env (or the user has run `wandb login` previously). All scalar
    # metrics from the training step + eval are logged.
    use_wandb: bool = False
    wandb_project: str = "glr-stage-a"
    wandb_run_name: str | None = None       # default: f"seed{seed}"
    wandb_tags: tuple[str, ...] = ()


class StageATrainer:
    def __init__(self, config: StageATrainerConfig) -> None:
        self.config = config
        # Enable cuDNN auto-tuner for the heavy 5x5 convs in the spatial-broadcast
        # decoder. Picks the fastest cuDNN algorithm after a few iterations of
        # warmup. Big speedup on A100 for our fixed-shape workload.
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.model = StageAModel(
            image_size=config.image_size,
            in_channels=config.in_channels,
            feat_dim=config.feat_dim,
            slot_dim=config.slot_dim,
            num_slots=config.num_slots,
            num_roles=config.num_roles,
            filler_dim=config.filler_dim,
            slot_iters=config.slot_iters,
            sinkhorn_iters=config.sinkhorn_iters,
            decoder_hidden=config.decoder_hidden,
            decoder_type=config.decoder_type,
            decoder_layers=config.decoder_layers,
            patch_size=config.patch_size,
            use_checkpointing=config.use_checkpointing,
        )
        # torch.compile: 1.3-1.8x typical speedup on A100. First forward triggers
        # compilation (1-3 min), subsequent steps are fast. Runs the whole module.
        if config.use_compile and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode=config.compile_mode)
                print(f"[compile] torch.compile enabled (mode={config.compile_mode})")
            except Exception as e:
                print(f"[compile] failed ({e}); running in eager mode.")
        self.optim: torch.optim.Optimizer | None = None
        self.step = 0
        self.history: list[dict[str, float]] = []
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)
        # W&B: lazy init on first step (so model param count etc. is available)
        self._wandb = None
        self._wandb_initialized = False

    # ---- learning rate ----
    def _lr(self, step: int) -> float:
        warm = self.config.warmup_steps
        total = self.config.total_steps
        if step < warm:
            return self.config.lr * (step + 1) / max(warm, 1)
        # cosine decay
        progress = (step - warm) / max(total - warm, 1)
        progress = min(max(progress, 0.0), 1.0)
        return self.config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _aux_scale(self, step: int) -> float:
        warm = max(int(self.config.aux_warmup_frac * self.config.total_steps), 1)
        return min(step / warm, 1.0)

    # ---- W&B (lazy, optional) ----
    def _maybe_init_wandb(self) -> None:
        """Initialize a W&B run if configured. Idempotent."""
        if self._wandb_initialized or not self.config.use_wandb:
            return
        self._wandb_initialized = True
        try:
            import wandb  # local import: only required when use_wandb=True
        except ImportError:
            print("[wandb] not installed; skipping. `pip install wandb` to enable.")
            return
        run_name = self.config.wandb_run_name or f"seed{self.config.seed}"
        try:
            self._wandb = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                tags=list(self.config.wandb_tags) if self.config.wandb_tags else None,
                config={k: v for k, v in self.config.__dict__.items() if not k.startswith("_")},
                reinit="finish_previous",
            )
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self._wandb.summary["model/n_params"] = int(n_params)
            print(f"[wandb] logging to project={self.config.wandb_project} run={run_name}")
        except Exception as e:  # auth, network, anything
            print(f"[wandb] init failed ({e}); continuing without W&B.")
            self._wandb = None

    def _wandb_log(self, payload: dict, step: int) -> None:
        if self._wandb is not None:
            try:
                self._wandb.log(payload, step=step)
            except Exception as e:  # never let W&B break training
                print(f"[wandb] log failed ({e}); disabling.")
                self._wandb = None

    def _wandb_finish(self) -> None:
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
            self._wandb = None

    # ---- core step ----
    def _make_optim(self) -> None:
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )

    def _compute_target_features(self, image: torch.Tensor) -> torch.Tensor:
        """Run encoder+pos on the un-masked image to get the masked-prediction target.

        We detach this — it's the JEPA target, no gradient through it.
        """
        with torch.no_grad():
            from einops import rearrange
            feat_map = self.model.absorb.encoder(image)
            feat_map = rearrange(feat_map, "b d h w -> b h w d")
            feat_map = self.model.absorb.pos_embed(feat_map)
            feats = rearrange(feat_map, "b h w d -> b (h w) d")
            feats = self.model.absorb.feat_mlp(self.model.absorb.feat_norm(feats))
        return feats

    def step_batch(self, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
        cfg = self.config
        self.model.train()
        if self.optim is None:
            self._make_optim()
        for pg in self.optim.param_groups:
            pg["lr"] = self._lr(self.step)
        aux_scale = self._aux_scale(self.step)

        # Memory-debug switch. Enabled per-step at low cost; only the first
        # few steps print full traces. Set GLR_DEBUG_MEMORY=1 to enable.
        debug_mem = env_debug_enabled() and self.step < 3
        if debug_mem:
            reset_peak()
            print(f"\n=== step {self.step} memory trace ===")
            print(f"[mem] step start                                  {mem_snapshot()}")

        image = batch["image"].to(device)
        view1 = batch.get("view1")
        view2 = batch.get("view2")
        if view1 is not None:
            view1 = view1.to(device)
            view2 = view2.to(device)

        if debug_mem:
            print(f"[mem] after batch->device                         {mem_snapshot()}")

        # --- masking ---
        n_patches = (cfg.image_size // cfg.patch_size) ** 2
        mask = random_token_mask(image.size(0), n_patches, cfg.mask_ratio, device=device)
        image_masked = apply_image_mask(image, mask, cfg.patch_size)

        # bf16 autocast over the heavy compute. Keeps optimizer/master weights in fp32.
        amp_enabled = bool(cfg.use_amp) and device.type == "cuda"
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled)

        with autocast_ctx:
            with mem_track("target_feats (no_grad)", enabled=debug_mem):
                target_feats = self._compute_target_features(image)

            with mem_track("main forward (model)", enabled=debug_mem):
                out_main = self.model(image_masked)

            with mem_track("losses: pred + recon + vic + balance", enabled=debug_mem):
                ph = cfg.image_size // cfg.patch_size
                mask_pixel = (
                    mask.view(image.size(0), ph, ph)
                    .repeat_interleave(cfg.patch_size, dim=1)
                    .repeat_interleave(cfg.patch_size, dim=2)
                    .reshape(image.size(0), cfg.image_size * cfg.image_size)
                )
                loss_pred = masked_prediction_loss(out_main["pred_feats"], target_feats, mask_pixel)

                loss_recon = F.mse_loss(out_main["recon"], image)

                with torch.autocast(device_type="cuda", enabled=False):
                    vic = vicreg_loss(
                        out_main["slots"].float(),
                        var_weight=cfg.vicreg_var_weight,
                        cov_weight=cfg.vicreg_cov_weight,
                    )

                loss_balance = slot_usage_balance_loss(out_main["attn"])

            with mem_track("consistency (2x absorb on views)", enabled=debug_mem):
                if view1 is not None and view2 is not None:
                    out_a = self.model.absorb(view1)
                    out_b = self.model.absorb(view2)
                    loss_consistency = slot_consistency_loss(out_a["slots"], out_b["slots"])
                else:
                    loss_consistency = image.new_zeros(())

            total = (
                loss_recon
                + cfg.pred_weight * loss_pred
                + aux_scale * cfg.vicreg_total_weight * vic["total"]
                + aux_scale * cfg.usage_balance_weight * loss_balance
                + aux_scale * cfg.consistency_weight * loss_consistency
            )

        if debug_mem:
            print(f"[mem] before backward                             {mem_snapshot()}")

        with mem_track("backward", enabled=debug_mem):
            self.optim.zero_grad(set_to_none=True)
            total.backward()

        with mem_track("optim.step + clip", enabled=debug_mem):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optim.step()

        if debug_mem:
            print(f"[mem] step end                                    {mem_snapshot()}")

        log = {
            "step": float(self.step),
            "lr": self._lr(self.step),
            "aux_scale": aux_scale,
            "loss/total": float(total.detach()),
            "loss/recon": float(loss_recon.detach()),
            "loss/pred": float(loss_pred.detach()),
            "loss/vic_var": float(vic["var"].detach()),
            "loss/vic_cov": float(vic["cov"].detach()),
            "loss/balance": float(loss_balance.detach()),
            "loss/consistency": float(loss_consistency.detach()),
        }
        self.step += 1
        return log

    # ---- main loop ----
    def fit(
        self,
        train_loader: DataLoader,
        device: torch.device,
        eval_fn: Any | None = None,
    ) -> None:
        self.model.to(device)
        self._make_optim()
        self._maybe_init_wandb()
        t0 = time.time()
        train_iter = iter(train_loader)
        try:
            while self.step < self.config.total_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                log = self.step_batch(batch, device)
                self.history.append(log)
                self._wandb_log(log, step=self.step)

                if self.step % self.config.log_every == 0:
                    tps = (self.step + 1) / max(time.time() - t0, 1e-6)
                    print(
                        f"[step {self.step:6d}] "
                        f"loss={log['loss/total']:.4f} "
                        f"recon={log['loss/recon']:.4f} "
                        f"pred={log['loss/pred']:.4f} "
                        f"vic={log['loss/vic_var']:.3f}/{log['loss/vic_cov']:.3f} "
                        f"bal={log['loss/balance']:.3f} "
                        f"cons={log['loss/consistency']:.3f} "
                        f"lr={log['lr']:.2e} "
                        f"({tps:.1f} steps/s)"
                    )
                    self._wandb_log({"perf/steps_per_sec": float(tps)}, step=self.step)

                if eval_fn is not None and self.step % self.config.eval_every == 0:
                    eval_log = eval_fn(self.model)
                    print(f"[step {self.step:6d}] eval: {eval_log}")
                    # flatten eval dict into "eval/<name>" keys for W&B
                    flat: dict[str, float] = {}
                    for k, v in eval_log.items():
                        if isinstance(v, dict):
                            for sk, sv in v.items():
                                flat[f"eval/{k}/{sk}"] = float(sv)
                        else:
                            flat[f"eval/{k}"] = float(v)
                    self._wandb_log(flat, step=self.step)
        finally:
            self._wandb_finish()
