"""Stage A model: Absorb-only architecture for concept-emergence experiments.

Forward pass:
    image -> Absorb (CNN + slot attention + TPR) -> slots
    slots -> decoder -> reconstructed image (per-slot + alpha mask)
    slots -> masked-prediction head -> predicted feature for masked patches

Two decoder choices:

  - ``token`` (default, DINOSAUR-style): each slot attends to a small grid of
    patch tokens (e.g. 8x8 = 64 patches for a 64x64 image), produces per-patch
    features, then unpatchifies to pixels. Compute scales as B*K*num_patches,
    typically 64x cheaper than spatial broadcast on 64x64 images.
  - ``spatial`` (Locatello et al.): broadcasts every slot to every pixel and
    runs a 4-layer conv stack. Simpler, but ~50 TFLOPs per conv layer at
    plan-spec K=64 — too expensive in practice on a 64x64 grid.

Both decoders produce the same outputs (recon, masks, per_slot_recon) and are
swappable via the ``decoder_type`` argument on ``StageAModel``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from glr.models.bcs import AbsorbBlock


class SpatialBroadcastDecoder(nn.Module):
    """Decode each slot independently onto a spatial grid, then mix via alpha softmax.

    Memory note: this is the heaviest activation source in the model — every
    slot is broadcast to a full HxW grid before the conv stack, so activations
    grow as `B × K × hidden_dim × H × W`.

    Implementation detail: the broadcast tensor is built in **one allocation**
    (slot expand + position-bias add) instead of `repeat → rearrange → pos_embed
    → rearrange`. The naive sequence creates a transient extra 17 GB copy at
    plan-spec K=64, d=512, B=64. The single-alloc version doesn't.
    """

    def __init__(
        self,
        slot_dim: int,
        out_channels: int,
        image_size: int,
        hidden_dim: int = 64,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.slot_dim = slot_dim
        self.use_checkpointing = use_checkpointing
        # Position grid is fixed; project to slot_dim via a small Linear so
        # we get a (1, slot_dim, H, W) bias we add into the broadcast tensor.
        h = w = image_size
        xs = torch.linspace(0.0, 1.0, w)
        ys = torch.linspace(0.0, 1.0, h)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, 1.0 - xx, yy, 1.0 - yy], dim=-1)  # (H, W, 4)
        self.register_buffer("pos_grid", grid.unsqueeze(0))  # (1, H, W, 4)
        self.pos_proj = nn.Linear(4, slot_dim)
        # tiny conv-decoder; outputs out_channels + 1 alpha per pixel per slot
        self.net = nn.Sequential(
            nn.Conv2d(slot_dim, hidden_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels + 1, 3, padding=1),
        )
        self.out_channels = out_channels

    def _pos_bias(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return the (1, slot_dim, H, W) position bias in the requested dtype."""
        pos = self.pos_proj(self.pos_grid.to(dtype=dtype, device=device))  # (1, H, W, d)
        return pos.permute(0, 3, 1, 2)  # (1, d, H, W)

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """slots: (B, K, slot_dim) -> (recon, masks, per_slot_recon)."""
        b, k, d = slots.shape
        h = w = self.image_size
        # Single big allocation: expand slots to (BK, d, H, W) via view+expand
        # (no copy), then add the small (1, d, H, W) position bias to materialize.
        pos_bias = self._pos_bias(slots.dtype, slots.device)
        slots_flat = slots.reshape(b * k, d, 1, 1)
        broadcast = slots_flat.expand(b * k, d, h, w) + pos_bias  # (BK, d, H, W)

        if self.use_checkpointing and self.training:
            out = torch.utils.checkpoint.checkpoint_sequential(
                self.net, segments=3, input=broadcast, use_reentrant=False
            )
        else:
            out = self.net(broadcast)
        out = rearrange(out, "(b k) c h w -> b k c h w", b=b, k=k)
        rgb, alpha = out[:, :, : self.out_channels], out[:, :, self.out_channels : self.out_channels + 1]
        masks = alpha.softmax(dim=1)
        recon = (rgb * masks).sum(dim=1)
        return recon, masks, rgb


class TokenDecoder(nn.Module):
    """DINOSAUR-style token decoder: ~64x cheaper than spatial broadcast at 64x64.

    Each slot is broadcast to a small grid of patch tokens (e.g. 8x8 = 64
    patches for a 64x64 image), passed through a few MLP blocks with patch
    position embeddings, and projected to per-patch (RGB+alpha) values that
    are unpatchified to image space. Per-slot rgb/alpha are mixed via softmax
    over slots, exactly like the spatial broadcast decoder — same outputs,
    same training signal, much less compute.

    Compute scaling: O(B * K * num_patches * hidden_dim^2) per MLP block.
    With num_patches=64 vs 4096 spatial positions, this is 64x cheaper than
    spatial broadcast on 64x64 images.
    """

    def __init__(
        self,
        slot_dim: int,
        out_channels: int,
        image_size: int,
        patch_size: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 4,
        mlp_ratio: float = 2.0,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.use_checkpointing = use_checkpointing

        # Per-patch positional bias, shared across slots and batch
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Project slot vectors to decoder hidden dim once
        self.slot_in = nn.Linear(slot_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)

        # Stack of pre-norm residual MLP blocks
        hidden_inner = int(hidden_dim * mlp_ratio)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_inner),
                    nn.GELU(),
                    nn.Linear(hidden_inner, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        # Output: hidden_dim -> patch_size^2 * (C + 1) per patch token
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, patch_size * patch_size * (out_channels + 1))

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """slots: (B, K, slot_dim) -> (recon, masks, per_slot_recon)."""
        b, k, _ = slots.shape

        # (B, K, slot_dim) -> (B, K, hidden) -> (B, K, num_patches, hidden) via expand
        x = self.norm_in(self.slot_in(slots))
        x = x.unsqueeze(2).expand(-1, -1, self.num_patches, -1).contiguous()
        x = x + self.pos_embed  # (1, 1, num_patches, hidden) broadcasts

        # Residual MLP blocks
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = x + torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = x + block(x)

        # Project to per-patch (RGB + alpha) pixels
        x = self.head(self.norm_out(x))
        # (B, K, num_patches, ph*pw*(C+1)) -> (B, K, h_p, w_p, ph, pw, C+1)
        h_p = w_p = self.num_patches_per_side
        x = x.view(b, k, h_p, w_p, self.patch_size, self.patch_size, self.out_channels + 1)
        # Unpatchify: combine (h_p, ph) -> H and (w_p, pw) -> W
        # (B, K, h_p, w_p, ph, pw, C+1) -> (B, K, C+1, h_p, ph, w_p, pw) -> (B, K, C+1, H, W)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        x = x.view(b, k, self.out_channels + 1, self.image_size, self.image_size)

        rgb = x[:, :, : self.out_channels]
        alpha = x[:, :, self.out_channels : self.out_channels + 1]
        masks = alpha.softmax(dim=1)
        recon = (rgb * masks).sum(dim=1)
        return recon, masks, rgb


class MaskedPredictionHead(nn.Module):
    """Predict masked-patch features from slot states (simple JEPA-ish surrogate)."""

    def __init__(self, slot_dim: int, feat_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, slots: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Predict per-token features as an attention-weighted mix of slot decodings.

        slots:     (B, K, slot_dim)
        attn_mask: (B, N, K) — soft routing from input tokens to slots (from slot attention).

        Returns features of shape (B, N, feat_dim). Caller compares against ground-truth
        (e.g., the un-masked target encoder's output) under a smooth-L1 loss.
        """
        per_slot = self.net(slots)  # (B, K, feat_dim)
        attn_norm = attn_mask + 1e-8
        attn_norm = attn_norm / attn_norm.sum(dim=-1, keepdim=True)
        return torch.einsum("b n k, b k d -> b n d", attn_norm, per_slot)


class StageAModel(nn.Module):
    """Stage A: Absorb + reconstruction + masked-prediction heads.

    Single-image inputs only. Two augmented views (for slot-consistency loss) are
    handled by the training loop, not this module — it operates on one view at a time.
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 1,
        feat_dim: int = 64,
        slot_dim: int = 128,
        num_slots: int = 8,
        num_roles: int = 8,
        filler_dim: int = 16,
        slot_iters: int = 3,
        sinkhorn_iters: int = 3,
        decoder_hidden: int = 64,
        decoder_type: str = "token",  # "token" (default, fast) or "spatial" (legacy)
        decoder_layers: int = 4,        # only used by token decoder
        patch_size: int = 8,            # only used by token decoder
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.absorb = AbsorbBlock(
            image_size=image_size,
            in_channels=in_channels,
            feat_dim=feat_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            num_roles=num_roles,
            filler_dim=filler_dim,
            slot_iters=slot_iters,
            sinkhorn_iters=sinkhorn_iters,
            use_checkpointing=use_checkpointing,
        )
        if decoder_type == "token":
            self.decoder = TokenDecoder(
                slot_dim=slot_dim,
                out_channels=in_channels,
                image_size=image_size,
                patch_size=patch_size,
                hidden_dim=decoder_hidden,
                num_layers=decoder_layers,
                use_checkpointing=use_checkpointing,
            )
        elif decoder_type == "spatial":
            self.decoder = SpatialBroadcastDecoder(
                slot_dim=slot_dim,
                out_channels=in_channels,
                image_size=image_size,
                hidden_dim=decoder_hidden,
                use_checkpointing=use_checkpointing,
            )
        else:
            raise ValueError(f"decoder_type must be 'token' or 'spatial', got {decoder_type!r}")
        self.masked_pred = MaskedPredictionHead(slot_dim, feat_dim)
        self.config = dict(
            image_size=image_size,
            in_channels=in_channels,
            feat_dim=feat_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            num_roles=num_roles,
            filler_dim=filler_dim,
            decoder_type=decoder_type,
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        absorbed = self.absorb(image)
        recon, masks, per_slot = self.decoder(absorbed["slots"])
        pred_feats = self.masked_pred(absorbed["slots"], absorbed["attn"])
        return {
            **absorbed,
            "recon": recon,
            "masks": masks,
            "per_slot_recon": per_slot,
            "pred_feats": pred_feats,
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
