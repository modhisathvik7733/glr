"""Stage A four-gate evaluation script.

Loads a final.pt checkpoint and runs the four Stage A gates:
  1. Linear probe accuracy on held-out test set      (mean >= 0.80)
  2. Mutual Information Gap across slots             (> 0.40)
  3. Compositional held-out probe gap                (<= 0.05 = within 5pt)
  4. Causal slot-swap intervention success rate      (>= 0.70)

The "shape" factor is used as the primary target for the compositional and
swap gates because it's the most discriminating dSprites factor.

Usage (single seed):
    python scripts/eval_stage_a.py --checkpoint outputs/stage_a_seed0/final.pt

Usage (all seeds):
    for s in 0 1 2; do
      python scripts/eval_stage_a.py --checkpoint outputs/stage_a_seed$s/final.pt
    done

Outputs:
  - Stdout: full per-gate report + verdict
  - <checkpoint_dir>/gate_results.json: machine-readable summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from glr.data import DSpritesDataset
from glr.data.dsprites import DSPRITES_FACTORS
from glr.eval.causal_swap import causal_slot_swap
from glr.eval.compositional import compositional_probe
from glr.eval.mig import mutual_information_gap
from glr.eval.probe import LinearProbe, _flatten, probe_all_factors
from glr.models import StageAModel


GATE_TARGETS = {
    "probe_mean": 0.80,
    "mig": 0.40,
    "compositional_gap": 0.05,
    "causal_swap": 0.70,
}


def collect_slots(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iterate the loader, collect slot tensors and ground-truth latents."""
    feats, latents = [], []
    n = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            feats.append(out["slots"].detach().cpu())
            latents.append(batch["latents"])
            n += batch["image"].size(0)
            if n >= max_samples:
                break
    feats = torch.cat(feats)[:max_samples]
    latents = torch.cat(latents)[:max_samples]
    return feats, latents


def filter_dataset_to_held_out(
    ds: DSpritesDataset, held_out_pairs: list, held_out_keys: tuple[str, str]
) -> None:
    """Mutate the dataset in place to keep only samples whose (k_a, k_b)
    falls in `held_out_pairs`. The opposite of what DSpritesDataset's
    held_out_pairs argument does at construction time.
    """
    ki = DSPRITES_FACTORS.index(held_out_keys[0])
    kj = DSPRITES_FACTORS.index(held_out_keys[1])
    held_set = set(map(tuple, [list(map(int, p)) for p in held_out_pairs]))
    mask = np.array(
        [
            (int(ds.latents[n, ki]), int(ds.latents[n, kj])) in held_set
            for n in range(len(ds))
        ]
    )
    ds.imgs = ds.imgs[mask]
    ds.latents = ds.latents[mask]


def build_model_from_checkpoint(ckpt: dict) -> StageAModel:
    """Recreate the model architecture from the trainer config saved in ckpt."""
    cfg = ckpt["config"]  # StageATrainerConfig.__dict__
    model = StageAModel(
        image_size=cfg["image_size"],
        in_channels=cfg["in_channels"],
        feat_dim=cfg["feat_dim"],
        slot_dim=cfg["slot_dim"],
        num_slots=cfg["num_slots"],
        num_roles=cfg["num_roles"],
        filler_dim=cfg["filler_dim"],
        slot_iters=cfg["slot_iters"],
        sinkhorn_iters=cfg["sinkhorn_iters"],
        decoder_hidden=cfg["decoder_hidden"],
        decoder_type=cfg.get("decoder_type", "spatial"),
        decoder_layers=cfg.get("decoder_layers", 4),
        patch_size=cfg.get("patch_size", 8),
    )
    state_dict = ckpt["model"]
    # Strip torch.compile prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[load] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to final.pt")
    parser.add_argument("--data-root", default="data/raw", help="dSprites file directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-eval", type=int, default=4096, help="Per-split eval samples")
    parser.add_argument("--swap-pairs", type=int, default=300, help="Causal-swap A,B pairs")
    parser.add_argument(
        "--target-factor", default="shape", choices=DSPRITES_FACTORS,
        help="Factor used for compositional and swap gates",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    seed = cfg["seed"]
    held_out_pairs = ckpt.get("held_out_pairs") or []
    held_out_keys = ("shape", "scale")  # default; could also be in cfg

    print(f"[load] seed={seed}, held_out_pairs={held_out_pairs}")
    model = build_model_from_checkpoint(ckpt).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params/1e6:.2f}M params, decoder_type={cfg.get('decoder_type', 'spatial')}")

    # Build two views of dSprites: in-distribution and held-out
    in_dist_ds = DSpritesDataset(
        root=args.data_root,
        two_view=False,
        held_out_pairs=held_out_pairs if held_out_pairs else None,
        held_out_keys=held_out_keys,
    )
    held_out_ds = DSpritesDataset(root=args.data_root, two_view=False)
    if held_out_pairs:
        filter_dataset_to_held_out(held_out_ds, held_out_pairs, held_out_keys)
    print(f"[data] in-dist: {len(in_dist_ds)}, held-out: {len(held_out_ds)}")

    in_loader = DataLoader(in_dist_ds, batch_size=128, shuffle=True, num_workers=2)
    held_loader = DataLoader(held_out_ds, batch_size=128, shuffle=True, num_workers=2)

    print("\n[1/4] Collecting in-distribution slots...")
    in_feats, in_latents = collect_slots(model, in_loader, device, args.n_eval)
    print(f"        got {in_feats.shape[0]} samples, slots {tuple(in_feats.shape)}")

    print("[2/4] Collecting held-out slots...")
    ho_max = min(args.n_eval, len(held_out_ds))
    ho_feats, ho_latents = collect_slots(model, held_loader, device, ho_max)
    print(f"        got {ho_feats.shape[0]} samples")

    # ----- Gate 1: Linear probe accuracy -----
    print("\n[Gate 1/4] Linear probe accuracy")
    cut = in_feats.size(0) // 2
    train_feat, test_feat = in_feats[:cut], in_feats[cut:]
    train_lat, test_lat = in_latents[:cut], in_latents[cut:]
    probe_acc = probe_all_factors(
        train_feat, train_lat, test_feat, test_lat, list(DSPRITES_FACTORS)
    )
    probe_mean = float(np.mean(list(probe_acc.values()))) if probe_acc else 0.0
    print(f"  per-factor accuracy:")
    for name, acc in probe_acc.items():
        print(f"    {name:14s} {acc:.3f}")
    print(f"  mean: {probe_mean:.3f}  (target >= {GATE_TARGETS['probe_mean']:.2f})")

    # ----- Gate 2: MIG -----
    print("\n[Gate 2/4] Mutual Information Gap")
    mig = mutual_information_gap(in_feats, in_latents, factor_names=list(DSPRITES_FACTORS))
    print(f"  per-factor MIG:")
    for name, g in mig["per_factor"].items():
        print(f"    {name:14s} {g:.3f}")
    print(f"  mean MIG: {mig['mig']:.3f}  (target > {GATE_TARGETS['mig']:.2f})")

    # ----- Gate 3: Compositional held-out probe -----
    print("\n[Gate 3/4] Compositional held-out probe")
    factor_idx = DSPRITES_FACTORS.index(args.target_factor)
    print(f"  target factor: {args.target_factor} (index {factor_idx})")
    if ho_feats.size(0) > 0:
        comp = compositional_probe(
            train_feat, train_lat, test_feat, test_lat, ho_feats, ho_latents, factor_idx
        )
        print(f"  in-distribution accuracy: {comp['in_dist_acc']:.3f}")
        print(f"  held-out accuracy:        {comp['held_out_acc']:.3f}")
        print(f"  gap (in - held):          {comp['gap']:.3f}  (target <= {GATE_TARGETS['compositional_gap']:.2f})")
    else:
        comp = {"in_dist_acc": float("nan"), "held_out_acc": float("nan"), "gap": float("nan")}
        print("  no held-out samples available — skipped")

    # ----- Gate 4: Causal slot-swap -----
    print("\n[Gate 4/4] Causal slot-swap")
    fitted_probe = LinearProbe()
    fitted_probe.fit(_flatten(train_feat), train_lat[:, factor_idx].numpy())

    n_pairs = min(args.swap_pairs, in_feats.size(0) // 2)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(in_dist_ds))
    a_idx = perm[:n_pairs]
    b_idx = perm[n_pairs : 2 * n_pairs]
    a_imgs = torch.stack([in_dist_ds[int(i)]["image"] for i in a_idx])
    b_imgs = torch.stack([in_dist_ds[int(i)]["image"] for i in b_idx])
    a_lats = torch.stack([in_dist_ds[int(i)]["latents"] for i in a_idx])
    b_lats = torch.stack([in_dist_ds[int(i)]["latents"] for i in b_idx])

    swap = causal_slot_swap(
        model, a_imgs, b_imgs, a_lats, b_lats,
        factor_idx=factor_idx, probe=fitted_probe, device=device,
    )
    print(f"  pairs with different {args.target_factor}: {swap['n_pairs']}/{n_pairs}")
    print(f"  successful swaps: {swap['n_successes']}")
    print(f"  swap success rate: {swap['swap_success_rate']:.3f}  (target >= {GATE_TARGETS['causal_swap']:.2f})")

    # ----- Verdict -----
    gates = [
        ("Linear probe (mean)", probe_mean, GATE_TARGETS["probe_mean"], "ge"),
        ("MIG", mig["mig"], GATE_TARGETS["mig"], "gt"),
        ("Compositional gap", comp["gap"], GATE_TARGETS["compositional_gap"], "le"),
        ("Causal slot-swap", swap["swap_success_rate"], GATE_TARGETS["causal_swap"], "ge"),
    ]
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  STAGE A GATE EVALUATION  —  seed {seed}  —  ckpt: {ckpt_path.name}")
    print(bar)
    passed = 0
    pass_flags = {}
    for name, val, target, op in gates:
        if val != val:  # NaN
            ok = False
            sym = "?"
        elif op == "ge":
            ok = val >= target
        elif op == "gt":
            ok = val > target
        elif op == "le":
            ok = val <= target
        else:
            ok = False
        if ok:
            passed += 1
        sym = "PASS" if ok else "FAIL" if val == val else "SKIP"
        cmp = {"ge": ">=", "gt": ">", "le": "<="}[op]
        print(f"  [{sym}] {name:24s}  {val:.3f}  {cmp} {target:.2f}")
        pass_flags[name] = ok
    print(bar)
    if passed == 4:
        verdict = f"STAGE_A_GATE: PASSED ({passed}/4) — proceed to Stage B"
    elif passed >= 2:
        verdict = f"STAGE_A_GATE: PARTIAL ({passed}/4) — diagnose and re-tune"
    else:
        verdict = f"STAGE_A_GATE: FAILED ({passed}/4) — architecture in question"
    print(f"  {verdict}")
    print(bar)

    # Save JSON next to the checkpoint
    out_path = ckpt_path.parent / "gate_results.json"
    results = {
        "seed": int(seed),
        "checkpoint": str(ckpt_path),
        "n_params": int(n_params),
        "passed_count": int(passed),
        "verdict": verdict,
        "probe_per_factor": {k: float(v) for k, v in probe_acc.items()},
        "probe_mean": float(probe_mean),
        "mig": float(mig["mig"]),
        "mig_per_factor": {k: float(v) for k, v in mig["per_factor"].items()},
        "compositional": {k: float(v) for k, v in comp.items()},
        "causal_swap": {k: float(v) if isinstance(v, (int, float)) else v for k, v in swap.items()},
        "gate_targets": GATE_TARGETS,
        "pass_flags": pass_flags,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
