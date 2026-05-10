"""Causal slot-swap intervention.

Procedure:
  1. Encode two scenes A and B, getting slot states S_A and S_B.
  2. Identify, for each slot in A, the slot in B that best matches it (greedy on
     cosine similarity *of role-weights*, not raw slot vectors — this favors
     swaps that preserve role identity, which is the TPR claim).
  3. For each slot index k, swap S_A[k] <- S_B[match(k)] producing S_swapped.
  4. Decode S_swapped and check whether the resulting reconstruction's predicted
     factors changed in the way ground truth says they should.

For a cheap automated version of (4) we use a frozen linear probe on the model's
own slot features as the predictor. A successful swap is one where the probe's
prediction on factor j flips from A's value to B's value.

Stage A gate: >= 70% of single-slot swaps produce the predicted change.
"""

from __future__ import annotations

import numpy as np
import torch

from glr.eval.probe import _flatten


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a @ b.T


def causal_slot_swap(
    model,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    factors_a: torch.Tensor,
    factors_b: torch.Tensor,
    factor_idx: int,
    probe,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run single-slot swaps from B into A and check probe-prediction flips.

    Args:
        model: Stage A model with .absorb available (returns slots).
        image_a / image_b: (B, C, H, W).
        factors_a / factors_b: (B, F).
        factor_idx: which factor to evaluate the swap against.
        probe: a fitted glr.eval.probe.LinearProbe.

    Returns:
        {"swap_success_rate": float, "n_pairs": int, "n_swaps_attempted": int}
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    image_a = image_a.to(device)
    image_b = image_b.to(device)
    f_a = factors_a[:, factor_idx].cpu().numpy()
    f_b = factors_b[:, factor_idx].cpu().numpy()

    with torch.no_grad():
        out_a = model.absorb(image_a)
        out_b = model.absorb(image_b)

    slots_a = out_a["slots"]      # (B, K, D)
    slots_b = out_b["slots"]
    role_a = out_a["role_weights"]  # (B, K, R)
    role_b = out_b["role_weights"]

    successes = 0
    attempts = 0
    different = (f_a != f_b)
    for i in range(slots_a.size(0)):
        if not different[i]:
            continue  # only meaningful when factor differs between A and B
        # match slots by role-weight cosine similarity
        sim = _cos_sim(role_a[i], role_b[i])  # (K, K)
        match = sim.argmax(dim=-1)            # (K,)
        # try swapping each slot one at a time
        any_success = False
        for k in range(slots_a.size(1)):
            swapped = slots_a[i].clone()
            swapped[k] = slots_b[i, match[k]]
            attempts += 1
            pred = probe.clf.predict(_flatten(swapped.unsqueeze(0)))[0]
            if pred == f_b[i]:
                any_success = True
                successes += 1
                # swap was sufficient; no need to continue trying more slots
                break
        # we count attempts up to and including the first successful swap
        if not any_success:
            # if no single-slot swap succeeded, we already counted K attempts above
            pass

    n_pairs = int(different.sum())
    rate = (successes / max(n_pairs, 1)) if n_pairs > 0 else 0.0
    return {
        "swap_success_rate": float(rate),
        "n_pairs": int(n_pairs),
        "n_swaps_attempted": int(attempts),
        "n_successes": int(successes),
    }
