"""Memory tracking utilities for diagnosing OOM and per-step VRAM accounting.

Usage:
    from glr.utils.memory import mem_track, mem_snapshot, reset_peak

    reset_peak()  # at the start of a step
    with mem_track("forward"):
        out = model(x)
    print(mem_snapshot())

Set the env var `GLR_DEBUG_MEMORY=1` to enable from the trainer.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch


def _gb(n_bytes: int) -> float:
    return n_bytes / (1024**3)


def mem_snapshot(prefix: str = "") -> str:
    if not torch.cuda.is_available():
        return f"{prefix}cpu"
    alloc = _gb(torch.cuda.memory_allocated())
    reserved = _gb(torch.cuda.memory_reserved())
    peak = _gb(torch.cuda.max_memory_allocated())
    free, total = torch.cuda.mem_get_info()
    return (
        f"{prefix}alloc={alloc:.2f}GB peak={peak:.2f}GB "
        f"reserved={reserved:.2f}GB device_free={_gb(free):.2f}GB / {_gb(total):.0f}GB"
    )


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def mem_track(label: str, enabled: bool = True):
    """Log allocated and peak memory delta around a block.

    Logs:
      [mem] <label>: <before>GB -> <after>GB (delta) | peak in-block: <peak>GB
    """
    if not enabled or not torch.cuda.is_available():
        yield
        return

    torch.cuda.synchronize()
    before = _gb(torch.cuda.memory_allocated())
    peak_before = _gb(torch.cuda.max_memory_allocated())
    try:
        yield
    finally:
        torch.cuda.synchronize()
        after = _gb(torch.cuda.memory_allocated())
        peak_after = _gb(torch.cuda.max_memory_allocated())
        delta = after - before
        peak_in_block = peak_after - peak_before
        # device-wide free for context
        free, total = torch.cuda.mem_get_info()
        sign = "+" if delta >= 0 else ""
        print(
            f"[mem] {label:<40s} "
            f"{before:6.2f} -> {after:6.2f} GB ({sign}{delta:5.2f})  "
            f"peak-in-block {peak_in_block:5.2f} GB  "
            f"free={_gb(free):5.2f}GB"
        )


def env_debug_enabled() -> bool:
    """Read GLR_DEBUG_MEMORY env var. '1', 'true', 'yes' -> True."""
    val = os.environ.get("GLR_DEBUG_MEMORY", "").strip().lower()
    return val in {"1", "true", "yes", "on"}
