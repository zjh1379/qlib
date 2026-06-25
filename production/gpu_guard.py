"""GPU VRAM pre-check + CPU fallback for the neural trainers.

The ALSTM/TRA models already run on cuda:0 (pytorch_alstm.py:73 maps
GPU>=0 + cuda available -> cuda:0). This guard only prevents a CUDA OOM
when something else (e.g. LM Studio) has eaten the VRAM: if free VRAM is
below the threshold, force CPU by returning GPU id -1 (which pytorch_alstm
interprets as cpu)."""
from __future__ import annotations

import logging

_log = logging.getLogger("gpu_guard")


def _default_probe():
    """Return (free_bytes, total_bytes) for the current CUDA device."""
    import torch
    return torch.cuda.mem_get_info()  # raises if no CUDA


def effective_gpu(requested_gpu: int, *, min_free_gb: float = 4.0, probe=_default_probe) -> int:
    """Return the GPU id to actually use. -1 means CPU.

    requested_gpu < 0 -> already CPU, keep it.
    Otherwise probe free VRAM; if < min_free_gb (or probe fails) -> -1 (CPU)."""
    if requested_gpu < 0:
        return -1
    try:
        free_bytes, _total = probe()
        free_gb = free_bytes / 2**30
        if free_gb < min_free_gb:
            _log.warning("gpu_low_vram free=%.1fGB < %.1fGB -> falling back to CPU", free_gb, min_free_gb)
            return -1
        _log.info("gpu_ok free=%.1fGB -> using cuda:%d", free_gb, requested_gpu)
        return requested_gpu
    except Exception as exc:
        _log.warning("gpu_probe_failed (%s) -> falling back to CPU", exc)
        return -1
