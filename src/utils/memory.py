"""
Memory management utilities for Colab.

Helps prevent OOM errors during long-running extraction and training.
"""

import gc
import torch
from typing import Optional


def clear_memory(verbose: bool = False) -> None:
    """
    Aggressively clear GPU and CPU memory.

    Call between major operations to prevent OOM.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if verbose:
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"GPU memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")


def get_memory_info() -> dict:
    """Get current memory usage info."""
    info = {'cpu_available': True}

    if torch.cuda.is_available():
        info.update({
            'gpu_allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'gpu_cached_gb': torch.cuda.memory_reserved() / 1e9,
            'gpu_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        })
        info['gpu_free_gb'] = info['gpu_total_gb'] - info['gpu_allocated_gb']

    return info


def get_optimal_batch_size(
    sample_size_bytes: int,
    target_memory_gb: float = 2.0,
    min_batch: int = 32,
    max_batch: int = 4096
) -> int:
    """
    Estimate optimal batch size based on available GPU memory.

    Args:
        sample_size_bytes: Size of one sample in bytes
        target_memory_gb: Target GPU memory usage
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Recommended batch size
    """
    target_bytes = target_memory_gb * 1e9

    # Account for model, gradients, optimizer states (~3x sample memory)
    effective_bytes = target_bytes / 4

    batch_size = int(effective_bytes / sample_size_bytes)
    batch_size = max(min_batch, min(max_batch, batch_size))

    # Round to power of 2 for efficiency
    return 2 ** int(batch_size).bit_length() // 2


def move_to_device(
    data,
    device: Optional[torch.device] = None,
    non_blocking: bool = True
):
    """
    Move tensor or nested structure to device.

    Args:
        data: Tensor, list, tuple, or dict of tensors
        device: Target device (default: cuda if available)
        non_blocking: Use async transfer

    Returns:
        Data on target device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(v, device, non_blocking) for v in data)
    else:
        return data
