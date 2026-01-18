"""
System capability detection for automatic resource optimization.

Auto-detects RAM, VRAM, CPU cores to compute optimal chunk sizes,
worker counts, and batch sizes for memory-efficient processing.

Usage:
    from src.utils.system import SystemCapabilities
    
    caps = SystemCapabilities.detect()
    print(f"Available RAM: {caps.available_ram_gb:.1f} GB")
    print(f"Optimal chunk size: {caps.optimal_chunk_size(sample_dim=256)}")
"""

import os
from dataclasses import dataclass
from typing import Optional

import psutil
import torch


@dataclass
class SystemCapabilities:
    """System resource information for optimization decisions."""
    
    cpu_count: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_name: Optional[str]
    gpu_mem_gb: float
    
    # Budget parameters
    memory_fraction: float = 0.5  # Use 50% of available RAM for data chunks
                                   # Remaining 50% for model, optimizer, intermediates
    
    @classmethod
    def detect(cls) -> 'SystemCapabilities':
        """Auto-detect system resources."""
        # CPU
        cpu_count = os.cpu_count() or 4
        
        # RAM
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        available_ram_gb = mem.available / (1024**3)
        
        # GPU
        gpu_name = None
        gpu_mem_gb = 0.0
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif torch.backends.mps.is_available():
            gpu_name = "Apple MPS"
            # MPS shares system RAM, estimate 50%
            gpu_mem_gb = available_ram_gb * 0.5
        
        return cls(
            cpu_count=cpu_count,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            gpu_name=gpu_name,
            gpu_mem_gb=gpu_mem_gb,
        )
    
    def optimal_chunk_size(
        self,
        sample_dim: int,
        dtype_bytes: int = 4,  # float32 = 4 bytes
        min_chunk: int = 1000,
        max_chunk: int = 100000,
    ) -> int:
        """
        Compute optimal chunk size based on available memory.
        
        Args:
            sample_dim: Feature dimension per sample
            dtype_bytes: Bytes per element (4 for float32)
            min_chunk: Minimum chunk size
            max_chunk: Maximum chunk size
            
        Returns:
            Optimal number of samples per chunk
        """
        target_bytes = self.available_ram_gb * (1024**3) * self.memory_fraction
        chunk_size = int(target_bytes / (sample_dim * dtype_bytes))
        return max(min_chunk, min(chunk_size, max_chunk))
    
    def optimal_workers(self, low_ram_threshold_gb: float = 8.0) -> int:
        """
        Compute optimal DataLoader worker count.
        
        Returns fewer workers when RAM is low to avoid OOM from
        worker memory duplication.
        """
        base_workers = min(self.cpu_count - 1, 8)
        
        if self.available_ram_gb < low_ram_threshold_gb:
            return min(base_workers, 2)
        
        return max(0, base_workers)
    
    def optimal_batch_size(self, base_batch_size: int = 4096) -> int:
        """
        Scale batch size based on GPU memory.
        
        Returns:
            Scaled batch size
        """
        if self.gpu_mem_gb >= 16:
            # A100, V100: can use larger batches
            return int(base_batch_size * 2)
        elif self.gpu_mem_gb >= 8:
            # T4, RTX 3070: standard
            return base_batch_size
        else:
            # Low GPU: smaller batches
            return int(base_batch_size * 0.5)
    
    def should_use_full_load(
        self,
        n_samples: int,
        sample_dim: int,
        dtype_bytes: int = 4,
    ) -> bool:
        """
        Determine if dataset is small enough to load fully.
        
        Returns True if entire dataset fits in one optimal chunk.
        """
        data_bytes = n_samples * sample_dim * dtype_bytes
        chunk_bytes = self.optimal_chunk_size(sample_dim, dtype_bytes) * sample_dim * dtype_bytes
        return data_bytes <= chunk_bytes
    
    def summary(self) -> str:
        """Human-readable summary of capabilities."""
        lines = [
            "=== System Capabilities ===",
            f"CPU cores: {self.cpu_count}",
            f"Total RAM: {self.total_ram_gb:.1f} GB",
            f"Available RAM: {self.available_ram_gb:.1f} GB",
            f"Memory budget: {self.memory_fraction*100:.0f}%",
        ]
        if self.gpu_name:
            lines.append(f"GPU: {self.gpu_name} ({self.gpu_mem_gb:.1f} GB)")
        else:
            lines.append("GPU: None")
        return "\n".join(lines)


def get_system_capabilities() -> SystemCapabilities:
    """Convenience function to detect and return system capabilities."""
    return SystemCapabilities.detect()
