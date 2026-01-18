"""
Unit tests for SystemCapabilities auto-detection.

Tests system resource detection, optimal parameter computation,
and fallback behavior for memory-efficient data loading.

Priority: 🟠 High - incorrect sizing can cause OOM or inefficiency.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.system import SystemCapabilities, get_system_capabilities


# =============================================================================
# Basic Detection Tests
# =============================================================================

class TestSystemCapabilitiesDetection:
    """Tests for basic system detection functionality."""

    def test_detect_returns_capabilities(self):
        """Detection should return SystemCapabilities instance."""
        caps = SystemCapabilities.detect()
        
        assert isinstance(caps, SystemCapabilities)
        assert isinstance(caps.cpu_count, int)
        assert isinstance(caps.total_ram_gb, float)
        assert isinstance(caps.available_ram_gb, float)

    def test_cpu_count_positive(self):
        """CPU count should be positive."""
        caps = SystemCapabilities.detect()
        assert caps.cpu_count >= 1

    def test_ram_values_positive(self):
        """RAM values should be positive."""
        caps = SystemCapabilities.detect()
        assert caps.total_ram_gb > 0
        assert caps.available_ram_gb > 0

    def test_available_less_than_total(self):
        """Available RAM should be <= total RAM."""
        caps = SystemCapabilities.detect()
        assert caps.available_ram_gb <= caps.total_ram_gb

    def test_gpu_detection_consistent(self):
        """GPU detection should return consistent types."""
        caps = SystemCapabilities.detect()
        
        if caps.gpu_name is not None:
            assert isinstance(caps.gpu_name, str)
            assert caps.gpu_mem_gb > 0
        else:
            # No GPU case
            assert caps.gpu_mem_gb == 0 or caps.gpu_mem_gb > 0  # MPS case

    def test_memory_fraction_default(self):
        """Default memory fraction should be 0.5."""
        caps = SystemCapabilities.detect()
        assert caps.memory_fraction == 0.5


# =============================================================================
# Optimal Chunk Size Tests
# =============================================================================

class TestOptimalChunkSize:
    """Tests for optimal_chunk_size computation."""

    def test_returns_positive_integer(self):
        """Chunk size should be a positive integer."""
        caps = SystemCapabilities.detect()
        chunk_size = caps.optimal_chunk_size(sample_dim=256)
        
        assert isinstance(chunk_size, int)
        assert chunk_size > 0

    def test_respects_min_chunk(self):
        """Chunk size should not go below minimum."""
        caps = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=0.001,  # Very low RAM
            available_ram_gb=0.0001,
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        chunk_size = caps.optimal_chunk_size(sample_dim=256, min_chunk=1000)
        assert chunk_size >= 1000

    def test_respects_max_chunk(self):
        """Chunk size should not exceed maximum."""
        caps = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=1000.0,  # Very high RAM
            available_ram_gb=500.0,
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        chunk_size = caps.optimal_chunk_size(sample_dim=256, max_chunk=50000)
        assert chunk_size <= 50000

    def test_larger_dim_smaller_chunk(self):
        """Larger sample dimensions should result in smaller chunks."""
        caps = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        # Use high max_chunk to avoid ceiling
        chunk_small = caps.optimal_chunk_size(sample_dim=128, max_chunk=1000000)
        chunk_large = caps.optimal_chunk_size(sample_dim=4096, max_chunk=1000000)
        
        assert chunk_large < chunk_small

    def test_more_ram_larger_chunk(self):
        """More available RAM should result in larger chunks."""
        caps_low = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=4.0,
            available_ram_gb=1.0,  # Low RAM
            gpu_name=None,
            gpu_mem_gb=0,
        )
        caps_high = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=64.0,
            available_ram_gb=32.0,  # High RAM
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        # Use high max_chunk to avoid ceiling
        chunk_low = caps_low.optimal_chunk_size(sample_dim=256, max_chunk=1000000)
        chunk_high = caps_high.optimal_chunk_size(sample_dim=256, max_chunk=1000000)
        
        assert chunk_high > chunk_low


# =============================================================================
# Optimal Workers Tests
# =============================================================================

class TestOptimalWorkers:
    """Tests for optimal_workers computation."""

    def test_returns_non_negative(self):
        """Worker count should be non-negative."""
        caps = SystemCapabilities.detect()
        workers = caps.optimal_workers()
        assert workers >= 0

    def test_respects_cpu_count(self):
        """Workers should not exceed cpu_count - 1."""
        caps = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        workers = caps.optimal_workers()
        assert workers <= 3  # cpu_count - 1

    def test_low_ram_reduces_workers(self):
        """Low available RAM should reduce worker count."""
        caps = SystemCapabilities(
            cpu_count=16,
            total_ram_gb=8.0,
            available_ram_gb=4.0,  # Below threshold
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        workers = caps.optimal_workers(low_ram_threshold_gb=8.0)
        assert workers <= 2

    def test_high_ram_more_workers(self):
        """High RAM should allow more workers."""
        caps = SystemCapabilities(
            cpu_count=16,
            total_ram_gb=64.0,
            available_ram_gb=32.0,
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        workers = caps.optimal_workers()
        assert workers > 2  # Should allow more than low-RAM case


# =============================================================================
# Optimal Batch Size Tests
# =============================================================================

class TestOptimalBatchSize:
    """Tests for optimal_batch_size computation."""

    def test_returns_positive(self):
        """Batch size should be positive."""
        caps = SystemCapabilities.detect()
        batch = caps.optimal_batch_size()
        assert batch > 0

    def test_large_gpu_larger_batch(self):
        """Large GPU memory should scale up batch size."""
        caps = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            gpu_name="A100",
            gpu_mem_gb=40.0,  # Large GPU
        )
        
        batch = caps.optimal_batch_size(base_batch_size=4096)
        assert batch > 4096

    def test_small_gpu_smaller_batch(self):
        """Small GPU memory should scale down batch size."""
        caps = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            gpu_name="GTX 1050",
            gpu_mem_gb=2.0,  # Small GPU
        )
        
        batch = caps.optimal_batch_size(base_batch_size=4096)
        assert batch < 4096


# =============================================================================
# Full Load Decision Tests
# =============================================================================

class TestShouldUseFullLoad:
    """Tests for should_use_full_load decision."""

    def test_small_dataset_uses_full_load(self):
        """Small datasets should use full-load mode."""
        caps = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        # Very small dataset
        should_full = caps.should_use_full_load(n_samples=100, sample_dim=256)
        assert should_full is True

    def test_large_dataset_uses_chunked(self):
        """Large datasets should use chunked mode."""
        caps = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=4.0,
            available_ram_gb=2.0,
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        # Very large dataset
        should_full = caps.should_use_full_load(n_samples=10000000, sample_dim=4096)
        assert should_full is False


# =============================================================================
# Summary and Convenience Function Tests
# =============================================================================

class TestSummaryAndConvenience:
    """Tests for summary output and convenience functions."""

    def test_summary_contains_key_info(self):
        """Summary should contain key system info."""
        caps = SystemCapabilities.detect()
        summary = caps.summary()
        
        assert "CPU cores" in summary
        assert "RAM" in summary
        assert "Memory budget" in summary

    def test_get_system_capabilities_works(self):
        """Convenience function should work."""
        caps = get_system_capabilities()
        assert isinstance(caps, SystemCapabilities)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_available_ram(self):
        """Should handle near-zero available RAM gracefully."""
        caps = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=16.0,
            available_ram_gb=0.001,
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        # Should still return valid chunk size (min_chunk)
        chunk_size = caps.optimal_chunk_size(sample_dim=256)
        assert chunk_size >= 1000

    def test_single_cpu(self):
        """Should handle single CPU system."""
        caps = SystemCapabilities(
            cpu_count=1,
            total_ram_gb=4.0,
            available_ram_gb=2.0,
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        workers = caps.optimal_workers()
        assert workers >= 0

    def test_very_large_dimension(self):
        """Should handle very large sample dimensions."""
        caps = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=8.0,
            available_ram_gb=4.0,
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        # Very high dimension (like flattened image)
        chunk_size = caps.optimal_chunk_size(sample_dim=1000000)
        assert chunk_size >= 1000  # Should be at min_chunk

    def test_dtype_bytes_affects_chunk(self):
        """Different dtype sizes should affect chunk size."""
        caps = SystemCapabilities(
            cpu_count=4,
            total_ram_gb=0.5,  # Very low RAM to avoid max_chunk ceiling
            available_ram_gb=0.2,
            gpu_name=None,
            gpu_mem_gb=0,
        )
        
        chunk_float32 = caps.optimal_chunk_size(sample_dim=256, dtype_bytes=4)
        chunk_float64 = caps.optimal_chunk_size(sample_dim=256, dtype_bytes=8)
        
        # float64 takes twice the memory, so chunk should be smaller
        # Both should be well below max_chunk with low RAM
        assert chunk_float64 < chunk_float32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
