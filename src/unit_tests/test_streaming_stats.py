"""
Unit tests for streaming statistics (Welford's algorithm).

These tests validate that streaming mean/variance computation matches
full-dataset computation for activation normalization.

Priority: 🟠 High - incorrect normalization affects all downstream analysis.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.streaming_stats import StreamingStats, compute_stats_from_chunks


# =============================================================================
# Basic Properties
# =============================================================================

class TestStreamingStatsProperties:
    """Tests for basic StreamingStats properties."""

    def test_init(self):
        """Should initialize with correct dimensions."""
        stats = StreamingStats(n_features=256)
        
        assert stats.n_features == 256
        assert stats.count == 0
        assert len(stats.mean) == 256

    def test_mean_shape(self):
        """Mean should have correct shape."""
        stats = StreamingStats(n_features=128)
        data = np.random.randn(100, 128).astype(np.float32)
        stats.update_batch(data)
        
        assert stats.mean.shape == (128,)
        assert stats.mean.dtype == np.float32

    def test_std_shape(self):
        """Std should have correct shape."""
        stats = StreamingStats(n_features=64)
        data = np.random.randn(100, 64).astype(np.float32)
        stats.update_batch(data)
        
        assert stats.std.shape == (64,)


# =============================================================================
# Mathematical Correctness
# =============================================================================

class TestStreamingStatsCorrectness:
    """Tests that streaming computation matches numpy."""

    def test_matches_numpy_mean(self):
        """Streaming mean should match numpy computation."""
        np.random.seed(42)
        data = np.random.randn(10000, 256).astype(np.float32)
        
        # Streaming
        stats = StreamingStats(n_features=256)
        for i in range(0, 10000, 1000):
            stats.update_batch(data[i:i+1000])
        
        # Numpy
        np_mean = data.mean(axis=0)
        
        assert np.allclose(stats.mean, np_mean, atol=1e-4), \
            f"Mean diff: {np.abs(stats.mean - np_mean).max()}"

    def test_matches_numpy_variance(self):
        """Streaming variance should match numpy computation."""
        np.random.seed(42)
        data = np.random.randn(10000, 256).astype(np.float32)
        
        # Streaming
        stats = StreamingStats(n_features=256)
        for i in range(0, 10000, 1000):
            stats.update_batch(data[i:i+1000])
        
        # Numpy (population variance)
        np_var = data.var(axis=0, ddof=0)
        
        assert np.allclose(stats.variance, np_var, atol=1e-3), \
            f"Variance diff: {np.abs(stats.variance - np_var).max()}"

    def test_matches_numpy_std(self):
        """Streaming std should match numpy computation."""
        np.random.seed(42)
        data = np.random.randn(5000, 128).astype(np.float32)
        
        # Streaming
        stats = StreamingStats(n_features=128)
        stats.update_batch(data)
        
        # Numpy
        np_std = data.std(axis=0, ddof=0)
        
        # Note: StreamingStats adds 1e-8 for numerical stability
        assert np.allclose(stats.std, np.sqrt(np_std**2 + 1e-8), atol=1e-3)

    def test_chunked_equals_batch(self):
        """Chunked updates should equal single batch update."""
        np.random.seed(42)
        data = np.random.randn(1000, 50).astype(np.float32)
        
        # Single batch
        stats_batch = StreamingStats(n_features=50)
        stats_batch.update_batch(data)
        
        # Chunked
        stats_chunked = StreamingStats(n_features=50)
        for i in range(0, 1000, 100):
            stats_chunked.update_batch(data[i:i+100])
        
        assert np.allclose(stats_batch.mean, stats_chunked.mean, rtol=1e-5)
        assert np.allclose(stats_batch.variance, stats_chunked.variance, rtol=1e-4)


# =============================================================================
# Normalization
# =============================================================================

class TestNormalization:
    """Tests for normalization functionality."""

    def test_normalize_produces_zero_mean(self):
        """Normalized data should have approximately zero mean."""
        np.random.seed(42)
        data = np.random.randn(1000, 256).astype(np.float32) * 5 + 10
        
        stats = StreamingStats(n_features=256)
        stats.update_batch(data)
        
        normalized = stats.normalize(data)
        
        assert np.allclose(normalized.mean(axis=0), np.zeros(256), atol=0.1)

    def test_normalize_produces_unit_variance(self):
        """Normalized data should have approximately unit variance."""
        np.random.seed(42)
        data = np.random.randn(1000, 128).astype(np.float32) * 3 + 7
        
        stats = StreamingStats(n_features=128)
        stats.update_batch(data)
        
        normalized = stats.normalize(data)
        
        assert np.allclose(normalized.std(axis=0), np.ones(128), atol=0.1)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_sample(self):
        """Should handle single sample correctly."""
        stats = StreamingStats(n_features=10)
        sample = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.float32)
        
        stats.update_batch(sample)
        
        assert stats.count == 1
        assert np.allclose(stats.mean, sample[0])
        # Variance should be 0 for single sample
        assert np.allclose(stats.variance, np.zeros(10))

    def test_constant_features(self):
        """Constant features should have zero variance."""
        stats = StreamingStats(n_features=5)
        data = np.ones((100, 5), dtype=np.float32) * 7
        
        stats.update_batch(data)
        
        assert np.allclose(stats.mean, np.full(5, 7.0))
        assert np.allclose(stats.variance, np.zeros(5))

    def test_large_values(self):
        """Should handle large values without overflow."""
        stats = StreamingStats(n_features=10)
        data = np.ones((100, 10), dtype=np.float32) * 1e6
        
        stats.update_batch(data)
        
        assert np.allclose(stats.mean, np.full(10, 1e6))
        assert not np.isnan(stats.mean).any()


# =============================================================================
# compute_stats_from_chunks
# =============================================================================

class TestComputeStatsFromChunks:
    """Tests for the helper function."""

    def test_compute_from_iterator(self):
        """Should work with iterator of chunks."""
        np.random.seed(42)
        data = np.random.randn(1000, 64).astype(np.float32)
        
        def chunk_iterator():
            for i in range(0, 1000, 100):
                yield data[i:i+100]
        
        stats = compute_stats_from_chunks(chunk_iterator(), n_features=64)
        
        assert np.allclose(stats.mean, data.mean(axis=0), atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
