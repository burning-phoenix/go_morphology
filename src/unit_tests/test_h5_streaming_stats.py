"""
Unit tests for H5 streaming statistics (data/streaming_stats.py).

Validates Welford's and Chan's algorithms for streaming mean/std
computation from HDF5 files without loading all data into memory.

Priority: 🔴 Critical - incorrect stats corrupt normalization.
"""

import pytest
import numpy as np
import h5py
import tempfile
import sys
import importlib.util
from pathlib import Path

# Direct module loading to bypass data/__init__.py import cascade
_src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_src_dir))

# Load utils.system first (dependency)
_system_spec = importlib.util.spec_from_file_location(
    "utils.system", _src_dir / "utils" / "system.py"
)
_system_module = importlib.util.module_from_spec(_system_spec)
sys.modules['utils.system'] = _system_module
_system_spec.loader.exec_module(_system_module)

# Load data.streaming_stats directly (bypasses data/__init__.py)
_stats_spec = importlib.util.spec_from_file_location(
    "data.streaming_stats", _src_dir / "data" / "streaming_stats.py"
)
_stats_module = importlib.util.module_from_spec(_stats_spec)
sys.modules['data.streaming_stats'] = _stats_module

try:
    _stats_spec.loader.exec_module(_stats_module)
    compute_h5_stats_streaming = _stats_module.compute_h5_stats_streaming
    compute_h5_stats_batch = _stats_module.compute_h5_stats_batch
    save_stats_to_h5 = _stats_module.save_stats_to_h5
    load_stats_from_h5 = _stats_module.load_stats_from_h5
    _combine_stats = _stats_module._combine_stats
    HAS_STREAMING_STATS = True
except Exception as e:
    print(f"Import error: {e}")
    HAS_STREAMING_STATS = False

# Skip all tests if not available
pytestmark = pytest.mark.skipif(
    not HAS_STREAMING_STATS,
    reason="data.streaming_stats not available"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_h5():
    """Fixture that creates a temp HDF5 file and cleans up after test."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_h5_data(temp_h5):
    """Create HDF5 with known statistical properties."""
    np.random.seed(42)
    n_samples = 5000
    n_features = 128
    
    # Generate data with known mean and std
    true_mean = np.random.randn(n_features).astype(np.float32) * 5
    true_std = np.abs(np.random.randn(n_features).astype(np.float32)) + 0.5
    
    # Generate samples from N(true_mean, true_std)
    data = true_mean + true_std * np.random.randn(n_samples, n_features)
    data = data.astype(np.float32)
    
    with h5py.File(temp_h5, 'w') as f:
        f.create_dataset('test', data=data, chunks=(1000, n_features))
    
    return {
        'path': temp_h5,
        'data': data,
        'true_mean': true_mean,
        'true_std': true_std,
        'n_samples': n_samples,
        'n_features': n_features,
    }


# =============================================================================
# Welford's Algorithm Tests
# =============================================================================

class TestWelfordStreaming:
    """Tests for compute_h5_stats_streaming (Welford's algorithm)."""

    def test_matches_numpy_mean(self, sample_h5_data):
        """Streaming mean should match numpy computation."""
        stats = compute_h5_stats_streaming(
            sample_h5_data['path'], 
            'test',
            chunk_size=500  # Force chunking
        )
        
        np_mean = sample_h5_data['data'].mean(axis=0)
        
        assert np.allclose(stats['mean'], np_mean, atol=1e-4), \
            f"Mean diff: {np.abs(stats['mean'] - np_mean).max()}"

    def test_matches_numpy_std(self, sample_h5_data):
        """Streaming std should match numpy (with Bessel's correction)."""
        stats = compute_h5_stats_streaming(
            sample_h5_data['path'],
            'test',
            chunk_size=500
        )
        
        # Bessel's correction: ddof=1
        np_std = sample_h5_data['data'].std(axis=0, ddof=1)
        
        assert np.allclose(stats['std'], np_std, rtol=1e-3), \
            f"Std diff: {np.abs(stats['std'] - np_std).max()}"

    def test_returns_correct_n_samples(self, sample_h5_data):
        """Should return correct sample count."""
        stats = compute_h5_stats_streaming(
            sample_h5_data['path'],
            'test'
        )
        
        assert stats['n_samples'] == sample_h5_data['n_samples']

    def test_dtypes_are_float32(self, sample_h5_data):
        """Results should be float32."""
        stats = compute_h5_stats_streaming(
            sample_h5_data['path'],
            'test'
        )
        
        assert stats['mean'].dtype == np.float32
        assert stats['std'].dtype == np.float32

    def test_raises_on_missing_dataset(self, temp_h5):
        """Should raise KeyError for missing dataset."""
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('other', data=np.zeros(10))
        
        with pytest.raises(KeyError, match="not found"):
            compute_h5_stats_streaming(temp_h5, 'nonexistent')


# =============================================================================
# Chan's Algorithm Tests
# =============================================================================

class TestChanBatch:
    """Tests for compute_h5_stats_batch (Chan's parallel algorithm)."""

    def test_matches_numpy_mean(self, sample_h5_data):
        """Batch mean should match numpy computation."""
        stats = compute_h5_stats_batch(
            sample_h5_data['path'],
            'test',
            chunk_size=500
        )
        
        np_mean = sample_h5_data['data'].mean(axis=0)
        
        assert np.allclose(stats['mean'], np_mean, atol=1e-4)

    def test_matches_numpy_std(self, sample_h5_data):
        """Batch std should match numpy (with Bessel's correction)."""
        stats = compute_h5_stats_batch(
            sample_h5_data['path'],
            'test',
            chunk_size=500
        )
        
        np_std = sample_h5_data['data'].std(axis=0, ddof=1)
        
        assert np.allclose(stats['std'], np_std, rtol=1e-3)

    def test_welford_equals_chan(self, sample_h5_data):
        """Welford and Chan should produce same results."""
        stats_welford = compute_h5_stats_streaming(
            sample_h5_data['path'], 'test', chunk_size=500
        )
        stats_chan = compute_h5_stats_batch(
            sample_h5_data['path'], 'test', chunk_size=500
        )
        
        assert np.allclose(stats_welford['mean'], stats_chan['mean'], atol=1e-5)
        assert np.allclose(stats_welford['std'], stats_chan['std'], atol=1e-4)


# =============================================================================
# _combine_stats Helper Tests
# =============================================================================

class TestCombineStats:
    """Tests for the Chan's parallel combination helper."""

    def test_combines_two_populations(self):
        """Should correctly combine two population statistics."""
        np.random.seed(42)
        
        # Two separate populations
        data_a = np.random.randn(100, 10) * 2 + 5
        data_b = np.random.randn(150, 10) * 3 - 2
        
        stats_a = {
            'n': len(data_a),
            'mean': data_a.mean(axis=0),
            'var': data_a.var(axis=0, ddof=0),
        }
        stats_b = {
            'n': len(data_b),
            'mean': data_b.mean(axis=0),
            'var': data_b.var(axis=0, ddof=0),
        }
        
        combined = _combine_stats(stats_a, stats_b)
        
        # Compare to numpy on combined data
        combined_data = np.vstack([data_a, data_b])
        np_mean = combined_data.mean(axis=0)
        np_var = combined_data.var(axis=0, ddof=0)
        
        assert np.allclose(combined['mean'], np_mean, atol=1e-10)
        assert np.allclose(combined['var'], np_var, atol=1e-10)
        assert combined['n'] == 250

    def test_combine_preserves_numerical_stability(self):
        """Should handle large mean differences without overflow."""
        stats_a = {
            'n': 1000,
            'mean': np.array([1e6]),
            'var': np.array([1.0]),
        }
        stats_b = {
            'n': 1000,
            'mean': np.array([1e-6]),
            'var': np.array([1.0]),
        }
        
        combined = _combine_stats(stats_a, stats_b)
        
        # Mean should be average of the two means
        expected_mean = (1e6 + 1e-6) / 2
        assert np.allclose(combined['mean'], expected_mean, rtol=1e-6)
        assert not np.isnan(combined['var']).any()


# =============================================================================
# Save/Load Stats Tests
# =============================================================================

class TestStatsPersistence:
    """Tests for save_stats_to_h5 and load_stats_from_h5."""

    def test_roundtrip(self, temp_h5):
        """Stats should survive save/load roundtrip."""
        # Create file with dataset
        np.random.seed(42)
        mean = np.random.randn(64).astype(np.float32)
        std = np.abs(np.random.randn(64)).astype(np.float32) + 0.1
        
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=np.zeros((10, 64)))
        
        # Save stats
        save_stats_to_h5(temp_h5, 'test', mean, std)
        
        # Load stats
        loaded = load_stats_from_h5(temp_h5, 'test')
        
        assert loaded is not None
        assert np.allclose(loaded['mean'], mean)
        assert np.allclose(loaded['std'], std)

    def test_load_returns_none_if_missing(self, temp_h5):
        """Should return None if stats not saved."""
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=np.zeros((10, 64)))
        
        loaded = load_stats_from_h5(temp_h5, 'test')
        assert loaded is None


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample(self, temp_h5):
        """Should handle single sample gracefully."""
        data = np.array([[1.0, 2.0, 3.0]])
        
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=data.astype(np.float32))
        
        stats = compute_h5_stats_streaming(temp_h5, 'test')
        
        assert np.allclose(stats['mean'], data[0])
        assert stats['n_samples'] == 1

    def test_constant_values(self, temp_h5):
        """Constant values should have zero std."""
        data = np.ones((100, 10), dtype=np.float32) * 42
        
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=data)
        
        stats = compute_h5_stats_streaming(temp_h5, 'test')
        
        assert np.allclose(stats['mean'], 42.0)
        assert np.allclose(stats['std'], 0.0, atol=1e-6)

    def test_large_values_no_overflow(self, temp_h5):
        """Should handle large values without overflow."""
        data = np.ones((100, 10), dtype=np.float32) * 1e6
        
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=data)
        
        stats = compute_h5_stats_streaming(temp_h5, 'test')
        
        assert np.allclose(stats['mean'], 1e6)
        assert not np.isnan(stats['mean']).any()
        assert not np.isinf(stats['mean']).any()

    def test_small_chunk_size(self, temp_h5):
        """Should work with very small chunk size."""
        np.random.seed(42)
        data = np.random.randn(100, 20).astype(np.float32)
        
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=data)
        
        stats = compute_h5_stats_streaming(temp_h5, 'test', chunk_size=10)
        
        np_mean = data.mean(axis=0)
        assert np.allclose(stats['mean'], np_mean, atol=1e-4)

    def test_chunk_size_larger_than_data(self, temp_h5):
        """Should work when chunk size exceeds data size."""
        np.random.seed(42)
        data = np.random.randn(50, 20).astype(np.float32)
        
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=data)
        
        stats = compute_h5_stats_streaming(temp_h5, 'test', chunk_size=1000)
        
        np_mean = data.mean(axis=0)
        assert np.allclose(stats['mean'], np_mean, atol=1e-4)


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability of algorithms."""

    def test_welford_catastrophic_cancellation(self, temp_h5):
        """Welford should avoid catastrophic cancellation."""
        # Data where naive algorithm would struggle
        # Mean ~1e6, variance ~1 (1e9 loses precision in float32)
        np.random.seed(42)
        data = 1e6 + np.random.randn(1000, 10)
        
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=data.astype(np.float32))
        
        stats = compute_h5_stats_streaming(temp_h5, 'test', chunk_size=100)
        
        np_std = data.std(axis=0, ddof=1).astype(np.float32)
        
        # Welford should get this right
        assert np.allclose(stats['std'], np_std, rtol=0.1)

    def test_multiple_chunk_sizes_consistent(self, temp_h5):
        """Results should be consistent across different chunk sizes."""
        np.random.seed(42)
        data = np.random.randn(1000, 50).astype(np.float32)
        
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=data)
        
        # Try different chunk sizes
        stats_10 = compute_h5_stats_streaming(temp_h5, 'test', chunk_size=10)
        stats_100 = compute_h5_stats_streaming(temp_h5, 'test', chunk_size=100)
        stats_500 = compute_h5_stats_streaming(temp_h5, 'test', chunk_size=500)
        
        # All should match
        assert np.allclose(stats_10['mean'], stats_100['mean'], atol=1e-5)
        assert np.allclose(stats_100['mean'], stats_500['mean'], atol=1e-5)
        assert np.allclose(stats_10['std'], stats_100['std'], atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
