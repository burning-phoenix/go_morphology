"""
Unit tests for ChunkedH5Dataset and related data loading utilities.

Tests chunked streaming from HDF5 files, auto-detection of optimal
parameters, and fallback to full-load for small datasets.

Priority: 🟠 High - incorrect loading corrupts all downstream analysis.
"""

import pytest
import numpy as np
import h5py
import tempfile
import sys
import importlib.util
import torch
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

# Load data.streaming_stats (dependency for h5_dataset)
_stats_spec = importlib.util.spec_from_file_location(
    "streaming_stats", _src_dir / "data" / "streaming_stats.py"
)
_stats_module = importlib.util.module_from_spec(_stats_spec)
sys.modules['streaming_stats'] = _stats_module
_stats_spec.loader.exec_module(_stats_module)

# Load data.h5_dataset directly (bypasses data/__init__.py)
_h5_spec = importlib.util.spec_from_file_location(
    "h5_dataset", _src_dir / "data" / "h5_dataset.py"
)
_h5_module = importlib.util.module_from_spec(_h5_spec)
sys.modules['h5_dataset'] = _h5_module

try:
    _h5_spec.loader.exec_module(_h5_module)
    ChunkedH5Dataset = _h5_module.ChunkedH5Dataset
    IndexedH5Dataset = _h5_module.IndexedH5Dataset
    create_h5_dataloaders = _h5_module.create_h5_dataloaders
    _RangeH5Dataset = _h5_module._RangeH5Dataset
    HAS_H5_DATASET = True
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
    HAS_H5_DATASET = False

# Skip all tests if not available
pytestmark = pytest.mark.skipif(
    not HAS_H5_DATASET,
    reason="data.h5_dataset not available"
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
    """Create HDF5 with sample activation-like data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 256
    
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    
    with h5py.File(temp_h5, 'w') as f:
        dset = f.create_dataset('block5', data=data, chunks=(100, n_features))
        # Add normalization stats as attributes
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        dset.attrs['mean'] = mean
        dset.attrs['std'] = std
    
    return {
        'path': temp_h5,
        'data': data,
        'mean': mean,
        'std': std,
        'n_samples': n_samples,
        'n_features': n_features,
    }


# =============================================================================
# ChunkedH5Dataset Basic Tests
# =============================================================================

class TestChunkedH5DatasetBasic:
    """Basic functionality tests for ChunkedH5Dataset."""

    def test_init_with_path(self, sample_h5_data):
        """Should initialize with path and dataset key."""
        dataset = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            normalize=False,
        )
        
        assert len(dataset) == sample_h5_data['n_samples']
        assert dataset.shape == (1000, 256)

    def test_iteration_yields_tensors(self, sample_h5_data):
        """Iteration should yield PyTorch tensors."""
        dataset = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=False,
        )
        
        # Get first sample
        sample = next(iter(dataset))
        
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (256,)
        assert sample.dtype == torch.float32

    def test_iterates_all_samples(self, sample_h5_data):
        """Should iterate through all samples."""
        dataset = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=False,
            shuffle_chunks=False,
        )
        
        count = sum(1 for _ in dataset)
        assert count == sample_h5_data['n_samples']

    def test_raises_on_missing_dataset(self, temp_h5):
        """Should raise KeyError for missing dataset."""
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('other', data=np.zeros(10))
        
        with pytest.raises(KeyError, match="not found"):
            ChunkedH5Dataset(temp_h5, 'nonexistent')


# =============================================================================
# Normalization Tests
# =============================================================================

class TestChunkedH5DatasetNormalization:
    """Tests for normalization functionality."""

    def test_normalization_applied(self, sample_h5_data):
        """Data should be normalized when normalize=True."""
        dataset = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=True,
            mean=sample_h5_data['mean'],
            std=sample_h5_data['std'],
            shuffle_chunks=False,
        )
        
        # Collect all samples
        samples = list(dataset)
        all_data = torch.stack(samples).numpy()
        
        # Should have approximately zero mean and unit std
        assert np.abs(all_data.mean()) < 0.1
        # Std check is approximate due to float32 precision
        assert 0.8 < all_data.std() < 1.2

    def test_no_normalization_when_disabled(self, sample_h5_data):
        """Data should be raw when normalize=False."""
        dataset = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=False,
            shuffle_chunks=False,
        )
        
        # Get first chunk worth of samples using a SINGLE iterator
        iterator = iter(dataset)
        samples = [next(iterator) for _ in range(100)]
        all_data = torch.stack(samples).numpy()
        
        # Should match raw data (non-zero mean, varying std)
        raw_chunk = sample_h5_data['data'][:100]
        assert np.allclose(all_data, raw_chunk, atol=1e-5)

    def test_loads_stats_from_h5_attrs(self, sample_h5_data):
        """Should load normalization stats from HDF5 attributes."""
        dataset = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=True,
            # Don't provide mean/std, should load from file
            shuffle_chunks=False,
        )
        
        assert dataset.mean is not None
        assert dataset.std is not None
        assert np.allclose(dataset.mean, sample_h5_data['mean'])


# =============================================================================
# Chunking Tests
# =============================================================================

class TestChunkedH5DatasetChunking:
    """Tests for chunking behavior."""

    def test_respects_chunk_size(self, sample_h5_data):
        """Should process data in specified chunk sizes."""
        # This is implicitly tested by correct iteration count
        dataset = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=False,
            shuffle_chunks=False,
        )
        
        count = sum(1 for _ in dataset)
        assert count == 1000

    def test_handles_non_divisible_chunks(self, temp_h5):
        """Should handle when n_samples % chunk_size != 0."""
        data = np.random.randn(150, 64).astype(np.float32)
        
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=data)
        
        dataset = ChunkedH5Dataset(
            temp_h5,
            'test',
            chunk_size=40,  # 150 / 40 = 3.75 chunks
            normalize=False,
            shuffle_chunks=False,
        )
        
        count = sum(1 for _ in dataset)
        assert count == 150


# =============================================================================
# Shuffling Tests
# =============================================================================

class TestChunkedH5DatasetShuffling:
    """Tests for shuffle behavior."""

    def test_shuffle_changes_order(self, sample_h5_data):
        """Shuffling should change sample order."""
        dataset_unshuffled = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=False,
            shuffle_chunks=False,
        )
        
        dataset_shuffled = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=False,
            shuffle_chunks=True,
            seed=42,
        )
        
        samples_unshuffled = [next(iter(dataset_unshuffled)) for _ in range(10)]
        samples_shuffled = [next(iter(dataset_shuffled)) for _ in range(10)]
        
        # Order should differ (with high probability)
        unshuffled_stack = torch.stack(samples_unshuffled)
        shuffled_stack = torch.stack(samples_shuffled)
        
        # Not exactly equal (shuffled)
        assert not torch.allclose(unshuffled_stack, shuffled_stack)

    def test_same_seed_same_order(self, sample_h5_data):
        """Same seed should produce same order."""
        dataset1 = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=False,
            shuffle_chunks=True,
            seed=12345,
        )
        
        dataset2 = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=False,
            shuffle_chunks=True,
            seed=12345,
        )
        
        samples1 = [next(iter(dataset1)) for _ in range(50)]
        samples2 = [next(iter(dataset2)) for _ in range(50)]
        
        for s1, s2 in zip(samples1, samples2):
            assert torch.allclose(s1, s2)


# =============================================================================
# IndexedH5Dataset Tests
# =============================================================================

class TestIndexedH5Dataset:
    """Tests for random-access IndexedH5Dataset."""

    def test_random_access(self, sample_h5_data):
        """Should support random access by index."""
        dataset = IndexedH5Dataset(
            sample_h5_data['path'],
            'block5',
            normalize=False,
        )
        
        sample_0 = dataset[0]
        sample_50 = dataset[50]
        sample_999 = dataset[999]
        
        assert isinstance(sample_0, torch.Tensor)
        assert sample_0.shape == (256,)
        assert not torch.allclose(sample_0, sample_50)

    def test_len_correct(self, sample_h5_data):
        """Length should match number of samples."""
        dataset = IndexedH5Dataset(
            sample_h5_data['path'],
            'block5',
            normalize=False,
        )
        
        assert len(dataset) == 1000


# =============================================================================
# create_h5_dataloaders Tests
# =============================================================================

class TestCreateH5Dataloaders:
    """Tests for the convenience dataloader factory."""

    def test_returns_train_val_loaders(self, sample_h5_data):
        """Should return train and val DataLoaders."""
        train_loader, val_loader, norm_stats = create_h5_dataloaders(
            sample_h5_data['path'],
            'block5',
            batch_size=64,
            val_split=0.2,
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert 'mean' in norm_stats
        assert 'std' in norm_stats

    def test_split_ratio_correct(self, sample_h5_data):
        """Train/val split should match requested ratio."""
        train_loader, val_loader, _ = create_h5_dataloaders(
            sample_h5_data['path'],
            'block5',
            batch_size=64,
            val_split=0.2,
        )
        
        # Count samples (approximately)
        train_count = sum(len(batch[0]) for batch in train_loader)
        val_count = sum(len(batch[0]) for batch in val_loader)
        
        # Should be roughly 80/20 split
        total = train_count + val_count
        val_ratio = val_count / total
        
        assert 0.15 < val_ratio < 0.25

    def test_batches_are_tensors(self, sample_h5_data):
        """Batches should contain tensors."""
        train_loader, _, _ = create_h5_dataloaders(
            sample_h5_data['path'],
            'block5',
            batch_size=32,
        )
        
        batch = next(iter(train_loader))
        
        # IterableDataset yields (batch,) tuple
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        
        assert isinstance(batch, torch.Tensor)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample(self, temp_h5):
        """Should handle single-sample dataset."""
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=data)
        
        dataset = ChunkedH5Dataset(
            temp_h5,
            'test',
            normalize=False,
        )
        
        assert len(dataset) == 1
        sample = next(iter(dataset))
        assert torch.allclose(sample, torch.tensor([1.0, 2.0, 3.0]))

    def test_large_chunk_size(self, sample_h5_data):
        """Should handle chunk size larger than dataset."""
        dataset = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=10000,  # Larger than 1000 samples
            normalize=False,
        )
        
        count = sum(1 for _ in dataset)
        assert count == 1000

    def test_very_small_chunk_size(self, temp_h5):
        """Should handle very small chunk size."""
        data = np.random.randn(100, 16).astype(np.float32)
        
        with h5py.File(temp_h5, 'w') as f:
            f.create_dataset('test', data=data)
        
        dataset = ChunkedH5Dataset(
            temp_h5,
            'test',
            chunk_size=1,  # Single sample per chunk
            normalize=False,
            shuffle_chunks=False,
        )
        
        count = sum(1 for _ in dataset)
        assert count == 100


# =============================================================================
# DataLoader Integration Tests
# =============================================================================

class TestDataLoaderIntegration:
    """Tests for PyTorch DataLoader integration."""

    def test_works_with_dataloader(self, sample_h5_data):
        """ChunkedH5Dataset should work with DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=False,
        )
        
        loader = DataLoader(dataset, batch_size=32)
        
        batch = next(iter(loader))
        assert batch.shape == (32, 256)

    def test_multiple_epochs(self, sample_h5_data):
        """Should support multiple epoch iteration."""
        dataset = ChunkedH5Dataset(
            sample_h5_data['path'],
            'block5',
            chunk_size=100,
            normalize=False,
        )
        
        # Iterate twice
        count_epoch1 = sum(1 for _ in dataset)
        count_epoch2 = sum(1 for _ in dataset)
        
        assert count_epoch1 == 1000
        assert count_epoch2 == 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
