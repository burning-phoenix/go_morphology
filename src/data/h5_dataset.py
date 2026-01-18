"""
HDF5 Dataset for PyTorch with chunked streaming.

Provides memory-efficient iteration over large HDF5 datasets
without loading all data into memory.

Usage:
    from src.data.h5_dataset import ChunkedH5Dataset
    from torch.utils.data import DataLoader
    
    dataset = ChunkedH5Dataset(
        h5_path='activations.h5',
        dataset_key='block5',
        chunk_size=10000,  # or None for auto
    )
    
    loader = DataLoader(dataset, batch_size=4096)
    for batch in loader:
        # Process batch
        pass

References:
- h5py_guide.md: Chunking (L:698-727), iter_chunks (L:793-800)
"""

from pathlib import Path
from typing import Optional, Tuple, Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset

# Handle both package and standalone import scenarios
try:
    from ..utils.system import get_system_capabilities
except ImportError:
    from utils.system import get_system_capabilities


class ChunkedH5Dataset(IterableDataset):
    """
    Stream data from HDF5 without loading all into memory.
    
    Uses h5py's efficient chunked reading to iterate through
    large datasets piece by piece.
    
    Note: IterableDataset does not support random access, so
    shuffle must be done at the batch level or with a buffer.
    """
    
    def __init__(
        self,
        h5_path: str,
        dataset_key: str,
        chunk_size: Optional[int] = None,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        shuffle_chunks: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            h5_path: Path to HDF5 file
            dataset_key: Key of dataset within file (e.g., 'block5')
            chunk_size: Samples per chunk (None = auto-detect)
            normalize: Whether to normalize data
            mean: Precomputed mean (load from file if None and normalize=True)
            std: Precomputed std (load from file if None and normalize=True)
            shuffle_chunks: Whether to randomize chunk order each epoch
            seed: Random seed for reproducibility
        """
        self.h5_path = str(h5_path)
        self.dataset_key = dataset_key
        self.normalize = normalize
        self.shuffle_chunks = shuffle_chunks
        self.seed = seed
        
        # Get shape without loading data
        with h5py.File(self.h5_path, 'r') as f:
            if dataset_key not in f:
                raise KeyError(f"Dataset '{dataset_key}' not found in {h5_path}")
            
            dset = f[dataset_key]
            self.shape = dset.shape
            self.dtype = dset.dtype
            self.h5_chunks = dset.chunks  # HDF5 storage chunks
        
        # Auto chunk size based on system
        if chunk_size is None:
            caps = get_system_capabilities()
            chunk_size = caps.optimal_chunk_size(self.shape[1])
        self.chunk_size = chunk_size
        
        # Normalization
        self.mean = mean
        self.std = std
        if normalize and (mean is None or std is None):
            self._load_or_compute_stats()
    
    def _load_or_compute_stats(self):
        """Load normalization stats from file or compute streaming."""
        # Try to load from HDF5 attributes
        with h5py.File(self.h5_path, 'r') as f:
            dset = f[self.dataset_key]
            if 'mean' in dset.attrs and 'std' in dset.attrs:
                self.mean = dset.attrs['mean']
                self.std = dset.attrs['std']
                return
        
        # Compute using streaming stats
        try:
            from .streaming_stats import compute_h5_stats_streaming
        except ImportError:
            from streaming_stats import compute_h5_stats_streaming
        stats = compute_h5_stats_streaming(self.h5_path, self.dataset_key)
        self.mean = stats['mean']
        self.std = stats['std']
    
    def __len__(self) -> int:
        return self.shape[0]
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield samples one at a time, reading in chunks.

        Properly handles multi-worker DataLoader by splitting data across workers.
        Each worker processes a disjoint subset of chunks.
        """
        n_samples = self.shape[0]

        # Handle multi-worker DataLoader
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process loading - use all data
            iter_start = 0
            iter_end = n_samples
            worker_seed = self.seed
        else:
            # Multi-process loading - split data among workers
            per_worker = (n_samples + worker_info.num_workers - 1) // worker_info.num_workers
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, n_samples)
            worker_seed = self.seed + worker_id

            # Handle edge case where worker has no data
            if iter_start >= n_samples:
                return

        # Calculate chunks for this worker's range
        n_worker_samples = iter_end - iter_start
        n_chunks = (n_worker_samples + self.chunk_size - 1) // self.chunk_size

        # Generate chunk indices
        chunk_indices = list(range(n_chunks))

        if self.shuffle_chunks:
            rng = np.random.default_rng(worker_seed)
            rng.shuffle(chunk_indices)

        with h5py.File(self.h5_path, 'r') as f:
            dset = f[self.dataset_key]

            for chunk_idx in chunk_indices:
                start = iter_start + chunk_idx * self.chunk_size
                end = min(start + self.chunk_size, iter_end)

                # Read chunk
                chunk = dset[start:end].astype(np.float32)

                # Normalize
                if self.normalize and self.mean is not None:
                    chunk = (chunk - self.mean) / (self.std + 1e-8)

                # Shuffle within chunk for better training
                if self.shuffle_chunks:
                    rng = np.random.default_rng(worker_seed + chunk_idx)
                    rng.shuffle(chunk)

                # Yield individual samples
                for sample in chunk:
                    yield torch.from_numpy(sample)


class IndexedH5Dataset(Dataset):
    """
    Random-access HDF5 dataset for validation/testing.
    
    Unlike ChunkedH5Dataset, this supports random indexing
    but may be slower for large datasets due to many small reads.
    
    Best for smaller datasets or when random access is required.
    """
    
    def __init__(
        self,
        h5_path: str,
        dataset_key: str,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.h5_path = str(h5_path)
        self.dataset_key = dataset_key
        self.normalize = normalize
        
        # Get shape
        with h5py.File(self.h5_path, 'r') as f:
            self.shape = f[dataset_key].shape
        
        self.mean = mean
        self.std = std
        if normalize and (mean is None or std is None):
            self._load_stats()
    
    def _load_stats(self):
        """Load normalization stats."""
        with h5py.File(self.h5_path, 'r') as f:
            dset = f[self.dataset_key]
            if 'mean' in dset.attrs and 'std' in dset.attrs:
                self.mean = dset.attrs['mean']
                self.std = dset.attrs['std']
    
    def __len__(self) -> int:
        return self.shape[0]
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.h5_path, 'r') as f:
            sample = f[self.dataset_key][idx].astype(np.float32)
        
        if self.normalize and self.mean is not None:
            sample = (sample - self.mean) / (self.std + 1e-8)
        
        return torch.from_numpy(sample)


def create_h5_dataloaders(
    h5_path: str,
    dataset_key: str,
    batch_size: int = 4096,
    val_split: float = 0.1,
    normalize: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, dict]:
    """
    Create train/val dataloaders from HDF5 file with memory-efficient streaming.
    
    Args:
        h5_path: Path to HDF5 file
        dataset_key: Key of dataset
        batch_size: Batch size
        val_split: Fraction for validation
        normalize: Whether to normalize
        num_workers: DataLoader workers
        seed: Random seed
        
    Returns:
        (train_loader, val_loader, norm_stats)
    """
    from torch.utils.data import DataLoader
    try:
        from .streaming_stats import compute_h5_stats_streaming
    except ImportError:
        from streaming_stats import compute_h5_stats_streaming
    
    # Get dataset info
    with h5py.File(h5_path, 'r') as f:
        n_samples = f[dataset_key].shape[0]
        sample_dim = f[dataset_key].shape[1]
    
    # Compute normalization stats (streaming)
    stats = compute_h5_stats_streaming(h5_path, dataset_key)
    mean = stats['mean']
    std = stats['std']
    
    # Check if we can just load everything
    caps = get_system_capabilities()
    if caps.should_use_full_load(n_samples, sample_dim):
        print(f"Dataset fits in memory, using full-load mode")
        return _create_full_load_dataloaders(
            h5_path, dataset_key, batch_size, val_split,
            normalize, mean, std, num_workers, seed
        )
    
    # Chunked streaming mode
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    print(f"Using chunked streaming: {n_samples:,} samples")
    print(f"  Train: {n_train:,}, Val: {n_val:,}")
    print(f"  Chunk size: {caps.optimal_chunk_size(sample_dim):,}")
    
    # For streaming, we'll handle train/val split within the dataset
    # by specifying index ranges
    train_dataset = _RangeH5Dataset(
        h5_path, dataset_key, 0, n_train,
        normalize, mean, std, shuffle=True, seed=seed
    )
    
    val_dataset = _RangeH5Dataset(
        h5_path, dataset_key, n_train, n_samples,
        normalize, mean, std, shuffle=False, seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    norm_stats = {'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std)}
    
    return train_loader, val_loader, norm_stats


class _RangeH5Dataset(IterableDataset):
    """Internal dataset for a range of indices within HDF5 file.

    Properly handles multi-worker DataLoader by splitting data across workers.
    Each worker processes a disjoint subset of the data.
    """

    def __init__(
        self,
        h5_path: str,
        dataset_key: str,
        start_idx: int,
        end_idx: int,
        normalize: bool,
        mean: np.ndarray,
        std: np.ndarray,
        shuffle: bool,
        seed: int,
    ):
        self.h5_path = h5_path
        self.dataset_key = dataset_key
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.shuffle = shuffle
        self.seed = seed

        caps = get_system_capabilities()
        with h5py.File(h5_path, 'r') as f:
            sample_dim = f[dataset_key].shape[1]
        self.chunk_size = caps.optimal_chunk_size(sample_dim)

    def __len__(self):
        return self.end_idx - self.start_idx

    def __iter__(self):
        # Handle multi-worker DataLoader by splitting data across workers
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process loading - use full range
            iter_start = self.start_idx
            iter_end = self.end_idx
        else:
            # Multi-process loading - split range among workers
            total_samples = self.end_idx - self.start_idx
            per_worker = (total_samples + worker_info.num_workers - 1) // worker_info.num_workers
            worker_id = worker_info.id
            iter_start = self.start_idx + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end_idx)

            # Handle edge case where worker has no data
            if iter_start >= self.end_idx:
                return

        n_samples = iter_end - iter_start
        n_chunks = (n_samples + self.chunk_size - 1) // self.chunk_size

        chunk_indices = list(range(n_chunks))

        # Use worker-specific seed for shuffling to ensure different order per worker
        worker_seed = self.seed
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id

        if self.shuffle:
            rng = np.random.default_rng(worker_seed)
            rng.shuffle(chunk_indices)

        with h5py.File(self.h5_path, 'r') as f:
            dset = f[self.dataset_key]

            for chunk_idx in chunk_indices:
                start = iter_start + chunk_idx * self.chunk_size
                end = min(start + self.chunk_size, iter_end)

                chunk = dset[start:end].astype(np.float32)

                if self.normalize:
                    chunk = (chunk - self.mean) / (self.std + 1e-8)

                if self.shuffle:
                    rng = np.random.default_rng(worker_seed + chunk_idx)
                    rng.shuffle(chunk)

                for sample in chunk:
                    yield torch.from_numpy(sample)


def _create_full_load_dataloaders(
    h5_path: str,
    dataset_key: str,
    batch_size: int,
    val_split: float,
    normalize: bool,
    mean: np.ndarray,
    std: np.ndarray,
    num_workers: int,
    seed: int,
):
    """Fallback for small datasets that fit in memory."""
    from torch.utils.data import DataLoader, TensorDataset
    
    with h5py.File(h5_path, 'r') as f:
        data = f[dataset_key][()].astype(np.float32)
    
    if normalize:
        data = (data - mean) / (std + 1e-8)
    
    data = torch.from_numpy(data)
    
    # Split
    n_samples = len(data)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator)
    
    train_data = data[indices[:n_train]]
    val_data = data[indices[n_train:]]
    
    del data
    
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    norm_stats = {'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std)}
    
    return train_loader, val_loader, norm_stats
