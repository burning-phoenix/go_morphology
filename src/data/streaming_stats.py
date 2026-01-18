"""
Streaming statistics computation for large datasets.

Uses Welford's online algorithm to compute mean and variance
in a single pass without loading all data into memory.

Reference:
- Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums 
  of Squares and Products". Technometrics. 4(3): 419–420.

Usage:
    from src.data.streaming_stats import compute_h5_stats_streaming
    
    stats = compute_h5_stats_streaming('activations.h5', 'block5')
    print(f"Mean: {stats['mean'].mean():.4f}")
    print(f"Std: {stats['std'].mean():.4f}")
"""

from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np

# Handle both package and standalone import scenarios
try:
    from ..utils.system import get_system_capabilities
except ImportError:
    from utils.system import get_system_capabilities


def compute_h5_stats_streaming(
    h5_path: str,
    dataset_key: str,
    chunk_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute mean and std of HDF5 dataset without loading all into memory.
    
    Uses Welford's online algorithm for numerical stability.
    
    Args:
        h5_path: Path to HDF5 file
        dataset_key: Key of dataset
        chunk_size: Samples per chunk (None = auto-detect)
        
    Returns:
        Dict with 'mean' and 'std' as float32 arrays
    """
    with h5py.File(h5_path, 'r') as f:
        if dataset_key not in f:
            raise KeyError(f"Dataset '{dataset_key}' not found in {h5_path}")
        
        dset = f[dataset_key]
        n_samples, n_features = dset.shape
        
        # Auto chunk size
        if chunk_size is None:
            caps = get_system_capabilities()
            chunk_size = caps.optimal_chunk_size(n_features)
        
        # Initialize Welford accumulators
        n = 0
        mean = np.zeros(n_features, dtype=np.float64)
        M2 = np.zeros(n_features, dtype=np.float64)
        
        # Process in chunks
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = dset[start:end].astype(np.float64)
            
            # Welford's online update for each sample
            for x in chunk:
                n += 1
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                M2 += delta * delta2
        
        # Compute final variance and std
        if n > 1:
            variance = M2 / (n - 1)  # Bessel's correction
        else:
            variance = M2
        
        std = np.sqrt(variance)
    
    return {
        'mean': mean.astype(np.float32),
        'std': std.astype(np.float32),
        'n_samples': n,
    }


def compute_h5_stats_batch(
    h5_path: str,
    dataset_key: str,
    chunk_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute mean and std using batched parallel update.
    
    Faster than per-sample Welford for very large dimensions.
    Uses Chan's parallel algorithm for combining chunk statistics.
    
    Args:
        h5_path: Path to HDF5 file
        dataset_key: Key of dataset
        chunk_size: Samples per chunk (None = auto-detect)
        
    Returns:
        Dict with 'mean' and 'std' as float32 arrays
    """
    with h5py.File(h5_path, 'r') as f:
        if dataset_key not in f:
            raise KeyError(f"Dataset '{dataset_key}' not found in {h5_path}")
        
        dset = f[dataset_key]
        n_samples, n_features = dset.shape
        
        if chunk_size is None:
            caps = get_system_capabilities()
            chunk_size = caps.optimal_chunk_size(n_features)
        
        # Accumulate chunk statistics for parallel combination
        chunk_stats = []
        
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = dset[start:end].astype(np.float64)
            
            n_chunk = len(chunk)
            mean_chunk = chunk.mean(axis=0)
            var_chunk = chunk.var(axis=0, ddof=0)  # Population variance
            
            chunk_stats.append({
                'n': n_chunk,
                'mean': mean_chunk,
                'var': var_chunk,
            })
        
        # Combine using Chan's parallel algorithm
        if len(chunk_stats) == 1:
            combined = chunk_stats[0]
        else:
            combined = chunk_stats[0]
            for stats in chunk_stats[1:]:
                combined = _combine_stats(combined, stats)
        
        # Bessel's correction for sample variance
        std = np.sqrt(combined['var'] * combined['n'] / (combined['n'] - 1))
    
    return {
        'mean': combined['mean'].astype(np.float32),
        'std': std.astype(np.float32),
        'n_samples': combined['n'],
    }


def _combine_stats(a: dict, b: dict) -> dict:
    """
    Combine two sets of statistics using Chan's parallel algorithm.
    
    Reference:
    Chan, Tony F et al. (1983). Algorithms for Computing the Sample Variance.
    """
    n_a = a['n']
    n_b = b['n']
    n_ab = n_a + n_b
    
    delta = b['mean'] - a['mean']
    
    mean_ab = a['mean'] + delta * n_b / n_ab
    
    # M2_a = var_a * n_a, M2_b = var_b * n_b
    M2_a = a['var'] * n_a
    M2_b = b['var'] * n_b
    M2_ab = M2_a + M2_b + delta**2 * n_a * n_b / n_ab
    
    var_ab = M2_ab / n_ab
    
    return {
        'n': n_ab,
        'mean': mean_ab,
        'var': var_ab,
    }


def save_stats_to_h5(
    h5_path: str,
    dataset_key: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    """Save normalization stats as HDF5 attributes."""
    with h5py.File(h5_path, 'a') as f:
        dset = f[dataset_key]
        dset.attrs['mean'] = mean
        dset.attrs['std'] = std


def load_stats_from_h5(
    h5_path: str,
    dataset_key: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Load normalization stats from HDF5 attributes if present."""
    with h5py.File(h5_path, 'r') as f:
        dset = f[dataset_key]
        if 'mean' in dset.attrs and 'std' in dset.attrs:
            return {
                'mean': dset.attrs['mean'],
                'std': dset.attrs['std'],
            }
    return None
