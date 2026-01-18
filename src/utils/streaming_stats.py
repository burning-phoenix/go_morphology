"""
Streaming statistics using Welford's algorithm.

Computes mean and variance without loading all data into memory.
Essential for normalizing activations on Colab with limited RAM.

Reference: docs/msae_paper.md
"""

import numpy as np
from typing import Optional


class StreamingStats:
    """
    Online computation of mean and variance using Welford's algorithm.

    Memory efficient: O(n_features) instead of O(n_samples * n_features)
    """

    def __init__(self, n_features: int):
        """
        Args:
            n_features: Number of features (e.g., 256 for Leela Zero channels)
        """
        self.n_features = n_features
        self.count = 0
        self._mean = np.zeros(n_features, dtype=np.float64)
        self._m2 = np.zeros(n_features, dtype=np.float64)  # Sum of squared deviations

    def update(self, x: np.ndarray) -> None:
        """
        Update statistics with a single sample.

        Args:
            x: Array of shape (n_features,)
        """
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def update_batch(self, batch: np.ndarray) -> None:
        """
        Update statistics with a batch of samples.

        Uses parallel algorithm for combining batch stats with running stats.

        Args:
            batch: Array of shape (batch_size, n_features)
        """
        if len(batch) == 0:
            return

        batch = np.asarray(batch, dtype=np.float64)
        n_batch = len(batch)

        # Compute batch statistics
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0, ddof=0)  # Population variance
        batch_m2 = batch_var * n_batch

        # Combine with running statistics (parallel algorithm)
        if self.count == 0:
            self._mean = batch_mean
            self._m2 = batch_m2
            self.count = n_batch
        else:
            n_total = self.count + n_batch
            delta = batch_mean - self._mean

            # Update mean
            self._mean = (self.count * self._mean + n_batch * batch_mean) / n_total

            # Update M2 using parallel variance formula
            self._m2 = self._m2 + batch_m2 + delta**2 * self.count * n_batch / n_total

            self.count = n_total

    @property
    def mean(self) -> np.ndarray:
        """Current mean estimate."""
        return self._mean.astype(np.float32)

    @property
    def variance(self) -> np.ndarray:
        """Current variance estimate (population variance)."""
        if self.count < 2:
            return np.zeros(self.n_features, dtype=np.float32)
        return (self._m2 / self.count).astype(np.float32)

    @property
    def std(self) -> np.ndarray:
        """Current standard deviation estimate."""
        return np.sqrt(self.variance + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize data using computed statistics.

        Args:
            x: Data to normalize, shape (..., n_features)

        Returns:
            Normalized data: (x - mean) / std
        """
        return (x - self.mean) / self.std

    def save(self, path: str) -> None:
        """Save statistics to file."""
        np.savez(path,
                 mean=self.mean,
                 std=self.std,
                 variance=self.variance,
                 count=self.count,
                 n_features=self.n_features)

    @classmethod
    def load(cls, path: str) -> 'StreamingStats':
        """Load statistics from file."""
        data = np.load(path)
        stats = cls(int(data['n_features']))
        stats._mean = data['mean'].astype(np.float64)
        stats._m2 = data['variance'].astype(np.float64) * data['count']
        stats.count = int(data['count'])
        return stats


def compute_stats_from_chunks(
    chunk_iterator,
    n_features: int = 256
) -> StreamingStats:
    """
    Compute statistics from an iterator of data chunks.

    Args:
        chunk_iterator: Iterator yielding numpy arrays of shape (n, n_features)
        n_features: Number of features

    Returns:
        StreamingStats with computed mean/variance
    """
    stats = StreamingStats(n_features)

    for chunk in chunk_iterator:
        stats.update_batch(chunk)

    return stats
