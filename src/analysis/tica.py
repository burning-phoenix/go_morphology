"""
Time-lagged Independent Component Analysis (TICA) for SAE feature reduction.

Finds slow collective variables before clustering for attractor analysis.
TICA identifies linear combinations of features that maximize autocorrelation
at a given lag time, revealing the slowest dynamical modes.

Reference:
- docs/dynamical_systems_chaos/INDEX.md L:643-664
- PyEMMA Markov State Modeling documentation

Key equations:
- C(0) = Cov(X_t, X_t) - instantaneous covariance
- C(τ) = Cov(X_t, X_{t+τ}) - time-lagged covariance
- Generalized eigenvalue problem: C(τ) @ u = λ * C(0) @ u
- Eigenvalues sorted descending → slowest modes first
"""

import numpy as np
from scipy import linalg
from typing import Optional, Tuple, List, Iterator, Union
from dataclasses import dataclass
from pathlib import Path
import h5py


@dataclass
class TICAResult:
    """Results from TICA transformation."""
    eigenvectors: np.ndarray      # [input_dim, n_components]
    eigenvalues: np.ndarray       # [n_components]
    mean: np.ndarray              # [input_dim]
    lag: int
    n_components: int
    kinetic_variance: np.ndarray  # λ² - contribution to slow dynamics


class TICATransformer:
    """
    Time-lagged Independent Component Analysis.

    Finds slow collective variables by solving the generalized eigenvalue
    problem C(τ) @ u = λ * C(0) @ u, where C(τ) is the time-lagged covariance.

    Usage:
        tica = TICATransformer(n_components=50, lag=1)
        tica.fit(features, game_ids)  # Respects game boundaries
        slow_features = tica.transform(features)

    Critical: TICA must respect game boundaries - don't compute lagged
    covariance across game transitions.
    """

    def __init__(self, n_components: int = 50, lag: int = 1):
        """
        Args:
            n_components: Number of slow modes to keep (default 50)
            lag: Time lag in steps (default 1 = consecutive positions)
        """
        self.n_components = n_components
        self.lag = lag
        self.result_: Optional[TICAResult] = None

    def fit(
        self,
        X: np.ndarray,
        game_ids: Optional[np.ndarray] = None
    ) -> 'TICATransformer':
        """
        Fit TICA on slow collective variables.

        Args:
            X: [n_positions, n_features] feature matrix (SAE features)
            game_ids: [n_positions] game index for each position
                      If provided, only computes lagged covariance within games

        Returns:
            self for method chaining
        """
        n_samples, n_features = X.shape
        print(f"Fitting TICA: {n_samples:,} samples × {n_features} features")
        print(f"  Lag: {self.lag}, Components: {self.n_components}")

        # Center data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrices
        if game_ids is not None:
            print("  Respecting game boundaries...")
            C0, C_tau = self._compute_covariances_with_boundaries(X_centered, game_ids)
        else:
            C0, C_tau = self._compute_covariances_simple(X_centered)

        # Solve generalized eigenvalue problem: C_tau @ u = λ * C0 @ u
        print("  Solving generalized eigenvalue problem...")

        # Regularize C0 for numerical stability
        C0_reg = C0 + 1e-6 * np.eye(n_features)

        try:
            eigenvalues, eigenvectors = linalg.eigh(C_tau, C0_reg)
        except linalg.LinAlgError as e:
            print(f"  Warning: eigh failed, using svd fallback: {e}")
            # Fallback: solve via SVD
            C0_inv_sqrt = linalg.sqrtm(linalg.inv(C0_reg))
            M = C0_inv_sqrt @ C_tau @ C0_inv_sqrt
            eigenvalues, V = linalg.eigh(M)
            eigenvectors = C0_inv_sqrt @ V

        # Sort by eigenvalue (descending - slowest first)
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Keep top n_components
        n_keep = min(self.n_components, n_features)
        eigenvalues = eigenvalues[:n_keep]
        eigenvectors = eigenvectors[:, :n_keep]

        # Kinetic variance: λ² measures contribution to slow dynamics
        kinetic_variance = eigenvalues ** 2

        self.result_ = TICAResult(
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            mean=self.mean_,
            lag=self.lag,
            n_components=n_keep,
            kinetic_variance=kinetic_variance,
        )

        print(f"  Top 5 eigenvalues: {eigenvalues[:5]}")
        print(f"  Kinetic variance explained: {kinetic_variance.sum():.4f}")

        return self

    def _compute_covariances_simple(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute covariances without respecting game boundaries.

        Args:
            X: [n_samples, n_features] centered feature matrix

        Returns:
            C0: [n_features, n_features] instantaneous covariance
            C_tau: [n_features, n_features] time-lagged covariance
        """
        n_samples = len(X)

        # Instantaneous covariance
        C0 = (X.T @ X) / n_samples

        # Time-lagged covariance
        X_t = X[:-self.lag]
        X_tau = X[self.lag:]
        C_tau = (X_t.T @ X_tau) / len(X_t)

        # Symmetrize C_tau for numerical stability
        C_tau = (C_tau + C_tau.T) / 2

        return C0, C_tau

    def _compute_covariances_with_boundaries(
        self,
        X: np.ndarray,
        game_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute covariances respecting game boundaries.

        Critical: Don't include transitions across games in the lagged covariance.

        Args:
            X: [n_samples, n_features] centered feature matrix
            game_ids: [n_samples] game index for each position

        Returns:
            C0: [n_features, n_features] instantaneous covariance
            C_tau: [n_features, n_features] time-lagged covariance
        """
        n_samples, n_features = X.shape
        unique_games = np.unique(game_ids)

        # Instantaneous covariance (all samples)
        C0 = (X.T @ X) / n_samples

        # Time-lagged covariance (within games only)
        C_tau = np.zeros((n_features, n_features))
        n_pairs = 0

        for game_id in unique_games:
            mask = game_ids == game_id
            X_game = X[mask]

            if len(X_game) <= self.lag:
                continue

            X_t = X_game[:-self.lag]
            X_tau = X_game[self.lag:]

            C_tau += X_t.T @ X_tau
            n_pairs += len(X_t)

        if n_pairs > 0:
            C_tau /= n_pairs

        # Symmetrize
        C_tau = (C_tau + C_tau.T) / 2

        print(f"    Used {n_pairs:,} lagged pairs from {len(unique_games)} games")

        return C0, C_tau

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to TICA space.

        Args:
            X: [n_samples, n_features] feature matrix

        Returns:
            X_tica: [n_samples, n_components] transformed features
        """
        if self.result_ is None:
            raise ValueError("Must call fit() before transform()")

        X_centered = X - self.result_.mean
        return X_centered @ self.result_.eigenvectors

    def transform_streaming(
        self,
        h5_path: Union[str, Path],
        dataset_key: str,
        output_path: Union[str, Path],
        output_key: str = 'tica_features',
        chunk_size: int = 10000,
    ) -> None:
        """
        Transform features to TICA space, streaming from/to h5py.

        Memory-efficient: Processes in chunks without loading all data.

        Args:
            h5_path: Path to input HDF5 file with SAE features
            dataset_key: Dataset key in input file
            output_path: Path to output HDF5 file for TICA features
            output_key: Dataset key for output
            chunk_size: Number of samples to process at once
        """
        if self.result_ is None:
            raise ValueError("Must call fit() before transform()")

        h5_path = Path(h5_path)
        output_path = Path(output_path)

        with h5py.File(h5_path, 'r') as f_in:
            dset = f_in[dataset_key]
            n_samples, n_features = dset.shape
            n_components = self.result_.n_components

            print(f"Streaming TICA transform: {n_samples:,} samples")
            print(f"  {n_features} features → {n_components} components")
            print(f"  Output: {output_path}")

            with h5py.File(output_path, 'w') as f_out:
                # Create output dataset with chunking
                out_dset = f_out.create_dataset(
                    output_key,
                    shape=(n_samples, n_components),
                    dtype=np.float32,
                    chunks=(min(chunk_size, n_samples), n_components),
                    compression='gzip',
                    compression_opts=4,
                )

                # Store TICA parameters as attributes
                out_dset.attrs['n_components'] = n_components
                out_dset.attrs['lag'] = self.result_.lag
                out_dset.attrs['eigenvalues'] = self.result_.eigenvalues
                out_dset.attrs['kinetic_variance'] = self.result_.kinetic_variance

                # Process chunks
                mean = self.result_.mean
                eigenvectors = self.result_.eigenvectors

                for start in range(0, n_samples, chunk_size):
                    end = min(start + chunk_size, n_samples)
                    chunk = dset[start:end].astype(np.float32)

                    # Transform
                    chunk_centered = chunk - mean
                    chunk_tica = chunk_centered @ eigenvectors

                    # Write
                    out_dset[start:end] = chunk_tica

                    if (start // chunk_size) % 10 == 0:
                        print(f"  Processed {end:,} / {n_samples:,}")

        print(f"  Saved TICA features to {output_path}")

    def fit_transform(
        self,
        X: np.ndarray,
        game_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit TICA and transform in one call.

        Args:
            X: [n_samples, n_features] feature matrix
            game_ids: [n_samples] optional game indices

        Returns:
            X_tica: [n_samples, n_components] transformed features
        """
        self.fit(X, game_ids)
        return self.transform(X)

    def save(self, output_path: str):
        """
        Save fitted transformer to disk.

        Args:
            output_path: Path to save file (npz format)
        """
        if self.result_ is None:
            raise ValueError("Must call fit() before save()")

        np.savez(
            output_path,
            eigenvectors=self.result_.eigenvectors,
            eigenvalues=self.result_.eigenvalues,
            mean=self.result_.mean,
            lag=self.result_.lag,
            n_components=self.result_.n_components,
            kinetic_variance=self.result_.kinetic_variance,
        )
        print(f"Saved TICA transformer to {output_path}")

    @classmethod
    def load(cls, input_path: str) -> 'TICATransformer':
        """
        Load fitted transformer from disk.

        Args:
            input_path: Path to saved file

        Returns:
            Loaded TICATransformer
        """
        data = np.load(input_path)

        transformer = cls(
            n_components=int(data['n_components']),
            lag=int(data['lag']),
        )

        transformer.result_ = TICAResult(
            eigenvectors=data['eigenvectors'],
            eigenvalues=data['eigenvalues'],
            mean=data['mean'],
            lag=int(data['lag']),
            n_components=int(data['n_components']),
            kinetic_variance=data['kinetic_variance'],
        )

        print(f"Loaded TICA transformer from {input_path}")
        return transformer

    def get_implied_timescales(self, lag_time: float = 1.0) -> np.ndarray:
        """
        Compute implied timescales from eigenvalues.

        t_i = -τ / ln(|λ_i|)

        Args:
            lag_time: Duration of one lag step

        Returns:
            Implied timescales for each component
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        eigenvalues = self.result_.eigenvalues

        # Avoid log(0) or log(negative)
        valid = eigenvalues > 0
        timescales = np.full_like(eigenvalues, np.inf)
        timescales[valid] = -lag_time / np.log(eigenvalues[valid])

        return timescales

    def get_cumulative_kinetic_variance(self) -> np.ndarray:
        """
        Compute cumulative kinetic variance explained by components.

        Per PyEMMA paper: Use 95% kinetic variance cutoff for component selection.

        Returns:
            Cumulative variance ratio [0, 1] for each component
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        kinetic_var = self.result_.kinetic_variance
        cumsum = np.cumsum(kinetic_var)
        return cumsum / cumsum[-1]

    def fit_streaming(
        self,
        h5_path: Union[str, Path],
        dataset_key: str,
        game_ids: Optional[np.ndarray] = None,
        chunk_size: int = 10000,
    ) -> 'TICATransformer':
        """
        Fit TICA using streaming covariance accumulation.

        Memory-efficient: Only needs O(d²) for covariance matrices,
        not O(n×d) for all data. Streams through h5py file in chunks.

        Args:
            h5_path: Path to HDF5 file with SAE features
            dataset_key: Dataset key in HDF5 file (e.g., 'block5')
            game_ids: [n_positions] game index for each position
            chunk_size: Number of samples to process at once

        Returns:
            self for method chaining
        """
        h5_path = Path(h5_path)

        with h5py.File(h5_path, 'r') as f:
            dset = f[dataset_key]
            n_samples, n_features = dset.shape

            print(f"Streaming TICA fit: {n_samples:,} samples × {n_features} features")
            print(f"  Chunk size: {chunk_size:,}, Lag: {self.lag}")
            print(f"  Memory for covariances: {2 * n_features**2 * 8 / 1e6:.1f} MB")

            # Pass 1: Compute mean
            print("  Pass 1: Computing mean...")
            mean_acc = np.zeros(n_features, dtype=np.float64)
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                chunk = dset[start:end].astype(np.float64)
                mean_acc += chunk.sum(axis=0)
            mean = (mean_acc / n_samples).astype(np.float32)
            self.mean_ = mean

            # Pass 2: Accumulate covariances
            print("  Pass 2: Accumulating covariances...")
            C0 = np.zeros((n_features, n_features), dtype=np.float64)
            C_tau = np.zeros((n_features, n_features), dtype=np.float64)
            n_pairs = 0

            if game_ids is not None:
                # With game boundaries - need to process per-game
                C0, C_tau, n_pairs = self._accumulate_covariances_with_boundaries_streaming(
                    dset, mean, game_ids, chunk_size
                )
            else:
                # Simple case - can process in overlapping windows
                prev_chunk_tail = None

                for start in range(0, n_samples, chunk_size):
                    end = min(start + chunk_size, n_samples)
                    chunk = dset[start:end].astype(np.float32) - mean

                    # Instantaneous covariance
                    C0 += chunk.T @ chunk

                    # Time-lagged covariance within this chunk
                    if len(chunk) > self.lag:
                        X_t = chunk[:-self.lag]
                        X_tau = chunk[self.lag:]
                        C_tau += X_t.T @ X_tau
                        n_pairs += len(X_t)

                    # Handle boundary between chunks for lagged covariance
                    if prev_chunk_tail is not None and len(chunk) >= self.lag:
                        # prev_chunk_tail: last `lag` rows of previous chunk
                        # chunk[:lag]: first `lag` rows of this chunk
                        for i in range(min(self.lag, len(prev_chunk_tail))):
                            if i < len(chunk):
                                C_tau += np.outer(prev_chunk_tail[i], chunk[i])
                                n_pairs += 1

                    # Save tail for next iteration
                    prev_chunk_tail = chunk[-self.lag:].copy() if len(chunk) >= self.lag else chunk.copy()

                C0 /= n_samples
                if n_pairs > 0:
                    C_tau /= n_pairs

        # Symmetrize C_tau
        C_tau = (C_tau + C_tau.T) / 2

        print(f"  Used {n_pairs:,} lagged pairs")

        # Solve generalized eigenvalue problem
        self._solve_eigenvalue_problem(C0, C_tau, n_features)

        return self

    def _accumulate_covariances_with_boundaries_streaming(
        self,
        dset: h5py.Dataset,
        mean: np.ndarray,
        game_ids: np.ndarray,
        chunk_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Accumulate covariances respecting game boundaries, streaming from h5py.

        Args:
            dset: h5py Dataset to stream from
            mean: Pre-computed mean vector
            game_ids: [n_samples] game index for each position
            chunk_size: Chunk size for streaming

        Returns:
            C0, C_tau, n_pairs
        """
        n_samples, n_features = dset.shape

        C0 = np.zeros((n_features, n_features), dtype=np.float64)
        C_tau = np.zeros((n_features, n_features), dtype=np.float64)
        n_pairs = 0

        # Process each game separately for lagged covariance
        unique_games = np.unique(game_ids)
        print(f"    Processing {len(unique_games)} games...")

        # First, accumulate C0 over all data
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = dset[start:end].astype(np.float32) - mean
            C0 += chunk.T @ chunk
        C0 /= n_samples

        # For C_tau, process game by game
        for game_id in unique_games:
            mask = game_ids == game_id
            indices = np.where(mask)[0]

            if len(indices) <= self.lag:
                continue

            # Load this game's data
            game_data = dset[indices].astype(np.float32) - mean

            X_t = game_data[:-self.lag]
            X_tau = game_data[self.lag:]

            C_tau += X_t.T @ X_tau
            n_pairs += len(X_t)

        if n_pairs > 0:
            C_tau /= n_pairs

        return C0, C_tau, n_pairs

    def _solve_eigenvalue_problem(
        self,
        C0: np.ndarray,
        C_tau: np.ndarray,
        n_features: int,
    ):
        """
        Solve the generalized eigenvalue problem and store results.

        Args:
            C0: Instantaneous covariance matrix
            C_tau: Time-lagged covariance matrix
            n_features: Number of features
        """
        print("  Solving generalized eigenvalue problem...")

        # Regularize C0 for numerical stability
        C0_reg = C0 + 1e-6 * np.eye(n_features)

        try:
            eigenvalues, eigenvectors = linalg.eigh(C_tau, C0_reg)
        except linalg.LinAlgError as e:
            print(f"  Warning: eigh failed, using svd fallback: {e}")
            C0_inv_sqrt = linalg.sqrtm(linalg.inv(C0_reg))
            M = C0_inv_sqrt @ C_tau @ C0_inv_sqrt
            eigenvalues, V = linalg.eigh(M)
            eigenvectors = C0_inv_sqrt @ V

        # Sort by eigenvalue (descending - slowest first)
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Keep top n_components
        n_keep = min(self.n_components, n_features)
        eigenvalues = eigenvalues[:n_keep]
        eigenvectors = eigenvectors[:, :n_keep]

        # Kinetic variance
        kinetic_variance = eigenvalues ** 2

        self.result_ = TICAResult(
            eigenvectors=eigenvectors.astype(np.float32),
            eigenvalues=eigenvalues.astype(np.float32),
            mean=self.mean_,
            lag=self.lag,
            n_components=n_keep,
            kinetic_variance=kinetic_variance.astype(np.float32),
        )

        print(f"  Top 5 eigenvalues: {eigenvalues[:5]}")
        print(f"  Kinetic variance explained: {kinetic_variance.sum():.4f}")

    def select_n_components_for_variance(self, threshold: float = 0.95) -> int:
        """
        Find minimum components needed for target kinetic variance.

        Per PyEMMA paper (L:240): "95% kinetic variance cutoff"

        Args:
            threshold: Target cumulative variance (default 0.95)

        Returns:
            Number of components needed
        """
        cumvar = self.get_cumulative_kinetic_variance()
        n_components = int(np.searchsorted(cumvar, threshold) + 1)
        return min(n_components, len(cumvar))


def fit_tica_with_variance_cutoff(
    X: np.ndarray,
    game_ids: Optional[np.ndarray] = None,
    lag: int = 1,
    variance_threshold: float = 0.95,
    max_components: int = 1000
) -> Tuple['TICATransformer', int]:
    """
    Fit TICA and automatically select components for target kinetic variance.

    Per PyEMMA paper (L:240): "TICA...yields a four dimensional subspace
    using a 95% kinetic variance cutoff"

    This is the recommended approach - don't use fixed n_components.

    Args:
        X: [n_samples, n_features] feature matrix
        game_ids: [n_samples] optional game indices for boundary handling
        lag: Time lag in steps
        variance_threshold: Target cumulative kinetic variance (default 0.95)
        max_components: Maximum components to compute initially

    Returns:
        tica: Fitted TICATransformer
        n_selected: Number of components for target variance
    """
    n_features = X.shape[1]
    n_initial = min(max_components, n_features)

    print(f"Fitting TICA with {variance_threshold*100:.0f}% kinetic variance cutoff")
    print(f"  Initial components: {n_initial}")

    # Fit with max components first
    tica = TICATransformer(n_components=n_initial, lag=lag)
    tica.fit(X, game_ids)

    # Select components for target variance
    n_selected = tica.select_n_components_for_variance(variance_threshold)
    cumvar = tica.get_cumulative_kinetic_variance()

    print(f"\nVariance analysis:")
    print(f"  Components for {variance_threshold*100:.0f}% variance: {n_selected}")
    print(f"  Actual variance at {n_selected} components: {cumvar[n_selected-1]*100:.2f}%")
    print(f"  Variance at 50 components: {cumvar[min(49, len(cumvar)-1)]*100:.2f}%")

    return tica, n_selected


def compute_implied_timescales_multi_lag(
    X: np.ndarray,
    game_ids: Optional[np.ndarray] = None,
    lags: List[int] = [1, 2, 5, 10],
    n_components: int = 10,
    n_timescales: int = 5
) -> dict:
    """
    Compute implied timescales at multiple lag times for validation.

    Per PyEMMA paper (L:250): "ITS are approximately constant as a function of τ"
    Select smallest τ where ITS converge (plateau).

    Args:
        X: [n_samples, n_features] feature matrix
        game_ids: [n_samples] optional game indices
        lags: List of lag times to test
        n_components: Number of TICA components to compute
        n_timescales: Number of slowest timescales to track

    Returns:
        dict with:
            'lags': list of lag values
            'timescales': [n_lags, n_timescales] array
            'eigenvalues': [n_lags, n_timescales] array
    """
    print(f"Computing implied timescales at lags: {lags}")

    timescales_all = []
    eigenvalues_all = []

    for lag in lags:
        print(f"  Lag {lag}...", end=" ")

        tica = TICATransformer(n_components=n_components, lag=lag)
        tica.fit(X, game_ids)

        # Get eigenvalues (already sorted descending)
        eigs = tica.result_.eigenvalues[:n_timescales]

        # Compute implied timescales: t_i = -τ / ln(λ_i)
        # Only for 0 < λ < 1
        ts = []
        for ev in eigs:
            if 0 < ev < 1:
                ts.append(-lag / np.log(ev))
            else:
                ts.append(np.inf)

        timescales_all.append(ts)
        eigenvalues_all.append(eigs)
        print(f"ITS = {ts[:3]}")

    return {
        'lags': lags,
        'timescales': np.array(timescales_all),
        'eigenvalues': np.array(eigenvalues_all),
    }
