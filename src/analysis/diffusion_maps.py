"""
Diffusion Maps for geometry-aware dimensionality reduction.

Uses pydiffmap for Nyström-accelerated diffusion maps with α=1
(Laplace-Beltrami operator) to recover intrinsic geometry independent
of data density.

Reference:
- docs/manifold_representation/Diffusion_maps.md (Coifman & Lafon)
- α=1: geometry-only, independent of sampling density
- Nyström extension: 2-4x speedup for large datasets

Example:
    >>> from src.analysis.diffusion_maps import DiffusionMapAnalyzer
    >>> analyzer = DiffusionMapAnalyzer(n_components=50, alpha=1.0)
    >>> coords, eigenvalues = analyzer.fit_transform(features)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)

# Check for pydiffmap availability
HAS_PYDIFFMAP = False
try:
    from pydiffmap import diffusion_map as dm
    HAS_PYDIFFMAP = True
except ImportError:
    logger.warning("pydiffmap not installed. Install with: pip install pydiffmap")


@dataclass
class DiffusionMapResult:
    """Result of diffusion map analysis."""
    coordinates: np.ndarray  # [n_samples, n_components]
    eigenvalues: np.ndarray  # [n_components]
    n_components: int
    alpha: float
    epsilon: float
    n_landmarks: Optional[int]
    diffusion_time: int = 1


class DiffusionMapAnalyzer:
    """
    Geometry-aware dimensionality reduction via Diffusion Maps.
    
    Uses α=1 normalization to approximate the Laplace-Beltrami operator,
    recovering intrinsic geometry independent of sampling density.
    
    For large datasets (N > 10k), uses Nyström approximation via landmarks.
    
    Attributes:
        n_components: Number of diffusion coordinates to extract
        alpha: Normalization parameter (1.0 = Laplace-Beltrami)
        epsilon: Kernel bandwidth ('auto' for adaptive)
        n_landmarks: Number of landmarks for Nyström (None = full computation)
    """
    
    def __init__(
        self,
        n_components: int = 50,
        alpha: float = 1.0,
        epsilon: str = 'bgh',
        n_landmarks: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize Diffusion Map analyzer.
        
        Args:
            n_components: Number of diffusion coordinates
            alpha: Normalization (1.0 = Laplace-Beltrami, density-independent)
            epsilon: Kernel bandwidth ('bgh' = auto via Belkin, Graham, Hoffmann)
            n_landmarks: Landmarks for Nyström (None = use all points)
            random_state: For reproducibility of landmark selection
        """
        if not HAS_PYDIFFMAP:
            raise ImportError(
                "pydiffmap required. Install with: pip install pydiffmap"
            )
        
        self.n_components = n_components
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_landmarks = n_landmarks
        self.random_state = random_state
        
        self._dmap = None
        self._landmark_idx = None
        self._result = None
    
    def fit(self, X: np.ndarray) -> 'DiffusionMapAnalyzer':
        """
        Fit diffusion map to data.
        
        For large datasets, uses Nyström approximation via landmarks.
        
        Args:
            X: [n_samples, n_features] feature matrix
            
        Returns:
            Self for chaining
        """
        n_samples = X.shape[0]
        logger.info(f"Fitting diffusion map to {n_samples} samples, {X.shape[1]} features")
        
        # Decide on Nyström vs full computation
        use_nystrom = self.n_landmarks is not None and n_samples > self.n_landmarks
        
        if use_nystrom:
            logger.info(f"Using Nyström with {self.n_landmarks} landmarks")
            self._fit_nystrom(X)
        else:
            logger.info("Using full diffusion map computation")
            self._fit_full(X)
        
        return self
    
    def _fit_full(self, X: np.ndarray):
        """Fit full diffusion map (all pairwise distances)."""
        self._dmap = dm.DiffusionMap.from_sklearn(
            n_evecs=self.n_components,
            epsilon=self.epsilon,
            alpha=self.alpha,
            k=min(200, X.shape[0] - 1)  # k-nearest neighbors
        )
        self._dmap.fit(X)
        self._landmark_idx = None
    
    def _fit_nystrom(self, X: np.ndarray):
        """Fit diffusion map on landmarks for Nyström extension."""
        rng = np.random.default_rng(self.random_state)
        self._landmark_idx = rng.choice(
            len(X), self.n_landmarks, replace=False
        )
        X_landmarks = X[self._landmark_idx]
        
        self._dmap = dm.DiffusionMap.from_sklearn(
            n_evecs=self.n_components,
            epsilon=self.epsilon,
            alpha=self.alpha,
            k=min(200, len(X_landmarks) - 1)
        )
        self._dmap.fit(X_landmarks)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to diffusion coordinates.
        
        Args:
            X: [n_samples, n_features] feature matrix
            
        Returns:
            [n_samples, n_components] diffusion coordinates
        """
        if self._dmap is None:
            raise ValueError("Must call fit() before transform()")
        
        if self._landmark_idx is not None:
            # Nyström extension
            return self._dmap.transform(X)
        else:
            # Already fit on full data
            return self._dmap.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.
        
        Args:
            X: [n_samples, n_features] feature matrix
            
        Returns:
            Tuple of:
                - [n_samples, n_components] diffusion coordinates
                - [n_components] eigenvalues
        """
        self.fit(X)
        coords = self.transform(X)
        eigenvalues = self.eigenvalues
        
        self._result = DiffusionMapResult(
            coordinates=coords,
            eigenvalues=eigenvalues,
            n_components=self.n_components,
            alpha=self.alpha,
            epsilon=float(self._dmap.epsilon) if hasattr(self._dmap, 'epsilon') else 0.0,
            n_landmarks=self.n_landmarks
        )
        
        return coords, eigenvalues
    
    @property
    def raw_eigenvalues(self) -> np.ndarray:
        """Get raw eigenvalues from pydiffmap (log-scaled)."""
        if self._dmap is None:
            raise ValueError("Must call fit() first")
        return self._dmap.evals

    @property
    def eigenvalues(self) -> np.ndarray:
        """
        Get eigenvalues of diffusion transition matrix P.

        Note: pydiffmap returns eigenvalues directly (NOT log-scaled).
        The previous implementation incorrectly applied exp(), which caused
        all eigenvalues to appear as ~1.0 when raw values were already in (0,1].

        According to Coifman & Lafon (2006):
        - Eigenvalues should satisfy 1 = λ₀ > |λ₁| ≥ |λ₂| ≥ ...
        - The trivial eigenvalue λ₀ = 1 is typically excluded by pydiffmap

        Returns:
            [n_components] eigenvalues of P
        """
        if self._dmap is None:
            raise ValueError("Must call fit() first")

        # pydiffmap returns eigenvalues directly (NOT log-scaled)
        # BUG FIX: Previous code incorrectly applied exp() which caused
        # eigenvalues near 0 to become ~1.0, destroying the spectral gap
        eigenvalues = self._dmap.evals.copy()

        # Log diagnostic info on first access
        logger.info(
            f"Raw pydiffmap eigenvalues: min={eigenvalues.min():.6f}, "
            f"max={eigenvalues.max():.6f}, first 5: {eigenvalues[:5]}"
        )

        # Validate results - eigenvalues should be in (0, 1] for proper diffusion map
        # If all eigenvalues are ~0, the kernel bandwidth (epsilon) may be too small
        # If all eigenvalues are ~1, epsilon may be too large (all points connected)
        if np.all(np.abs(eigenvalues) < 1e-6):
            logger.warning(
                "All eigenvalues are ~0. Kernel bandwidth (epsilon) may be too small. "
                "Try increasing epsilon or reducing input dimensionality with PCA."
            )
        elif np.all(eigenvalues > 0.99):
            logger.warning(
                "All eigenvalues are ~1. Kernel bandwidth (epsilon) may be too large "
                "or input dimensionality is too high. Try PCA preprocessing."
            )

        return eigenvalues
    
    @property
    def result(self) -> Optional[DiffusionMapResult]:
        """Get full result object."""
        return self._result
    
    def compute_implied_timescales(self, t: int = 1) -> np.ndarray:
        """
        Convert eigenvalues to implied timescales.

        t_i = -t / ln(λ_i)

        For valid eigenvalues λ ∈ (0, 1):
        - λ close to 1 → large timescale (slow process)
        - λ close to 0 → small timescale (fast process)

        Reference: PyEMMA_Markov_State_modeling.md, L:54-62

        Args:
            t: Diffusion time (lag time)

        Returns:
            [n_components] implied timescales
        """
        eigenvalues = self.eigenvalues

        # Validate eigenvalues are in expected range
        if np.any(eigenvalues <= 0) or np.any(eigenvalues >= 1):
            logger.warning(
                f"Some eigenvalues outside valid range (0,1): "
                f"min={eigenvalues.min():.4e}, max={eigenvalues.max():.4e}. "
                f"Timescales may be inaccurate."
            )

        # Compute implied timescales (handle edge cases)
        with np.errstate(divide='ignore', invalid='ignore'):
            timescales = -t / np.log(eigenvalues)

        # Replace invalid values (NaN, inf) with large finite values
        timescales = np.where(
            np.isfinite(timescales) & (timescales > 0),
            timescales,
            np.inf
        )

        return timescales


def compute_diffusion_map(
    X: np.ndarray,
    n_components: int = 50,
    alpha: float = 1.0,
    n_landmarks: Optional[int] = 5000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for diffusion map computation.
    
    Args:
        X: [n_samples, n_features] feature matrix
        n_components: Number of diffusion coordinates
        alpha: Normalization (1.0 = Laplace-Beltrami)
        n_landmarks: Landmarks for Nyström (None = full)
        random_state: Random seed
        
    Returns:
        Tuple of (coordinates, eigenvalues)
    """
    # Auto-enable Nyström for large datasets
    if n_landmarks is None and len(X) > 10000:
        n_landmarks = min(5000, len(X) // 10)
        logger.info(f"Auto-enabling Nyström with {n_landmarks} landmarks")
    
    analyzer = DiffusionMapAnalyzer(
        n_components=n_components,
        alpha=alpha,
        n_landmarks=n_landmarks,
        random_state=random_state
    )
    return analyzer.fit_transform(X)


def compute_diffusion_map_with_pca(
    X: np.ndarray,
    n_components: int = 50,
    pca_components: int = 100,
    alpha: float = 1.0,
    n_landmarks: Optional[int] = 5000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diffusion map with PCA preprocessing for high-dimensional data.

    High-dimensional data (e.g., 4096-D SAE features) causes problems for
    diffusion maps because:
    1. Distance concentration in high dimensions
    2. Kernel bandwidth estimation becomes unreliable
    3. All eigenvalues tend toward 1.0 (no spectral gap)

    This function applies PCA first to reduce to a manageable dimensionality
    where diffusion map kernels work properly.

    Args:
        X: [n_samples, n_features] feature matrix (can be high-dimensional)
        n_components: Number of diffusion coordinates to extract
        pca_components: Number of PCA components for preprocessing
        alpha: Normalization (1.0 = Laplace-Beltrami)
        n_landmarks: Landmarks for Nyström (None = full)
        random_state: Random seed

    Returns:
        Tuple of (diffusion_coordinates, eigenvalues, pca_explained_variance_ratio)
    """
    from sklearn.decomposition import PCA

    n_samples, n_features = X.shape
    logger.info(f"Diffusion map with PCA: {n_samples} samples, {n_features} features")

    # Apply PCA preprocessing if input is high-dimensional
    if n_features > pca_components:
        logger.info(f"Applying PCA: {n_features} → {pca_components} dimensions")
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_pca = pca.fit_transform(X)
        pca_variance = pca.explained_variance_ratio_
        logger.info(f"PCA variance explained: {pca_variance.sum():.2%}")
    else:
        X_pca = X
        pca_variance = np.ones(n_features)  # All variance retained

    # Now run diffusion map on reduced data
    coords, eigenvalues = compute_diffusion_map(
        X_pca,
        n_components=n_components,
        alpha=alpha,
        n_landmarks=n_landmarks,
        random_state=random_state
    )

    return coords, eigenvalues, pca_variance


# Fallback implementation using scipy (no pydiffmap dependency)
def compute_diffusion_map_scipy(
    X: np.ndarray,
    n_components: int = 50,
    alpha: float = 1.0,
    epsilon: Optional[float] = None,
    k_neighbors: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scipy-based diffusion map (fallback when pydiffmap unavailable).
    
    This is slower and doesn't support Nyström, but works without
    additional dependencies.
    
    Args:
        X: [n_samples, n_features] feature matrix
        n_components: Number of diffusion coordinates
        alpha: Normalization parameter
        epsilon: Kernel bandwidth (auto if None)
        k_neighbors: Number of neighbors for sparse kernel
        
    Returns:
        Tuple of (coordinates, eigenvalues)
    """
    from scipy.spatial.distance import cdist
    from scipy.sparse.linalg import eigsh
    from scipy import sparse as sp
    from sklearn.neighbors import NearestNeighbors
    
    n_samples = X.shape[0]
    logger.info(f"Computing diffusion map via scipy ({n_samples} samples)")
    
    # Adaptive epsilon based on median k-th neighbor distance
    if epsilon is None:
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X)
        distances, _ = nbrs.kneighbors(X)
        epsilon = np.median(distances[:, -1]) ** 2
        logger.info(f"Auto epsilon: {epsilon:.4f}")
    
    # Compute sparse kernel (k-nearest neighbors)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X)
    distances_sq, indices = nbrs.kneighbors(X)
    distances_sq = distances_sq ** 2
    
    # Build sparse kernel matrix
    from scipy.sparse import lil_matrix
    K = lil_matrix((n_samples, n_samples))
    
    for i in range(n_samples):
        for j_idx, j in enumerate(indices[i]):
            K[i, j] = np.exp(-distances_sq[i, j_idx] / epsilon)
    
    K = K.tocsr()
    K = (K + K.T) / 2  # Symmetrize
    
    # Alpha normalization
    d = np.array(K.sum(axis=1)).flatten()
    d_alpha = d ** (-alpha)
    D_alpha = sp.diags(d_alpha)
    K_alpha = D_alpha @ K @ D_alpha
    
    # Row-stochastic normalization
    d_new = np.array(K_alpha.sum(axis=1)).flatten()
    D_inv = sp.diags(1.0 / d_new)
    P = D_inv @ K_alpha
    
    # Eigendecomposition
    eigenvalues, eigenvectors = eigsh(P, k=n_components + 1, which='LM')
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Skip trivial eigenvector
    return eigenvectors[:, 1:n_components+1], eigenvalues[1:n_components+1]
