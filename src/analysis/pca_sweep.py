"""
PCA Component Sweep for Diffusion Map Optimization.

Tests various N_PCA_COMPONENTS to find optimal dimensionality
that maximizes diffusion map eigenvalue quality (spectral gap).

Reference: Berry & Harlim (2016) Variable bandwidth diffusion kernels.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional imports
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.decomposition import PCA, IncrementalPCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class PCASweepResult:
    """Result of PCA component sweep analysis.
    
    Attributes:
        n_components_tested: List of PCA component counts tested
        variance_explained: Cumulative variance explained at each level
        dm_eigenvalues: DM eigenvalues for each PCA level [n_levels, n_dm_components]
        spectral_gaps: Spectral gap (λ₁ - λ₂) at each level
        first_eigenvalues: First non-trivial eigenvalue at each level
        recommended_components: Optimal n_components based on spectral gap
        recommendation_reason: Text explanation of recommendation
    """
    n_components_tested: List[int]
    variance_explained: List[float]
    dm_eigenvalues: List[np.ndarray]
    spectral_gaps: List[float]
    first_eigenvalues: List[float]
    recommended_components: int
    recommendation_reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'n_components_tested': self.n_components_tested,
            'variance_explained': self.variance_explained,
            'spectral_gaps': self.spectral_gaps,
            'first_eigenvalues': self.first_eigenvalues,
            'recommended_components': self.recommended_components,
            'recommendation_reason': self.recommendation_reason,
        }


@dataclass
class SpectralValidation:
    """Result of spectral quality validation.
    
    Success criteria from Coifman & Lafon (2006):
    - First eigenvalue should be close to 1 (slow mode exists)
    - Spectral gap indicates metastable structure
    """
    valid: bool
    first_eigenvalue: float
    spectral_gap: float
    n_valid_eigenvalues: int  # Count of eigenvalues > 0
    message: str


class PCASweepAnalyzer:
    """Analyzer for finding optimal PCA dimensionality for Diffusion Maps.
    
    The curse of dimensionality causes diffusion map kernels to underflow
    in high dimensions. This analyzer tests various PCA component counts
    to find the optimal balance between:
    
    1. Variance explained (higher is better)
    2. DM eigenvalue quality (spectral gap indicates metastable structure)
    
    Usage:
        analyzer = PCASweepAnalyzer(n_dm_components=10, alpha=0.5)
        result = analyzer.sweep(data, component_range=[10, 20, 30, 50])
        analyzer.visualize_sweep(result, save_path='pca_sweep.png')
    """
    
    def __init__(
        self,
        n_dm_components: int = 10,
        alpha: float = 0.5,
        epsilon: str = 'bgh',
        n_landmarks: int = 5000,
        random_state: int = 42,
        min_spectral_gap: float = 0.05,
        min_first_eigenvalue: float = 0.8,
    ):
        """
        Initialize PCA sweep analyzer.
        
        Args:
            n_dm_components: Number of diffusion map components to compute
            alpha: DM normalization (0.5 = Fokker-Planck, 1.0 = Laplace-Beltrami)
            epsilon: Bandwidth selection ('bgh' for Berry-Harlim adaptive)
            n_landmarks: Number of landmarks for Nyström approximation
            random_state: Random seed for reproducibility
            min_spectral_gap: Minimum acceptable spectral gap
            min_first_eigenvalue: Minimum acceptable first eigenvalue
        """
        self.n_dm_components = n_dm_components
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_landmarks = n_landmarks
        self.random_state = random_state
        self.min_spectral_gap = min_spectral_gap
        self.min_first_eigenvalue = min_first_eigenvalue
        
    def sweep(
        self,
        data: np.ndarray,
        component_range: List[int] = [10, 20, 30, 50, 100],
        verbose: bool = True,
    ) -> PCASweepResult:
        """
        Test DM with each PCA component count and return eigenvalue quality.
        
        Args:
            data: [n_samples, n_features] raw feature matrix
            component_range: List of PCA component counts to test
            verbose: Print progress
            
        Returns:
            PCASweepResult with eigenvalue analysis for each component count
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for PCA sweep")
            
        # Import here to avoid circular dependency
        from .diffusion_maps import DiffusionMapAnalyzer
        
        # Filter component_range to valid values
        max_components = min(data.shape[0], data.shape[1])
        component_range = [c for c in component_range if c <= max_components]
        
        if verbose:
            logger.info(f"PCA Sweep: Testing {component_range} components on {data.shape}")
        
        # Select landmarks
        n_landmarks = min(self.n_landmarks, data.shape[0])
        rng = np.random.RandomState(self.random_state)
        landmark_idx = rng.choice(data.shape[0], n_landmarks, replace=False)
        landmark_data = data[landmark_idx]
        
        results = {
            'variance_explained': [],
            'dm_eigenvalues': [],
            'spectral_gaps': [],
            'first_eigenvalues': [],
        }
        
        for n_components in component_range:
            if verbose:
                logger.info(f"  Testing n_components={n_components}")
            
            # Fit PCA
            pca = PCA(n_components=n_components, random_state=self.random_state)
            pca_coords = pca.fit_transform(landmark_data)
            variance_explained = pca.explained_variance_ratio_.sum()
            
            # Fit Diffusion Map
            dm_analyzer = DiffusionMapAnalyzer(
                n_components=self.n_dm_components,
                alpha=self.alpha,
                epsilon=self.epsilon,
                random_state=self.random_state,
            )
            
            try:
                dm_analyzer.fit(pca_coords)
                eigenvalues = dm_analyzer.eigenvalues
                
                # Compute spectral gap
                if len(eigenvalues) >= 2:
                    spectral_gap = eigenvalues[0] - eigenvalues[1]
                    first_eigenvalue = eigenvalues[0]
                else:
                    spectral_gap = 0.0
                    first_eigenvalue = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
                    
            except Exception as e:
                logger.warning(f"DM failed for n_components={n_components}: {e}")
                eigenvalues = np.zeros(self.n_dm_components)
                spectral_gap = 0.0
                first_eigenvalue = 0.0
            
            results['variance_explained'].append(variance_explained)
            results['dm_eigenvalues'].append(eigenvalues)
            results['spectral_gaps'].append(spectral_gap)
            results['first_eigenvalues'].append(first_eigenvalue)
            
            if verbose:
                logger.info(
                    f"    Variance: {variance_explained:.2%}, "
                    f"λ₁: {first_eigenvalue:.4f}, "
                    f"Gap: {spectral_gap:.4f}"
                )
        
        # Find optimal components
        recommended, reason = self._find_optimal_components(
            component_range, results
        )
        
        return PCASweepResult(
            n_components_tested=component_range,
            variance_explained=results['variance_explained'],
            dm_eigenvalues=results['dm_eigenvalues'],
            spectral_gaps=results['spectral_gaps'],
            first_eigenvalues=results['first_eigenvalues'],
            recommended_components=recommended,
            recommendation_reason=reason,
        )
    
    def _find_optimal_components(
        self,
        component_range: List[int],
        results: Dict[str, List],
    ) -> Tuple[int, str]:
        """Find optimal n_components based on spectral gap."""
        
        # First check: any valid configurations?
        valid_indices = [
            i for i, (gap, ev1) in enumerate(
                zip(results['spectral_gaps'], results['first_eigenvalues'])
            )
            if gap >= self.min_spectral_gap and ev1 >= self.min_first_eigenvalue
        ]
        
        if valid_indices:
            # Pick the one with highest spectral gap among valid
            best_idx = max(valid_indices, key=lambda i: results['spectral_gaps'][i])
            return (
                component_range[best_idx],
                f"Best spectral gap ({results['spectral_gaps'][best_idx]:.4f}) "
                f"with λ₁={results['first_eigenvalues'][best_idx]:.4f}"
            )
        
        # No valid configurations - pick best spectral gap anyway
        best_idx = np.argmax(results['spectral_gaps'])
        gap = results['spectral_gaps'][best_idx]
        ev1 = results['first_eigenvalues'][best_idx]
        
        return (
            component_range[best_idx],
            f"WARNING: No config meets criteria. Best: gap={gap:.4f}, λ₁={ev1:.4f}. "
            f"Consider increasing n_landmarks or checking data quality."
        )
    
    def validate_spectral_quality(
        self,
        eigenvalues: np.ndarray,
    ) -> SpectralValidation:
        """
        Validate eigenvalue spectrum quality.
        
        Success criteria from Coifman & Lafon (2006):
        1. First eigenvalue λ₁ > min_first_eigenvalue (meaningful slow mode)
        2. Spectral gap (λ₁ - λ₂) > min_spectral_gap (metastable structure)
        3. No negative eigenvalues (numerical stability)
        
        Args:
            eigenvalues: Array of DM eigenvalues
            
        Returns:
            SpectralValidation with validity status and diagnostics
        """
        if len(eigenvalues) < 2:
            return SpectralValidation(
                valid=False,
                first_eigenvalue=eigenvalues[0] if len(eigenvalues) > 0 else 0.0,
                spectral_gap=0.0,
                n_valid_eigenvalues=len(eigenvalues),
                message="Insufficient eigenvalues computed"
            )
        
        first_ev = eigenvalues[0]
        spectral_gap = eigenvalues[0] - eigenvalues[1]
        n_valid = np.sum(eigenvalues > 0)
        
        # Check criteria
        issues = []
        if first_ev < self.min_first_eigenvalue:
            issues.append(f"λ₁={first_ev:.4f} < {self.min_first_eigenvalue}")
        if spectral_gap < self.min_spectral_gap:
            issues.append(f"gap={spectral_gap:.4f} < {self.min_spectral_gap}")
        if np.any(eigenvalues < -1e-10):  # Small tolerance for numerical noise
            issues.append("Negative eigenvalues detected (numerical instability)")
        
        valid = len(issues) == 0
        message = "Spectral quality OK" if valid else "; ".join(issues)
        
        return SpectralValidation(
            valid=valid,
            first_eigenvalue=first_ev,
            spectral_gap=spectral_gap,
            n_valid_eigenvalues=n_valid,
            message=message,
        )
    
    def visualize_sweep(
        self,
        result: PCASweepResult,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
    ):
        """
        Create 2x2 visualization of PCA sweep results.
        
        Panels:
        - Top-left: Variance explained vs n_components
        - Top-right: First DM eigenvalue vs n_components
        - Bottom-left: Spectral gap vs n_components
        - Bottom-right: Eigenvalue spectrum for recommended n_components
        
        Args:
            result: PCASweepResult from sweep()
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        n_comps = result.n_components_tested
        rec_idx = n_comps.index(result.recommended_components)
        
        # Top-left: Variance explained
        ax = axes[0, 0]
        ax.plot(n_comps, result.variance_explained, 'o-', linewidth=2, markersize=8)
        ax.axvline(result.recommended_components, color='green', linestyle='--', 
                   label=f'Recommended: {result.recommended_components}')
        ax.set_xlabel('N PCA Components')
        ax.set_ylabel('Cumulative Variance Explained')
        ax.set_title('PCA Variance Explained')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Top-right: First eigenvalue
        ax = axes[0, 1]
        ax.plot(n_comps, result.first_eigenvalues, 'o-', linewidth=2, markersize=8)
        ax.axhline(self.min_first_eigenvalue, color='red', linestyle='--', 
                   label=f'Threshold: {self.min_first_eigenvalue}')
        ax.axvline(result.recommended_components, color='green', linestyle='--')
        ax.set_xlabel('N PCA Components')
        ax.set_ylabel('First DM Eigenvalue (λ₁)')
        ax.set_title('DM First Eigenvalue Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom-left: Spectral gap
        ax = axes[1, 0]
        ax.plot(n_comps, result.spectral_gaps, 'o-', linewidth=2, markersize=8)
        ax.axhline(self.min_spectral_gap, color='red', linestyle='--',
                   label=f'Threshold: {self.min_spectral_gap}')
        ax.axvline(result.recommended_components, color='green', linestyle='--')
        ax.set_xlabel('N PCA Components')
        ax.set_ylabel('Spectral Gap (λ₁ - λ₂)')
        ax.set_title('Spectral Gap (Metastability Indicator)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom-right: Eigenvalue spectrum at recommended
        ax = axes[1, 1]
        eigenvalues = result.dm_eigenvalues[rec_idx]
        ax.bar(range(len(eigenvalues)), eigenvalues, alpha=0.7)
        ax.axhline(self.min_first_eigenvalue, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(f'DM Eigenspectrum (n_pca={result.recommended_components})')
        ax.grid(True, alpha=0.3)
        
        # Add recommendation text
        fig.suptitle(
            f'PCA Sweep Results\n{result.recommendation_reason}',
            fontsize=12, y=1.02
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved PCA sweep visualization to {save_path}")
        
        return fig, axes


def run_pca_sweep(
    data: np.ndarray,
    component_range: List[int] = [10, 20, 30, 50],
    n_landmarks: int = 5000,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
) -> PCASweepResult:
    """
    Convenience function for PCA sweep analysis.
    
    Args:
        data: [n_samples, n_features] feature matrix
        component_range: PCA component counts to test
        n_landmarks: Number of landmarks for DM
        alpha: DM normalization parameter
        save_path: Path to save visualization
        
    Returns:
        PCASweepResult with recommendations
    """
    analyzer = PCASweepAnalyzer(
        n_landmarks=n_landmarks,
        alpha=alpha,
    )
    
    result = analyzer.sweep(data, component_range)
    
    if save_path and HAS_MATPLOTLIB:
        analyzer.visualize_sweep(result, save_path)
    
    return result


# =============================================================================
# LANDMARK SELECTION STRATEGIES
# =============================================================================

def select_landmarks_uniform(
    n_samples: int,
    n_landmarks: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Select landmarks via uniform random sampling.
    
    Simple and fast. Good for temporally-ordered data where
    we want to preserve coverage across all timesteps.
    
    Args:
        n_samples: Total number of samples
        n_landmarks: Number of landmarks to select
        random_state: Random seed
        
    Returns:
        Array of landmark indices
    """
    rng = np.random.RandomState(random_state)
    return rng.choice(n_samples, min(n_landmarks, n_samples), replace=False)


def select_landmarks_temporal(
    n_samples: int,
    n_landmarks: int,
) -> np.ndarray:
    """
    Select landmarks via uniform temporal spacing.
    
    Ensures coverage across entire sequence. Best for
    trajectory data where early/middle/late phases matter.
    
    Args:
        n_samples: Total number of samples
        n_landmarks: Number of landmarks to select
        
    Returns:
        Array of landmark indices (evenly spaced)
    """
    return np.linspace(0, n_samples - 1, min(n_landmarks, n_samples), dtype=int)


def select_landmarks_kmeans(
    data: np.ndarray,
    n_landmarks: int,
    random_state: int = 42,
    max_iter: int = 10,
) -> np.ndarray:
    """
    Select landmarks via k-means++ initialization.
    
    Better geometric coverage - ensures rare/edge strategies
    are represented, not just densely sampled regions.
    
    Trade-offs vs uniform:
    - Pro: Better coverage of rare states (edges of manifold)
    - Pro: More informative landmarks for geometry
    - Con: O(n * k) complexity vs O(1) for uniform
    - Con: May oversample outliers/noise
    - Con: Destroys temporal ordering
    
    Args:
        data: [n_samples, n_features] feature matrix
        n_landmarks: Number of landmarks to select
        random_state: Random seed
        max_iter: Max iterations for k-means initialization
        
    Returns:
        Array of landmark indices
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for k-means++ landmark selection")
    
    from sklearn.cluster import MiniBatchKMeans
    
    n_landmarks = min(n_landmarks, data.shape[0])
    
    # Use MiniBatchKMeans for efficiency
    kmeans = MiniBatchKMeans(
        n_clusters=n_landmarks,
        random_state=random_state,
        max_iter=max_iter,
        n_init=1,
        init='k-means++',
        batch_size=min(1024, data.shape[0]),
    )
    kmeans.fit(data)
    
    # Find nearest data point to each cluster center
    from scipy.spatial.distance import cdist
    distances = cdist(kmeans.cluster_centers_, data)
    landmark_indices = np.argmin(distances, axis=1)
    
    # Remove duplicates
    landmark_indices = np.unique(landmark_indices)
    
    return landmark_indices


@dataclass
class LandmarkComparisonResult:
    """Result of comparing landmark selection strategies."""
    strategy_names: List[str]
    n_landmarks_actual: List[int]
    dm_eigenvalues: List[np.ndarray]
    spectral_gaps: List[float]
    first_eigenvalues: List[float]
    computation_times: List[float]
    best_strategy: str
    summary: str


def compare_landmark_strategies(
    data: np.ndarray,
    n_landmarks: int = 5000,
    n_pca_components: int = 20,
    n_dm_components: int = 5,
    alpha: float = 0.5,
    random_state: int = 42,
    verbose: bool = True,
) -> LandmarkComparisonResult:
    """
    Compare uniform vs k-means++ landmark selection strategies.
    
    Runs diffusion maps with each strategy and compares eigenvalue quality.
    
    Args:
        data: [n_samples, n_features] feature matrix
        n_landmarks: Target number of landmarks
        n_pca_components: PCA components for preprocessing
        n_dm_components: Number of DM components
        alpha: DM normalization parameter
        random_state: Random seed
        verbose: Print progress
        
    Returns:
        LandmarkComparisonResult with comparison metrics
    """
    import time
    from .diffusion_maps import DiffusionMapAnalyzer
    
    strategies = {
        'uniform': lambda: select_landmarks_uniform(len(data), n_landmarks, random_state),
        'temporal': lambda: select_landmarks_temporal(len(data), n_landmarks),
        'kmeans++': lambda: select_landmarks_kmeans(data, n_landmarks, random_state),
    }
    
    results = {
        'names': [],
        'n_actual': [],
        'eigenvalues': [],
        'spectral_gaps': [],
        'first_eigenvalues': [],
        'times': [],
    }
    
    for name, select_fn in strategies.items():
        if verbose:
            logger.info(f"Testing {name} landmark selection...")
        
        t0 = time.time()
        
        try:
            # Select landmarks
            landmark_idx = select_fn()
            landmark_data = data[landmark_idx]
            
            # PCA preprocessing
            pca = PCA(n_components=n_pca_components, random_state=random_state)
            pca_data = pca.fit_transform(landmark_data)
            
            # Fit DM
            dm = DiffusionMapAnalyzer(
                n_components=n_dm_components,
                alpha=alpha,
                random_state=random_state,
            )
            dm.fit(pca_data)
            eigenvalues = dm.eigenvalues
            
            spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) >= 2 else 0.0
            first_ev = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Strategy {name} failed: {e}")
            landmark_idx = np.array([])
            eigenvalues = np.zeros(n_dm_components)
            spectral_gap = 0.0
            first_ev = 0.0
        
        elapsed = time.time() - t0
        
        results['names'].append(name)
        results['n_actual'].append(len(landmark_idx))
        results['eigenvalues'].append(eigenvalues)
        results['spectral_gaps'].append(spectral_gap)
        results['first_eigenvalues'].append(first_ev)
        results['times'].append(elapsed)
        
        if verbose:
            logger.info(f"  {name}: n={len(landmark_idx)}, λ₁={first_ev:.4f}, "
                       f"gap={spectral_gap:.4f}, time={elapsed:.2f}s")
    
    # Find best strategy
    best_idx = np.argmax(results['spectral_gaps'])
    best_strategy = results['names'][best_idx]
    
    summary = (
        f"Best strategy: {best_strategy} "
        f"(gap={results['spectral_gaps'][best_idx]:.4f}, "
        f"λ₁={results['first_eigenvalues'][best_idx]:.4f})"
    )
    
    return LandmarkComparisonResult(
        strategy_names=results['names'],
        n_landmarks_actual=results['n_actual'],
        dm_eigenvalues=results['eigenvalues'],
        spectral_gaps=results['spectral_gaps'],
        first_eigenvalues=results['first_eigenvalues'],
        computation_times=results['times'],
        best_strategy=best_strategy,
        summary=summary,
    )


def visualize_landmark_comparison(
    result: LandmarkComparisonResult,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
):
    """
    Visualize comparison of landmark selection strategies.
    
    Args:
        result: LandmarkComparisonResult from compare_landmark_strategies()
        save_path: Path to save figure
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    names = result.strategy_names
    x = np.arange(len(names))
    
    # Panel 1: First eigenvalue
    ax = axes[0]
    bars = ax.bar(x, result.first_eigenvalues, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.axhline(0.8, color='red', linestyle='--', label='Threshold (0.8)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('First Eigenvalue (λ₁)')
    ax.set_title('Eigenvalue Quality')
    ax.legend()
    
    # Highlight best
    best_idx = result.strategy_names.index(result.best_strategy)
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(3)
    
    # Panel 2: Spectral gap
    ax = axes[1]
    bars = ax.bar(x, result.spectral_gaps, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.axhline(0.05, color='red', linestyle='--', label='Threshold (0.05)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Spectral Gap (λ₁ - λ₂)')
    ax.set_title('Metastability Indicator')
    ax.legend()
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(3)
    
    # Panel 3: Computation time
    ax = axes[2]
    ax.bar(x, result.computation_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Cost')
    
    fig.suptitle(f'Landmark Selection Comparison\n{result.summary}', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved landmark comparison to {save_path}")
    
    return fig, axes

