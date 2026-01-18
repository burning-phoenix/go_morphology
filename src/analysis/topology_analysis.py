"""
Topological Data Analysis for SAE Feature Space.

Computes persistent homology to detect topological features (loops, voids)
in the strategic landscape. H1 loops may correspond to "strategic cycles".

Reference:
- MASTER_IMPLEMENTATION_PLAN.md: Lines 591-763 (NB07 spec)
- docs/topological_data_analysis/INDEX.md: Lines 147-168 (Vietoris-Rips)
- docs/topological_data_analysis/INDEX.md: Lines 221-242 (persistence)
- docs/topological_data_analysis/INDEX.md: Lines 409-431 (attractors)

Key concepts:
- Vietoris-Rips filtration: Grows balls around points, tracks topology changes
- H0: Connected components (clusters)
- H1: Loops (1-dimensional holes)
- H2: Voids (2-dimensional cavities)
- Persistence: (birth, death) pairs showing when features appear/disappear
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import time


@dataclass
class PersistenceResult:
    """Results from persistent homology computation."""
    diagrams: Dict[int, np.ndarray]   # dim -> [n_features, 2] (birth, death)
    max_dim: int                      # Maximum homology dimension computed
    max_edge: float                   # Filtration threshold used
    computation_time: float           # Seconds for ripser computation
    n_points: int                     # Number of points used


@dataclass
class TopologyResult:
    """Complete topology analysis with significance testing."""
    observed: PersistenceResult
    null_distribution: List[PersistenceResult]  # From shuffled surrogates
    p_values: Dict[int, float]        # Per-dimension p-values
    significant_features: Dict[int, np.ndarray]  # dim -> features passing threshold
    n_shuffles: int


def compute_persistent_homology(
    point_cloud: np.ndarray,
    max_dim: int = 2,
    max_edge: Optional[float] = None,
    subsample: int = 2000,
    seed: int = 42,
) -> PersistenceResult:
    """
    Compute Vietoris-Rips persistence diagram using ripser.

    Args:
        point_cloud: [N, D] points in feature space
        max_dim: Maximum homology dimension (0, 1, or 2)
        max_edge: Maximum edge length (None = auto from data)
        subsample: Max points to use (ripser is O(n^3))
        seed: Random seed for subsampling

    Returns:
        PersistenceResult with persistence diagrams
    """
    try:
        import ripser
    except ImportError:
        raise ImportError("ripser not installed. Run: pip install ripser")

    n_points, n_dims = point_cloud.shape
    print(f"Computing persistent homology: {n_points:,} points × {n_dims} dims")
    print(f"  max_dim: {max_dim}")

    # Subsample if necessary (ripser is O(n^3))
    if n_points > subsample:
        print(f"  Subsampling to {subsample} points")
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_points, subsample, replace=False)
        point_cloud = point_cloud[indices]
        n_points = subsample

    # Auto-compute max_edge if not specified
    if max_edge is None:
        # Use 95th percentile of pairwise distances
        from scipy.spatial.distance import pdist
        if n_points <= 5000:
            dists = pdist(point_cloud)
            max_edge = np.percentile(dists, 95)
        else:
            # Sample distances for large point clouds
            sample_size = 5000
            rng = np.random.default_rng(seed)
            sample_indices = rng.choice(n_points, sample_size, replace=False)
            dists = pdist(point_cloud[sample_indices])
            max_edge = np.percentile(dists, 95)
        print(f"  auto max_edge: {max_edge:.4f}")

    # Compute persistence
    start_time = time.time()
    result = ripser.ripser(
        point_cloud,
        maxdim=max_dim,
        thresh=max_edge,
    )
    computation_time = time.time() - start_time

    print(f"  Computation time: {computation_time:.2f}s")

    # Extract diagrams
    diagrams = {}
    for dim in range(max_dim + 1):
        dgm = result['dgms'][dim]
        # Filter out infinite persistence (birth at infinity)
        finite_mask = np.isfinite(dgm[:, 1])
        diagrams[dim] = dgm[finite_mask]
        print(f"  H{dim}: {len(diagrams[dim])} features")

    return PersistenceResult(
        diagrams=diagrams,
        max_dim=max_dim,
        max_edge=max_edge,
        computation_time=computation_time,
        n_points=n_points,
    )


def shuffled_null_distribution(
    point_cloud: np.ndarray,
    n_shuffles: int = 100,
    max_dim: int = 2,
    subsample: int = 2000,
    seed: int = 42,
) -> List[PersistenceResult]:
    """
    Generate null distribution via coordinate shuffling.

    Shuffles each dimension independently to destroy structure
    while preserving marginal distributions.

    Args:
        point_cloud: [N, D] points in feature space
        n_shuffles: Number of shuffled versions
        max_dim: Maximum homology dimension
        subsample: Max points per computation
        seed: Base random seed

    Returns:
        List of PersistenceResult for shuffled data
    """
    print(f"Generating null distribution: {n_shuffles} shuffles")

    rng = np.random.default_rng(seed)
    null_results = []

    for i in range(n_shuffles):
        # Shuffle each dimension independently
        shuffled = point_cloud.copy()
        for d in range(point_cloud.shape[1]):
            rng.shuffle(shuffled[:, d])

        # Compute persistence (suppress output)
        result = compute_persistent_homology(
            shuffled,
            max_dim=max_dim,
            subsample=subsample,
            seed=seed + i,
        )
        null_results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_shuffles}")

    return null_results


def compute_significance(
    observed: PersistenceResult,
    null: List[PersistenceResult],
    dimension: int = 1,
    persistence_threshold: float = 0.1,
) -> Tuple[float, np.ndarray]:
    """
    Compute p-value for observed features vs null distribution.

    Tests whether the observed number of high-persistence features
    is significantly greater than expected by chance.

    Args:
        observed: PersistenceResult for real data
        null: List of PersistenceResult for shuffled data
        dimension: Which homology dimension to test (usually 1 for loops)
        persistence_threshold: Minimum persistence (death - birth) to count

    Returns:
        (p_value, significant_features) where significant_features
        is [n_features, 2] array of (birth, death) pairs
    """
    # Count features above threshold in observed data
    obs_diagram = observed.diagrams.get(dimension, np.empty((0, 2)))
    obs_persistence = obs_diagram[:, 1] - obs_diagram[:, 0]
    obs_count = np.sum(obs_persistence > persistence_threshold)

    # Count features in null distribution
    null_counts = []
    for null_result in null:
        null_diagram = null_result.diagrams.get(dimension, np.empty((0, 2)))
        null_persistence = null_diagram[:, 1] - null_diagram[:, 0]
        null_counts.append(np.sum(null_persistence > persistence_threshold))

    null_counts = np.array(null_counts)

    # One-tailed p-value: P(null >= observed)
    p_value = np.mean(null_counts >= obs_count)

    # Extract significant features
    significant_mask = obs_persistence > persistence_threshold
    significant_features = obs_diagram[significant_mask]

    print(f"H{dimension} significance test:")
    print(f"  Observed count: {obs_count}")
    print(f"  Null mean: {np.mean(null_counts):.1f} +/- {np.std(null_counts):.1f}")
    print(f"  p-value: {p_value:.4f}")

    return p_value, significant_features


def compute_persistence_stats(diagram: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for a persistence diagram.

    Args:
        diagram: [n_features, 2] array of (birth, death) pairs

    Returns:
        Dict with 'n_features', 'mean_persistence', 'max_persistence', etc.
    """
    if len(diagram) == 0:
        return {
            'n_features': 0,
            'mean_persistence': 0.0,
            'max_persistence': 0.0,
            'total_persistence': 0.0,
            'mean_birth': 0.0,
            'mean_death': 0.0,
        }

    persistence = diagram[:, 1] - diagram[:, 0]

    return {
        'n_features': len(diagram),
        'mean_persistence': float(np.mean(persistence)),
        'max_persistence': float(np.max(persistence)),
        'total_persistence': float(np.sum(persistence)),
        'mean_birth': float(np.mean(diagram[:, 0])),
        'mean_death': float(np.mean(diagram[:, 1])),
    }


class TopologyAnalyzer:
    """
    Complete topology analysis pipeline.

    Computes persistent homology, generates null distribution,
    and tests for significant topological features.

    Usage:
        analyzer = TopologyAnalyzer(subsample_size=2000)
        result = analyzer.analyze(point_cloud, n_shuffles=100)

        # Check significance
        if result.p_values[1] < 0.05:
            print("Significant H1 loops detected!")

        # Compare conditions
        comparison = analyzer.compare_conditions(pc1, pc2)
    """

    def __init__(
        self,
        subsample_size: int = 2000,
        max_dim: int = 2,
        persistence_threshold: float = 0.1,
    ):
        """
        Args:
            subsample_size: Maximum points for ripser computation
            max_dim: Maximum homology dimension (0, 1, or 2)
            persistence_threshold: Minimum persistence for significance
        """
        self.subsample_size = subsample_size
        self.max_dim = max_dim
        self.persistence_threshold = persistence_threshold

    def analyze(
        self,
        point_cloud: np.ndarray,
        n_shuffles: int = 100,
        seed: int = 42,
    ) -> TopologyResult:
        """
        Complete topology analysis with significance testing.

        Args:
            point_cloud: [N, D] points in feature space
            n_shuffles: Number of shuffles for null distribution
            seed: Random seed

        Returns:
            TopologyResult with observed, null, and p-values
        """
        print("=" * 60)
        print("TOPOLOGY ANALYSIS")
        print("=" * 60)

        # Compute observed persistence
        print("\n1. Computing observed persistence...")
        observed = compute_persistent_homology(
            point_cloud,
            max_dim=self.max_dim,
            subsample=self.subsample_size,
            seed=seed,
        )

        # Generate null distribution
        print("\n2. Generating null distribution...")
        null_distribution = shuffled_null_distribution(
            point_cloud,
            n_shuffles=n_shuffles,
            max_dim=self.max_dim,
            subsample=self.subsample_size,
            seed=seed,
        )

        # Compute significance for each dimension
        print("\n3. Computing significance...")
        p_values = {}
        significant_features = {}

        for dim in range(self.max_dim + 1):
            p_val, sig_feat = compute_significance(
                observed,
                null_distribution,
                dimension=dim,
                persistence_threshold=self.persistence_threshold,
            )
            p_values[dim] = p_val
            significant_features[dim] = sig_feat

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for dim in range(self.max_dim + 1):
            status = "SIGNIFICANT" if p_values[dim] < 0.05 else "not significant"
            n_sig = len(significant_features[dim])
            print(f"  H{dim}: p={p_values[dim]:.4f} [{status}], {n_sig} significant features")

        return TopologyResult(
            observed=observed,
            null_distribution=null_distribution,
            p_values=p_values,
            significant_features=significant_features,
            n_shuffles=n_shuffles,
        )

    def compare_conditions(
        self,
        point_cloud_1: np.ndarray,
        point_cloud_2: np.ndarray,
        labels: Tuple[str, str] = ('Condition 1', 'Condition 2'),
        seed: int = 42,
    ) -> Dict:
        """
        Compare topological features between two conditions.

        Args:
            point_cloud_1: [N1, D] first condition
            point_cloud_2: [N2, D] second condition
            labels: Names for the two conditions
            seed: Random seed

        Returns:
            Dict with comparison statistics
        """
        print(f"Comparing topology: {labels[0]} vs {labels[1]}")

        # Compute persistence for both
        result_1 = compute_persistent_homology(
            point_cloud_1,
            max_dim=self.max_dim,
            subsample=self.subsample_size,
            seed=seed,
        )

        result_2 = compute_persistent_homology(
            point_cloud_2,
            max_dim=self.max_dim,
            subsample=self.subsample_size,
            seed=seed + 1,
        )

        # Compare each dimension
        comparison = {
            'labels': labels,
            'dimensions': {},
        }

        for dim in range(self.max_dim + 1):
            stats_1 = compute_persistence_stats(result_1.diagrams.get(dim, np.empty((0, 2))))
            stats_2 = compute_persistence_stats(result_2.diagrams.get(dim, np.empty((0, 2))))

            comparison['dimensions'][dim] = {
                labels[0]: stats_1,
                labels[1]: stats_2,
                'diff_n_features': stats_2['n_features'] - stats_1['n_features'],
                'diff_total_persistence': stats_2['total_persistence'] - stats_1['total_persistence'],
            }

            print(f"\nH{dim}:")
            print(f"  {labels[0]}: {stats_1['n_features']} features, "
                  f"total persistence = {stats_1['total_persistence']:.3f}")
            print(f"  {labels[1]}: {stats_2['n_features']} features, "
                  f"total persistence = {stats_2['total_persistence']:.3f}")

        return comparison

    def visualize_persistence(
        self,
        result: Union[TopologyResult, PersistenceResult],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize persistence diagrams.

        Args:
            result: TopologyResult or PersistenceResult
            save_path: If specified, save figure
        """
        import matplotlib.pyplot as plt

        if isinstance(result, TopologyResult):
            persistence_result = result.observed
            has_null = True
        else:
            persistence_result = result
            has_null = False

        fig, axes = plt.subplots(1, self.max_dim + 1, figsize=(5 * (self.max_dim + 1), 5))
        if self.max_dim == 0:
            axes = [axes]

        colors = ['blue', 'orange', 'green']

        for dim, ax in enumerate(axes):
            diagram = persistence_result.diagrams.get(dim, np.empty((0, 2)))

            if len(diagram) > 0:
                ax.scatter(diagram[:, 0], diagram[:, 1],
                          c=colors[dim], alpha=0.6, s=20)

            # Diagonal line
            max_val = max(
                diagram[:, 1].max() if len(diagram) > 0 else 1,
                persistence_result.max_edge
            )
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

            ax.set_xlabel('Birth', fontsize=12)
            ax.set_ylabel('Death', fontsize=12)

            title = f'H{dim}'
            if isinstance(result, TopologyResult):
                p_val = result.p_values.get(dim, 1.0)
                title += f' (p={p_val:.3f})'

            ax.set_title(title, fontsize=14)
            ax.set_xlim(0, max_val * 1.05)
            ax.set_ylim(0, max_val * 1.05)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved persistence diagram to {save_path}")

        plt.show()

    def visualize_barcode(
        self,
        result: Union[TopologyResult, PersistenceResult],
        dimension: int = 1,
        max_features: int = 50,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize persistence barcode.

        Args:
            result: TopologyResult or PersistenceResult
            dimension: Which homology dimension to show
            max_features: Maximum number of bars to display
            save_path: If specified, save figure
        """
        import matplotlib.pyplot as plt

        if isinstance(result, TopologyResult):
            persistence_result = result.observed
        else:
            persistence_result = result

        diagram = persistence_result.diagrams.get(dimension, np.empty((0, 2)))

        if len(diagram) == 0:
            print(f"No H{dimension} features to display")
            return

        # Sort by persistence (descending)
        persistence = diagram[:, 1] - diagram[:, 0]
        sorted_indices = np.argsort(persistence)[::-1][:max_features]
        sorted_diagram = diagram[sorted_indices]

        fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_diagram) * 0.15)))

        for i, (birth, death) in enumerate(sorted_diagram):
            ax.plot([birth, death], [i, i], 'b-', linewidth=2)
            ax.plot(birth, i, 'bo', markersize=4)
            ax.plot(death, i, 'ro', markersize=4)

        ax.set_xlabel('Filtration Value', fontsize=12)
        ax.set_ylabel('Feature Index', fontsize=12)
        ax.set_title(f'H{dimension} Persistence Barcode', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved barcode to {save_path}")

        plt.show()

    def save(self, result: TopologyResult, output_path: str) -> None:
        """
        Save topology results to disk.

        Args:
            result: TopologyResult to save
            output_path: Path to output file (npz format)
        """
        # Prepare diagrams for saving
        obs_diagrams = {f'obs_h{k}': v for k, v in result.observed.diagrams.items()}
        sig_features = {f'sig_h{k}': v for k, v in result.significant_features.items()}

        np.savez(
            output_path,
            **obs_diagrams,
            **sig_features,
            max_dim=result.observed.max_dim,
            max_edge=result.observed.max_edge,
            computation_time=result.observed.computation_time,
            n_points=result.observed.n_points,
            p_values=np.array(list(result.p_values.values())),
            p_value_dims=np.array(list(result.p_values.keys())),
            n_shuffles=result.n_shuffles,
        )
        print(f"Saved topology results to {output_path}")

    @classmethod
    def load(cls, input_path: str) -> TopologyResult:
        """
        Load topology results from disk.

        Args:
            input_path: Path to saved file

        Returns:
            TopologyResult (without null distribution)
        """
        data = np.load(input_path)

        # Reconstruct diagrams
        max_dim = int(data['max_dim'])
        diagrams = {}
        significant_features = {}

        for dim in range(max_dim + 1):
            if f'obs_h{dim}' in data:
                diagrams[dim] = data[f'obs_h{dim}']
            if f'sig_h{dim}' in data:
                significant_features[dim] = data[f'sig_h{dim}']

        observed = PersistenceResult(
            diagrams=diagrams,
            max_dim=max_dim,
            max_edge=float(data['max_edge']),
            computation_time=float(data['computation_time']),
            n_points=int(data['n_points']),
        )

        p_values = dict(zip(data['p_value_dims'], data['p_values']))

        return TopologyResult(
            observed=observed,
            null_distribution=[],  # Not saved
            p_values=p_values,
            significant_features=significant_features,
            n_shuffles=int(data['n_shuffles']),
        )
