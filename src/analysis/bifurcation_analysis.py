"""
Bifurcation Detection in SAE Feature Space.

Identifies "bifurcation points" - moves where the strategic trajectory
diverges significantly from the expected path, indicating critical decisions.

Reference:
- MASTER_IMPLEMENTATION_PLAN.md: Lines 495-589 (NB06 spec)
- docs/dynamical_systems_chaos/INDEX.md: Lines 195-242 (bifurcation theory)

Key concepts:
- Bifurcation: Qualitative change in system dynamics at critical parameter values
- Trajectory divergence: Cosine distance between consecutive trajectory segments
- Threshold: 95th percentile of divergence scores
- Outcome correlation: Permutation test for bifurcation-outcome relationship
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import h5py


@dataclass
class BifurcationResult:
    """Results from bifurcation detection."""
    move_indices: np.ndarray          # Indices where bifurcations detected
    divergence_scores: np.ndarray     # Cosine divergence at each move
    threshold: float                  # 95th percentile threshold
    game_ids: np.ndarray              # Game index for each move
    game_boundaries: List[Tuple[int, int]]  # (start, end) for each game
    outcomes: Optional[np.ndarray] = None   # Game outcomes if available


@dataclass
class CorrelationResult:
    """Results from bifurcation-outcome correlation test."""
    correlation: float                # Observed correlation
    p_value: float                    # Permutation test p-value
    null_mean: float                  # Mean of null distribution
    null_std: float                   # Std of null distribution
    n_permutations: int               # Number of permutations used
    significant: bool                 # p_value < 0.05


def compute_trajectory_divergence(
    trajectory: np.ndarray,
    window: int = 5,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute divergence between consecutive trajectory segments.

    At each position i, compares the direction of travel from
    [i-window, i] to [i, i+window]. High divergence indicates
    a "turn" or bifurcation in the trajectory.

    Args:
        trajectory: [T, D] - one game's trajectory through feature space
        window: Number of moves to average for direction vectors
        metric: 'cosine' (recommended) or 'euclidean'

    Returns:
        [T] divergence score for each move (NaN at boundaries)
    """
    T, D = trajectory.shape
    divergences = np.full(T, np.nan)

    if T < 2 * window + 1:
        return divergences

    for i in range(window, T - window):
        # Direction vector before position i
        vec_before = trajectory[i] - trajectory[i - window]

        # Direction vector after position i
        vec_after = trajectory[i + window] - trajectory[i]

        # Handle zero vectors
        norm_before = np.linalg.norm(vec_before)
        norm_after = np.linalg.norm(vec_after)

        if norm_before < 1e-8 or norm_after < 1e-8:
            divergences[i] = 0.0
            continue

        if metric == 'cosine':
            # Cosine distance (1 - cosine similarity)
            # Range: [0, 2], where 0 = same direction, 2 = opposite
            divergences[i] = cosine(vec_before, vec_after)
        else:
            # Normalized Euclidean distance
            vec_before_norm = vec_before / norm_before
            vec_after_norm = vec_after / norm_after
            divergences[i] = np.linalg.norm(vec_before_norm - vec_after_norm)

    return divergences


def detect_bifurcations(
    divergences: np.ndarray,
    threshold_percentile: float = 95,
    min_distance: int = 5,
) -> Tuple[np.ndarray, float]:
    """
    Identify bifurcation points as moves with divergence above threshold.

    Args:
        divergences: [T] divergence score for each move
        threshold_percentile: Percentile for threshold (default 95)
        min_distance: Minimum distance between bifurcations (to avoid clustering)

    Returns:
        (bifurcation_indices, threshold_value)
    """
    # Filter out NaN values for threshold computation
    valid_divergences = divergences[~np.isnan(divergences)]

    if len(valid_divergences) == 0:
        return np.array([], dtype=int), 0.0

    threshold = np.percentile(valid_divergences, threshold_percentile)

    # Find all candidates above threshold
    candidates = np.where(divergences > threshold)[0]

    if len(candidates) == 0:
        return np.array([], dtype=int), threshold

    # Filter by minimum distance (keep highest divergence in each cluster)
    bifurcations = []
    last_idx = -min_distance - 1

    # Sort by divergence (descending) to prioritize highest
    sorted_candidates = candidates[np.argsort(divergences[candidates])[::-1]]

    for idx in sorted_candidates:
        if idx - last_idx >= min_distance:
            bifurcations.append(idx)
            last_idx = idx

    return np.sort(np.array(bifurcations)), threshold


def correlate_with_outcomes(
    bifurcation_result: BifurcationResult,
    outcomes: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> CorrelationResult:
    """
    Test if bifurcation timing/magnitude correlates with game outcomes.

    Uses permutation test: shuffles outcome labels while preserving
    bifurcation structure, computes null distribution.

    Args:
        bifurcation_result: Result from detect_all_games
        outcomes: [n_games] outcome labels (e.g., win=1, loss=0)
        n_permutations: Number of permutations for null distribution
        seed: Random seed for reproducibility

    Returns:
        CorrelationResult with correlation, p-value, etc.
    """
    rng = np.random.default_rng(seed)

    # Compute per-game bifurcation statistics
    game_boundaries = bifurcation_result.game_boundaries
    n_games = len(game_boundaries)

    if n_games != len(outcomes):
        raise ValueError(f"Mismatch: {n_games} games vs {len(outcomes)} outcomes")

    # Per-game stats: number of bifurcations, max divergence
    n_bifurcations = np.zeros(n_games)
    max_divergence = np.zeros(n_games)

    for game_idx, (start, end) in enumerate(game_boundaries):
        # Find bifurcations in this game
        game_mask = (bifurcation_result.move_indices >= start) & \
                    (bifurcation_result.move_indices < end)
        game_bifurcations = bifurcation_result.move_indices[game_mask]

        n_bifurcations[game_idx] = len(game_bifurcations)

        # Max divergence in this game
        game_divergences = bifurcation_result.divergence_scores[start:end]
        valid = ~np.isnan(game_divergences)
        if valid.any():
            max_divergence[game_idx] = np.max(game_divergences[valid])

    # Use combined feature for correlation
    # z-score and combine (equal weight)
    z_n = (n_bifurcations - n_bifurcations.mean()) / (n_bifurcations.std() + 1e-8)
    z_max = (max_divergence - max_divergence.mean()) / (max_divergence.std() + 1e-8)
    bifurcation_score = (z_n + z_max) / 2

    # Observed correlation
    observed_corr = np.corrcoef(bifurcation_score, outcomes)[0, 1]

    # Permutation null distribution
    null_corrs = []
    for _ in range(n_permutations):
        shuffled_outcomes = rng.permutation(outcomes)
        null_corr = np.corrcoef(bifurcation_score, shuffled_outcomes)[0, 1]
        null_corrs.append(null_corr)

    null_corrs = np.array(null_corrs)

    # Two-tailed p-value
    p_value = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))

    return CorrelationResult(
        correlation=observed_corr,
        p_value=p_value,
        null_mean=np.mean(null_corrs),
        null_std=np.std(null_corrs),
        n_permutations=n_permutations,
        significant=p_value < 0.05,
    )


class BifurcationAnalyzer:
    """
    Complete bifurcation analysis pipeline.

    Identifies critical decision points in game trajectories by detecting
    where the feature-space trajectory diverges significantly.

    Usage:
        analyzer = BifurcationAnalyzer()
        result = analyzer.analyze_all_games(features, game_ids)

        # With outcomes
        corr = analyzer.correlate_with_outcomes(result, outcomes)

        # Visualization
        analyzer.visualize(result, save_path='bifurcations.png')
    """

    def __init__(
        self,
        window: int = 5,
        threshold_percentile: float = 95,
        min_distance: int = 5,
        metric: str = 'cosine',
    ):
        """
        Args:
            window: Moves to average for direction vectors
            threshold_percentile: Percentile for bifurcation threshold
            min_distance: Minimum moves between bifurcations
            metric: 'cosine' (recommended) or 'euclidean'
        """
        self.window = window
        self.threshold_percentile = threshold_percentile
        self.min_distance = min_distance
        self.metric = metric

    def analyze_all_games(
        self,
        features: np.ndarray,
        game_ids: np.ndarray,
    ) -> BifurcationResult:
        """
        Detect bifurcations across all games.

        Args:
            features: [n_samples, n_features] SAE or TICA features
            game_ids: [n_samples] game index for each sample

        Returns:
            BifurcationResult with all detected bifurcations
        """
        n_samples, n_features = features.shape
        print(f"Bifurcation analysis: {n_samples:,} samples × {n_features} features")
        print(f"  Window: {self.window}, Threshold: {self.threshold_percentile}th percentile")

        # Identify game boundaries
        unique_games = np.unique(game_ids)
        game_boundaries = []

        for game_id in unique_games:
            mask = game_ids == game_id
            indices = np.where(mask)[0]
            game_boundaries.append((indices[0], indices[-1] + 1))

        # Compute divergences for all samples
        all_divergences = np.full(n_samples, np.nan)

        for game_id, (start, end) in zip(unique_games, game_boundaries):
            game_trajectory = features[start:end]
            game_divergences = compute_trajectory_divergence(
                game_trajectory,
                window=self.window,
                metric=self.metric
            )
            all_divergences[start:end] = game_divergences

        # Detect bifurcations globally (threshold across all games)
        bifurcation_indices, threshold = detect_bifurcations(
            all_divergences,
            threshold_percentile=self.threshold_percentile,
            min_distance=self.min_distance,
        )

        n_bifurcations = len(bifurcation_indices)
        print(f"  Found {n_bifurcations} bifurcations ({n_bifurcations/n_samples*100:.2f}%)")
        print(f"  Threshold: {threshold:.4f}")

        return BifurcationResult(
            move_indices=bifurcation_indices,
            divergence_scores=all_divergences,
            threshold=threshold,
            game_ids=game_ids,
            game_boundaries=game_boundaries,
        )

    def analyze_streaming(
        self,
        h5_path: Union[str, Path],
        dataset_key: str,
        game_ids: np.ndarray,
        chunk_size: int = 10000,
    ) -> BifurcationResult:
        """
        Detect bifurcations with streaming h5py access.

        Memory-efficient version that processes one game at a time.

        Args:
            h5_path: Path to HDF5 file with features
            dataset_key: Dataset key in HDF5 file
            game_ids: [n_samples] game index for each sample
            chunk_size: Not used (kept for API consistency)

        Returns:
            BifurcationResult
        """
        h5_path = Path(h5_path)

        with h5py.File(h5_path, 'r') as f:
            dset = f[dataset_key]
            n_samples, n_features = dset.shape

            print(f"Streaming bifurcation analysis: {n_samples:,} samples")

            # Identify game boundaries
            unique_games = np.unique(game_ids)
            game_boundaries = []

            for game_id in unique_games:
                mask = game_ids == game_id
                indices = np.where(mask)[0]
                game_boundaries.append((indices[0], indices[-1] + 1))

            # Compute divergences game by game (streaming)
            all_divergences = np.full(n_samples, np.nan)

            for game_id, (start, end) in zip(unique_games, game_boundaries):
                # Load only this game's data
                game_trajectory = dset[start:end].astype(np.float32)
                game_divergences = compute_trajectory_divergence(
                    game_trajectory,
                    window=self.window,
                    metric=self.metric
                )
                all_divergences[start:end] = game_divergences

        # Detect bifurcations
        bifurcation_indices, threshold = detect_bifurcations(
            all_divergences,
            threshold_percentile=self.threshold_percentile,
            min_distance=self.min_distance,
        )

        n_bifurcations = len(bifurcation_indices)
        print(f"  Found {n_bifurcations} bifurcations ({n_bifurcations/n_samples*100:.2f}%)")

        return BifurcationResult(
            move_indices=bifurcation_indices,
            divergence_scores=all_divergences,
            threshold=threshold,
            game_ids=game_ids,
            game_boundaries=game_boundaries,
        )

    def get_bifurcation_contexts(
        self,
        features: np.ndarray,
        result: BifurcationResult,
        context_window: int = 10,
    ) -> Dict[int, np.ndarray]:
        """
        Extract feature context around each bifurcation point.

        Args:
            features: [n_samples, n_features] feature matrix
            result: BifurcationResult from analyze
            context_window: Moves before/after bifurcation

        Returns:
            Dict mapping bifurcation index -> [2*context_window+1, n_features]
        """
        contexts = {}

        for idx in result.move_indices:
            start = max(0, idx - context_window)
            end = min(len(features), idx + context_window + 1)
            contexts[idx] = features[start:end]

        return contexts

    def visualize(
        self,
        result: BifurcationResult,
        game_idx: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize bifurcation detection results.

        Args:
            result: BifurcationResult from analyze
            game_idx: If specified, show only this game
            save_path: If specified, save figure to this path
        """
        import matplotlib.pyplot as plt

        if game_idx is not None:
            # Single game visualization
            start, end = result.game_boundaries[game_idx]
            divergences = result.divergence_scores[start:end]
            x = np.arange(end - start)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x, divergences, 'b-', alpha=0.7, label='Divergence')
            ax.axhline(y=result.threshold, color='r', linestyle='--',
                       label=f'Threshold (p{self.threshold_percentile})')

            # Mark bifurcations
            game_bifurcations = result.move_indices[
                (result.move_indices >= start) & (result.move_indices < end)
            ] - start
            ax.scatter(game_bifurcations, divergences[game_bifurcations],
                      c='red', s=100, zorder=5, label='Bifurcations')

            ax.set_xlabel('Move Number', fontsize=12)
            ax.set_ylabel('Trajectory Divergence', fontsize=12)
            ax.set_title(f'Game {game_idx}: Bifurcation Detection', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        else:
            # Summary across all games
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Divergence distribution
            ax1 = axes[0]
            valid_div = result.divergence_scores[~np.isnan(result.divergence_scores)]
            ax1.hist(valid_div, bins=50, edgecolor='black', alpha=0.7)
            ax1.axvline(x=result.threshold, color='r', linestyle='--', linewidth=2,
                        label=f'Threshold: {result.threshold:.3f}')
            ax1.set_xlabel('Divergence Score', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title('Divergence Distribution', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Bifurcations per game
            ax2 = axes[1]
            bif_per_game = []
            for start, end in result.game_boundaries:
                n = np.sum((result.move_indices >= start) & (result.move_indices < end))
                bif_per_game.append(n)

            ax2.hist(bif_per_game, bins=range(max(bif_per_game) + 2),
                     edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Bifurcations per Game', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Bifurcation Count Distribution', fontsize=14)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved bifurcation visualization to {save_path}")

        plt.show()

    def save(self, result: BifurcationResult, output_path: str) -> None:
        """
        Save bifurcation results to disk.

        Args:
            result: BifurcationResult to save
            output_path: Path to output file (npz format)
        """
        np.savez(
            output_path,
            move_indices=result.move_indices,
            divergence_scores=result.divergence_scores,
            threshold=result.threshold,
            game_ids=result.game_ids,
            game_boundaries=np.array(result.game_boundaries),
        )
        print(f"Saved bifurcation results to {output_path}")

    @classmethod
    def load(cls, input_path: str) -> BifurcationResult:
        """
        Load bifurcation results from disk.

        Args:
            input_path: Path to saved file

        Returns:
            BifurcationResult
        """
        data = np.load(input_path)

        return BifurcationResult(
            move_indices=data['move_indices'],
            divergence_scores=data['divergence_scores'],
            threshold=float(data['threshold']),
            game_ids=data['game_ids'],
            game_boundaries=[tuple(b) for b in data['game_boundaries']],
        )
