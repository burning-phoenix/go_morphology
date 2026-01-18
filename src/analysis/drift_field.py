"""Drift field computation for Fokker-Planck analysis.

This module computes drift vectors (mean displacement) at each state,
which forms the basis for Fokker-Planck analysis of probability flow.

For Fokker-Planck equation:
    ∂ρ/∂t = -∇·(v·ρ) + D∇²ρ

Where:
    - ρ(x,t) is probability density
    - v(x) is drift field (computed here)
    - D is diffusion coefficient

References:
- Risken (1996) "The Fokker-Planck Equation"
- PyEMMA documentation on MSM dynamics

Usage:
    >>> from src.analysis.drift_field import DriftFieldAnalyzer
    >>> analyzer = DriftFieldAnalyzer()
    >>> result = analyzer.compute_drift_field(features, labels, game_ids)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftFieldResult:
    """Result of drift field computation."""
    drift_vectors: np.ndarray  # [n_states, n_dims] mean displacement per state
    diffusion_coefficients: np.ndarray  # [n_states] variance of displacement
    counts: np.ndarray  # [n_states] number of transitions from each state
    mean_drift_magnitude: float  # Average ||drift||
    max_drift_magnitude: float  # Maximum ||drift||
    coverage: float  # Fraction of states with observations


@dataclass
class MacrostateDriftResult:
    """Drift field aggregated to macrostate level."""
    drift_vectors: np.ndarray  # [n_macrostates, n_dims]
    diffusion_coefficients: np.ndarray  # [n_macrostates]
    transition_rates: np.ndarray  # [n_macrostates, n_macrostates] inter-state rates
    counts: np.ndarray  # [n_macrostates]


class DriftFieldAnalyzer:
    """Analyzer for computing drift fields from trajectory data.

    The drift field v(x) represents the expected direction and magnitude
    of movement from each state, which characterizes the deterministic
    component of the dynamics in a Fokker-Planck framework.
    """

    def __init__(self):
        """Initialize drift field analyzer."""
        pass

    def compute_drift_field(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        game_ids: np.ndarray,
        min_counts: int = 10,
    ) -> DriftFieldResult:
        """Compute mean drift (velocity) at each microstate.

        Respects game boundaries - only computes transitions within games.

        Args:
            features: [N, D] feature vectors (e.g., TICA coordinates)
            labels: [N] cluster assignments (microstates)
            game_ids: [N] game identifiers
            min_counts: Minimum transitions required for reliable drift estimate

        Returns:
            DriftFieldResult with drift vectors and statistics
        """
        n_dims = features.shape[1]
        n_states = labels.max() + 1

        # Initialize accumulators
        drift_sum = np.zeros((n_states, n_dims))
        drift_sq_sum = np.zeros((n_states, n_dims))  # For variance
        counts = np.zeros(n_states)

        # Process each game separately
        unique_games = np.unique(game_ids)
        for game_id in unique_games:
            mask = game_ids == game_id
            game_features = features[mask]
            game_labels = labels[mask]

            # Compute displacements within game
            for t in range(len(game_features) - 1):
                state = game_labels[t]
                if state >= 0:  # Skip noise points (label -1)
                    displacement = game_features[t + 1] - game_features[t]
                    drift_sum[state] += displacement
                    drift_sq_sum[state] += displacement ** 2
                    counts[state] += 1

        # Compute mean drift (avoid division by zero)
        nonzero = counts > 0
        drift_vectors = np.zeros((n_states, n_dims))
        drift_vectors[nonzero] = drift_sum[nonzero] / counts[nonzero, np.newaxis]

        # Compute diffusion coefficients (variance of displacement)
        diffusion = np.zeros(n_states)
        for s in range(n_states):
            if counts[s] > 1:
                mean_sq = drift_sq_sum[s] / counts[s]
                sq_mean = drift_vectors[s] ** 2
                var = np.mean(mean_sq - sq_mean)  # Average over dimensions
                diffusion[s] = max(0, var)  # Ensure non-negative

        # Compute summary statistics
        drift_magnitudes = np.linalg.norm(drift_vectors, axis=1)
        mean_magnitude = np.mean(drift_magnitudes[nonzero]) if nonzero.any() else 0
        max_magnitude = np.max(drift_magnitudes) if drift_magnitudes.size > 0 else 0
        coverage = np.mean(counts >= min_counts)

        # Log states with low counts
        low_count_states = np.sum(counts < min_counts)
        if low_count_states > 0:
            logger.warning(
                f"{low_count_states} states have fewer than {min_counts} transitions. "
                f"Drift estimates may be unreliable."
            )

        logger.info(
            f"Drift field computed: {n_states} states, "
            f"mean ||drift||={mean_magnitude:.4f}, "
            f"coverage={coverage:.1%}"
        )

        return DriftFieldResult(
            drift_vectors=drift_vectors,
            diffusion_coefficients=diffusion,
            counts=counts,
            mean_drift_magnitude=mean_magnitude,
            max_drift_magnitude=max_magnitude,
            coverage=coverage,
        )

    def aggregate_to_macrostates(
        self,
        drift_result: DriftFieldResult,
        memberships: np.ndarray,
        stationary_distribution: Optional[np.ndarray] = None,
    ) -> MacrostateDriftResult:
        """Aggregate microstate drift field to macrostate level.

        Uses fuzzy membership probabilities for weighted averaging.

        Args:
            drift_result: Microstate drift field result
            memberships: [n_micro, n_macro] PCCA+ membership probabilities
            stationary_distribution: [n_micro] optional weights

        Returns:
            MacrostateDriftResult
        """
        n_micro, n_macro = memberships.shape
        n_dims = drift_result.drift_vectors.shape[1]

        # Default weights: uniform or counts
        if stationary_distribution is None:
            weights = drift_result.counts
        else:
            weights = stationary_distribution * drift_result.counts

        # Normalize weights
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n_micro) / n_micro

        # Weighted aggregation
        macro_drift = np.zeros((n_macro, n_dims))
        macro_diffusion = np.zeros(n_macro)
        macro_counts = np.zeros(n_macro)

        for m in range(n_macro):
            # Membership-weighted average
            state_weights = memberships[:, m] * weights
            total_weight = state_weights.sum()

            if total_weight > 0:
                state_weights /= total_weight
                macro_drift[m] = (drift_result.drift_vectors.T @ state_weights)
                macro_diffusion[m] = state_weights @ drift_result.diffusion_coefficients
                macro_counts[m] = memberships[:, m] @ drift_result.counts

        # Compute inter-macrostate transition rates
        # Rate from macrostate i to j based on drift direction
        transition_rates = self._compute_transition_rates(
            macro_drift, memberships, drift_result.drift_vectors
        )

        return MacrostateDriftResult(
            drift_vectors=macro_drift,
            diffusion_coefficients=macro_diffusion,
            transition_rates=transition_rates,
            counts=macro_counts,
        )

    def _compute_transition_rates(
        self,
        macro_drift: np.ndarray,
        memberships: np.ndarray,
        micro_drift: np.ndarray,
    ) -> np.ndarray:
        """Compute effective transition rates between macrostates.

        Based on mean drift direction and membership gradients.
        """
        n_macro = memberships.shape[1]
        rates = np.zeros((n_macro, n_macro))

        # Simple model: rate ~ flux from membership change
        # This is a placeholder; proper rates come from MSM
        for i in range(n_macro):
            for j in range(n_macro):
                if i != j:
                    # Approximate rate as fraction of drift pointing toward j
                    rates[i, j] = 0.0  # Placeholder

        return rates

    def compute_probability_current(
        self,
        drift_vectors: np.ndarray,
        stationary_distribution: np.ndarray,
    ) -> np.ndarray:
        """Compute probability current J = ρ * v.

        This is the flux of probability through feature space.

        Args:
            drift_vectors: [n_states, n_dims] drift field
            stationary_distribution: [n_states] equilibrium distribution

        Returns:
            [n_states, n_dims] probability current vectors
        """
        return drift_vectors * stationary_distribution[:, np.newaxis]

    def compute_potential_energy(
        self,
        stationary_distribution: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Compute effective potential from stationary distribution.

        U(x) = -kT * ln(ρ(x))

        Args:
            stationary_distribution: [n_states] equilibrium distribution
            temperature: Temperature parameter (default 1.0 = kT units)

        Returns:
            [n_states] potential energy at each state
        """
        # Avoid log(0) with small epsilon
        eps = 1e-10
        pi = stationary_distribution + eps
        return -temperature * np.log(pi)


def save_drift_field(result: DriftFieldResult, filepath: str) -> None:
    """Save drift field result to npz file."""
    np.savez(
        filepath,
        drift_vectors=result.drift_vectors,
        diffusion_coefficients=result.diffusion_coefficients,
        counts=result.counts,
        mean_drift_magnitude=result.mean_drift_magnitude,
        max_drift_magnitude=result.max_drift_magnitude,
        coverage=result.coverage,
    )
    logger.info(f"Saved drift field to {filepath}")


def load_drift_field(filepath: str) -> DriftFieldResult:
    """Load drift field result from npz file."""
    data = np.load(filepath)
    return DriftFieldResult(
        drift_vectors=data['drift_vectors'],
        diffusion_coefficients=data['diffusion_coefficients'],
        counts=data['counts'],
        mean_drift_magnitude=float(data['mean_drift_magnitude']),
        max_drift_magnitude=float(data['max_drift_magnitude']),
        coverage=float(data['coverage']),
    )


def visualize_drift_field_2d(
    drift_result: DriftFieldResult,
    coords_2d: np.ndarray,
    labels: np.ndarray,
    ax=None,
    scale: float = 1.0,
    min_counts: int = 10,
    cmap: str = 'viridis',
) -> None:
    """Visualize drift field as quiver plot in 2D projection.

    Args:
        drift_result: Drift field result
        coords_2d: [N, 2] 2D coordinates (e.g., UMAP projection)
        labels: [N] cluster assignments
        ax: Matplotlib axes (created if None)
        scale: Arrow scaling factor
        min_counts: Minimum counts to show arrow
        cmap: Colormap for drift magnitude
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Compute centroid positions for each state
    n_states = drift_result.drift_vectors.shape[0]
    centroids = np.zeros((n_states, 2))
    for s in range(n_states):
        mask = labels == s
        if mask.any():
            centroids[s] = coords_2d[mask].mean(axis=0)

    # Project drift vectors to 2D (take first 2 components or project)
    # Simplified: if drift is higher dimensional, use direction toward next state
    drift_2d = drift_result.drift_vectors[:, :2] if drift_result.drift_vectors.shape[1] >= 2 else \
               np.zeros((n_states, 2))

    # Filter by counts
    valid = drift_result.counts >= min_counts
    magnitudes = np.linalg.norm(drift_2d, axis=1)

    # Plot background scatter
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c='lightgray', s=1, alpha=0.3)

    # Plot drift arrows
    ax.quiver(
        centroids[valid, 0],
        centroids[valid, 1],
        drift_2d[valid, 0],
        drift_2d[valid, 1],
        magnitudes[valid],
        cmap=cmap,
        scale=1.0 / scale,
        width=0.003,
        alpha=0.8,
    )

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('Drift Field')

    return ax
