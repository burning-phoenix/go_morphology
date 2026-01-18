"""PCCA+ coarse-graining for MSM macrostate identification.

This module provides spectral clustering of Markov State Model microstates
into metastable macrostates using the PCCA+ algorithm.

References:
- Röblitz & Weber (2013) "Fuzzy spectral clustering by PCCA+"
- Deuflhard & Weber (2005) "Robust Perron cluster analysis"
- PyEMMA documentation L:262-266, 439
- Deeptime documentation: https://deeptime-ml.github.io/latest/notebooks/pcca.html

Usage:
    >>> from src.analysis.pcca import PCCAAnalyzer
    >>> analyzer = PCCAAnalyzer(transition_matrix)
    >>> gap_result = analyzer.spectral_gap_analysis()
    >>> pcca_result = analyzer.fit(n_macrostates=5)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpectralGapResult:
    """Result of spectral gap analysis for macrostate selection."""
    eigenvalues: np.ndarray  # MSM eigenvalues in descending order
    gaps: np.ndarray  # Gaps between consecutive eigenvalues
    suggested_n_macrostates: List[int]  # Top suggestions based on gap size
    gap_ratios: np.ndarray  # Ratio of gap to eigenvalue (relative gap)


@dataclass
class PCCAResult:
    """Result of PCCA+ coarse-graining."""
    memberships: np.ndarray  # [n_microstates, n_macrostates] fuzzy membership
    assignments: np.ndarray  # [n_microstates] hard assignment (argmax)
    coarse_transition_matrix: np.ndarray  # [n_macro, n_macro]
    coarse_stationary: np.ndarray  # [n_macro] stationary distribution
    n_macrostates: int
    metastable_sets: List[np.ndarray]  # List of microstate indices per macrostate
    membership_entropy: float  # Average entropy of membership (lower = crisper)

    # Optional: store eigenvalues used
    eigenvalues_used: Optional[np.ndarray] = None


class PCCAAnalyzer:
    """PCCA+ analyzer for MSM coarse-graining.

    Uses deeptime library when available, with scipy fallback.

    Attributes:
        transition_matrix: Row-stochastic transition matrix
        stationary_distribution: Equilibrium distribution (computed if not provided)
        eigenvalues: MSM eigenvalues
        eigenvectors: MSM right eigenvectors
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        stationary_distribution: Optional[np.ndarray] = None,
    ):
        """Initialize PCCA+ analyzer.

        Args:
            transition_matrix: Row-stochastic [n, n] transition matrix
            stationary_distribution: Optional stationary distribution
        """
        self.transition_matrix = np.asarray(transition_matrix)
        self.n_states = self.transition_matrix.shape[0]

        # Validate transition matrix
        self._validate_transition_matrix()

        # Compute or store stationary distribution
        if stationary_distribution is not None:
            self.stationary_distribution = np.asarray(stationary_distribution)
        else:
            self.stationary_distribution = self._compute_stationary_distribution()

        # Compute eigendecomposition
        self.eigenvalues, self.eigenvectors = self._compute_eigendecomposition()

        # Check for deeptime availability
        self._has_deeptime = self._check_deeptime()

    def _validate_transition_matrix(self) -> None:
        """Validate transition matrix properties."""
        # Check square
        if self.transition_matrix.shape[0] != self.transition_matrix.shape[1]:
            raise ValueError("Transition matrix must be square")

        # Check row-stochastic (rows sum to 1)
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-5):
            logger.warning(
                f"Transition matrix rows don't sum to 1. "
                f"Max deviation: {np.abs(row_sums - 1).max():.6f}"
            )

    def _compute_stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution via power iteration."""
        # Left eigenvector of eigenvalue 1
        # Use power iteration for numerical stability
        pi = np.ones(self.n_states) / self.n_states

        for _ in range(1000):
            pi_new = pi @ self.transition_matrix
            if np.allclose(pi, pi_new, rtol=1e-10):
                break
            pi = pi_new

        # Normalize
        pi = pi / pi.sum()
        return pi

    def _compute_eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and right eigenvectors of transition matrix."""
        from scipy.linalg import eig

        eigenvalues, eigenvectors = eig(self.transition_matrix)

        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real

        return eigenvalues, eigenvectors

    def _check_deeptime(self) -> bool:
        """Check if deeptime library is available."""
        try:
            import deeptime
            return True
        except ImportError:
            logger.info("deeptime not available, using scipy fallback")
            return False

    def spectral_gap_analysis(self, max_states: int = 20) -> SpectralGapResult:
        """Analyze spectral gap to suggest number of macrostates.

        The gap after eigenvalue m suggests m metastable states.
        A large gap indicates timescale separation between fast and slow processes.

        Args:
            max_states: Maximum number of states to consider

        Returns:
            SpectralGapResult with eigenvalues, gaps, and suggestions
        """
        # Use real parts, sorted descending
        eigs = self.eigenvalues[:max_states]

        # Compute gaps (difference between consecutive eigenvalues)
        gaps = -np.diff(eigs)  # Negative because eigenvalues are descending

        # Compute relative gaps (gap / eigenvalue)
        gap_ratios = gaps / np.abs(eigs[:-1])

        # Find largest gaps (excluding first eigenvalue = 1)
        # The gap after eigenvalue i suggests i+1 metastable states
        gap_indices = np.argsort(gaps[1:])[::-1] + 2  # +2 for 1-indexing and skip λ₁=1

        # Filter to reasonable range
        suggestions = [i for i in gap_indices if 2 <= i <= min(max_states, 15)][:5]

        logger.info(
            f"Spectral gap analysis: top eigenvalues = {eigs[:6]}, "
            f"suggested n_macrostates = {suggestions[:3]}"
        )

        return SpectralGapResult(
            eigenvalues=eigs,
            gaps=gaps,
            suggested_n_macrostates=suggestions,
            gap_ratios=gap_ratios,
        )

    def fit(self, n_macrostates: int) -> PCCAResult:
        """Apply PCCA+ to identify metastable macrostates.

        Args:
            n_macrostates: Number of metastable sets to identify

        Returns:
            PCCAResult with memberships, assignments, coarse matrices
        """
        if n_macrostates < 2:
            raise ValueError("n_macrostates must be >= 2")
        if n_macrostates > self.n_states:
            raise ValueError(f"n_macrostates ({n_macrostates}) > n_states ({self.n_states})")

        if self._has_deeptime:
            return self._fit_deeptime(n_macrostates)
        else:
            return self._fit_scipy(n_macrostates)

    def _fit_deeptime(self, n_macrostates: int) -> PCCAResult:
        """Fit PCCA+ using deeptime library."""
        from deeptime.markov.msm import MarkovStateModel

        # Create MSM from transition matrix
        msm = MarkovStateModel(self.transition_matrix)

        # Apply PCCA+
        pcca = msm.pcca(n_metastable_sets=n_macrostates)

        # Extract results
        memberships = pcca.memberships
        assignments = pcca.assignments

        # Coarse-grained matrices
        coarse_T = pcca.coarse_grained_transition_matrix
        coarse_pi = pcca.coarse_grained_stationary_probability

        # Extract metastable sets
        metastable_sets = [
            np.where(assignments == i)[0] for i in range(n_macrostates)
        ]

        # Compute membership entropy (measure of crispness)
        entropy = self._compute_membership_entropy(memberships)

        logger.info(
            f"PCCA+ (deeptime): {n_macrostates} macrostates, "
            f"membership entropy = {entropy:.3f}"
        )

        return PCCAResult(
            memberships=memberships,
            assignments=assignments,
            coarse_transition_matrix=coarse_T,
            coarse_stationary=coarse_pi,
            n_macrostates=n_macrostates,
            metastable_sets=metastable_sets,
            membership_entropy=entropy,
            eigenvalues_used=self.eigenvalues[:n_macrostates],
        )

    def _fit_scipy(self, n_macrostates: int) -> PCCAResult:
        """Fit PCCA+ using scipy (fallback implementation).

        This is a simplified PCCA+ implementation based on the Schur decomposition
        approach. For production use, prefer deeptime.
        """
        from scipy.cluster.vq import kmeans2

        # Take top m eigenvectors
        X = self.eigenvectors[:, :n_macrostates]

        # Normalize rows to unit length (for clustering)
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        X_normalized = X / row_norms

        # Use k-means to get initial clustering
        # Then convert to soft memberships
        _, assignments = kmeans2(X_normalized, n_macrostates, minit='++')

        # Convert hard assignments to soft memberships
        # (simplified: just use one-hot encoding as initial memberships)
        memberships = np.zeros((self.n_states, n_macrostates))
        for i, a in enumerate(assignments):
            memberships[i, a] = 1.0

        # Compute coarse-grained matrices
        coarse_T = self._compute_coarse_transition_matrix(memberships)
        coarse_pi = self._compute_coarse_stationary(memberships)

        # Extract metastable sets
        metastable_sets = [
            np.where(assignments == i)[0] for i in range(n_macrostates)
        ]

        # Compute membership entropy
        entropy = self._compute_membership_entropy(memberships)

        logger.warning(
            f"PCCA+ (scipy fallback): Using simplified implementation. "
            f"Install deeptime for full PCCA+ optimization."
        )

        return PCCAResult(
            memberships=memberships,
            assignments=assignments,
            coarse_transition_matrix=coarse_T,
            coarse_stationary=coarse_pi,
            n_macrostates=n_macrostates,
            metastable_sets=metastable_sets,
            membership_entropy=entropy,
            eigenvalues_used=self.eigenvalues[:n_macrostates],
        )

    def _compute_coarse_transition_matrix(
        self, memberships: np.ndarray
    ) -> np.ndarray:
        """Compute coarse-grained transition matrix from memberships.

        T_coarse[i,j] = sum_k sum_l pi_k * chi_k,i * T_kl * chi_l,j / pi_i_coarse

        Simplified: T_coarse = (M^T * diag(pi) * T * M) / (M^T * diag(pi) * ones)
        """
        n_macro = memberships.shape[1]
        pi = self.stationary_distribution

        # Weighted membership sums
        weighted_M = memberships * pi[:, np.newaxis]  # [n_micro, n_macro]

        # Coarse stationary: sum of weighted memberships
        coarse_pi = weighted_M.sum(axis=0)  # [n_macro]

        # Coarse transition matrix
        # T_coarse[i,j] = sum over microstates
        coarse_T = np.zeros((n_macro, n_macro))
        for i in range(n_macro):
            for j in range(n_macro):
                # Sum over all microstate pairs
                flux_ij = 0.0
                for k in range(self.n_states):
                    for l in range(self.n_states):
                        flux_ij += (
                            pi[k] * memberships[k, i] *
                            self.transition_matrix[k, l] * memberships[l, j]
                        )
                coarse_T[i, j] = flux_ij / coarse_pi[i] if coarse_pi[i] > 0 else 0

        # Normalize rows
        row_sums = coarse_T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        coarse_T = coarse_T / row_sums

        return coarse_T

    def _compute_coarse_stationary(self, memberships: np.ndarray) -> np.ndarray:
        """Compute coarse-grained stationary distribution."""
        pi = self.stationary_distribution
        coarse_pi = (memberships.T @ pi)
        return coarse_pi / coarse_pi.sum()

    def _compute_membership_entropy(self, memberships: np.ndarray) -> float:
        """Compute average entropy of membership probabilities.

        Lower entropy = crisper assignments (closer to 0 or 1).
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        M = memberships + eps

        # Normalize rows (should already be normalized)
        M = M / M.sum(axis=1, keepdims=True)

        # Compute entropy per row
        entropies = -np.sum(M * np.log(M), axis=1)

        return float(np.mean(entropies))

    def fit_multiple(
        self,
        n_values: List[int] = [5, 7, 10]
    ) -> Dict[int, PCCAResult]:
        """Fit PCCA+ at multiple macrostate counts for comparison.

        Args:
            n_values: List of n_macrostates to try

        Returns:
            Dictionary mapping n_macrostates -> PCCAResult
        """
        results = {}
        for n in n_values:
            try:
                results[n] = self.fit(n)
                logger.info(
                    f"PCCA+ n={n}: entropy={results[n].membership_entropy:.3f}"
                )
            except Exception as e:
                logger.warning(f"PCCA+ failed for n={n}: {e}")

        return results

    def compute_mfpt_matrix(self, pcca_result: PCCAResult) -> np.ndarray:
        """Compute mean first passage times between macrostates.

        Args:
            pcca_result: Result from fit()

        Returns:
            [n_macro, n_macro] matrix of MFPTs
        """
        coarse_T = pcca_result.coarse_transition_matrix
        coarse_pi = pcca_result.coarse_stationary
        n = pcca_result.n_macrostates

        # Fundamental matrix approach
        # MFPT[i,j] = (Z[j,j] - Z[i,j]) / pi[j]
        # where Z = (I - T + W)^{-1}, W[i,:] = pi

        I = np.eye(n)
        W = np.tile(coarse_pi, (n, 1))

        try:
            Z = np.linalg.inv(I - coarse_T + W)
        except np.linalg.LinAlgError:
            # Add regularization if singular
            Z = np.linalg.inv(I - coarse_T + W + 1e-10 * I)

        mfpt = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if coarse_pi[j] > 0:
                    mfpt[i, j] = (Z[j, j] - Z[i, j]) / coarse_pi[j]

        return mfpt


def save_pcca_result(result: PCCAResult, filepath: str) -> None:
    """Save PCCA+ result to npz file."""
    np.savez(
        filepath,
        memberships=result.memberships,
        assignments=result.assignments,
        coarse_transition_matrix=result.coarse_transition_matrix,
        coarse_stationary=result.coarse_stationary,
        n_macrostates=result.n_macrostates,
        membership_entropy=result.membership_entropy,
        eigenvalues_used=result.eigenvalues_used,
    )
    logger.info(f"Saved PCCA+ result to {filepath}")


def load_pcca_result(filepath: str) -> PCCAResult:
    """Load PCCA+ result from npz file."""
    data = np.load(filepath, allow_pickle=True)

    assignments = data['assignments']
    n_macrostates = int(data['n_macrostates'])

    # Reconstruct metastable sets
    metastable_sets = [
        np.where(assignments == i)[0] for i in range(n_macrostates)
    ]

    return PCCAResult(
        memberships=data['memberships'],
        assignments=assignments,
        coarse_transition_matrix=data['coarse_transition_matrix'],
        coarse_stationary=data['coarse_stationary'],
        n_macrostates=n_macrostates,
        metastable_sets=metastable_sets,
        membership_entropy=float(data['membership_entropy']),
        eigenvalues_used=data.get('eigenvalues_used'),
    )
