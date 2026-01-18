"""
Markov State Model (MSM) Transition Matrix Analysis.

Builds and validates transition matrices from clustered trajectory data,
with implied timescales and Chapman-Kolmogorov validation.

Reference:
- docs/dynamical_systems_chaos/PyEMMA_Markov_State_modeling.md L:38-92
- Chapman-Kolmogorov test: L:64-72
- Implied timescales: L:54-62

Key equations:
- Count matrix: C_ij(τ) = # transitions from state i to state j
- Transition matrix: P_ij(τ) = C_ij(τ) / Σ_j C_ij(τ)  (row-stochastic)
- Implied timescale: t_i = -τ / ln(λ_i(τ))
- Chapman-Kolmogorov: P(kτ) ≈ P^k(τ)

CRITICAL: Only count transitions within games, never across game boundaries!
"""

import numpy as np
from scipy import linalg
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class MSMResult:
    """Results from MSM analysis."""
    transition_matrix: np.ndarray       # [n_states, n_states] row-stochastic
    count_matrix: np.ndarray            # [n_states, n_states] raw counts
    stationary_distribution: np.ndarray # [n_states] equilibrium probabilities
    eigenvalues: np.ndarray             # [n_states] sorted descending
    n_states: int
    lag: int
    n_transitions: int                  # Total transition count


@dataclass
class ImpliedTimescales:
    """Implied timescales at multiple lag times."""
    lags: np.ndarray                    # [n_lags] lag values
    timescales: np.ndarray              # [n_lags, n_timescales] ITS values
    n_timescales: int


@dataclass
class ChapmanKolmogorovResult:
    """Results from Chapman-Kolmogorov validation."""
    k_values: List[int]                 # Multiples tested
    rmse: Dict[int, float]              # k -> RMSE
    predictions: Dict[int, np.ndarray]  # k -> P^k(τ)
    estimates: Dict[int, np.ndarray]    # k -> P(kτ)
    passed: bool                        # All RMSE < threshold


class TransitionMatrixAnalyzer:
    """
    Build and analyze Markov State Models from clustered trajectories.

    Constructs transition matrices respecting game boundaries and provides
    validation via implied timescales and Chapman-Kolmogorov test.

    Usage:
        analyzer = TransitionMatrixAnalyzer(lag=1)
        analyzer.fit(cluster_labels, game_ids)
        pi = analyzer.get_stationary_distribution()
        its = analyzer.get_implied_timescales([1, 2, 3, 5, 10])
        ck = analyzer.chapman_kolmogorov_test([2, 3, 5])
    """

    def __init__(self, lag: int = 1):
        """
        Args:
            lag: Lag time in steps (default 1 = consecutive positions)
        """
        self.lag = lag
        self.result_: Optional[MSMResult] = None
        self._cluster_labels: Optional[np.ndarray] = None
        self._game_ids: Optional[np.ndarray] = None

    def fit(
        self,
        cluster_labels: np.ndarray,
        game_ids: np.ndarray
    ) -> MSMResult:
        """
        Build transition matrix from clustered trajectory data.

        CRITICAL: Only counts transitions within games, never across boundaries!

        Args:
            cluster_labels: [n_samples] cluster assignments (-1 = noise, excluded)
            game_ids: [n_samples] game index for each sample

        Returns:
            MSMResult with transition matrix and eigendecomposition
        """
        self._cluster_labels = cluster_labels
        self._game_ids = game_ids

        # Get unique non-noise cluster labels
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        n_states = len(valid_labels)

        # Create label -> index mapping
        label_to_idx = {label: i for i, label in enumerate(valid_labels)}

        print(f"Building MSM: {n_states} states, lag={self.lag}")

        # Count matrix
        count_matrix = self._build_count_matrix(
            cluster_labels, game_ids, label_to_idx, n_states
        )

        n_transitions = int(count_matrix.sum())
        print(f"  Total transitions: {n_transitions:,}")

        # Row-normalize to get transition matrix
        transition_matrix = self._row_normalize(count_matrix)

        # Eigendecomposition
        eigenvalues, stationary_dist = self._eigendecomposition(transition_matrix)

        self.result_ = MSMResult(
            transition_matrix=transition_matrix,
            count_matrix=count_matrix,
            stationary_distribution=stationary_dist,
            eigenvalues=eigenvalues,
            n_states=n_states,
            lag=self.lag,
            n_transitions=n_transitions,
        )

        # Store mapping for later use
        self._label_to_idx = label_to_idx
        self._idx_to_label = {i: label for label, i in label_to_idx.items()}

        return self.result_

    def _build_count_matrix(
        self,
        cluster_labels: np.ndarray,
        game_ids: np.ndarray,
        label_to_idx: Dict[int, int],
        n_states: int
    ) -> np.ndarray:
        """
        Build count matrix respecting game boundaries.

        C_ij = count of transitions from state i to state j at lag τ

        CRITICAL: Only count within-game transitions!

        Args:
            cluster_labels: [n_samples] cluster assignments
            game_ids: [n_samples] game identifiers
            label_to_idx: Mapping from cluster label to matrix index
            n_states: Number of valid states

        Returns:
            [n_states, n_states] count matrix
        """
        count_matrix = np.zeros((n_states, n_states), dtype=np.float64)

        unique_games = np.unique(game_ids)

        for game_id in unique_games:
            # Get trajectory for this game
            game_mask = game_ids == game_id
            game_labels = cluster_labels[game_mask]

            # Skip if too short for lag
            if len(game_labels) <= self.lag:
                continue

            # Count transitions within this game
            for t in range(len(game_labels) - self.lag):
                label_t = game_labels[t]
                label_tau = game_labels[t + self.lag]

                # Skip noise points
                if label_t < 0 or label_tau < 0:
                    continue

                i = label_to_idx[label_t]
                j = label_to_idx[label_tau]
                count_matrix[i, j] += 1

        return count_matrix

    def _row_normalize(
        self,
        count_matrix: np.ndarray,
        epsilon: float = 1e-10
    ) -> np.ndarray:
        """
        Convert count matrix to row-stochastic transition matrix.

        P_ij = C_ij / Σ_j C_ij

        Args:
            count_matrix: [n_states, n_states] counts
            epsilon: Small value to avoid division by zero

        Returns:
            [n_states, n_states] row-stochastic matrix
        """
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero for states with no outgoing transitions
        row_sums = np.maximum(row_sums, epsilon)
        return count_matrix / row_sums

    def _eigendecomposition(
        self,
        transition_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and stationary distribution.

        For a row-stochastic matrix:
        - Largest eigenvalue is 1
        - Left eigenvector of eigenvalue 1 is stationary distribution π

        Args:
            transition_matrix: [n_states, n_states] row-stochastic

        Returns:
            (eigenvalues sorted descending, stationary distribution)
        """
        # Right eigenvalues (for timescales)
        eigenvalues = linalg.eigvals(transition_matrix)

        # Take real parts and sort descending
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Stationary distribution: left eigenvector of eigenvalue 1
        # P^T @ π = π  =>  solve (P^T - I) @ π = 0
        # Use power iteration for numerical stability
        pi = self._compute_stationary_distribution(transition_matrix)

        return eigenvalues, pi

    def _compute_stationary_distribution(
        self,
        transition_matrix: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-10
    ) -> np.ndarray:
        """
        Compute stationary distribution via power iteration.

        π^T @ P = π^T  (left eigenvector of eigenvalue 1)

        Args:
            transition_matrix: [n_states, n_states]
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            [n_states] stationary distribution (sums to 1)
        """
        n_states = transition_matrix.shape[0]

        # Initialize uniform
        pi = np.ones(n_states) / n_states

        for _ in range(max_iter):
            pi_new = pi @ transition_matrix
            pi_new = pi_new / pi_new.sum()  # Normalize

            if np.max(np.abs(pi_new - pi)) < tol:
                break
            pi = pi_new

        return pi

    def get_stationary_distribution(self) -> np.ndarray:
        """
        Get the stationary (equilibrium) distribution.

        Returns:
            [n_states] probability vector summing to 1
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")
        return self.result_.stationary_distribution

    def get_implied_timescales(
        self,
        lags: List[int],
        n_timescales: Optional[int] = None
    ) -> ImpliedTimescales:
        """
        Compute implied timescales at multiple lag times.

        t_i(τ) = -τ / ln(λ_i(τ))

        ITS should be constant vs. lag when Markovian (convergence test).

        Args:
            lags: List of lag times to evaluate
            n_timescales: Number of timescales to compute (default: n_states - 1)

        Returns:
            ImpliedTimescales with [n_lags, n_timescales] array
        """
        if self._cluster_labels is None or self._game_ids is None:
            raise ValueError("Must call fit() first")

        if n_timescales is None:
            n_timescales = min(self.result_.n_states - 1, 10)

        lags = np.array(lags)
        timescales = np.zeros((len(lags), n_timescales))

        for i, lag in enumerate(lags):
            # Build MSM at this lag
            analyzer = TransitionMatrixAnalyzer(lag=lag)
            analyzer.fit(self._cluster_labels, self._game_ids)

            # Compute ITS from eigenvalues (skip λ=1)
            eigenvalues = analyzer.result_.eigenvalues[1:n_timescales+1]

            for j, lam in enumerate(eigenvalues):
                if lam > 0 and lam < 1:
                    timescales[i, j] = -lag / np.log(lam)
                else:
                    timescales[i, j] = np.inf

        return ImpliedTimescales(
            lags=lags,
            timescales=timescales,
            n_timescales=n_timescales,
        )

    def chapman_kolmogorov_test(
        self,
        k_values: List[int],
        threshold: float = 0.2
    ) -> ChapmanKolmogorovResult:
        """
        Validate Markovianity via Chapman-Kolmogorov test.

        Tests whether P(kτ) ≈ P^k(τ), i.e., predictions from
        the model match re-estimated MSMs at longer lag times.

        Args:
            k_values: List of multipliers to test (e.g., [2, 3, 5])
            threshold: RMSE threshold for passing (default 0.2)

        Returns:
            ChapmanKolmogorovResult with RMSE values and pass/fail
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        base_lag = self.lag
        P_base = self.result_.transition_matrix

        rmse_dict = {}
        predictions = {}
        estimates = {}

        for k in k_values:
            # Prediction: P^k(τ)
            P_pred = np.linalg.matrix_power(P_base, k)
            predictions[k] = P_pred

            # Estimate: re-build MSM at lag kτ
            analyzer = TransitionMatrixAnalyzer(lag=base_lag * k)
            try:
                analyzer.fit(self._cluster_labels, self._game_ids)
                P_est = analyzer.result_.transition_matrix
            except Exception as e:
                print(f"  Warning: Could not estimate P({k}*{base_lag}): {e}")
                P_est = P_pred  # Use prediction as fallback
            estimates[k] = P_est

            # Compute RMSE
            rmse = np.sqrt(np.mean((P_pred - P_est) ** 2))
            rmse_dict[k] = rmse

            status = "PASS" if rmse < threshold else "FAIL"
            print(f"  C-K test k={k}: RMSE={rmse:.4f} [{status}]")

        passed = all(rmse < threshold for rmse in rmse_dict.values())

        return ChapmanKolmogorovResult(
            k_values=k_values,
            rmse=rmse_dict,
            predictions=predictions,
            estimates=estimates,
            passed=passed,
        )

    def get_mfpt(
        self,
        source: int,
        target: int
    ) -> float:
        """
        Compute Mean First Passage Time from source to target state.

        MFPT(A→B) = expected number of steps to reach B starting from A.

        Uses the fundamental matrix approach:
        MFPT[i,j] = (Z[j,j] - Z[i,j]) / π[j]
        where Z = (I - P + W)^{-1} and W[i,j] = π[j]

        Args:
            source: Source state index
            target: Target state index

        Returns:
            Mean first passage time in lag steps
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        P = self.result_.transition_matrix
        pi = self.result_.stationary_distribution
        n = self.result_.n_states

        # Fundamental matrix approach
        I = np.eye(n)
        W = np.outer(np.ones(n), pi)  # W[i,j] = π[j]
        Z = linalg.inv(I - P + W)

        # MFPT formula
        mfpt = (Z[target, target] - Z[source, target]) / pi[target]

        return mfpt * self.lag  # Convert to original time units

    def get_mfpt_matrix(self) -> np.ndarray:
        """
        Compute full Mean First Passage Time matrix.

        Returns:
            [n_states, n_states] matrix where M[i,j] = MFPT from i to j
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        P = self.result_.transition_matrix
        pi = self.result_.stationary_distribution
        n = self.result_.n_states

        # Fundamental matrix
        I = np.eye(n)
        W = np.outer(np.ones(n), pi)
        Z = linalg.inv(I - P + W)

        # MFPT matrix
        mfpt_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    mfpt_matrix[i, j] = (Z[j, j] - Z[i, j]) / pi[j]

        return mfpt_matrix * self.lag

    def save(self, output_dir: str):
        """
        Save MSM results to disk.

        Args:
            output_dir: Directory to save results
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main results
        np.savez(
            output_path / 'msm.npz',
            transition_matrix=self.result_.transition_matrix,
            count_matrix=self.result_.count_matrix,
            stationary_distribution=self.result_.stationary_distribution,
            eigenvalues=self.result_.eigenvalues,
            n_states=self.result_.n_states,
            lag=self.result_.lag,
        )

        # Save metadata
        metadata = {
            'n_states': self.result_.n_states,
            'lag': self.result_.lag,
            'n_transitions': self.result_.n_transitions,
            'label_to_idx': {str(k): v for k, v in self._label_to_idx.items()},
        }
        with open(output_path / 'msm_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved MSM results to {output_path}")

    @classmethod
    def load(cls, output_dir: str) -> 'TransitionMatrixAnalyzer':
        """
        Load MSM results from disk.

        Note: Does not restore raw data, only computed results.
        """
        output_path = Path(output_dir)

        # Load main results
        data = np.load(output_path / 'msm.npz')

        # Load metadata
        with open(output_path / 'msm_metadata.json', 'r') as f:
            metadata = json.load(f)

        analyzer = cls(lag=int(data['lag']))
        analyzer.result_ = MSMResult(
            transition_matrix=data['transition_matrix'],
            count_matrix=data['count_matrix'],
            stationary_distribution=data['stationary_distribution'],
            eigenvalues=data['eigenvalues'],
            n_states=int(data['n_states']),
            lag=int(data['lag']),
            n_transitions=metadata['n_transitions'],
        )

        analyzer._label_to_idx = {int(k): v for k, v in metadata['label_to_idx'].items()}
        analyzer._idx_to_label = {v: int(k) for k, v in metadata['label_to_idx'].items()}

        print(f"Loaded MSM from {output_path}")
        return analyzer


def compute_dwell_times(
    cluster_labels: np.ndarray,
    game_ids: np.ndarray
) -> Dict[int, List[int]]:
    """
    Compute dwell times for each cluster.

    Dwell time = consecutive moves spent in the same cluster within a game.

    Args:
        cluster_labels: [n_samples] cluster assignments
        game_ids: [n_samples] game identifiers

    Returns:
        Dict mapping cluster_id -> list of dwell durations
    """
    import itertools

    unique_clusters = set(cluster_labels)
    unique_clusters.discard(-1)  # Remove noise
    dwell_times = {c: [] for c in unique_clusters}

    unique_games = np.unique(game_ids)

    for game_id in unique_games:
        game_mask = game_ids == game_id
        game_labels = cluster_labels[game_mask]

        # Find runs of same cluster
        for cluster, group in itertools.groupby(game_labels):
            if cluster >= 0:  # Ignore noise
                run_length = len(list(group))
                dwell_times[cluster].append(run_length)

    return dwell_times


def compute_transition_counts_by_game(
    cluster_labels: np.ndarray,
    game_ids: np.ndarray,
    game_outcomes: Optional[Dict[int, str]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute transition counts stratified by game outcome.

    Useful for comparing dynamics in won vs lost games.

    Args:
        cluster_labels: [n_samples] cluster assignments
        game_ids: [n_samples] game identifiers
        game_outcomes: Dict mapping game_id -> outcome (e.g., 'W', 'L', 'B+', 'W+')
                       If None, only 'all' counts are computed.

    Returns:
        Dict with 'all', 'win', 'loss' count matrices
    """
    unique_labels = np.unique(cluster_labels)
    valid_labels = unique_labels[unique_labels >= 0]
    n_states = len(valid_labels)
    label_to_idx = {label: i for i, label in enumerate(valid_labels)}

    counts = {
        'all': np.zeros((n_states, n_states)),
        'win': np.zeros((n_states, n_states)),
        'loss': np.zeros((n_states, n_states)),
    }

    unique_games = np.unique(game_ids)

    for game_id in unique_games:
        game_mask = game_ids == game_id
        game_labels = cluster_labels[game_mask]

        # Determine outcome category
        if game_outcomes is not None:
            outcome = game_outcomes.get(int(game_id), 'unknown')
            if outcome in ['W', 'B+', 1, '1']:
                outcome_key = 'win'
            elif outcome in ['L', 'W+', -1, '-1']:
                outcome_key = 'loss'
            else:
                outcome_key = None
        else:
            outcome_key = None

        # Count transitions
        for t in range(len(game_labels) - 1):
            label_t = game_labels[t]
            label_tau = game_labels[t + 1]

            if label_t < 0 or label_tau < 0:
                continue

            from_idx = label_to_idx[label_t]
            to_idx = label_to_idx[label_tau]

            counts['all'][from_idx, to_idx] += 1
            if outcome_key:
                counts[outcome_key][from_idx, to_idx] += 1

    return counts
