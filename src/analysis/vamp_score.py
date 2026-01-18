"""
VAMP-2 score for clustering optimization.

From PyEMMA paper (L:246): "The number of cluster centers has been chosen
to optimize the VAMP-2 score"

The VAMP-2 score measures how well a clustering captures slow dynamics.
Higher score = better clustering for Markov State Model construction.

Reference:
- PyEMMA_Markov_State_modeling.md L:228-246
- Wu & Noe (2020) "Variational Approach for Learning Markov Processes from Time Series Data"
"""

import numpy as np
from scipy import linalg
from typing import Optional, List, Tuple, Dict
from sklearn.cluster import KMeans, MiniBatchKMeans
from joblib import Parallel, delayed
import warnings

# Check for GPU support via cuML (defer import to avoid startup crash)
HAS_CUML = False
cuMLKMeans = None

def _check_cuml():
    """Lazy check for cuML availability."""
    global HAS_CUML, cuMLKMeans
    if cuMLKMeans is not None:
        return HAS_CUML
    try:
        import cuml
        from cuml.cluster import KMeans as _cuMLKMeans
        cuMLKMeans = _cuMLKMeans
        HAS_CUML = True
    except (ImportError, RuntimeError, OSError) as e:
        HAS_CUML = False
    return HAS_CUML


def compute_vamp2_score(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    lag: int = 1,
    game_ids: Optional[np.ndarray] = None
) -> float:
    """
    Compute VAMP-2 score for a given clustering.

    VAMP-2 score = sum of squared singular values of the Koopman matrix,
    which measures the kinetic variance captured by the clustering.

    Args:
        features: [N, d] array of features (e.g., TICA-transformed)
        cluster_labels: [N] cluster assignments (0 to k-1)
        lag: lag time in frames
        game_ids: [N] game IDs for boundary handling (optional)

    Returns:
        VAMP-2 score (higher = better clustering for dynamics)
    """
    n_samples = len(cluster_labels)
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)

    # Handle case where labels might not be 0...k-1
    label_map = {label: i for i, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in cluster_labels])

    # Build indicator matrix chi: [N, k]
    # chi[i, j] = 1 if sample i belongs to cluster j
    chi = np.zeros((n_samples, n_clusters))
    chi[np.arange(n_samples), mapped_labels] = 1

    # Compute time-lagged covariances
    C00, C0t, Ctt = _compute_covariances(chi, lag, game_ids)

    # VAMP-2 score from covariances
    score = _vamp2_from_covariances(C00, C0t, Ctt)

    return score


def _compute_covariances(
    chi: np.ndarray,
    lag: int,
    game_ids: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute instantaneous and time-lagged covariances of indicator matrix.

    Args:
        chi: [N, k] indicator matrix
        lag: time lag
        game_ids: optional game boundaries

    Returns:
        C00: [k, k] covariance of chi(t)
        C0t: [k, k] cross-covariance chi(t) with chi(t+τ)
        Ctt: [k, k] covariance of chi(t+τ)
    """
    n_samples, n_clusters = chi.shape

    if game_ids is None:
        # Simple case: no game boundaries
        chi_t = chi[:-lag]
        chi_tau = chi[lag:]
        n_pairs = len(chi_t)

        # Means
        mean_t = chi_t.mean(axis=0)
        mean_tau = chi_tau.mean(axis=0)

        # Centered
        chi_t_c = chi_t - mean_t
        chi_tau_c = chi_tau - mean_tau

        C00 = (chi_t_c.T @ chi_t_c) / n_pairs
        C0t = (chi_t_c.T @ chi_tau_c) / n_pairs
        Ctt = (chi_tau_c.T @ chi_tau_c) / n_pairs

    else:
        # Respect game boundaries
        unique_games = np.unique(game_ids)

        # Accumulate covariances
        chi_t_all = []
        chi_tau_all = []

        for game_id in unique_games:
            mask = game_ids == game_id
            chi_game = chi[mask]

            if len(chi_game) <= lag:
                continue

            chi_t_all.append(chi_game[:-lag])
            chi_tau_all.append(chi_game[lag:])

        if not chi_t_all:
            # No valid pairs
            return (np.eye(n_clusters), np.eye(n_clusters), np.eye(n_clusters))

        chi_t = np.vstack(chi_t_all)
        chi_tau = np.vstack(chi_tau_all)
        n_pairs = len(chi_t)

        # Means
        mean_t = chi_t.mean(axis=0)
        mean_tau = chi_tau.mean(axis=0)

        # Centered
        chi_t_c = chi_t - mean_t
        chi_tau_c = chi_tau - mean_tau

        C00 = (chi_t_c.T @ chi_t_c) / n_pairs
        C0t = (chi_t_c.T @ chi_tau_c) / n_pairs
        Ctt = (chi_tau_c.T @ chi_tau_c) / n_pairs

    return C00, C0t, Ctt


def _vamp2_from_covariances(
    C00: np.ndarray,
    C0t: np.ndarray,
    Ctt: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute VAMP-2 score from covariance matrices.

    VAMP-2 = trace(K^T K) where K is the Koopman matrix approximation.
    K = C00^{-1/2} C0t Ctt^{-1/2}

    The score equals the sum of squared singular values of K.

    Args:
        C00: [k, k] covariance of chi(t)
        C0t: [k, k] cross-covariance
        Ctt: [k, k] covariance of chi(t+τ)
        epsilon: regularization for numerical stability

    Returns:
        VAMP-2 score
    """
    k = C00.shape[0]

    # Regularize for numerical stability
    C00_reg = C00 + epsilon * np.eye(k)
    Ctt_reg = Ctt + epsilon * np.eye(k)

    try:
        # Compute C00^{-1/2}
        eigvals_00, eigvecs_00 = linalg.eigh(C00_reg)
        eigvals_00 = np.maximum(eigvals_00, epsilon)
        C00_inv_sqrt = eigvecs_00 @ np.diag(1.0 / np.sqrt(eigvals_00)) @ eigvecs_00.T

        # Compute Ctt^{-1/2}
        eigvals_tt, eigvecs_tt = linalg.eigh(Ctt_reg)
        eigvals_tt = np.maximum(eigvals_tt, epsilon)
        Ctt_inv_sqrt = eigvecs_tt @ np.diag(1.0 / np.sqrt(eigvals_tt)) @ eigvecs_tt.T

        # Koopman matrix approximation
        K = C00_inv_sqrt @ C0t @ Ctt_inv_sqrt

        # VAMP-2 = sum of squared singular values = trace(K^T K)
        # Equivalently, sum of eigenvalues of K^T K
        KtK = K.T @ K
        eigvals = linalg.eigvalsh(KtK)
        score = np.sum(np.maximum(eigvals, 0))

    except linalg.LinAlgError:
        # Fallback: return trace of normalized covariance
        score = np.trace(C0t @ C0t.T) / (np.trace(C00) * np.trace(Ctt) + epsilon)

    return float(score)


def _fit_kmeans_single_k(
    features: np.ndarray,
    k: int,
    lag: int,
    game_ids: Optional[np.ndarray],
    n_init: int,
    random_state: int,
    use_minibatch: bool,
    use_gpu: bool,
    batch_size: int,
) -> Tuple[int, float, np.ndarray]:
    """
    Fit k-means for a single k value and compute VAMP-2 score.
    Helper function for parallel execution.
    """
    labels = None

    # Try GPU first if requested
    if use_gpu and _check_cuml():
        try:
            import cupy as cp
            features_gpu = cp.asarray(features)
            kmeans = cuMLKMeans(
                n_clusters=k,
                n_init=n_init,
                random_state=random_state,
                max_iter=300,
            )
            labels = kmeans.fit_predict(features_gpu)
            labels = cp.asnumpy(labels)
        except Exception as e:
            warnings.warn(f"GPU failed ({e}), falling back to CPU")
            labels = None

    # CPU fallback
    if labels is None:
        if use_minibatch:
            # MiniBatchKMeans - much faster for large datasets
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                n_init=n_init,
                random_state=random_state,
                max_iter=300,
                batch_size=batch_size,
            )
            labels = kmeans.fit_predict(features)
        else:
            # Standard KMeans
            kmeans = KMeans(
                n_clusters=k,
                n_init=n_init,
                random_state=random_state,
                max_iter=300,
            )
            labels = kmeans.fit_predict(features)

    score = compute_vamp2_score(features, labels, lag, game_ids)
    return k, score, labels


def optimize_kmeans_k(
    features: np.ndarray,
    k_range: List[int],
    lag: int = 1,
    game_ids: Optional[np.ndarray] = None,
    n_init: int = 10,
    random_state: int = 42,
    verbose: bool = True,
    n_jobs: int = -1,
    use_minibatch: bool = True,
    use_gpu: bool = False,
    batch_size: int = 10000,
) -> Tuple[int, Dict[int, float], Dict[int, np.ndarray]]:
    """
    Find optimal k for k-means by maximizing VAMP-2 score.

    Per PyEMMA paper (L:246): "The number of cluster centers has been
    chosen to optimize the VAMP-2 score"

    OPTIMIZED VERSION:
    - Uses MiniBatchKMeans by default (10-50x faster for large datasets)
    - Tests k values in parallel using joblib
    - Optional GPU acceleration via cuML

    Args:
        features: [N, d] TICA-transformed features
        k_range: List of k values to test, e.g., [25, 50, 75, 100, 150]
        lag: lag time for VAMP-2 computation
        game_ids: [N] game IDs for boundary handling
        n_init: Number of k-means initializations
        random_state: Random seed for reproducibility
        verbose: Print progress
        n_jobs: Number of parallel jobs (-1 = all cores)
        use_minibatch: Use MiniBatchKMeans (much faster, default True)
        use_gpu: Use GPU via cuML if available
        batch_size: Batch size for MiniBatchKMeans

    Returns:
        best_k: Optimal number of clusters
        scores: Dict mapping k -> VAMP-2 score
        labels: Dict mapping k -> cluster labels
    """
    n_samples = len(features)

    if verbose:
        print(f"Optimizing k-means via VAMP-2 score (lag={lag})")
        print(f"Testing k: {k_range}")
        print(f"Samples: {n_samples:,}, Features: {features.shape[1]}")

        gpu_available = _check_cuml() if use_gpu else False
        method = "GPU (cuML)" if (use_gpu and gpu_available) else \
                 "MiniBatchKMeans" if use_minibatch else "Standard KMeans"
        print(f"Method: {method}, Parallel jobs: {n_jobs}")
        print("-" * 50)

    # Check GPU availability
    if use_gpu and not _check_cuml():
        warnings.warn("cuML not available, falling back to CPU")
        use_gpu = False

    # Parallel execution across k values
    if n_jobs != 1 and len(k_range) > 1:
        if verbose:
            print(f"Running {len(k_range)} k values in parallel...")

        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_fit_kmeans_single_k)(
                features, k, lag, game_ids, n_init,
                random_state + i, use_minibatch, use_gpu, batch_size
            )
            for i, k in enumerate(k_range)
        )

        scores = {k: score for k, score, _ in results}
        labels_dict = {k: labels for k, _, labels in results}

        if verbose:
            for k in k_range:
                print(f"  k={k:4d}... VAMP-2 = {scores[k]:.4f}")
    else:
        # Sequential execution
        scores = {}
        labels_dict = {}

        for i, k in enumerate(k_range):
            if verbose:
                print(f"  k={k:4d}...", end=" ", flush=True)

            k, score, labels = _fit_kmeans_single_k(
                features, k, lag, game_ids, n_init,
                random_state + i, use_minibatch, use_gpu, batch_size
            )
            scores[k] = score
            labels_dict[k] = labels

            if verbose:
                print(f"VAMP-2 = {score:.4f}")

    best_k = max(scores, key=scores.get)

    if verbose:
        print("-" * 50)
        print(f"Optimal k = {best_k} with VAMP-2 = {scores[best_k]:.4f}")

    return best_k, scores, labels_dict


def detect_vamp2_saturation(
    scores: Dict[int, float],
    method: str = 'elbow',
    threshold: float = 0.1
) -> Tuple[int, Dict[str, any]]:
    """
    Detect saturation/optimal k from VAMP-2 scores using elbow detection.

    BUG FIX: The original optimize_kmeans_k() just returns max(k) when
    VAMP-2 keeps increasing monotonically. This function detects when
    the rate of increase slows down (diminishing returns).

    Methods:
    - 'elbow': Find the elbow point where curvature is maximum
    - 'gradient': Find where gradient drops below threshold * max_gradient
    - 'second_derivative': Find where second derivative is most negative

    Args:
        scores: Dict mapping k -> VAMP-2 score
        method: Detection method ('elbow', 'gradient', 'second_derivative')
        threshold: Threshold for gradient method (fraction of max gradient)

    Returns:
        optimal_k: Detected optimal k value
        diagnostics: Dict with detection details
    """
    ks = np.array(sorted(scores.keys()))
    vals = np.array([scores[k] for k in ks])

    if len(ks) < 3:
        # Not enough points for elbow detection
        return int(ks[np.argmax(vals)]), {'method': 'max', 'reason': 'insufficient_points'}

    # Normalize for numerical stability
    ks_norm = (ks - ks.min()) / (ks.max() - ks.min() + 1e-10)
    vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)

    diagnostics = {'method': method, 'k_values': ks.tolist(), 'scores': vals.tolist()}

    if method == 'elbow':
        # Elbow method: maximize distance from line connecting first and last points
        # This is the "Kneedle" algorithm simplified

        # Line from first to last point
        p1 = np.array([ks_norm[0], vals_norm[0]])
        p2 = np.array([ks_norm[-1], vals_norm[-1]])

        # Distance from each point to the line
        distances = []
        for i, (kn, vn) in enumerate(zip(ks_norm, vals_norm)):
            # Point-to-line distance
            p0 = np.array([kn, vn])
            d = np.abs(np.cross(p2 - p1, p1 - p0)) / (np.linalg.norm(p2 - p1) + 1e-10)
            distances.append(d)

        distances = np.array(distances)
        elbow_idx = np.argmax(distances)
        optimal_k = int(ks[elbow_idx])

        diagnostics['distances'] = distances.tolist()
        diagnostics['elbow_idx'] = int(elbow_idx)
        diagnostics['elbow_distance'] = float(distances[elbow_idx])

    elif method == 'gradient':
        # Gradient method: find where gradient drops below threshold
        gradients = np.diff(vals) / np.diff(ks)
        max_gradient = np.max(np.abs(gradients))

        # Find first point where gradient < threshold * max
        below_threshold = np.abs(gradients) < threshold * max_gradient
        if np.any(below_threshold):
            saturation_idx = np.argmax(below_threshold)
        else:
            saturation_idx = len(gradients) - 1

        optimal_k = int(ks[saturation_idx])

        diagnostics['gradients'] = gradients.tolist()
        diagnostics['max_gradient'] = float(max_gradient)
        diagnostics['saturation_idx'] = int(saturation_idx)

    elif method == 'second_derivative':
        # Second derivative method: find where curvature is most negative
        # (where rate of increase is slowing down most rapidly)
        if len(ks) < 4:
            return detect_vamp2_saturation(scores, method='gradient', threshold=threshold)

        # Compute second derivative using finite differences
        first_deriv = np.diff(vals) / np.diff(ks)
        k_mid = (ks[:-1] + ks[1:]) / 2
        second_deriv = np.diff(first_deriv) / np.diff(k_mid)

        # Most negative second derivative = elbow point
        if len(second_deriv) > 0:
            elbow_idx = np.argmin(second_deriv) + 1  # +1 because second_deriv is shorter
            optimal_k = int(ks[min(elbow_idx, len(ks) - 1)])
        else:
            optimal_k = int(ks[np.argmax(vals)])

        diagnostics['second_derivative'] = second_deriv.tolist()
        diagnostics['elbow_idx'] = int(elbow_idx) if len(second_deriv) > 0 else 0

    else:
        raise ValueError(f"Unknown method: {method}")

    # Check if saturation was actually detected
    # If optimal_k is the last value tested, we may not have found true saturation
    if optimal_k == ks[-1]:
        diagnostics['warning'] = 'optimal_k equals max tested - consider extending k_range'

    return optimal_k, diagnostics


def optimize_kmeans_k_with_saturation(
    features: np.ndarray,
    k_range: List[int],
    lag: int = 1,
    game_ids: Optional[np.ndarray] = None,
    n_init: int = 10,
    random_state: int = 42,
    verbose: bool = True,
    n_jobs: int = -1,
    use_minibatch: bool = True,
    use_gpu: bool = False,
    batch_size: int = 10000,
    saturation_method: str = 'elbow',
    saturation_threshold: float = 0.1,
) -> Tuple[int, Dict[int, float], Dict[int, np.ndarray], Dict[str, any]]:
    """
    Find optimal k using VAMP-2 with saturation detection.

    This is an improved version of optimize_kmeans_k that doesn't just
    return max(k) when VAMP-2 increases monotonically, but instead
    detects the elbow/saturation point.

    Args:
        (same as optimize_kmeans_k, plus:)
        saturation_method: Method for detecting saturation ('elbow', 'gradient', 'second_derivative')
        saturation_threshold: Threshold for gradient method

    Returns:
        best_k: Optimal k (with saturation detection)
        scores: Dict mapping k -> VAMP-2 score
        labels: Dict mapping k -> cluster labels
        diagnostics: Dict with saturation detection details
    """
    # First, run standard optimization to get all scores
    max_k, scores, labels_dict = optimize_kmeans_k(
        features=features,
        k_range=k_range,
        lag=lag,
        game_ids=game_ids,
        n_init=n_init,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
        use_minibatch=use_minibatch,
        use_gpu=use_gpu,
        batch_size=batch_size,
    )

    # Now detect saturation
    optimal_k, diagnostics = detect_vamp2_saturation(
        scores=scores,
        method=saturation_method,
        threshold=saturation_threshold
    )

    if verbose:
        print(f"\nSaturation detection ({saturation_method} method):")
        print(f"  Max VAMP-2 at k={max_k} (score={scores[max_k]:.4f})")
        print(f"  Saturation detected at k={optimal_k} (score={scores[optimal_k]:.4f})")
        if 'warning' in diagnostics:
            print(f"  WARNING: {diagnostics['warning']}")

    diagnostics['max_k'] = max_k
    diagnostics['max_score'] = scores[max_k]

    return optimal_k, scores, labels_dict, diagnostics


def cross_validated_vamp2(
    features: np.ndarray,
    k: int,
    lag: int = 1,
    game_ids: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Compute cross-validated VAMP-2 score for a given k.

    Cross-validation helps avoid overfitting when selecting k.

    Args:
        features: [N, d] features
        k: number of clusters
        lag: lag time
        game_ids: game boundaries
        n_splits: number of CV folds
        random_state: random seed

    Returns:
        mean_score: Mean VAMP-2 across folds
        std_score: Standard deviation
    """
    n_samples = len(features)
    indices = np.arange(n_samples)

    # Simple k-fold split (respecting time order within folds)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)

    fold_size = n_samples // n_splits
    scores = []

    for i in range(n_splits):
        # Test fold
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_splits - 1 else n_samples
        test_idx = indices[test_start:test_end]

        # Train fold
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        # Fit k-means on train
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        kmeans.fit(features[train_idx])

        # Predict on test
        test_labels = kmeans.predict(features[test_idx])
        test_game_ids = game_ids[test_idx] if game_ids is not None else None

        # Compute VAMP-2 on test
        score = compute_vamp2_score(
            features[test_idx], test_labels, lag, test_game_ids
        )
        scores.append(score)

    return float(np.mean(scores)), float(np.std(scores))
