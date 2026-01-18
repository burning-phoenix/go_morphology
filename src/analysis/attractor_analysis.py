"""
Attractor Basin Analysis using HDBSCAN clustering.

Identifies metastable strategic states (attractors) in the TICA-reduced
feature space using density-based clustering.

Reference:
- docs/topological_data_analysis/INDEX.md L:690-885
- HDBSCAN paper: "Accelerated Hierarchical Density Based Clustering"

Key concepts:
- HDBSCAN finds clusters of varying densities via density-based level sets
- Stability score σ(C) = excess of mass, measures cluster persistence
- Noise points labeled as -1
"""

import numpy as np
import hdbscan
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClusterMetrics:
    """Per-cluster statistics."""
    cluster_id: int
    size: int                           # Number of positions in cluster
    stability: float                    # HDBSCAN cluster persistence
    centroid: np.ndarray               # Cluster center in TICA space
    mean_features: Optional[np.ndarray] = None  # Mean SAE features [4096]
    concept_signature: Dict[str, float] = field(default_factory=dict)
    dwell_time_mean: float = 0.0
    dwell_time_std: float = 0.0


@dataclass
class ClusteringResult:
    """Full clustering result."""
    labels: np.ndarray                  # [n_samples] cluster assignments (-1 = noise)
    n_clusters: int                     # Number of clusters (excluding noise)
    noise_ratio: float                  # Fraction of points labeled as noise
    stability_scores: Dict[int, float]  # Cluster ID -> stability
    centroids: np.ndarray              # [n_clusters, n_features] cluster centers
    silhouette: Optional[float] = None  # Silhouette score if computable


class AttractorAnalyzer:
    """
    Identify and characterize strategic attractors via HDBSCAN.

    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications
    with Noise) finds clusters of varying densities without requiring
    the number of clusters to be specified.

    Usage:
        analyzer = AttractorAnalyzer(min_cluster_size=500, min_samples=250)
        result = analyzer.fit(tica_features)
        metrics = analyzer.compute_cluster_metrics(sae_features)
    """

    def __init__(
        self,
        min_cluster_size: int = 500,
        min_samples: int = 250,
        cluster_selection_method: str = 'eom',
        metric: str = 'euclidean',
        n_jobs: int = -1
    ):
        """
        Args:
            min_cluster_size: Minimum cluster size (default 500 = 0.1% of 500K)
            min_samples: Core point density threshold
            cluster_selection_method: 'eom' (excess of mass) or 'leaf'
            metric: Distance metric ('euclidean', 'cosine', etc.)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
        self.n_jobs = n_jobs

        self.clusterer_: Optional[hdbscan.HDBSCAN] = None
        self.result_: Optional[ClusteringResult] = None

    def fit(self, features: np.ndarray) -> ClusteringResult:
        """
        Cluster features using HDBSCAN.

        Args:
            features: [n_samples, n_features] TICA-transformed features

        Returns:
            ClusteringResult with labels, stability scores, etc.
        """
        n_samples, n_features = features.shape
        print(f"Clustering {n_samples:,} samples × {n_features} features")
        print(f"  min_cluster_size: {self.min_cluster_size}")
        print(f"  min_samples: {self.min_samples}")
        print(f"  selection: {self.cluster_selection_method}")
        print(f"  metric: {self.metric}")

        # Fit HDBSCAN
        self.clusterer_ = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            metric=self.metric,
            prediction_data=True,  # Enable soft clustering
            core_dist_n_jobs=self.n_jobs,
        )

        labels = self.clusterer_.fit_predict(features)

        # Extract results
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_mask = labels == -1
        noise_ratio = noise_mask.sum() / len(labels)

        print(f"  Found {n_clusters} clusters")
        print(f"  Noise ratio: {noise_ratio:.3f}")

        # Stability scores (cluster persistence)
        # HDBSCAN stores these in clusterer_.cluster_persistence_
        stability_scores = {}
        if hasattr(self.clusterer_, 'cluster_persistence_'):
            for i, stability in enumerate(self.clusterer_.cluster_persistence_):
                stability_scores[i] = stability

        # Compute centroids
        centroids = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            centroid = features[mask].mean(axis=0)
            centroids.append(centroid)

        centroids = np.array(centroids) if centroids else np.empty((0, n_features))

        # Compute silhouette score if we have enough clusters
        silhouette = None
        if n_clusters >= 2 and noise_ratio < 0.9:
            # Use non-noise points only
            valid_mask = ~noise_mask
            if valid_mask.sum() > n_clusters:
                try:
                    silhouette = silhouette_score(
                        features[valid_mask],
                        labels[valid_mask]
                    )
                    print(f"  Silhouette score: {silhouette:.4f}")
                except Exception as e:
                    print(f"  Could not compute silhouette: {e}")

        self.result_ = ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            noise_ratio=noise_ratio,
            stability_scores=stability_scores,
            centroids=centroids,
            silhouette=silhouette,
        )

        return self.result_

    def compute_cluster_metrics(
        self,
        sae_features: Optional[np.ndarray] = None,
        concepts: Optional[Dict[str, np.ndarray]] = None,
        dwell_times: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[int, ClusterMetrics]:
        """
        Compute comprehensive statistics for each cluster.

        Args:
            sae_features: [n_samples, 4096] full SAE features (optional)
            concepts: Dict mapping concept name -> [n_samples] activations
            dwell_times: Dict mapping cluster_id -> list of dwell durations

        Returns:
            Dict mapping cluster_id -> ClusterMetrics
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        metrics = {}

        for cluster_id in range(self.result_.n_clusters):
            mask = self.result_.labels == cluster_id
            size = mask.sum()

            # Stability
            stability = self.result_.stability_scores.get(cluster_id, 0.0)

            # Centroid
            centroid = self.result_.centroids[cluster_id]

            # Mean SAE features
            mean_features = None
            if sae_features is not None:
                mean_features = sae_features[mask].mean(axis=0)

            # Concept signature
            concept_signature = {}
            if concepts is not None:
                for name, values in concepts.items():
                    concept_signature[name] = values[mask].mean()

            # Dwell time stats
            dwell_mean = 0.0
            dwell_std = 0.0
            if dwell_times is not None and cluster_id in dwell_times:
                times = dwell_times[cluster_id]
                if times:
                    dwell_mean = np.mean(times)
                    dwell_std = np.std(times)

            metrics[cluster_id] = ClusterMetrics(
                cluster_id=cluster_id,
                size=size,
                stability=stability,
                centroid=centroid,
                mean_features=mean_features,
                concept_signature=concept_signature,
                dwell_time_mean=dwell_mean,
                dwell_time_std=dwell_std,
            )

        return metrics

    def get_cluster_prototypes(
        self,
        features: np.ndarray,
        k: int = 5
    ) -> Dict[int, List[int]]:
        """
        Get indices of k positions closest to each cluster centroid.

        Args:
            features: [n_samples, n_features] feature matrix
            k: Number of prototypes per cluster

        Returns:
            Dict mapping cluster_id -> list of position indices
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        prototypes = {}

        for cluster_id in range(self.result_.n_clusters):
            centroid = self.result_.centroids[cluster_id]
            mask = self.result_.labels == cluster_id
            indices = np.where(mask)[0]

            # Compute distances to centroid
            cluster_features = features[mask]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)

            # Get k closest
            closest_indices = np.argsort(distances)[:k]
            prototypes[cluster_id] = indices[closest_indices].tolist()

        return prototypes

    def save(self, output_dir: str):
        """
        Save clustering results to disk.

        Args:
            output_dir: Directory to save results
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main results
        np.savez(
            output_path / 'clusters.npz',
            labels=self.result_.labels,
            centroids=self.result_.centroids,
            n_clusters=self.result_.n_clusters,
            noise_ratio=self.result_.noise_ratio,
        )

        # Save stability scores separately (dict)
        import json
        with open(output_path / 'cluster_stability.json', 'w') as f:
            json.dump(
                {str(k): v for k, v in self.result_.stability_scores.items()},
                f, indent=2
            )

        print(f"Saved clustering results to {output_path}")

    @classmethod
    def load(cls, output_dir: str) -> 'AttractorAnalyzer':
        """
        Load clustering results from disk.

        Note: Does not restore the full HDBSCAN clusterer, only results.
        """
        import json

        output_path = Path(output_dir)

        # Load main results
        data = np.load(output_path / 'clusters.npz')

        # Load stability scores
        with open(output_path / 'cluster_stability.json', 'r') as f:
            stability_scores = {int(k): v for k, v in json.load(f).items()}

        analyzer = cls()
        analyzer.result_ = ClusteringResult(
            labels=data['labels'],
            centroids=data['centroids'],
            n_clusters=int(data['n_clusters']),
            noise_ratio=float(data['noise_ratio']),
            stability_scores=stability_scores,
        )

        return analyzer


@dataclass
class KMeansResult:
    """Results from K-means clustering for MSM."""
    labels: np.ndarray                  # [n_samples] cluster assignments (0 to k-1)
    n_clusters: int                     # Number of clusters
    centers: np.ndarray                 # [n_clusters, n_features] cluster centers
    inertia: float                      # Sum of squared distances to nearest center
    silhouette: Optional[float] = None  # Silhouette score if computable


class KMeansClusterer:
    """
    K-means clustering for MSM discretization.

    This is the standard method per PyEMMA literature (L:246):
    "we use the k-means algorithm to segment the four dimensional TICA space"

    K-means is preferred over HDBSCAN for MSM construction because:
    1. All points are assigned to a cluster (no noise label)
    2. Uniform partitioning of state space
    3. Compatible with VAMP-2 score optimization
    4. 20+ years of validated use in MSM literature

    HDBSCAN is kept separately in AttractorAnalyzer for density-based
    analysis where natural cluster structure is of interest.

    Usage:
        clusterer = KMeansClusterer(n_clusters=75)
        result = clusterer.fit(tica_features)

        # Or optimize k via VAMP-2:
        from vamp_score import optimize_kmeans_k
        best_k, scores, _ = optimize_kmeans_k(features, [50, 75, 100])
        clusterer = KMeansClusterer(n_clusters=best_k)
        result = clusterer.fit(features)
    """

    def __init__(
        self,
        n_clusters: int = 75,
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42,
        use_minibatch: bool = False,
        batch_size: int = 10000
    ):
        """
        Args:
            n_clusters: Number of clusters (optimize via VAMP-2 score)
            n_init: Number of random initializations
            max_iter: Maximum iterations per initialization
            random_state: Random seed for reproducibility
            use_minibatch: Use MiniBatchKMeans for faster processing on large datasets
            batch_size: Batch size for MiniBatchKMeans (default 10000)
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_minibatch = use_minibatch
        self.batch_size = batch_size

        self.kmeans_ = None  # KMeans or MiniBatchKMeans
        self.result_: Optional[KMeansResult] = None

    def fit(self, features: np.ndarray) -> KMeansResult:
        """
        Fit K-means clustering to TICA-transformed features.

        Args:
            features: [n_samples, n_features] TICA features

        Returns:
            KMeansResult with labels, centers, etc.
        """
        n_samples, n_features = features.shape
        print(f"K-means clustering: {n_samples:,} samples × {n_features} features")
        print(f"  n_clusters: {self.n_clusters}")
        print(f"  n_init: {self.n_init}")

        # Use MiniBatchKMeans for faster processing on large datasets
        if self.use_minibatch:
            print(f"  Using MiniBatchKMeans (batch_size={self.batch_size})")
            self.kmeans_ = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                batch_size=self.batch_size,
            )
        else:
            self.kmeans_ = KMeans(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )

        labels = self.kmeans_.fit_predict(features)
        centers = self.kmeans_.cluster_centers_
        inertia = self.kmeans_.inertia_

        print(f"  Inertia: {inertia:.2f}")

        # Compute silhouette score (subsample for large datasets to avoid O(n²) hang)
        silhouette = None
        MAX_SILHOUETTE_SAMPLES = 50000
        if self.n_clusters >= 2:
            try:
                if n_samples > MAX_SILHOUETTE_SAMPLES:
                    print(f"  Computing silhouette on {MAX_SILHOUETTE_SAMPLES:,} sample subset...")
                    sample_idx = np.random.choice(n_samples, MAX_SILHOUETTE_SAMPLES, replace=False)
                    silhouette = silhouette_score(features[sample_idx], labels[sample_idx])
                else:
                    silhouette = silhouette_score(features, labels)
                print(f"  Silhouette: {silhouette:.4f}")
            except Exception as e:
                print(f"  Could not compute silhouette: {e}")

        # Report cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        sizes = dict(zip(unique, counts))
        min_size = min(counts)
        max_size = max(counts)
        mean_size = np.mean(counts)
        print(f"  Cluster sizes: min={min_size}, max={max_size}, mean={mean_size:.1f}")

        self.result_ = KMeansResult(
            labels=labels,
            n_clusters=self.n_clusters,
            centers=centers,
            inertia=inertia,
            silhouette=silhouette,
        )

        return self.result_

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Assign new samples to nearest cluster centers.

        Args:
            features: [n_samples, n_features] feature matrix

        Returns:
            [n_samples] cluster labels
        """
        if self.kmeans_ is None:
            raise ValueError("Must call fit() first")
        return self.kmeans_.predict(features)

    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Get size of each cluster.

        Returns:
            Dict mapping cluster_id -> count
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        unique, counts = np.unique(self.result_.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def get_cluster_prototypes(
        self,
        features: np.ndarray,
        k: int = 5
    ) -> Dict[int, List[int]]:
        """
        Get indices of k positions closest to each cluster center.

        Args:
            features: [n_samples, n_features] feature matrix
            k: Number of prototypes per cluster

        Returns:
            Dict mapping cluster_id -> list of position indices
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        prototypes = {}

        for cluster_id in range(self.n_clusters):
            center = self.result_.centers[cluster_id]
            mask = self.result_.labels == cluster_id
            indices = np.where(mask)[0]

            # Compute distances to center
            cluster_features = features[mask]
            distances = np.linalg.norm(cluster_features - center, axis=1)

            # Get k closest
            n_select = min(k, len(distances))
            closest_indices = np.argsort(distances)[:n_select]
            prototypes[cluster_id] = indices[closest_indices].tolist()

        return prototypes

    def compute_transition_matrix(
        self,
        game_ids: np.ndarray,
        lag: int = 1
    ) -> np.ndarray:
        """
        Compute transition count matrix from cluster assignments.

        Respects game boundaries - no transitions across games.

        Args:
            game_ids: [n_samples] game index for each position
            lag: Time lag in steps

        Returns:
            [n_clusters, n_clusters] transition count matrix
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        labels = self.result_.labels
        n_clusters = self.n_clusters
        counts = np.zeros((n_clusters, n_clusters), dtype=np.int64)

        unique_games = np.unique(game_ids)

        for game_id in unique_games:
            mask = game_ids == game_id
            game_labels = labels[mask]

            if len(game_labels) <= lag:
                continue

            # Count transitions within this game
            for i in range(len(game_labels) - lag):
                from_state = game_labels[i]
                to_state = game_labels[i + lag]
                counts[from_state, to_state] += 1

        return counts

    def save(self, output_path: str):
        """
        Save clustering results to disk.

        Args:
            output_path: Path to save file (npz format)
        """
        if self.result_ is None:
            raise ValueError("Must call fit() first")

        np.savez(
            output_path,
            labels=self.result_.labels,
            centers=self.result_.centers,
            n_clusters=self.result_.n_clusters,
            inertia=self.result_.inertia,
            silhouette=self.result_.silhouette if self.result_.silhouette else -1,
        )
        print(f"Saved K-means results to {output_path}")

    @classmethod
    def load(cls, input_path: str) -> 'KMeansClusterer':
        """
        Load clustering results from disk.

        Note: Does not restore the sklearn KMeans object, only results.
        """
        data = np.load(input_path)

        n_clusters = int(data['n_clusters'])
        clusterer = cls(n_clusters=n_clusters)

        silhouette = float(data['silhouette'])
        if silhouette == -1:
            silhouette = None

        clusterer.result_ = KMeansResult(
            labels=data['labels'],
            centers=data['centers'],
            n_clusters=n_clusters,
            inertia=float(data['inertia']),
            silhouette=silhouette,
        )

        print(f"Loaded K-means results from {input_path}")
        return clusterer


def run_kmeans_baseline(
    features: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Run K-means clustering as a baseline comparison.

    Args:
        features: [n_samples, n_features] feature matrix
        n_clusters: Number of clusters (use same as HDBSCAN for comparison)
        random_state: Random seed

    Returns:
        (labels, silhouette_score)
    """
    print(f"Running K-means baseline with k={n_clusters}...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)

    silhouette = silhouette_score(features, labels)
    print(f"  K-means silhouette: {silhouette:.4f}")

    return labels, silhouette


# =============================================================================
# HDBSCAN Parameter Experiments
# =============================================================================

@dataclass
class HDBSCANExperimentResult:
    """Result from a single HDBSCAN experiment."""
    config: Dict
    n_clusters: int
    noise_ratio: float
    silhouette: Optional[float]
    labels: np.ndarray
    reduction_method: str  # 'pca' or 'umap'
    n_dims: int
    elapsed_time: float


@dataclass
class HDBSCANExperimentSummary:
    """Summary of all HDBSCAN parameter experiments."""
    results: List[HDBSCANExperimentResult]
    best_result: Optional[HDBSCANExperimentResult]  # Best by silhouette
    all_noise: bool  # True if ALL experiments found 100% noise
    conclusion: str  # Summary interpretation


def run_hdbscan_experiments(
    features: np.ndarray,
    configs: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> HDBSCANExperimentSummary:
    """
    Run HDBSCAN parameter sweep to validate clustering behavior.

    Per HDBSCAN documentation (https://hdbscan.readthedocs.io/en/latest/parameter_selection.html):
    - min_cluster_size is the primary parameter
    - High dimensions (>50-100) can cause issues
    - UMAP preprocessing often helps

    Default experiments test:
    1. Lower dimensionality (10-20D) via UMAP
    2. Various min_cluster_size values
    3. Different selection methods (eom vs leaf)

    Args:
        features: [n_samples, n_features] SAE or TICA features
        configs: List of experiment configurations. If None, uses defaults.
        verbose: Print progress

    Returns:
        HDBSCANExperimentSummary with all results and interpretation
    """
    import time

    if configs is None:
        # Default experiments per plan
        configs = [
            # UMAP + low dimensionality (Anthropic's approach)
            {'method': 'umap', 'dims': 10, 'min_cluster_size': 50, 'min_samples': 15},
            {'method': 'umap', 'dims': 20, 'min_cluster_size': 100, 'min_samples': 30},
            # PCA baseline
            {'method': 'pca', 'dims': 20, 'min_cluster_size': 100, 'min_samples': 50},
            # Allow single cluster (test for one big cluster vs noise)
            {'method': 'umap', 'dims': 20, 'min_cluster_size': 100, 'min_samples': 30,
             'allow_single_cluster': True},
        ]

    results = []

    for i, config in enumerate(configs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Experiment {i+1}/{len(configs)}: {config}")
            print('='*60)

        start_time = time.time()

        # Dimensionality reduction
        method = config.get('method', 'pca')
        n_dims = config.get('dims', 20)
        reduced = _reduce_dimensions(features, method, n_dims)

        # HDBSCAN parameters
        min_cluster_size = config.get('min_cluster_size', 100)
        min_samples = config.get('min_samples', min_cluster_size // 2)
        cluster_selection_method = config.get('cluster_selection_method', 'eom')
        allow_single_cluster = config.get('allow_single_cluster', False)

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=allow_single_cluster,
        )
        labels = clusterer.fit_predict(reduced)

        # Compute metrics
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_ratio = (labels == -1).sum() / len(labels)

        # Silhouette (only if we have clusters and non-noise points)
        silhouette = None
        if n_clusters >= 2 and noise_ratio < 0.95:
            valid_mask = labels >= 0
            if valid_mask.sum() > n_clusters:
                try:
                    silhouette = silhouette_score(reduced[valid_mask], labels[valid_mask])
                except Exception:
                    pass

        elapsed = time.time() - start_time

        if verbose:
            print(f"  Clusters found: {n_clusters}")
            print(f"  Noise ratio: {noise_ratio:.1%}")
            if silhouette is not None:
                print(f"  Silhouette: {silhouette:.4f}")
            print(f"  Time: {elapsed:.1f}s")

        results.append(HDBSCANExperimentResult(
            config=config,
            n_clusters=n_clusters,
            noise_ratio=noise_ratio,
            silhouette=silhouette,
            labels=labels,
            reduction_method=method,
            n_dims=n_dims,
            elapsed_time=elapsed,
        ))

    # Analyze results
    all_noise = all(r.n_clusters == 0 for r in results)

    # Find best result (by silhouette, preferring more clusters)
    best_result = None
    best_score = -1
    for r in results:
        if r.silhouette is not None and r.silhouette > best_score:
            best_score = r.silhouette
            best_result = r

    # Generate conclusion
    if all_noise:
        conclusion = (
            "All HDBSCAN experiments found 100% noise. "
            "This strongly suggests the data forms a CONTINUOUS MANIFOLD "
            "without natural density-based clusters. "
            "Proceeding with K-means discretization + PCCA+ is appropriate."
        )
    elif best_result is not None and best_result.n_clusters > 0:
        conclusion = (
            f"Found {best_result.n_clusters} clusters with "
            f"silhouette={best_result.silhouette:.3f} using "
            f"{best_result.reduction_method.upper()}({best_result.n_dims}D). "
            f"Density structure EXISTS at this scale. "
            f"Consider using these clusters alongside MSM analysis."
        )
    else:
        conclusion = (
            "Mixed results: some experiments found clusters, others didn't. "
            "Data may have scale-dependent density structure. "
            "Recommend using MSM for dynamics, HDBSCAN for density analysis."
        )

    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(conclusion)

    return HDBSCANExperimentSummary(
        results=results,
        best_result=best_result,
        all_noise=all_noise,
        conclusion=conclusion,
    )


def _reduce_dimensions(
    features: np.ndarray,
    method: str,
    n_dims: int,
) -> np.ndarray:
    """
    Reduce dimensionality for HDBSCAN experiments.

    Args:
        features: [n_samples, n_features] input features
        method: 'pca' or 'umap'
        n_dims: Target dimensionality

    Returns:
        [n_samples, n_dims] reduced features
    """
    n_samples, n_features = features.shape

    if n_features <= n_dims:
        return features

    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_dims, random_state=42)
        return reducer.fit_transform(features)

    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_dims,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42,
            )
            return reducer.fit_transform(features)
        except ImportError:
            print("Warning: umap-learn not installed, falling back to PCA")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_dims, random_state=42)
            return reducer.fit_transform(features)

    else:
        raise ValueError(f"Unknown reduction method: {method}")


def summarize_hdbscan_experiments(summary: HDBSCANExperimentSummary) -> Dict:
    """
    Create a JSON-serializable summary of HDBSCAN experiments.

    Args:
        summary: HDBSCANExperimentSummary from run_hdbscan_experiments

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        'n_experiments': len(summary.results),
        'all_noise': summary.all_noise,
        'conclusion': summary.conclusion,
        'experiments': [
            {
                'config': r.config,
                'n_clusters': r.n_clusters,
                'noise_ratio': float(r.noise_ratio),
                'silhouette': float(r.silhouette) if r.silhouette else None,
                'reduction_method': r.reduction_method,
                'n_dims': r.n_dims,
                'elapsed_time': float(r.elapsed_time),
            }
            for r in summary.results
        ],
        'best_config': summary.best_result.config if summary.best_result else None,
    }
