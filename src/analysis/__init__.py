# Analysis utilities

from .concepts import ConceptLabeler
from .parallel_concepts import (
    compute_labels,
    compute_labels_parallel,
    compute_labels_sequential,
    CONCEPTS,
)
from .probes import (
    LinearProbe,
    ProbeResult,
    ProbeEvaluator,
    train_probe,
    evaluate_probe,
    extract_sae_features,
)
from .hierarchy import (
    HierarchyMetrics,
    compute_nestedness,
    compute_reconstruction_r2,
    compute_feature_stability,
    compute_feature_importance_by_k,
    analyze_hierarchy,
    save_hierarchy_results,
    print_hierarchy_summary,
)
from .controls import (
    ControlResult,
    ControlSummary,
    ControlEvaluator,
    shuffle_labels,
    permute_features,
    create_random_network_activations,
    randomize_network_weights,
)
from .causal import (
    AblationResult,
    DifferentialImpactResult,
    SteeringResult,
    CausalMetrics,
    CausalAnalyzer,
    compute_policy_kl_divergence,
    compute_ablation_sparsity,
)

# TICA and attractor analysis
from .tica import (
    TICAResult,
    TICATransformer,
    fit_tica_with_variance_cutoff,
    compute_implied_timescales_multi_lag,
)
from .attractor_analysis import (
    ClusterMetrics,
    ClusteringResult,
    AttractorAnalyzer,
    KMeansResult,
    KMeansClusterer,
    run_kmeans_baseline,
)

# Diffusion Maps (geometry-aware reduction)
from .diffusion_maps import (
    DiffusionMapResult,
    DiffusionMapAnalyzer,
    compute_diffusion_map,
)

# PCA Sweep for optimal DM preprocessing (NB05)
from .pca_sweep import (
    PCASweepResult,
    SpectralValidation,
    PCASweepAnalyzer,
    run_pca_sweep,
    # Landmark selection strategies
    LandmarkComparisonResult,
    select_landmarks_uniform,
    select_landmarks_temporal,
    select_landmarks_kmeans,
    compare_landmark_strategies,
    visualize_landmark_comparison,
)

# Transition Path Theory (TPT)
from .tpt import (
    TPTResult,
    PathwayResult,
    TPTAnalyzer,
    compute_tpt,
)

# Bifurcation detection (NB06)
from .bifurcation_analysis import (
    BifurcationResult,
    CorrelationResult,
    compute_trajectory_divergence,
    detect_bifurcations,
    correlate_with_outcomes,
    BifurcationAnalyzer,
)

# Topology analysis (NB07)
from .topology_analysis import (
    PersistenceResult,
    TopologyResult,
    compute_persistent_homology,
    shuffled_null_distribution,
    compute_significance,
    compute_persistence_stats,
    TopologyAnalyzer,
)

# Generative models (NB08)
from .generative_models import (
    TrainingResult,
    TrajectoryPredictor,
    TrajectoryDataset,
    StreamingTrajectoryDataset,
    train_trajectory_predictor,
    sample_trajectories,
    GenerativeAnalyzer,
)

__all__ = [
    # Concepts
    'ConceptLabeler',
    'compute_labels',
    'compute_labels_parallel',
    'compute_labels_sequential',
    'CONCEPTS',

    # Probes
    'LinearProbe',
    'ProbeResult',
    'ProbeEvaluator',
    'train_probe',
    'evaluate_probe',
    'extract_sae_features',

    # Hierarchy
    'HierarchyMetrics',
    'compute_nestedness',
    'compute_reconstruction_r2',
    'compute_feature_stability',
    'compute_feature_importance_by_k',
    'analyze_hierarchy',
    'save_hierarchy_results',
    'print_hierarchy_summary',

    # Controls
    'ControlResult',
    'ControlSummary',
    'ControlEvaluator',
    'shuffle_labels',
    'permute_features',
    'create_random_network_activations',
    'randomize_network_weights',

    # Causal
    'AblationResult',
    'DifferentialImpactResult',
    'SteeringResult',
    'CausalMetrics',
    'CausalAnalyzer',
    'compute_policy_kl_divergence',
    'compute_ablation_sparsity',

    # TICA and attractor analysis
    'TICAResult',
    'TICATransformer',
    'fit_tica_with_variance_cutoff',
    'compute_implied_timescales_multi_lag',
    'ClusterMetrics',
    'ClusteringResult',
    'AttractorAnalyzer',
    'KMeansResult',
    'KMeansClusterer',
    'run_kmeans_baseline',

    # Diffusion Maps
    'DiffusionMapResult',
    'DiffusionMapAnalyzer',
    'compute_diffusion_map',

    # PCA Sweep
    'PCASweepResult',
    'SpectralValidation',
    'PCASweepAnalyzer',
    'run_pca_sweep',

    # TPT
    'TPTResult',
    'PathwayResult',
    'TPTAnalyzer',
    'compute_tpt',

    # Bifurcation detection (NB06)
    'BifurcationResult',
    'CorrelationResult',
    'compute_trajectory_divergence',
    'detect_bifurcations',
    'correlate_with_outcomes',
    'BifurcationAnalyzer',

    # Topology analysis (NB07)
    'PersistenceResult',
    'TopologyResult',
    'compute_persistent_homology',
    'shuffled_null_distribution',
    'compute_significance',
    'compute_persistence_stats',
    'TopologyAnalyzer',

    # Generative models (NB08)
    'TrainingResult',
    'TrajectoryPredictor',
    'TrajectoryDataset',
    'StreamingTrajectoryDataset',
    'train_trajectory_predictor',
    'sample_trajectories',
    'GenerativeAnalyzer',
]
