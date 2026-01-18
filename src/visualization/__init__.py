# Visualization utilities for MSAE Go analysis

from .go_board import (
    BOARD_SIZE,
    EMPTY,
    BLACK,
    WHITE,
    BoardPosition,
    GoBoardRenderer,
    render_top_activating_positions,
    create_feature_examples_folder,
)

from .heatmaps import (
    plot_spatial_activation_heatmap,
    plot_concept_auc_heatmap,
    plot_hierarchy_heatmap,
    plot_feature_activation_grid,
    plot_nestedness_heatmap,
    plot_layer_comparison_heatmap,
    plot_activation_distribution,
)

from .plots import (
    plot_reconstruction_r2,
    plot_sparsity_fidelity,
    plot_probe_comparison,
    plot_causal_steering,
    plot_dead_feature_rates,
    plot_nestedness_matrix,
    plot_control_comparison,
    plot_training_curves,
    create_summary_figure,
)

from .feature_analysis import (
    LivenessReport,
    compute_feature_liveness,
    generate_feature_examples_with_liveness,
    plot_liveness_histogram,
)

__all__ = [
    # Go board constants
    'BOARD_SIZE',
    'EMPTY',
    'BLACK',
    'WHITE',

    # Go board rendering
    'BoardPosition',
    'GoBoardRenderer',
    'render_top_activating_positions',
    'create_feature_examples_folder',

    # Heatmaps
    'plot_spatial_activation_heatmap',
    'plot_concept_auc_heatmap',
    'plot_hierarchy_heatmap',
    'plot_feature_activation_grid',
    'plot_nestedness_heatmap',
    'plot_layer_comparison_heatmap',
    'plot_activation_distribution',

    # Plots
    'plot_reconstruction_r2',
    'plot_sparsity_fidelity',
    'plot_probe_comparison',
    'plot_causal_steering',
    'plot_dead_feature_rates',
    'plot_nestedness_matrix',
    'plot_control_comparison',
    'plot_training_curves',
    'create_summary_figure',
    
    # Feature analysis
    'LivenessReport',
    'compute_feature_liveness',
    'generate_feature_examples_with_liveness',
    'plot_liveness_histogram',
]

