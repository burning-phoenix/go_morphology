"""
Result plots for SAE analysis.

Provides publication-quality plots for SAE experiment results:
- Reconstruction R² curves
- Probe comparison charts
- Causal steering visualizations
- Dead feature analysis

Based on CLAUDE.md requirements:
- reconstruction_plot.png: R² vs k level, one line per layer
- causal_steering.png: before/after policy for ablated features

Usage:
    from src.visualization.plots import plot_reconstruction_r2

    plot_reconstruction_r2(results, k_levels,
                          output_path='outputs/figures/reconstruction_plot.png')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .go_board import BoardPosition, GoBoardRenderer, BOARD_SIZE


# Consistent color scheme
LAYER_COLORS = {
    'block5': '#1f77b4',   # Blue (early)
    'block20': '#ff7f0e',  # Orange (middle)
    'block35': '#2ca02c',  # Green (late)
}

METHOD_COLORS = {
    'msae': '#1f77b4',
    'baseline': '#ff7f0e',
    'raw': '#7f7f7f',
}

METHOD_MARKERS = {
    'msae': 'o',
    'baseline': 's',
    'raw': '^',
}


def plot_reconstruction_r2(
    results: Dict[str, Dict[int, float]],
    k_levels: Optional[List[int]] = None,
    layers: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Plot reconstruction R² vs k level, one line per layer.

    Args:
        results: {layer: {k: r2_score}} dictionary
        k_levels: List of k values (auto-detected if None)
        layers: List of layers to plot (auto-detected if None)
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with R² curves
    """
    if layers is None:
        layers = list(results.keys())

    if k_levels is None:
        # Get k levels from first layer
        first_layer = list(results.values())[0]
        k_levels = sorted([k for k in first_layer.keys() if isinstance(k, int)])

    fig, ax = plt.subplots(figsize=figsize)

    for layer in layers:
        if layer not in results:
            continue

        layer_results = results[layer]
        r2_values = [layer_results.get(k, np.nan) for k in k_levels]

        color = LAYER_COLORS.get(layer, 'gray')
        ax.plot(k_levels, r2_values, 'o-', label=layer, color=color,
               linewidth=2, markersize=8)

    ax.set_xlabel('k (Number of Active Features)', fontsize=12)
    ax.set_ylabel('Reconstruction R²', fontsize=12)
    ax.set_title('MSAE Reconstruction Quality by k Level', fontsize=14)

    ax.legend(title='Layer', loc='lower right')
    ax.grid(True, alpha=0.3)

    # Set axis limits
    ax.set_ylim(0, 1)
    ax.set_xticks(k_levels)

    # Add target thresholds
    ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Target (k=128)')
    ax.axhline(y=0.40, color='orange', linestyle='--', alpha=0.5, label='Min (k=16)')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved reconstruction plot to {output_path}")

    return fig


def plot_sparsity_fidelity(
    results: Dict[str, Dict[str, float]],
    methods: List[str] = ['msae', 'baseline'],
    figsize: Tuple[float, float] = (8, 6),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Plot sparsity-fidelity tradeoff.

    Args:
        results: {method: {layer: {'sparsity': L0, 'r2': R²}}}
        methods: Methods to compare
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with scatter plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    for method in methods:
        if method not in results:
            continue

        sparsities = []
        r2_values = []
        labels = []

        for layer, metrics in results[method].items():
            sparsities.append(metrics.get('sparsity', 0))
            r2_values.append(metrics.get('r2', 0))
            labels.append(layer)

        color = METHOD_COLORS.get(method, 'gray')
        marker = METHOD_MARKERS.get(method, 'o')

        ax.scatter(sparsities, r2_values, c=color, marker=marker,
                  s=100, label=method, alpha=0.8)

        # Add labels
        for x, y, label in zip(sparsities, r2_values, labels):
            ax.annotate(label, (x, y), textcoords='offset points',
                       xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Sparsity (L₀)', fontsize=12)
    ax.set_ylabel('Reconstruction R²', fontsize=12)
    ax.set_title('Sparsity-Fidelity Tradeoff', fontsize=14)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved sparsity-fidelity plot to {output_path}")

    return fig


def plot_probe_comparison(
    probe_results: Dict,
    concepts: List[str],
    methods: List[str],
    layer: str,
    k: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 6),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Create grouped bar chart comparing probe AUC across methods.

    Args:
        probe_results: Results dict from probes.json
        concepts: List of concepts to compare
        methods: List of methods ('msae', 'baseline', 'raw')
        layer: Which layer to plot
        k: k level for MSAE (None for best)
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with grouped bar chart
    """
    n_concepts = len(concepts)
    n_methods = len(methods)

    # Prepare data
    x = np.arange(n_concepts)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        aucs = []
        errors = []

        for concept in concepts:
            try:
                if method == 'msae' and k is not None:
                    key = f'msae_k{k}'
                elif method == 'baseline':
                    key = 'baseline_k64'
                else:
                    key = method

                result = probe_results[concept][layer][key]
                auc = result['auc']
                ci_low = result.get('ci_low', auc)
                ci_high = result.get('ci_high', auc)
                error = (ci_high - ci_low) / 2
            except (KeyError, TypeError):
                auc = 0.5
                error = 0

            aucs.append(auc)
            errors.append(error)

        color = METHOD_COLORS.get(method, 'gray')
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, aucs, width, label=method, color=color,
                     yerr=errors, capsize=3, alpha=0.8)

    ax.set_xlabel('Concept', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'Probe Performance by Concept ({layer})', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha='right')
    ax.set_ylim(0.4, 1.0)

    # Add chance level line
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved probe comparison to {output_path}")

    return fig


def plot_causal_steering(
    original_policy: np.ndarray,
    steered_policy: np.ndarray,
    position: Optional[BoardPosition] = None,
    feature_idx: int = 0,
    figsize: Tuple[float, float] = (16, 6),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Visualize causal steering effect on policy.

    Shows side-by-side: original policy, steered policy, and difference.

    Args:
        original_policy: 19x19 or 361 original policy (probabilities)
        steered_policy: 19x19 or 361 steered policy
        position: Board position (optional)
        feature_idx: Which feature was ablated/steered
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with policy comparison
    """
    # Reshape to 19x19 if needed
    if original_policy.size == 361:
        original_policy = original_policy.reshape(19, 19)
    if steered_policy.size == 361:
        steered_policy = steered_policy.reshape(19, 19)

    # Compute difference
    diff = steered_policy - original_policy

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original policy
    renderer1 = GoBoardRenderer(figsize=(5, 5))
    if position is not None:
        renderer1.render_board(position, ax=axes[0], show_coordinates=False)
    else:
        renderer1.render_board(BoardPosition.empty(), ax=axes[0], show_coordinates=False)
    renderer1.add_heatmap_overlay(original_policy, cmap='viridis', alpha=0.7,
                                  colorbar_label='Probability')
    axes[0].set_title('Original Policy')

    # Steered policy
    renderer2 = GoBoardRenderer(figsize=(5, 5))
    if position is not None:
        renderer2.render_board(position, ax=axes[1], show_coordinates=False)
    else:
        renderer2.render_board(BoardPosition.empty(), ax=axes[1], show_coordinates=False)
    renderer2.add_heatmap_overlay(steered_policy, cmap='viridis', alpha=0.7,
                                  colorbar_label='Probability')
    axes[1].set_title('Steered Policy')

    # Difference
    renderer3 = GoBoardRenderer(figsize=(5, 5))
    if position is not None:
        renderer3.render_board(position, ax=axes[2], show_coordinates=False)
    else:
        renderer3.render_board(BoardPosition.empty(), ax=axes[2], show_coordinates=False)

    # Use diverging colormap for difference
    vmax = max(abs(diff.min()), abs(diff.max()))
    renderer3.add_heatmap_overlay(diff, cmap='RdBu', alpha=0.7,
                                  vmin=-vmax, vmax=vmax,
                                  colorbar_label='Δ Probability')
    axes[2].set_title('Difference (Steered - Original)')

    plt.suptitle(f'Causal Steering Effect (Feature {feature_idx})', fontsize=14)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved causal steering visualization to {output_path}")

    return fig


def plot_dead_feature_rates(
    dead_rates: Dict[str, Dict[int, float]],
    k_levels: Optional[List[int]] = None,
    figsize: Tuple[float, float] = (10, 6),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Plot dead feature rates by layer and k level.

    Args:
        dead_rates: {layer: {k: rate}} dictionary
        k_levels: List of k values
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with dead feature rates
    """
    layers = list(dead_rates.keys())

    if k_levels is None:
        first_layer = list(dead_rates.values())[0]
        k_levels = sorted([k for k in first_layer.keys() if isinstance(k, int)])

    fig, ax = plt.subplots(figsize=figsize)

    for layer in layers:
        rates = [dead_rates[layer].get(k, 0) * 100 for k in k_levels]  # Convert to %
        color = LAYER_COLORS.get(layer, 'gray')
        ax.plot(k_levels, rates, 'o-', label=layer, color=color,
               linewidth=2, markersize=8)

    ax.set_xlabel('k (Number of Active Features)', fontsize=12)
    ax.set_ylabel('Dead Feature Rate (%)', fontsize=12)
    ax.set_title('Dead Feature Rate by k Level', fontsize=14)

    # Add thresholds
    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Max threshold')
    ax.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Target')

    ax.legend(title='Layer')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_levels)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved dead feature plot to {output_path}")

    return fig


def plot_nestedness_matrix(
    nestedness: Dict[str, float],
    k_levels: List[int],
    figsize: Tuple[float, float] = (8, 6),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Plot nestedness matrix showing feature overlap.

    Args:
        nestedness: Dict mapping 'k{low}_in_k{high}' -> ratio
        k_levels: List of k values
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with nestedness matrix
    """
    from .heatmaps import plot_nestedness_heatmap
    return plot_nestedness_heatmap(nestedness, k_levels, figsize, output_path=output_path)


def plot_control_comparison(
    control_results: Dict,
    concepts: List[str],
    layer: str,
    figsize: Tuple[float, float] = (12, 6),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Compare control results (shuffled, permuted) to actual probes.

    Args:
        control_results: Results from controls.json
        concepts: Concepts to plot
        layer: Layer to analyze
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with control comparison
    """
    control_types = ['shuffled', 'permuted']
    n_concepts = len(concepts)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_concepts)
    width = 0.35

    for i, control_type in enumerate(control_types):
        aucs = []
        for concept in concepts:
            try:
                result = control_results[control_type][concept][layer]
                # Get first available method
                method_key = list(result.keys())[0]
                auc = result[method_key]['auc']
            except (KeyError, TypeError, IndexError):
                auc = 0.5
            aucs.append(auc)

        offset = (i - 0.5) * width
        ax.bar(x + offset, aucs, width, label=control_type.capitalize(), alpha=0.8)

    ax.set_xlabel('Concept', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'Negative Control Results ({layer})', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha='right')
    ax.set_ylim(0.3, 0.7)

    # Chance level
    ax.axhline(y=0.5, color='red', linestyle='--', label='Expected (chance)', linewidth=2)
    ax.axhspan(0.45, 0.55, color='green', alpha=0.2, label='Acceptable range')

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved control comparison to {output_path}")

    return fig


def plot_training_curves(
    loss_history: Dict[str, List[float]],
    figsize: Tuple[float, float] = (10, 6),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Plot training loss curves.

    Args:
        loss_history: {layer: [loss_per_epoch]} dictionary
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with loss curves
    """
    fig, ax = plt.subplots(figsize=figsize)

    for layer, losses in loss_history.items():
        color = LAYER_COLORS.get(layer, 'gray')
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, label=layer, color=color, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('MSAE Training Loss', fontsize=14)

    ax.legend(title='Layer')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {output_path}")

    return fig


def create_summary_figure(
    reconstruction_results: Dict,
    probe_results: Dict,
    hierarchy_results: Dict,
    k_levels: List[int],
    layers: List[str],
    concepts: List[str],
    figsize: Tuple[float, float] = (16, 12),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Create a comprehensive summary figure with multiple panels.

    Args:
        reconstruction_results: R² results
        probe_results: Probe AUC results
        hierarchy_results: Hierarchy metrics
        k_levels: k values
        layers: Layer names
        concepts: Concept names
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with summary panels
    """
    fig = plt.figure(figsize=figsize)

    # Panel 1: Reconstruction R²
    ax1 = fig.add_subplot(2, 2, 1)
    for layer in layers:
        if layer in reconstruction_results:
            r2_values = [reconstruction_results[layer].get(k, np.nan) for k in k_levels]
            color = LAYER_COLORS.get(layer, 'gray')
            ax1.plot(k_levels, r2_values, 'o-', label=layer, color=color)
    ax1.set_xlabel('k')
    ax1.set_ylabel('R²')
    ax1.set_title('Reconstruction Quality')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Reconstruction gap
    ax2 = fig.add_subplot(2, 2, 2)
    gaps = []
    for layer in layers:
        if layer in hierarchy_results.get('reconstruction_gap', {}):
            gaps.append(hierarchy_results['reconstruction_gap'][layer])
        else:
            gaps.append(0)
    ax2.bar(layers, gaps, color=[LAYER_COLORS.get(l, 'gray') for l in layers])
    ax2.set_ylabel('R² Gap (k_max - k_min)')
    ax2.set_title('Hierarchy Gap')
    ax2.axhline(y=0.25, color='orange', linestyle='--', label='Minimum')
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Best AUC by concept (first layer)
    ax3 = fig.add_subplot(2, 2, 3)
    first_layer = layers[0] if layers else 'block5'
    aucs = []
    for concept in concepts[:6]:  # Limit to 6 concepts
        try:
            # Get best AUC across k levels
            best_auc = 0.5
            for k in k_levels:
                key = f'msae_k{k}'
                auc = probe_results.get(concept, {}).get(first_layer, {}).get(key, {}).get('auc', 0.5)
                best_auc = max(best_auc, auc)
            aucs.append(best_auc)
        except:
            aucs.append(0.5)
    ax3.barh(concepts[:6], aucs, color='steelblue')
    ax3.set_xlabel('Best AUC')
    ax3.set_title(f'Concept Probe Performance ({first_layer})')
    ax3.axvline(x=0.5, color='red', linestyle='--')
    ax3.set_xlim(0.4, 1.0)

    # Panel 4: Nestedness
    ax4 = fig.add_subplot(2, 2, 4)
    if 'nestedness' in hierarchy_results and first_layer in hierarchy_results['nestedness']:
        nestedness = hierarchy_results['nestedness'][first_layer]
        keys = sorted(nestedness.keys())
        values = [nestedness[k] for k in keys]
        ax4.bar(range(len(keys)), values, color='forestgreen')
        ax4.set_xticks(range(len(keys)))
        ax4.set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Overlap Ratio')
        ax4.set_title(f'Feature Nestedness ({first_layer})')
        ax4.set_ylim(0, 1)
    else:
        ax4.text(0.5, 0.5, 'No nestedness data', ha='center', va='center')
        ax4.set_title('Feature Nestedness')

    plt.suptitle('MSAE Experiment Summary', fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary figure to {output_path}")

    return fig
