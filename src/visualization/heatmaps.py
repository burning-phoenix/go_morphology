"""
Heatmap visualizations for SAE analysis.

Provides various heatmap visualizations for analyzing SAE features:
- Spatial activation heatmaps on Go boards
- Concept-AUC alignment heatmaps
- Hierarchy/nestedness heatmaps
- Feature activation pattern grids

Based on CLAUDE.md requirements:
- concept_auc_heatmap.png: concepts × k levels, color = AUC
- hierarchy_heatmap.png: feature importance by k level

Usage:
    from src.visualization.heatmaps import plot_concept_auc_heatmap

    plot_concept_auc_heatmap(probe_results, concepts, k_levels, layers,
                            output_path='outputs/figures/concept_auc_heatmap.png')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .go_board import BoardPosition, GoBoardRenderer, BOARD_SIZE


def plot_spatial_activation_heatmap(
    activation_map: np.ndarray,
    position: Optional[BoardPosition] = None,
    title: str = 'Activation Heatmap',
    cmap: str = 'viridis',
    show_stones: bool = True,
    alpha: float = 0.7,
    figsize: Tuple[float, float] = (8, 8),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Plot a spatial activation heatmap, optionally overlaid on a Go board.

    Args:
        activation_map: 19x19 array of activation values
        position: Optional board position to show stones
        title: Figure title
        cmap: Colormap name
        show_stones: Whether to show stones from position
        alpha: Heatmap transparency
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with the heatmap
    """
    if activation_map.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"activation_map must be {BOARD_SIZE}x{BOARD_SIZE}")

    renderer = GoBoardRenderer(figsize=figsize)

    if show_stones and position is not None:
        renderer.render_board(position, show_coordinates=True)
    else:
        renderer.render_board(BoardPosition.empty(), show_coordinates=True)

    renderer.add_heatmap_overlay(
        activation_map,
        cmap=cmap,
        alpha=alpha,
        colorbar_label='Activation'
    )
    renderer.set_title(title)

    if output_path:
        renderer.save(output_path)

    return renderer.fig


def plot_concept_auc_heatmap(
    probe_results: Dict,
    concepts: List[str],
    k_levels: List[int],
    layers: List[str],
    method: str = 'msae',
    figsize: Tuple[float, float] = (12, 8),
    cmap: str = 'RdYlGn',
    vmin: float = 0.5,
    vmax: float = 1.0,
    output_path: Optional[str] = None,
) -> Figure:
    """
    Create a heatmap of concept probe AUC scores.

    Shows concepts × k levels with color indicating AUC.

    Args:
        probe_results: Results from probe training (from probes.json)
            Format: {concept: {layer: {method_k: {auc: float}}}}
        concepts: List of concept names to include
        k_levels: List of k values
        layers: List of layer names
        method: Method prefix ('msae', 'baseline')
        figsize: Figure size
        cmap: Colormap (RdYlGn: red=low, green=high)
        vmin: Minimum AUC for colormap
        vmax: Maximum AUC for colormap
        output_path: Where to save (optional)

    Returns:
        Figure with the heatmap
    """
    n_concepts = len(concepts)
    n_k = len(k_levels)
    n_layers = len(layers)

    # Create figure with subplots for each layer
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        # Build AUC matrix for this layer
        auc_matrix = np.zeros((n_concepts, n_k))

        for i, concept in enumerate(concepts):
            for j, k in enumerate(k_levels):
                key = f"{method}_k{k}"
                try:
                    auc = probe_results[concept][layer][key]['auc']
                except (KeyError, TypeError):
                    auc = 0.5  # Default to chance

                auc_matrix[i, j] = auc

        # Create heatmap
        im = ax.imshow(
            auc_matrix,
            cmap=cmap,
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )

        # Labels
        ax.set_xticks(range(n_k))
        ax.set_xticklabels([f'k={k}' for k in k_levels])
        ax.set_yticks(range(n_concepts))
        ax.set_yticklabels(concepts)
        ax.set_xlabel('k level')
        ax.set_ylabel('Concept')
        ax.set_title(f'{layer}')

        # Add text annotations
        for i in range(n_concepts):
            for j in range(n_k):
                value = auc_matrix[i, j]
                color = 'white' if value < 0.7 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=color, fontsize=8)

    # Add colorbar
    fig.colorbar(im, ax=axes, label='AUC', shrink=0.8)

    plt.suptitle('Concept Probe AUC by k Level and Layer', fontsize=14)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved concept AUC heatmap to {output_path}")

    return fig


def plot_hierarchy_heatmap(
    importance_matrix: np.ndarray,
    k_levels: List[int],
    top_n_features: int = 50,
    figsize: Tuple[float, float] = (10, 12),
    cmap: str = 'Blues',
    output_path: Optional[str] = None,
) -> Figure:
    """
    Create a heatmap of feature importance by k level.

    Shows which features are most important at each k level.

    Args:
        importance_matrix: (n_features, n_k_levels) activation frequency matrix
        k_levels: List of k values
        top_n_features: Number of top features to show
        figsize: Figure size
        cmap: Colormap
        output_path: Where to save (optional)

    Returns:
        Figure with the heatmap
    """
    n_features, n_k = importance_matrix.shape

    # Find top features by total importance
    total_importance = importance_matrix.sum(axis=1)
    top_indices = np.argsort(total_importance)[::-1][:top_n_features]

    # Extract top features
    top_matrix = importance_matrix[top_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        top_matrix,
        cmap=cmap,
        aspect='auto',
        interpolation='nearest'
    )

    # Labels
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f'k={k}' for k in k_levels])
    ax.set_xlabel('k level')

    ax.set_yticks(range(top_n_features))
    ax.set_yticklabels([f'F{idx}' for idx in top_indices], fontsize=6)
    ax.set_ylabel('Feature Index')

    ax.set_title(f'Top {top_n_features} Feature Importance by k Level')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Activation Frequency')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved hierarchy heatmap to {output_path}")

    return fig


def plot_feature_activation_grid(
    positions: List[BoardPosition],
    activations: np.ndarray,
    feature_idx: int,
    grid_size: Tuple[int, int] = (3, 3),
    figsize: Tuple[float, float] = (12, 12),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Plot a grid showing top activating positions for a feature.

    Args:
        positions: List of board positions
        activations: (n_samples, n_features) feature activations
        feature_idx: Which feature to visualize
        grid_size: (rows, cols) for the grid
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with grid of boards
    """
    n_rows, n_cols = grid_size
    n_show = n_rows * n_cols

    # Get feature activations and find top positions
    if activations.ndim == 2:
        feature_acts = activations[:, feature_idx]
    else:
        # Flatten spatial dimension
        feature_acts = activations.reshape(-1, activations.shape[-1])[:, feature_idx]

    top_indices = np.argsort(feature_acts)[::-1][:n_show]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for ax, top_idx in zip(axes, top_indices):
        # Determine position and spatial location
        if activations.ndim == 3:
            pos_idx = top_idx // 361
            spatial_idx = top_idx % 361
        else:
            pos_idx = top_idx
            spatial_idx = None

        if pos_idx < len(positions):
            position = positions[pos_idx]
            renderer = GoBoardRenderer(figsize=(4, 4))
            renderer.render_board(position, ax=ax, show_coordinates=False)

            # If we have spatial info, create heatmap
            if spatial_idx is not None and activations.ndim == 3:
                act_map = activations[pos_idx, :, feature_idx].reshape(19, 19)
                renderer.add_heatmap_overlay(act_map, alpha=0.5)

            act_value = feature_acts[top_idx]
            ax.set_title(f'Act: {act_value:.3f}', fontsize=10)
        else:
            ax.axis('off')

    plt.suptitle(f'Feature {feature_idx} Top Activations', fontsize=14)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature grid to {output_path}")

    return fig


def plot_nestedness_heatmap(
    nestedness: Dict[str, float],
    k_levels: List[int],
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = 'Greens',
    output_path: Optional[str] = None,
) -> Figure:
    """
    Plot nestedness matrix showing feature overlap between k levels.

    Args:
        nestedness: Dict mapping 'k{low}_in_k{high}' -> overlap ratio
        k_levels: List of k values
        figsize: Figure size
        cmap: Colormap
        output_path: Where to save (optional)

    Returns:
        Figure with nestedness heatmap
    """
    k_levels = sorted(k_levels)
    n_k = len(k_levels)

    # Build matrix
    matrix = np.zeros((n_k, n_k))
    for i, k_low in enumerate(k_levels):
        matrix[i, i] = 1.0  # Diagonal is 100% overlap
        for j, k_high in enumerate(k_levels):
            if k_low < k_high:
                key = f'k{k_low}_in_k{k_high}'
                matrix[i, j] = nestedness.get(key, 0.0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)

    # Labels
    labels = [f'k={k}' for k in k_levels]
    ax.set_xticks(range(n_k))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n_k))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Higher k (target)')
    ax.set_ylabel('Lower k (source)')
    ax.set_title('Feature Nestedness: Fraction of k_low features in k_high')

    # Add text annotations
    for i in range(n_k):
        for j in range(n_k):
            if i <= j:
                value = matrix[i, j]
                color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=color, fontsize=10)

    # Mask lower triangle
    for i in range(n_k):
        for j in range(i):
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor='gray', alpha=0.5
            ))

    fig.colorbar(im, ax=ax, label='Overlap Ratio')
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved nestedness heatmap to {output_path}")

    return fig


def plot_layer_comparison_heatmap(
    results: Dict[str, Dict],
    layers: List[str],
    metric: str = 'auc',
    concepts: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'RdYlGn',
    output_path: Optional[str] = None,
) -> Figure:
    """
    Compare a metric across layers for multiple concepts.

    Args:
        results: Nested dict of results
        layers: List of layer names
        metric: Which metric to plot
        concepts: List of concepts (auto-detected if None)
        figsize: Figure size
        cmap: Colormap
        output_path: Where to save (optional)

    Returns:
        Figure with comparison heatmap
    """
    if concepts is None:
        concepts = list(results.keys())

    n_concepts = len(concepts)
    n_layers = len(layers)

    # Build matrix
    matrix = np.zeros((n_concepts, n_layers))

    for i, concept in enumerate(concepts):
        for j, layer in enumerate(layers):
            try:
                value = results[concept][layer][metric]
            except (KeyError, TypeError):
                value = 0.0
            matrix[i, j] = value

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto')

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(n_concepts))
    ax.set_yticklabels(concepts)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Concept')
    ax.set_title(f'{metric.upper()} by Layer and Concept')

    # Text annotations
    for i in range(n_concepts):
        for j in range(n_layers):
            value = matrix[i, j]
            color = 'white' if value < 0.7 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label=metric.upper())
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved layer comparison to {output_path}")

    return fig


def plot_activation_distribution(
    activations: np.ndarray,
    feature_indices: Optional[List[int]] = None,
    n_features: int = 10,
    figsize: Tuple[float, float] = (12, 6),
    output_path: Optional[str] = None,
) -> Figure:
    """
    Plot distribution of activations for selected features.

    Args:
        activations: (n_samples, n_features) array
        feature_indices: Specific features to plot (or top-n by variance)
        n_features: Number of features if indices not specified
        figsize: Figure size
        output_path: Where to save (optional)

    Returns:
        Figure with activation distributions
    """
    if feature_indices is None:
        # Select features with highest variance
        variances = activations.var(axis=0)
        feature_indices = np.argsort(variances)[::-1][:n_features]

    n_show = len(feature_indices)
    n_cols = min(5, n_show)
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for ax, feat_idx in zip(axes[:n_show], feature_indices):
        values = activations[:, feat_idx]
        values = values[values > 0]  # Only non-zero activations

        if len(values) > 0:
            ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Activation')
            ax.set_ylabel('Count')
            ax.set_title(f'F{feat_idx}')
        else:
            ax.text(0.5, 0.5, 'Dead feature', ha='center', va='center')
            ax.set_title(f'F{feat_idx}')

    for ax in axes[n_show:]:
        ax.axis('off')

    plt.suptitle('Feature Activation Distributions', fontsize=14)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved activation distributions to {output_path}")

    return fig
