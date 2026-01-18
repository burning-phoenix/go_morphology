"""
Go board visualization for SAE feature analysis.

Renders Go boards with stones, activation overlays, and annotations
for visualizing which positions activate specific SAE features.

Based on Go concepts from docs/go_concepts.md:
- 19x19 board with star points (hoshi)
- Black stones (filled), white stones (hollow with outline)
- Coordinate labels (A-T, 1-19)

Usage:
    from src.visualization.go_board import GoBoardRenderer, BoardPosition

    renderer = GoBoardRenderer()
    fig = renderer.render_board(position)
    renderer.add_heatmap_overlay(activations)
    renderer.save('outputs/figures/feature_42.png')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path


# Board constants
BOARD_SIZE = 19
EMPTY = 0
BLACK = 1
WHITE = 2

# Coordinate labels (A-T, skipping I)
COL_LABELS = 'ABCDEFGHJKLMNOPQRST'  # Note: I is skipped in Go notation

# Star points (hoshi) positions on 19x19 board
STAR_POINTS = [
    (3, 3), (3, 9), (3, 15),
    (9, 3), (9, 9), (9, 15),
    (15, 3), (15, 9), (15, 15),
]


@dataclass
class BoardPosition:
    """
    A Go board position.

    Attributes:
        stones: 19x19 array where 0=empty, 1=black, 2=white
        to_play: Which color plays next (1=black, 2=white)
        move_number: Current move number in the game
    """
    stones: np.ndarray  # Shape: (19, 19)
    to_play: int = BLACK
    move_number: int = 0

    def __post_init__(self):
        if self.stones.shape != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(f"stones must be {BOARD_SIZE}x{BOARD_SIZE}, got {self.stones.shape}")

    @classmethod
    def empty(cls) -> 'BoardPosition':
        """Create an empty board."""
        return cls(stones=np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8))

    @classmethod
    def from_stones(cls, black: List[Tuple[int, int]], white: List[Tuple[int, int]]) -> 'BoardPosition':
        """
        Create board from lists of stone positions.

        Args:
            black: List of (x, y) positions for black stones
            white: List of (x, y) positions for white stones
        """
        stones = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for x, y in black:
            stones[x, y] = BLACK
        for x, y in white:
            stones[x, y] = WHITE
        return cls(stones=stones)


class GoBoardRenderer:
    """
    Renders Go boards with stones, heatmaps, and annotations.

    Supports:
    - Drawing standard 19x19 Go board with grid and star points
    - Placing black and white stones
    - Overlaying activation heatmaps
    - Adding markers and annotations
    - Publication-quality output
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (8, 8),
        dpi: int = 100,
        board_color: str = '#DEB887',  # Burlywood (traditional Go board color)
        grid_color: str = 'black',
        grid_linewidth: float = 0.5,
    ):
        """
        Initialize the renderer.

        Args:
            figsize: Figure size in inches
            dpi: Resolution for rendering
            board_color: Background color for the board
            grid_color: Color for grid lines
            grid_linewidth: Width of grid lines
        """
        self.figsize = figsize
        self.dpi = dpi
        self.board_color = board_color
        self.grid_color = grid_color
        self.grid_linewidth = grid_linewidth

        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self.position: Optional[BoardPosition] = None

    def _setup_axes(self, ax: Optional[Axes] = None) -> Axes:
        """Set up axes for board rendering."""
        if ax is None:
            self.fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            self.fig = ax.figure

        self.ax = ax

        # Set up axes limits and appearance
        ax.set_xlim(-0.8, BOARD_SIZE - 0.2)
        ax.set_ylim(-0.8, BOARD_SIZE - 0.2)
        ax.set_aspect('equal')
        ax.set_facecolor(self.board_color)

        # Remove default ticks
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    def _draw_grid(self, ax: Axes) -> None:
        """Draw the 19x19 grid."""
        for i in range(BOARD_SIZE):
            # Horizontal lines
            ax.axhline(y=i, xmin=0.02, xmax=0.98,
                      color=self.grid_color, linewidth=self.grid_linewidth)
            # Vertical lines
            ax.axvline(x=i, ymin=0.02, ymax=0.98,
                      color=self.grid_color, linewidth=self.grid_linewidth)

    def _draw_star_points(self, ax: Axes) -> None:
        """Draw star points (hoshi)."""
        for x, y in STAR_POINTS:
            ax.plot(x, y, 'o', color=self.grid_color, markersize=5)

    def _draw_coordinates(self, ax: Axes) -> None:
        """Draw coordinate labels (A-T, 1-19)."""
        # Column labels (A-T) at bottom
        for i, label in enumerate(COL_LABELS):
            ax.text(i, -0.5, label, ha='center', va='center', fontsize=8)

        # Row labels (1-19) on left
        for i in range(BOARD_SIZE):
            ax.text(-0.5, i, str(i + 1), ha='center', va='center', fontsize=8)

    def _draw_stones(self, ax: Axes, stones: np.ndarray) -> None:
        """Draw all stones on the board."""
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if stones[x, y] == BLACK:
                    circle = plt.Circle(
                        (x, y), 0.45,
                        facecolor='black',
                        edgecolor='black',
                        linewidth=1,
                        zorder=10
                    )
                    ax.add_patch(circle)
                elif stones[x, y] == WHITE:
                    circle = plt.Circle(
                        (x, y), 0.45,
                        facecolor='white',
                        edgecolor='black',
                        linewidth=1.5,
                        zorder=10
                    )
                    ax.add_patch(circle)

    def render_board(
        self,
        position: Optional[BoardPosition] = None,
        ax: Optional[Axes] = None,
        show_coordinates: bool = True,
    ) -> Figure:
        """
        Render a Go board position.

        Args:
            position: Board position to render (None for empty board)
            ax: Existing axes to draw on (creates new figure if None)
            show_coordinates: Whether to show coordinate labels

        Returns:
            The matplotlib Figure
        """
        ax = self._setup_axes(ax)

        # Draw board elements
        self._draw_grid(ax)
        self._draw_star_points(ax)

        if show_coordinates:
            self._draw_coordinates(ax)

        # Draw stones if position provided
        if position is not None:
            self.position = position
            self._draw_stones(ax, position.stones)

        return self.fig

    def add_heatmap_overlay(
        self,
        values: np.ndarray,
        cmap: str = 'hot',
        alpha: float = 0.5,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_colorbar: bool = True,
        colorbar_label: str = 'Activation',
    ) -> None:
        """
        Overlay a heatmap on the board.

        Args:
            values: 19x19 array of values to display
            cmap: Colormap name
            alpha: Transparency (0-1)
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            show_colorbar: Whether to add a colorbar
            colorbar_label: Label for colorbar
        """
        if self.ax is None:
            raise RuntimeError("Must call render_board() first")

        if values.shape != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(f"values must be {BOARD_SIZE}x{BOARD_SIZE}")

        # Create heatmap
        im = self.ax.imshow(
            values.T,  # Transpose to match board orientation
            cmap=cmap,
            alpha=alpha,
            extent=[-0.5, BOARD_SIZE - 0.5, -0.5, BOARD_SIZE - 0.5],
            origin='lower',
            vmin=vmin,
            vmax=vmax,
            zorder=5,  # Below stones
        )

        if show_colorbar and self.fig is not None:
            cbar = self.fig.colorbar(im, ax=self.ax, shrink=0.8)
            cbar.set_label(colorbar_label)

    def add_markers(
        self,
        positions: List[Tuple[int, int]],
        marker: str = 'x',
        color: str = 'red',
        size: float = 100,
        label: Optional[str] = None,
    ) -> None:
        """
        Add markers at specific positions.

        Args:
            positions: List of (x, y) positions to mark
            marker: Marker style ('x', 'o', 's', etc.)
            color: Marker color
            size: Marker size
            label: Legend label for these markers
        """
        if self.ax is None:
            raise RuntimeError("Must call render_board() first")

        if positions:
            xs, ys = zip(*positions)
            self.ax.scatter(xs, ys, marker=marker, c=color, s=size,
                          label=label, zorder=15)

    def add_text_annotations(
        self,
        positions: List[Tuple[int, int]],
        labels: List[str],
        fontsize: int = 10,
        color: str = 'blue',
    ) -> None:
        """
        Add text annotations at positions.

        Args:
            positions: List of (x, y) positions
            labels: Text labels for each position
            fontsize: Font size
            color: Text color
        """
        if self.ax is None:
            raise RuntimeError("Must call render_board() first")

        for (x, y), label in zip(positions, labels):
            self.ax.text(x, y, label, ha='center', va='center',
                        fontsize=fontsize, color=color, zorder=20,
                        bbox=dict(boxstyle='round,pad=0.2',
                                 facecolor='white', alpha=0.7))

    def add_territory_overlay(
        self,
        territory: np.ndarray,
        black_color: str = 'black',
        white_color: str = 'white',
        alpha: float = 0.3,
    ) -> None:
        """
        Overlay territory markings.

        Args:
            territory: 19x19 array where -1=black territory, 1=white territory, 0=neutral
            black_color: Color for black territory
            white_color: Color for white territory
            alpha: Transparency
        """
        if self.ax is None:
            raise RuntimeError("Must call render_board() first")

        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if territory[x, y] < -0.5:  # Black territory
                    rect = patches.Rectangle(
                        (x - 0.4, y - 0.4), 0.3, 0.3,
                        facecolor=black_color, alpha=alpha, zorder=8
                    )
                    self.ax.add_patch(rect)
                elif territory[x, y] > 0.5:  # White territory
                    rect = patches.Rectangle(
                        (x + 0.1, y + 0.1), 0.3, 0.3,
                        facecolor=white_color, alpha=alpha, zorder=8
                    )
                    self.ax.add_patch(rect)

    def set_title(self, title: str, fontsize: int = 12) -> None:
        """Set the figure title."""
        if self.ax is None:
            raise RuntimeError("Must call render_board() first")
        self.ax.set_title(title, fontsize=fontsize)

    def save(
        self,
        path: str,
        dpi: int = 300,
        bbox_inches: str = 'tight',
    ) -> None:
        """
        Save the figure to file.

        Args:
            path: Output file path
            dpi: Resolution for saved image
            bbox_inches: Bounding box option
        """
        if self.fig is None:
            raise RuntimeError("Must call render_board() first")

        # Ensure output directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self.fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved figure to {path}")

    def close(self) -> None:
        """Close the figure to free memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def render_top_activating_positions(
    positions: List[BoardPosition],
    activations: np.ndarray,
    feature_idx: int,
    n_top: int = 9,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 15),
) -> Figure:
    """
    Render a grid of top-activating positions for a feature.

    Args:
        positions: List of board positions
        activations: (n_positions, 361, n_features) or (n_positions * 361, n_features)
        feature_idx: Which feature to visualize
        n_top: Number of top positions to show
        output_path: Where to save (optional)
        figsize: Figure size

    Returns:
        Figure with grid of top-activating positions
    """
    # Reshape activations if needed
    if activations.ndim == 3:
        # (n_positions, 361, n_features) -> (n_positions * 361, n_features)
        n_positions = activations.shape[0]
        activations = activations.reshape(-1, activations.shape[-1])
    else:
        n_positions = len(positions)

    # Get feature activations
    feature_acts = activations[:, feature_idx]

    # Find top activating spatial positions
    top_indices = np.argsort(feature_acts)[::-1][:n_top]

    # Determine grid size
    n_cols = int(np.ceil(np.sqrt(n_top)))
    n_rows = int(np.ceil(n_top / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for idx, (ax, top_idx) in enumerate(zip(axes[:n_top], top_indices)):
        # Determine which position and spatial location
        pos_idx = top_idx // 361
        spatial_idx = top_idx % 361
        x, y = spatial_idx // 19, spatial_idx % 19

        if pos_idx < len(positions):
            position = positions[pos_idx]

            # Render board
            renderer = GoBoardRenderer(figsize=(4, 4))
            renderer.render_board(position, ax=ax, show_coordinates=False)

            # Mark the top-activating position
            renderer.add_markers([(x, y)], marker='*', color='red', size=200)

            # Add activation value as title
            act_value = feature_acts[top_idx]
            ax.set_title(f'Act: {act_value:.3f}', fontsize=10)

    # Hide unused axes
    for ax in axes[n_top:]:
        ax.axis('off')

    plt.suptitle(f'Top {n_top} Activating Positions for Feature {feature_idx}', fontsize=14)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")

    return fig


def create_feature_examples_folder(
    positions: List[BoardPosition],
    activations: np.ndarray,
    feature_indices: List[int],
    output_dir: str = 'outputs/figures/feature_examples',
    n_top: int = 9,
) -> None:
    """
    Create a folder with top-activating examples for multiple features.

    Args:
        positions: List of board positions
        activations: Feature activations
        feature_indices: Which features to visualize
        output_dir: Output directory
        n_top: Number of examples per feature
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for feature_idx in feature_indices:
        fig = render_top_activating_positions(
            positions, activations, feature_idx, n_top,
            output_path=str(output_path / f'feature_{feature_idx}.png')
        )
        plt.close(fig)

    print(f"Created {len(feature_indices)} feature example figures in {output_dir}")
