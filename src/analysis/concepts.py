"""
Go concept labeling functions.

Provides automatic labeling of Go positions for probe training.
Based on docs/go_concepts.md (Sensei's Library reference).

Concept categories:
1. Geometric (position-based): edge, corner, center
2. Tactical (local patterns): atari, liberty count, eye shapes
3. Strategic (group-level): living groups, territory, connections

Usage:
    from src.analysis.concepts import ConceptLabeler
    labeler = ConceptLabeler(board_size=19)
    labels = labeler.compute_all_labels(board_state)
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from enum import IntEnum


class StoneColor(IntEnum):
    """Stone colors on a Go board."""
    EMPTY = 0
    BLACK = 1
    WHITE = 2


@dataclass
class Group:
    """A connected group of stones."""
    color: StoneColor
    stones: Set[Tuple[int, int]]
    liberties: Set[Tuple[int, int]]

    @property
    def liberty_count(self) -> int:
        return len(self.liberties)

    @property
    def is_in_atari(self) -> bool:
        return len(self.liberties) == 1


class ConceptLabeler:
    """
    Compute concept labels for Go positions.

    All labels are computed per-point, producing (19, 19) boolean arrays.
    These can be flattened to match activation vectors.
    """

    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self._precompute_geometric_masks()

    def _precompute_geometric_masks(self):
        """Precompute static geometric masks."""
        n = self.board_size

        # Edge mask (first and last row/column)
        self.edge_mask = np.zeros((n, n), dtype=bool)
        self.edge_mask[0, :] = True
        self.edge_mask[n-1, :] = True
        self.edge_mask[:, 0] = True
        self.edge_mask[:, n-1] = True

        # Corner regions (4x4 in each corner)
        self.corner_mask = np.zeros((n, n), dtype=bool)
        self.corner_mask[:4, :4] = True  # Top-left
        self.corner_mask[:4, n-4:] = True  # Top-right
        self.corner_mask[n-4:, :4] = True  # Bottom-left
        self.corner_mask[n-4:, n-4:] = True  # Bottom-right

        # Center region (middle 5x5)
        center = n // 2
        self.center_mask = np.zeros((n, n), dtype=bool)
        self.center_mask[center-2:center+3, center-2:center+3] = True

        # Star points (for 19x19)
        self.star_mask = np.zeros((n, n), dtype=bool)
        if n == 19:
            for r in [3, 9, 15]:
                for c in [3, 9, 15]:
                    self.star_mask[r, c] = True

    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get orthogonal neighbors of a point."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                neighbors.append((nx, ny))
        return neighbors

    def _get_diagonal_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get diagonal neighbors of a point."""
        neighbors = []
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                neighbors.append((nx, ny))
        return neighbors

    def _find_group(
        self,
        board: np.ndarray,
        x: int,
        y: int
    ) -> Optional[Group]:
        """
        Find the group containing the stone at (x, y).

        Uses flood fill to find all connected stones.
        """
        color = board[x, y]
        if color == StoneColor.EMPTY:
            return None

        stones = set()
        liberties = set()
        visited = set()
        stack = [(x, y)]

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            if board[cx, cy] == color:
                stones.add((cx, cy))
                for nx, ny in self._get_neighbors(cx, cy):
                    if (nx, ny) not in visited:
                        if board[nx, ny] == StoneColor.EMPTY:
                            liberties.add((nx, ny))
                        elif board[nx, ny] == color:
                            stack.append((nx, ny))

        return Group(color=color, stones=stones, liberties=liberties)

    def _find_all_groups(self, board: np.ndarray) -> List[Group]:
        """Find all groups on the board."""
        visited = set()
        groups = []

        for x in range(self.board_size):
            for y in range(self.board_size):
                if (x, y) not in visited and board[x, y] != StoneColor.EMPTY:
                    group = self._find_group(board, x, y)
                    if group:
                        groups.append(group)
                        visited.update(group.stones)

        return groups

    # ==================== Geometric Concepts ====================

    def is_edge(self) -> np.ndarray:
        """Points on board edge (first/last row/column)."""
        return self.edge_mask.copy()

    def is_corner_region(self) -> np.ndarray:
        """Points in 4x4 corner regions."""
        return self.corner_mask.copy()

    def is_center_region(self) -> np.ndarray:
        """Points in center 5x5 region."""
        return self.center_mask.copy()

    def is_star_point(self) -> np.ndarray:
        """Star points (hoshi)."""
        return self.star_mask.copy()

    # ==================== Stone-Based Concepts ====================

    def stone_color(self, board: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get stone colors at each point.

        Returns:
            Dict with 'black', 'white', 'empty' boolean arrays
        """
        return {
            'black': board == StoneColor.BLACK,
            'white': board == StoneColor.WHITE,
            'empty': board == StoneColor.EMPTY,
        }

    def liberty_count(self, board: np.ndarray) -> np.ndarray:
        """
        Liberty count for each stone's group.

        Returns:
            (board_size, board_size) array with liberty counts (0 for empty)
        """
        liberties = np.zeros((self.board_size, self.board_size), dtype=np.int32)

        for group in self._find_all_groups(board):
            count = group.liberty_count
            for x, y in group.stones:
                liberties[x, y] = count

        return liberties

    def is_atari(self, board: np.ndarray) -> np.ndarray:
        """
        Points where a stone's group has exactly 1 liberty.

        From docs/go_concepts.md: "The state of a stone or group of stones
        that has only one liberty."
        """
        liberties = self.liberty_count(board)
        return (liberties == 1) & (board != StoneColor.EMPTY)

    def is_low_liberties(self, board: np.ndarray, threshold: int = 3) -> np.ndarray:
        """Points where group has fewer than threshold liberties."""
        liberties = self.liberty_count(board)
        return (liberties < threshold) & (liberties > 0)

    # ==================== Tactical Concepts ====================

    def is_eye_shape(self, board: np.ndarray, color: StoneColor) -> np.ndarray:
        """
        Empty points that could be eyes for the given color.

        An eye is an empty point surrounded orthogonally by stones of one color.
        
        False Eye Detection (from Sensei's Library):
        - A false eye can be filled by opponent without suicide
        - Determined by counting enemy stones on diagonal points
        
        Rules for real eyes:
        - Center: All 4 ortho same color, at most 1 enemy diagonal
        - Edge: All 3 ortho same color, 0 enemy diagonals (2 diag points)
        - Corner: All 2 ortho same color, 0 enemy diagonals (1 diag point)
        
        From docs/go_concepts.md: "an empty space surrounded by stones of one colour"
        """
        eye_candidates = np.zeros((self.board_size, self.board_size), dtype=bool)
        other = StoneColor.WHITE if color == StoneColor.BLACK else StoneColor.BLACK
        n = self.board_size

        for x in range(n):
            for y in range(n):
                if board[x, y] != StoneColor.EMPTY:
                    continue

                neighbors = self._get_neighbors(x, y)
                diagonals = self._get_diagonal_neighbors(x, y)

                # Rule 1: All orthogonal neighbors must be same color
                all_same_color = all(
                    board[nx, ny] == color
                    for nx, ny in neighbors
                )

                if not all_same_color:
                    continue

                # Rule 2: Count enemy diagonals for false eye detection
                enemy_diagonals = sum(
                    1 for dx, dy in diagonals
                    if board[dx, dy] == other
                )

                # Determine if point is on edge or corner
                on_edge = (x == 0 or x == n - 1 or y == 0 or y == n - 1)
                in_corner = (x == 0 or x == n - 1) and (y == 0 or y == n - 1)
                
                # Max allowed enemy diagonals:
                # - Corner (1 diagonal): 0 enemy allowed
                # - Edge (2 diagonals): 0 enemy allowed  
                # - Center (4 diagonals): 1 enemy allowed
                if in_corner:
                    max_enemy = 0
                elif on_edge:
                    max_enemy = 0
                else:
                    max_enemy = 1

                if enemy_diagonals <= max_enemy:
                    eye_candidates[x, y] = True

        return eye_candidates

    def is_potential_eye(
        self,
        board: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Get eye candidates for both colors."""
        return {
            'black': self.is_eye_shape(board, StoneColor.BLACK),
            'white': self.is_eye_shape(board, StoneColor.WHITE),
        }

    def is_cutting_point(self, board: np.ndarray) -> np.ndarray:
        """
        Empty points that would disconnect groups if opponent plays there.

        A cutting point is where playing would separate two or more
        enemy groups that currently share the point as a liberty.
        """
        cutting_points = np.zeros((self.board_size, self.board_size), dtype=bool)

        # Find all groups
        groups = self._find_all_groups(board)

        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x, y] != StoneColor.EMPTY:
                    continue

                # Check if this point is shared liberty of multiple same-color groups
                black_groups_sharing = []
                white_groups_sharing = []

                for group in groups:
                    if (x, y) in group.liberties:
                        if group.color == StoneColor.BLACK:
                            black_groups_sharing.append(group)
                        else:
                            white_groups_sharing.append(group)

                # Cutting point if it would separate 2+ groups of same color
                if len(black_groups_sharing) >= 2 or len(white_groups_sharing) >= 2:
                    cutting_points[x, y] = True

        return cutting_points

    def has_adjacent_enemy(self, board: np.ndarray) -> Dict[str, np.ndarray]:
        """Points adjacent to enemy stones (for each color)."""
        black_near_white = np.zeros((self.board_size, self.board_size), dtype=bool)
        white_near_black = np.zeros((self.board_size, self.board_size), dtype=bool)

        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x, y] == StoneColor.BLACK:
                    for nx, ny in self._get_neighbors(x, y):
                        if board[nx, ny] == StoneColor.WHITE:
                            black_near_white[x, y] = True
                            break
                elif board[x, y] == StoneColor.WHITE:
                    for nx, ny in self._get_neighbors(x, y):
                        if board[nx, ny] == StoneColor.BLACK:
                            white_near_black[x, y] = True
                            break

        return {
            'black': black_near_white,
            'white': white_near_black,
        }

    # ==================== Strategic Concepts ====================

    def estimate_territory(self, board: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Estimate territory ownership using flood-fill heuristic.

        Algorithm:
        - For each empty point, flood-fill to find the connected empty region
        - Check which colors the region borders
        - If region only borders one color, assign to that color
        - If region borders both colors, it's dame (neutral)

        This is a simplified version - proper territory scoring requires
        life/death analysis. Returns binary (0 or 1) values.

        From docs/go_concepts.md: Territory is the empty points surrounded,
        or rather 'controlled', by a player.
        """
        black_territory = np.zeros((self.board_size, self.board_size), dtype=float)
        white_territory = np.zeros((self.board_size, self.board_size), dtype=float)

        # For each empty point, count reachable stones of each color
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x, y] != StoneColor.EMPTY:
                    continue

                # BFS to find enclosed region
                visited = set()
                region = set()
                touches_black = False
                touches_white = False
                stack = [(x, y)]

                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) in visited:
                        continue
                    visited.add((cx, cy))

                    if board[cx, cy] == StoneColor.EMPTY:
                        region.add((cx, cy))
                        for nx, ny in self._get_neighbors(cx, cy):
                            stack.append((nx, ny))
                    elif board[cx, cy] == StoneColor.BLACK:
                        touches_black = True
                    else:
                        touches_white = True

                # Assign territory based on what the region touches
                if touches_black and not touches_white:
                    for rx, ry in region:
                        black_territory[rx, ry] = 1.0
                elif touches_white and not touches_black:
                    for rx, ry in region:
                        white_territory[rx, ry] = 1.0
                # Dame (neutral) if touches both

        return {
            'black_territory': black_territory,
            'white_territory': white_territory,
        }

    def group_size(self, board: np.ndarray) -> np.ndarray:
        """Size of the group each stone belongs to."""
        sizes = np.zeros((self.board_size, self.board_size), dtype=np.int32)

        for group in self._find_all_groups(board):
            size = len(group.stones)
            for x, y in group.stones:
                sizes[x, y] = size

        return sizes

    def is_large_group(self, board: np.ndarray, threshold: int = 5) -> np.ndarray:
        """Stones belonging to groups with >= threshold stones."""
        sizes = self.group_size(board)
        return sizes >= threshold

    # ==================== Label Computation ====================

    def compute_all_labels(
        self,
        board: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute all concept labels for a board position.

        Args:
            board: (board_size, board_size) array with StoneColor values

        Returns:
            Dict mapping concept names to (board_size, board_size) boolean arrays
        """
        labels = {}

        # Geometric (static)
        labels['is_edge'] = self.is_edge()
        labels['is_corner'] = self.is_corner_region()
        labels['is_center'] = self.is_center_region()
        labels['is_star_point'] = self.is_star_point()

        # Stone colors
        colors = self.stone_color(board)
        labels['has_black_stone'] = colors['black']
        labels['has_white_stone'] = colors['white']
        labels['is_empty'] = colors['empty']

        # Tactical
        labels['is_atari'] = self.is_atari(board)
        labels['is_low_liberty'] = self.is_low_liberties(board, threshold=3)
        labels['is_cutting_point'] = self.is_cutting_point(board)

        # Eyes
        eyes = self.is_potential_eye(board)
        labels['black_eye_shape'] = eyes['black']
        labels['white_eye_shape'] = eyes['white']

        # Contact
        contact = self.has_adjacent_enemy(board)
        labels['black_contact'] = contact['black']
        labels['white_contact'] = contact['white']

        # Strategic
        territory = self.estimate_territory(board)
        labels['black_territory'] = territory['black_territory'] > 0.5
        labels['white_territory'] = territory['white_territory'] > 0.5

        labels['is_large_group'] = self.is_large_group(board, threshold=5)

        return labels

    def flatten_labels(
        self,
        labels: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Flatten spatial labels to 1D arrays matching activation vectors.

        Args:
            labels: Dict of (board_size, board_size) arrays

        Returns:
            Dict of (board_size * board_size,) arrays
        """
        return {
            name: arr.flatten()
            for name, arr in labels.items()
        }


def create_concept_dataset(
    positions: np.ndarray,
    board_size: int = 19,
) -> Dict[str, np.ndarray]:
    """
    Create concept labels for a dataset of positions.

    Args:
        positions: (n_positions, 18, board_size, board_size) tensor
                   with encoded positions (from PositionEncoder)
        board_size: Size of Go board

    Returns:
        Dict mapping concept names to (n_positions * board_size^2,) arrays
    """
    labeler = ConceptLabeler(board_size)
    n_positions = len(positions)
    n_points = board_size * board_size

    # Initialize label arrays
    all_labels = {}

    for i in range(n_positions):
        # Extract board state from 18-plane encoding
        # Planes 0 and 8 are current black and white stones
        black_stones = positions[i, 0] > 0.5  # Current player's stones
        white_stones = positions[i, 8] > 0.5  # Opponent's stones

        # Determine which color is to play (plane 16 = black to play)
        black_to_play = positions[i, 16, 0, 0] > 0.5

        # Build board array
        board = np.zeros((board_size, board_size), dtype=np.int32)
        if black_to_play:
            board[black_stones] = StoneColor.BLACK
            board[white_stones] = StoneColor.WHITE
        else:
            # If white to play, swap colors (plane 0 = white, plane 8 = black)
            board[black_stones] = StoneColor.WHITE
            board[white_stones] = StoneColor.BLACK

        # Compute labels
        labels = labeler.compute_all_labels(board)
        flat_labels = labeler.flatten_labels(labels)

        # Store
        for name, arr in flat_labels.items():
            if name not in all_labels:
                all_labels[name] = np.zeros((n_positions * n_points,), dtype=bool)
            all_labels[name][i * n_points:(i + 1) * n_points] = arr

        if (i + 1) % 1000 == 0:
            print(f"  Labeled {i + 1}/{n_positions} positions...")

    return all_labels
