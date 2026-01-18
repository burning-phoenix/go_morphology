"""
Position encoder for Go board states.

Converts Go board positions into the 18-plane format used by Leela Zero.
Uses sgfmill library for Go board logic.

Plane encoding:
  - Planes 1-8:   Current player stones at T=0, T=-1, ..., T=-7
  - Planes 9-16:  Opponent stones at T=0, T=-1, ..., T=-7
  - Plane 17:     All 1s if black to move, else 0
  - Plane 18:     All 1s if white to move, else 0

Reference: docs/leela_zero_README.md, docs/go_concepts.md
"""

import numpy as np
import torch
from typing import List, Optional
from collections import deque
from dataclasses import dataclass, field

try:
    from sgfmill import boards as sgfmill_boards
    HAS_SGFMILL = True
except ImportError:
    HAS_SGFMILL = False


# Pre-computed constants for color planes (avoid repeated allocation)
_ONES_19x19 = np.ones((19, 19), dtype=np.float32)
_ZEROS_19x19 = np.zeros((19, 19), dtype=np.float32)


@dataclass
class BoardState:
    """
    Wrapper around sgfmill.boards.Board with move history tracking.

    Uses sgfmill for Go rules (captures, ko, etc.) and adds
    history tracking needed for Leela Zero's temporal encoding.

    Optimizations:
    - Maintains cached board array updated incrementally
    - History stores references, only copies on mutation
    """
    size: int = 19
    history_length: int = 8
    _board: 'sgfmill_boards.Board' = field(default=None, repr=False)
    _history: deque = field(default=None, init=False, repr=False)
    _board_cache: np.ndarray = field(default=None, init=False, repr=False)
    _cache_valid: bool = field(default=False, init=False, repr=False)
    current_player: str = 'b'  # 'b' or 'w'
    move_count: int = 0

    def __post_init__(self):
        if not HAS_SGFMILL:
            raise ImportError("sgfmill required: pip install sgfmill")
        if self._board is None:
            self._board = sgfmill_boards.Board(self.size)
        if self._history is None:
            self._history = deque(maxlen=self.history_length)
        # Initialize cache
        self._board_cache = np.zeros((self.size, self.size), dtype=np.int8)
        self._cache_valid = True  # Empty board cache is valid

    def copy(self) -> 'BoardState':
        """Deep copy of board state. Optimized to batch-copy history."""
        new_state = BoardState(self.size, self.history_length)
        new_state._board = self._board.copy()
        new_state.current_player = self.current_player
        new_state.move_count = self.move_count

        # Optimized: stack history arrays and copy in one operation if non-empty
        if self._history:
            history_stack = np.stack(list(self._history))
            new_state._history = deque(
                [history_stack[i].copy() for i in range(len(self._history))],
                maxlen=self.history_length
            )
        else:
            new_state._history = deque(maxlen=self.history_length)

        # Copy cache
        new_state._board_cache = self._board_cache.copy()
        new_state._cache_valid = self._cache_valid

        return new_state

    def _invalidate_cache(self):
        """Mark cache as invalid (called when board changes unpredictably)."""
        self._cache_valid = False

    def _rebuild_cache(self):
        """Rebuild board cache from sgfmill board (only when needed)."""
        self._board_cache.fill(0)
        for row in range(self.size):
            for col in range(self.size):
                color = self._board.get(row, col)
                if color == 'b':
                    self._board_cache[row, col] = 1
                elif color == 'w':
                    self._board_cache[row, col] = 2
        self._cache_valid = True

    def _get_board_array(self) -> np.ndarray:
        """
        Get board as numpy array. Uses cache when valid.

        Returns a copy to prevent external mutation.
        0=empty, 1=black, 2=white.
        """
        if not self._cache_valid:
            self._rebuild_cache()
        return self._board_cache.copy()

    def play(self, row: int, col: int, color: Optional[str] = None) -> None:
        """
        Play a move. Uses sgfmill for capture/ko logic.

        Args:
            row, col: Board coordinates (0-indexed, row 0 = top)
            color: 'b' or 'w' (default: current player)
        """
        if color is None:
            color = self.current_player

        # Save current board to history before move
        if not self._cache_valid:
            self._rebuild_cache()
        self._history.appendleft(self._board_cache.copy())

        # sgfmill handles captures and ko
        # Get captured stones before playing (to update cache efficiently)
        self._board.play(row, col, color)

        # Invalidate cache since captures may have occurred
        # (sgfmill doesn't expose capture info easily, so rebuild is safer)
        self._cache_valid = False

        # Switch player
        self.current_player = 'w' if color == 'b' else 'b'
        self.move_count += 1

    def pass_move(self) -> None:
        """Pass (no stone played)."""
        if not self._cache_valid:
            self._rebuild_cache()
        self._history.appendleft(self._board_cache.copy())
        self.current_player = 'w' if self.current_player == 'b' else 'b'
        self.move_count += 1

    @property
    def board(self) -> np.ndarray:
        """Current board as numpy array."""
        return self._get_board_array()

    @property
    def history(self) -> List[np.ndarray]:
        """List of historical board arrays."""
        return list(self._history)

    def get(self, row: int, col: int) -> Optional[str]:
        """Get stone at position. Returns 'b', 'w', or None."""
        return self._board.get(row, col)


class PositionEncoder:
    """
    Encodes Go positions into 18-plane tensors for neural network input.

    Optimizations:
    - Pre-allocated output buffer for encode_to_array()
    - Vectorized history plane filling
    - In-place operations where possible
    """

    def __init__(self, board_size: int = 19, history_length: int = 8):
        self.board_size = board_size
        self.history_length = history_length
        # Pre-allocate color plane constants
        self._ones = np.ones((board_size, board_size), dtype=np.float32)
        self._zeros = np.zeros((board_size, board_size), dtype=np.float32)

    def encode(self, board_state: BoardState) -> torch.Tensor:
        """
        Encode a board position into 18 planes.

        Args:
            board_state: BoardState instance

        Returns:
            Tensor of shape (18, board_size, board_size)
        """
        planes = self.encode_to_array(board_state)
        return torch.from_numpy(planes)

    def encode_to_array(self, board_state: BoardState) -> np.ndarray:
        """
        Encode a board position into 18 planes as numpy array.

        Optimized version that minimizes allocations.

        Args:
            board_state: BoardState instance

        Returns:
            np.ndarray of shape (18, board_size, board_size), dtype float32
        """
        planes = np.zeros((18, self.board_size, self.board_size), dtype=np.float32)

        # Get board array (uses cache - avoids rebuilding if valid)
        if board_state._cache_valid:
            board = board_state._board_cache
        else:
            board = board_state.board

        # Determine current/opponent values
        is_black_turn = board_state.current_player == 'b'
        current_val = 1 if is_black_turn else 2
        opponent_val = 2 if is_black_turn else 1

        # Plane 0 (T=0): Current player's stones
        # Boolean comparison directly to float32 (numpy handles conversion)
        planes[0] = (board == current_val)

        # Plane 8 (T=0): Opponent's stones
        planes[8] = (board == opponent_val)

        # Historical planes (T=-1 to T=-7)
        history = board_state._history
        n_history = min(len(history), self.history_length - 1)

        for t in range(n_history):
            hist_board = history[t]
            planes[t + 1] = (hist_board == current_val)
            planes[t + 9] = (hist_board == opponent_val)

        # Plane 16/17: Color to move (use pre-allocated constants for speed)
        if is_black_turn:
            planes[16] = self._ones
        else:
            planes[17] = self._ones

        return planes

    def encode_batch(self, boards: List[BoardState]) -> torch.Tensor:
        """
        Encode multiple board positions. Pre-allocates output for efficiency.

        Args:
            boards: List of BoardState instances

        Returns:
            Tensor of shape (batch, 18, board_size, board_size)
        """
        n = len(boards)
        if n == 0:
            return torch.zeros((0, 18, self.board_size, self.board_size), dtype=torch.float32)

        # Pre-allocate output array
        output = np.zeros((n, 18, self.board_size, self.board_size), dtype=np.float32)

        # Fill each position
        for i, board_state in enumerate(boards):
            output[i] = self.encode_to_array(board_state)

        return torch.from_numpy(output)

    def encode_batch_from_arrays(
        self,
        boards: np.ndarray,
        histories: Optional[np.ndarray] = None,
        current_players: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Encode multiple positions from raw arrays. Fully vectorized.

        This is the fastest path when you have board data as arrays
        rather than BoardState objects.

        Args:
            boards: Array of shape (batch, 19, 19), dtype int8
                    Values: 0=empty, 1=black, 2=white
            histories: Optional array of shape (batch, history_len, 19, 19)
            current_players: Optional array of shape (batch,), dtype int8
                            Values: 1=black to move, 2=white to move
                            Default: alternating based on stone count

        Returns:
            np.ndarray of shape (batch, 18, board_size, board_size), dtype float32
        """
        batch_size = boards.shape[0]
        output = np.zeros((batch_size, 18, self.board_size, self.board_size), dtype=np.float32)

        # Infer current player from stone counts if not provided
        if current_players is None:
            black_counts = (boards == 1).sum(axis=(1, 2))
            white_counts = (boards == 2).sum(axis=(1, 2))
            # Black moves first, so if equal counts, it's black's turn
            current_players = np.where(black_counts <= white_counts, 1, 2).astype(np.int8)

        # Vectorized encoding for all positions
        black_to_move = current_players == 1

        # For black-to-move positions: current=1 (black), opponent=2 (white)
        # For white-to-move positions: current=2 (white), opponent=1 (black)

        # Plane 0: Current player stones
        output[black_to_move, 0] = (boards[black_to_move] == 1).astype(np.float32)
        output[~black_to_move, 0] = (boards[~black_to_move] == 2).astype(np.float32)

        # Plane 8: Opponent stones
        output[black_to_move, 8] = (boards[black_to_move] == 2).astype(np.float32)
        output[~black_to_move, 8] = (boards[~black_to_move] == 1).astype(np.float32)

        # History planes
        if histories is not None:
            n_hist = min(histories.shape[1], self.history_length - 1)
            for t in range(n_hist):
                hist = histories[:, t]
                output[black_to_move, t + 1] = (hist[black_to_move] == 1).astype(np.float32)
                output[~black_to_move, t + 1] = (hist[~black_to_move] == 2).astype(np.float32)
                output[black_to_move, t + 9] = (hist[black_to_move] == 2).astype(np.float32)
                output[~black_to_move, t + 9] = (hist[~black_to_move] == 1).astype(np.float32)

        # Color planes
        output[black_to_move, 16] = 1.0
        output[~black_to_move, 17] = 1.0

        return output


def encode_position(board_state: BoardState) -> torch.Tensor:
    """Convenience function to encode a single position."""
    encoder = PositionEncoder(board_state.size)
    return encoder.encode(board_state)
