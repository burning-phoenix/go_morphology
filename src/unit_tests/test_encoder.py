"""
Unit tests for position encoding (18-plane format for Leela Zero).

Tests the BoardState wrapper and PositionEncoder to ensure correct encoding.
Priority: Critical - wrong encoding produces activations that don't match training.

Test Organization:
- TestBoardState: Core board state functionality
- TestPositionEncoder: Single position encoding
- TestPositionEncoderBatch: Batch encoding methods
- TestEncodingCorrectness: Format compliance verification
- TestEdgeCases: Boundary conditions and error handling
- TestOptimizations: Performance-related behavior
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.data.position_encoder import BoardState, PositionEncoder, encode_position
    HAS_ENCODER = True
except ImportError as e:
    print(f"Import error: {e}")
    HAS_ENCODER = False
    BoardState = None
    PositionEncoder = None
    encode_position = None

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(not HAS_ENCODER, reason="sgfmill not installed")


# ============== Fixtures ==============

@pytest.fixture
def empty_board():
    """Fresh empty board state."""
    return BoardState()


@pytest.fixture
def board_with_moves():
    """Board with a few moves played."""
    state = BoardState()
    state.play(3, 3, 'b')
    state.play(15, 15, 'w')
    state.play(3, 15, 'b')
    state.play(15, 3, 'w')
    return state


@pytest.fixture
def encoder():
    """Standard 19x19 encoder."""
    return PositionEncoder(board_size=19)


@pytest.fixture
def encoder_9x9():
    """9x9 board encoder."""
    return PositionEncoder(board_size=9)


# ============== BoardState Tests ==============

class TestBoardState:
    """Tests for BoardState wrapper around sgfmill."""

    def test_init_empty(self, empty_board):
        """New board should be empty with black to play."""
        assert empty_board.board.shape == (19, 19)
        assert (empty_board.board == 0).all()
        assert empty_board.current_player == 'b'
        assert empty_board.move_count == 0
        assert len(empty_board.history) == 0

    @pytest.mark.parametrize("size", [9, 13, 19])
    def test_init_custom_size(self, size):
        """Should support standard board sizes."""
        state = BoardState(size=size)
        assert state.board.shape == (size, size)
        assert state.size == size

    def test_play_stone_updates_board(self):
        """Playing a stone should update board array."""
        state = BoardState()
        state.play(3, 3, 'b')

        assert state.board[3, 3] == 1  # Black = 1
        assert state.current_player == 'w'
        assert state.move_count == 1

    def test_play_stone_updates_history(self):
        """Playing a stone should add previous state to history."""
        state = BoardState()

        # History empty before first move
        assert len(state.history) == 0

        state.play(3, 3, 'b')

        # History[0] = state before move (empty board)
        assert len(state.history) == 1
        assert (state.history[0] == 0).all()

    def test_play_alternates_colors(self):
        """Playing without color should alternate automatically."""
        state = BoardState()
        state.play(3, 3)   # Black (default first player)
        state.play(15, 15) # White (auto-alternate)

        assert state.board[3, 3] == 1   # Black
        assert state.board[15, 15] == 2 # White
        assert state.current_player == 'b'

    def test_history_length_limit(self):
        """History should be limited to configured length."""
        history_length = 8
        state = BoardState(history_length=history_length)

        # Play more moves than history length
        for i in range(15):
            r, c = i % 19, (i * 3) % 19
            state.play(r, c)

        assert len(state.history) <= history_length

    def test_pass_move(self):
        """Pass should switch player without placing stone."""
        state = BoardState()
        state.pass_move()

        assert (state.board == 0).all()
        assert state.current_player == 'w'
        assert state.move_count == 1
        assert len(state.history) == 1

    def test_copy_creates_independent_state(self):
        """Copy should create fully independent copy."""
        original = BoardState()
        original.play(3, 3, 'b')

        copy = original.copy()
        copy.play(15, 15, 'w')

        # Original unchanged
        assert original.board[15, 15] == 0
        assert original.move_count == 1
        # Copy modified
        assert copy.board[15, 15] == 2
        assert copy.move_count == 2

    def test_copy_preserves_history(self):
        """Copy should preserve history contents."""
        state = BoardState()
        state.play(3, 3, 'b')
        state.play(15, 15, 'w')

        copy = state.copy()

        assert len(copy.history) == len(state.history)
        for orig_hist, copy_hist in zip(state.history, copy.history):
            assert np.array_equal(orig_hist, copy_hist)

    def test_get_returns_stone_color(self):
        """get() should return stone color or None."""
        state = BoardState()
        state.play(3, 3, 'b')
        state.play(15, 15, 'w')

        assert state.get(3, 3) == 'b'
        assert state.get(15, 15) == 'w'
        assert state.get(0, 0) is None

    def test_cache_invalidation_on_play(self):
        """Cache should be invalidated after play (captures may occur)."""
        state = BoardState()
        state.play(3, 3, 'b')

        # After play, cache is invalidated (captures might have occurred)
        assert state._cache_valid == False


# ============== PositionEncoder Tests ==============

class TestPositionEncoder:
    """Tests for PositionEncoder single-position encoding."""

    def test_output_shape(self, empty_board, encoder):
        """Encoded position should have shape (18, 19, 19)."""
        planes = encoder.encode(empty_board)

        assert planes.shape == (18, 19, 19)
        assert planes.dtype == torch.float32

    def test_output_shape_9x9(self, encoder_9x9):
        """Should handle different board sizes."""
        state = BoardState(size=9)
        planes = encoder_9x9.encode(state)

        assert planes.shape == (18, 9, 9)

    def test_empty_board_stone_planes_zero(self, empty_board, encoder):
        """Empty board should have zeros in stone planes (0-15)."""
        planes = encoder.encode(empty_board)

        assert (planes[:16] == 0).all()

    def test_black_to_play_plane(self, empty_board, encoder):
        """When black to play, plane 16 should be all 1s, plane 17 all 0s."""
        planes = encoder.encode(empty_board)

        assert planes[16].sum() == 19 * 19
        assert planes[17].sum() == 0

    def test_white_to_play_plane(self, encoder):
        """When white to play, plane 17 should be all 1s, plane 16 all 0s."""
        state = BoardState()
        state.play(3, 3, 'b')  # Now white to play

        planes = encoder.encode(state)

        assert planes[16].sum() == 0
        assert planes[17].sum() == 19 * 19

    def test_current_player_stones_plane_0(self, board_with_moves, encoder):
        """Current player's stones should appear in plane 0."""
        # board_with_moves: 4 moves, black to play
        planes = encoder.encode(board_with_moves)

        # Black stones (current player) at (3,3) and (3,15)
        assert planes[0, 3, 3] == 1
        assert planes[0, 3, 15] == 1
        # White stones should not be in plane 0
        assert planes[0, 15, 15] == 0
        assert planes[0, 15, 3] == 0

    def test_opponent_stones_plane_8(self, board_with_moves, encoder):
        """Opponent's stones should appear in plane 8."""
        planes = encoder.encode(board_with_moves)

        # White stones (opponent) at (15,15) and (15,3)
        assert planes[8, 15, 15] == 1
        assert planes[8, 15, 3] == 1
        # Black stones should not be in plane 8
        assert planes[8, 3, 3] == 0

    def test_history_planes(self, encoder):
        """Historical positions should appear in planes 1-7 and 9-15."""
        state = BoardState()
        state.play(0, 0, 'b')  # Move 1
        state.play(1, 1, 'w')  # Move 2
        state.play(2, 2, 'b')  # Move 3 - now white to play

        planes = encoder.encode(state)

        # Current player is white, so:
        # - Plane 1 = white stones at T=-1 (position before move 3)
        # - Plane 9 = black stones at T=-1
        assert planes[1, 1, 1] == 1  # White was at (1,1) at T=-1
        assert planes[9, 0, 0] == 1  # Black was at (0,0) at T=-1
        assert planes[9, 2, 2] == 0  # Black (2,2) wasn't played yet at T=-1

    def test_encode_to_array_matches_encode(self, board_with_moves, encoder):
        """encode_to_array should produce same result as encode."""
        tensor_result = encoder.encode(board_with_moves).numpy()
        array_result = encoder.encode_to_array(board_with_moves)

        assert np.allclose(tensor_result, array_result)

    def test_convenience_function(self, board_with_moves):
        """encode_position convenience function should work."""
        planes = encode_position(board_with_moves)

        assert planes.shape == (18, 19, 19)
        assert isinstance(planes, torch.Tensor)


# ============== Batch Encoding Tests ==============

class TestPositionEncoderBatch:
    """Tests for batch encoding methods."""

    def test_encode_batch_shape(self, encoder):
        """encode_batch should return correct shape."""
        states = [BoardState() for _ in range(5)]
        for i, state in enumerate(states):
            if i > 0:
                state.play(i, i, 'b')

        batch = encoder.encode_batch(states)

        assert batch.shape == (5, 18, 19, 19)
        assert batch.dtype == torch.float32

    def test_encode_batch_empty_list(self, encoder):
        """encode_batch should handle empty list."""
        batch = encoder.encode_batch([])

        assert batch.shape == (0, 18, 19, 19)

    def test_encode_batch_matches_individual(self, encoder):
        """Batch encoding should match individual encoding."""
        states = []
        for i in range(3):
            state = BoardState()
            state.play(i, i, 'b')
            states.append(state)

        batch = encoder.encode_batch(states)

        for i, state in enumerate(states):
            individual = encoder.encode(state)
            assert torch.allclose(batch[i], individual)

    def test_encode_batch_from_arrays_basic(self, encoder):
        """encode_batch_from_arrays should encode raw board arrays."""
        # Create simple board arrays
        boards = np.zeros((3, 19, 19), dtype=np.int8)
        boards[0, 3, 3] = 1  # Black stone
        boards[1, 3, 3] = 1
        boards[1, 15, 15] = 2  # White stone
        boards[2, 9, 9] = 1

        result = encoder.encode_batch_from_arrays(boards)

        assert result.shape == (3, 18, 19, 19)
        assert result.dtype == np.float32

    def test_encode_batch_from_arrays_with_current_player(self, encoder):
        """encode_batch_from_arrays should respect current_players arg."""
        boards = np.zeros((2, 19, 19), dtype=np.int8)
        boards[0, 3, 3] = 1  # Black stone
        boards[1, 3, 3] = 1  # Same board

        # One black to move, one white to move
        current_players = np.array([1, 2], dtype=np.int8)

        result = encoder.encode_batch_from_arrays(boards, current_players=current_players)

        # Position 0: black to move, black stone in plane 0
        assert result[0, 0, 3, 3] == 1
        assert result[0, 16].sum() == 19 * 19  # Black to move plane

        # Position 1: white to move, black stone in plane 8 (opponent)
        assert result[1, 8, 3, 3] == 1
        assert result[1, 17].sum() == 19 * 19  # White to move plane

    def test_encode_batch_from_arrays_infers_player(self, encoder):
        """encode_batch_from_arrays should infer current player from stone count."""
        boards = np.zeros((2, 19, 19), dtype=np.int8)
        # Equal stones = black to move (black moves first)
        boards[0, 3, 3] = 1
        boards[0, 15, 15] = 2
        # More black = white to move
        boards[1, 3, 3] = 1
        boards[1, 3, 4] = 1
        boards[1, 15, 15] = 2

        result = encoder.encode_batch_from_arrays(boards)

        # Position 0: equal stones, black to move
        assert result[0, 16].sum() == 19 * 19
        # Position 1: more black, white to move
        assert result[1, 17].sum() == 19 * 19


# ============== Encoding Correctness Tests ==============

class TestEncodingCorrectness:
    """Tests verifying encoding matches Leela Zero format."""

    def test_plane_structure(self, encoder):
        """Encoding should follow documented 18-plane format."""
        state = BoardState()
        for r, c, color in [(3, 3, 'b'), (15, 15, 'w'), (3, 4, 'b'), (15, 16, 'w')]:
            state.play(r, c, color)

        planes = encoder.encode(state)

        # Verify structure
        assert planes.shape[0] == 18
        # Exactly one color plane should be full
        assert planes[16].sum() + planes[17].sum() == 19 * 19
        # Current positions in planes 0 and 8
        total_stones = (planes[0] + planes[8] > 0).sum()
        assert total_stones == 4

    def test_color_perspective_black_to_play(self, encoder):
        """When black to play, black stones in plane 0, white in plane 8."""
        state = BoardState()
        state.play(3, 3, 'b')
        state.play(15, 15, 'w')
        # Black to play

        planes = encoder.encode(state)

        assert planes[0, 3, 3] == 1    # Black in plane 0
        assert planes[8, 15, 15] == 1  # White in plane 8

    def test_color_perspective_white_to_play(self, encoder):
        """When white to play, white stones in plane 0, black in plane 8."""
        state = BoardState()
        state.play(3, 3, 'b')
        # White to play

        planes = encoder.encode(state)

        assert planes[0, 3, 3] == 0    # Black NOT in plane 0
        assert planes[8, 3, 3] == 1    # Black in plane 8 (opponent)

    def test_history_temporal_order(self, encoder):
        """History planes should have correct temporal ordering."""
        state = BoardState()
        # Play sequence of moves
        moves = [(0, 0, 'b'), (1, 0, 'w'), (0, 1, 'b'), (1, 1, 'w')]
        for r, c, color in moves:
            state.play(r, c, color)

        planes = encoder.encode(state)

        # T=-1 should have state before last move
        # At T=-1, we have moves 0,1,2 played (not move 3)
        # Black to play, so plane 1 = black at T=-1, plane 9 = white at T=-1
        assert planes[1, 0, 0] == 1  # Black (0,0) existed at T=-1
        assert planes[1, 0, 1] == 1  # Black (0,1) existed at T=-1
        assert planes[9, 1, 0] == 1  # White (1,0) existed at T=-1
        assert planes[9, 1, 1] == 0  # White (1,1) NOT at T=-1 (played at T=0)


# ============== Edge Cases ==============

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_capture_removes_stone(self, encoder):
        """Captured stones should not appear in encoding."""
        state = BoardState()
        # Create capture: white surrounds black at (1,0)
        state.play(1, 0, 'b')  # Black to capture
        state.play(0, 0, 'w')
        state.play(9, 9, 'b')  # Tenuki
        state.play(2, 0, 'w')
        state.play(9, 10, 'b')  # Tenuki
        state.play(1, 1, 'w')  # Captures black at (1,0)

        # Verify capture occurred
        assert state.board[1, 0] == 0

        planes = encoder.encode(state)
        # Captured stone should not be in any current plane
        assert planes[0, 1, 0] == 0
        assert planes[8, 1, 0] == 0

    def test_many_moves(self, encoder):
        """Should handle games with many moves without error."""
        state = BoardState()

        # Play 50 moves
        for i in range(50):
            r, c = (i * 7) % 19, (i * 11) % 19
            try:
                state.play(r, c)
            except Exception:
                pass  # Ignore illegal moves (occupied positions)

        planes = encoder.encode(state)
        assert planes.shape == (18, 19, 19)

    def test_all_history_planes_populated(self, encoder):
        """After 8+ moves, all history planes should have content."""
        state = BoardState()

        # Play 10 moves
        for i in range(10):
            r, c = i, i
            state.play(r, c)

        planes = encoder.encode(state)

        # Planes 1-7 should have some content (history)
        for t in range(1, 8):
            assert planes[t].sum() > 0 or planes[t + 8].sum() > 0

    @pytest.mark.parametrize("row,col", [
        (0, 0),   # Corner
        (0, 18),  # Corner
        (18, 0),  # Corner
        (18, 18), # Corner
        (0, 9),   # Edge
        (9, 0),   # Edge
        (9, 9),   # Center
    ])
    def test_stone_at_various_positions(self, encoder, row, col):
        """Stones at corners, edges, and center should encode correctly."""
        state = BoardState()
        state.play(row, col, 'b')

        planes = encoder.encode(state)

        # White to play, so black (opponent) is in plane 8
        assert planes[8, row, col] == 1


# ============== Optimization Verification ==============

class TestOptimizations:
    """Tests verifying optimized methods produce correct results."""

    def test_cache_reuse(self, encoder):
        """Cache should be reused when valid."""
        state = BoardState()
        state.play(3, 3, 'b')

        # Access board to rebuild cache
        _ = state.board
        assert state._cache_valid == True

        # Encode should use cache
        planes = encoder.encode(state)
        assert planes[8, 3, 3] == 1  # Correct result using cache

    def test_encode_to_array_same_as_encode(self, encoder):
        """encode_to_array optimization should match encode exactly."""
        state = BoardState()
        for i in range(5):
            state.play(i, i * 2 % 19)

        tensor_result = encoder.encode(state).numpy()
        array_result = encoder.encode_to_array(state)

        assert np.array_equal(tensor_result, array_result)

    def test_batch_pre_allocation(self, encoder):
        """Batch encoding should produce contiguous output."""
        states = [BoardState() for _ in range(10)]
        batch = encoder.encode_batch(states)

        # Output should be contiguous (efficient memory layout)
        assert batch.is_contiguous()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
