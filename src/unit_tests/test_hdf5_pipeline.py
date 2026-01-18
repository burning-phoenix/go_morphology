"""
Unit tests for HDF5 position loading pipeline.

Tests the load_positions_hdf5 function and streaming variants.
Priority: Critical - ensures pre-encoded positions match direct encoding.

Test Organization:
- TestHDF5Pipeline: Basic loading functionality
- TestHDF5EdgeCases: Error handling and boundary conditions
- TestGameIdMoveNumberCorrespondence: Metadata tracking verification
- TestEncodingRoundtrip: Encoding correctness through HDF5
- TestHDF5Creation: Write path verification
- TestHDF5Streaming: Streaming/chunked loading
"""

import pytest
import numpy as np
import tempfile
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from src.data import (
        load_positions_hdf5,
        load_positions_hdf5_streaming,
        BoardState,
        PositionEncoder,
    )
    HAS_DEPS = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Import error: {e}")
    HAS_DEPS = False
    BoardState = None
    PositionEncoder = None
    load_positions_hdf5 = None
    load_positions_hdf5_streaming = None

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not HAS_H5PY or not HAS_DEPS,
    reason="h5py or sgfmill not installed"
)


# ============== Helper Functions ==============

def create_test_hdf5(
    path: str,
    n_positions: int = 100,
    n_games: int = 10,
    include_game_ids: bool = True,
    include_move_numbers: bool = True,
    include_positions: bool = True,
    positions_shape: tuple = None,
    positions_dtype: np.dtype = np.float32,
    attrs: dict = None,
):
    """Helper to create test HDF5 files with various configurations."""
    with h5py.File(path, 'w') as f:
        if include_positions:
            shape = positions_shape or (n_positions, 18, 19, 19)
            positions = np.random.randn(*shape).astype(positions_dtype)
            f.create_dataset('positions', data=positions)

        if include_game_ids:
            # Create game_ids that map positions to games
            game_ids = np.repeat(np.arange(n_games), n_positions // n_games + 1)[:n_positions]
            f.create_dataset('game_ids', data=game_ids.astype(np.int32))

        if include_move_numbers:
            # Create sequential move numbers within each game
            move_numbers = np.zeros(n_positions, dtype=np.int16)
            if include_game_ids:
                for game_id in range(n_games):
                    mask = game_ids == game_id
                    move_numbers[mask] = np.arange(mask.sum())
            f.create_dataset('move_numbers', data=move_numbers)

        # Set attributes
        f.attrs['n_games'] = n_games
        f.attrs['n_positions'] = n_positions
        if attrs:
            for key, value in attrs.items():
                f.attrs[key] = value


@pytest.fixture
def temp_hdf5():
    """Fixture that creates a temp file and cleans up after test."""
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


# ============== Basic Loading Tests ==============

class TestHDF5Pipeline:
    """Tests for HDF5 position loading."""

    def test_load_positions_basic(self, temp_hdf5):
        """Test loading positions from HDF5 file."""
        n_positions = 100
        n_games = 10
        create_test_hdf5(temp_hdf5, n_positions=n_positions, n_games=n_games)

        # Load and verify
        loaded_positions, metadata = load_positions_hdf5(temp_hdf5)

        assert loaded_positions.shape == (n_positions, 18, 19, 19)
        assert loaded_positions.dtype == np.float32
        assert metadata['n_positions'] == n_positions
        assert metadata['n_games'] == n_games
        assert len(metadata['game_ids']) == n_positions
        assert len(metadata['move_numbers']) == n_positions
        assert metadata['game_ids'].dtype == np.int32
        assert metadata['move_numbers'].dtype == np.int16

    def test_load_positions_max_limit(self, temp_hdf5):
        """Test max_positions parameter limits loaded data."""
        n_positions = 100
        create_test_hdf5(temp_hdf5, n_positions=n_positions)

        # Load only 50
        loaded_positions, metadata = load_positions_hdf5(temp_hdf5, max_positions=50)

        assert loaded_positions.shape == (50, 18, 19, 19)
        assert metadata['n_positions'] == 50
        assert len(metadata['game_ids']) == 50
        assert len(metadata['move_numbers']) == 50

    def test_max_positions_exceeds_available(self, temp_hdf5):
        """Test max_positions > actual positions loads all available."""
        n_positions = 50
        create_test_hdf5(temp_hdf5, n_positions=n_positions)

        # Request more than available
        loaded_positions, metadata = load_positions_hdf5(temp_hdf5, max_positions=1000)

        assert loaded_positions.shape == (n_positions, 18, 19, 19)
        assert metadata['n_positions'] == n_positions

    def test_metadata_preservation(self, temp_hdf5):
        """Verify metadata attributes survive roundtrip."""
        custom_attrs = {
            'min_move': 10,
            'max_move': 150,
            'seed': 42,
            'source': 'test_file.sgf.xz',
        }
        create_test_hdf5(temp_hdf5, n_positions=10, n_games=5, attrs=custom_attrs)

        _, metadata = load_positions_hdf5(temp_hdf5)

        # Check top-level metadata
        assert metadata['n_positions'] == 10
        assert metadata['n_games'] == 5

        # Check config (attributes)
        assert metadata['config']['n_games'] == 5
        assert metadata['config']['min_move'] == 10
        assert metadata['config']['max_move'] == 150
        assert metadata['config']['seed'] == 42
        assert metadata['config']['source'] == 'test_file.sgf.xz'


# ============== Edge Case Tests ==============

class TestHDF5EdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_position(self, temp_hdf5):
        """Test loading a file with exactly one position."""
        create_test_hdf5(temp_hdf5, n_positions=1, n_games=1)

        loaded_positions, metadata = load_positions_hdf5(temp_hdf5)

        assert loaded_positions.shape == (1, 18, 19, 19)
        assert metadata['n_positions'] == 1
        assert len(metadata['game_ids']) == 1
        assert len(metadata['move_numbers']) == 1

    def test_missing_positions_dataset(self, temp_hdf5):
        """Test error when positions dataset is missing."""
        create_test_hdf5(temp_hdf5, include_positions=False)

        with pytest.raises(ValueError, match="missing required datasets.*positions"):
            load_positions_hdf5(temp_hdf5)

    def test_missing_game_ids_dataset(self, temp_hdf5):
        """Test error when game_ids dataset is missing."""
        create_test_hdf5(temp_hdf5, include_game_ids=False)

        with pytest.raises(ValueError, match="missing required datasets.*game_ids"):
            load_positions_hdf5(temp_hdf5)

    def test_missing_move_numbers_dataset(self, temp_hdf5):
        """Test error when move_numbers dataset is missing."""
        create_test_hdf5(temp_hdf5, include_move_numbers=False)

        with pytest.raises(ValueError, match="missing required datasets.*move_numbers"):
            load_positions_hdf5(temp_hdf5)

    def test_invalid_positions_shape_3d(self, temp_hdf5):
        """Test error when positions has wrong number of dimensions."""
        with h5py.File(temp_hdf5, 'w') as f:
            # 3D instead of 4D
            f.create_dataset('positions', data=np.random.randn(10, 18, 19).astype(np.float32))
            f.create_dataset('game_ids', data=np.zeros(10, dtype=np.int32))
            f.create_dataset('move_numbers', data=np.zeros(10, dtype=np.int16))

        with pytest.raises(ValueError, match="positions must be 4D"):
            load_positions_hdf5(temp_hdf5)

    def test_invalid_positions_shape_wrong_planes(self, temp_hdf5):
        """Test error when positions has wrong plane count."""
        with h5py.File(temp_hdf5, 'w') as f:
            # 16 planes instead of 18
            f.create_dataset('positions', data=np.random.randn(10, 16, 19, 19).astype(np.float32))
            f.create_dataset('game_ids', data=np.zeros(10, dtype=np.int32))
            f.create_dataset('move_numbers', data=np.zeros(10, dtype=np.int16))

        with pytest.raises(ValueError, match="positions must have shape.*18, 19, 19"):
            load_positions_hdf5(temp_hdf5)

    def test_mismatched_game_ids_length(self, temp_hdf5):
        """Test error when game_ids length doesn't match positions."""
        with h5py.File(temp_hdf5, 'w') as f:
            f.create_dataset('positions', data=np.random.randn(100, 18, 19, 19).astype(np.float32))
            f.create_dataset('game_ids', data=np.zeros(50, dtype=np.int32))  # Wrong length
            f.create_dataset('move_numbers', data=np.zeros(100, dtype=np.int16))

        with pytest.raises(ValueError, match="game_ids length.*doesn't match"):
            load_positions_hdf5(temp_hdf5)

    def test_mismatched_move_numbers_length(self, temp_hdf5):
        """Test error when move_numbers length doesn't match positions."""
        with h5py.File(temp_hdf5, 'w') as f:
            f.create_dataset('positions', data=np.random.randn(100, 18, 19, 19).astype(np.float32))
            f.create_dataset('game_ids', data=np.zeros(100, dtype=np.int32))
            f.create_dataset('move_numbers', data=np.zeros(50, dtype=np.int16))  # Wrong length

        with pytest.raises(ValueError, match="move_numbers length.*doesn't match"):
            load_positions_hdf5(temp_hdf5)

    def test_skip_validation(self, temp_hdf5):
        """Test that validation can be skipped."""
        with h5py.File(temp_hdf5, 'w') as f:
            # Wrong shape but validation disabled
            f.create_dataset('positions', data=np.random.randn(10, 16, 19, 19).astype(np.float32))
            f.create_dataset('game_ids', data=np.zeros(10, dtype=np.int32))
            f.create_dataset('move_numbers', data=np.zeros(10, dtype=np.int16))

        # Should not raise with validate=False
        loaded_positions, _ = load_positions_hdf5(temp_hdf5, validate=False)
        assert loaded_positions.shape == (10, 16, 19, 19)

    def test_dtype_conversion(self, temp_hdf5):
        """Test that float64 positions are converted to float32."""
        with h5py.File(temp_hdf5, 'w') as f:
            # Save as float64
            f.create_dataset('positions', data=np.random.randn(10, 18, 19, 19).astype(np.float64))
            f.create_dataset('game_ids', data=np.zeros(10, dtype=np.int64))  # Wrong dtype
            f.create_dataset('move_numbers', data=np.zeros(10, dtype=np.int32))  # Wrong dtype

        loaded_positions, metadata = load_positions_hdf5(temp_hdf5)

        # Should be converted to expected dtypes
        assert loaded_positions.dtype == np.float32
        assert metadata['game_ids'].dtype == np.int32
        assert metadata['move_numbers'].dtype == np.int16

    def test_path_object_support(self, temp_hdf5):
        """Test that Path objects are accepted."""
        create_test_hdf5(temp_hdf5, n_positions=10)

        # Pass as Path object
        loaded_positions, _ = load_positions_hdf5(Path(temp_hdf5))
        assert loaded_positions.shape == (10, 18, 19, 19)

    def test_n_games_fallback_to_unique(self, temp_hdf5):
        """Test n_games falls back to counting unique game_ids if attr missing."""
        with h5py.File(temp_hdf5, 'w') as f:
            f.create_dataset('positions', data=np.random.randn(10, 18, 19, 19).astype(np.float32))
            game_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3], dtype=np.int32)
            f.create_dataset('game_ids', data=game_ids)
            f.create_dataset('move_numbers', data=np.zeros(10, dtype=np.int16))
            # Deliberately don't set n_games attribute

        _, metadata = load_positions_hdf5(temp_hdf5)

        # Should count unique game_ids
        assert metadata['n_games'] == 4


# ============== Game ID / Move Number Tests ==============

class TestGameIdMoveNumberCorrespondence:
    """Tests verifying game_ids and move_numbers track positions correctly."""

    def test_game_ids_map_to_positions(self, temp_hdf5):
        """Verify game_ids correctly partition positions."""
        n_games = 5
        positions_per_game = 10
        n_positions = n_games * positions_per_game

        with h5py.File(temp_hdf5, 'w') as f:
            positions = np.random.randn(n_positions, 18, 19, 19).astype(np.float32)
            game_ids = np.repeat(np.arange(n_games), positions_per_game).astype(np.int32)
            move_numbers = np.tile(np.arange(positions_per_game), n_games).astype(np.int16)

            f.create_dataset('positions', data=positions)
            f.create_dataset('game_ids', data=game_ids)
            f.create_dataset('move_numbers', data=move_numbers)
            f.attrs['n_games'] = n_games

        loaded_positions, metadata = load_positions_hdf5(temp_hdf5)

        # Verify each game has correct number of positions
        for game_id in range(n_games):
            game_mask = metadata['game_ids'] == game_id
            assert game_mask.sum() == positions_per_game

            # Verify move numbers are sequential within game
            game_moves = metadata['move_numbers'][game_mask]
            assert np.array_equal(game_moves, np.arange(positions_per_game))

    def test_move_numbers_are_sequential_within_games(self, temp_hdf5):
        """Verify move numbers form valid sequences within each game."""
        # Create positions with varying game lengths
        game_lengths = [5, 10, 3, 7]
        n_positions = sum(game_lengths)

        with h5py.File(temp_hdf5, 'w') as f:
            positions = np.random.randn(n_positions, 18, 19, 19).astype(np.float32)

            game_ids = []
            move_numbers = []
            for game_id, length in enumerate(game_lengths):
                game_ids.extend([game_id] * length)
                move_numbers.extend(range(length))

            f.create_dataset('positions', data=positions)
            f.create_dataset('game_ids', data=np.array(game_ids, dtype=np.int32))
            f.create_dataset('move_numbers', data=np.array(move_numbers, dtype=np.int16))
            f.attrs['n_games'] = len(game_lengths)

        _, metadata = load_positions_hdf5(temp_hdf5)

        # Verify each game's move sequence
        for game_id, expected_length in enumerate(game_lengths):
            mask = metadata['game_ids'] == game_id
            moves = metadata['move_numbers'][mask]

            assert len(moves) == expected_length
            assert moves[0] == 0  # First move is 0
            assert np.all(np.diff(moves) == 1)  # Sequential


# ============== Encoding Roundtrip Tests ==============

class TestEncodingRoundtrip:
    """Tests verifying encoding correctness through HDF5 roundtrip."""

    def test_encoding_roundtrip(self, temp_hdf5):
        """Verify encoded positions match between direct encode and HDF5 load."""
        encoder = PositionEncoder(board_size=19)

        # Create board state with some moves
        board_state = BoardState(size=19)
        board_state.play(3, 3, 'b')
        board_state.play(15, 15, 'w')
        board_state.play(3, 15, 'b')
        board_state.play(15, 3, 'w')

        # Encode directly
        direct_encoded = encoder.encode(board_state).numpy()

        # Save to HDF5
        with h5py.File(temp_hdf5, 'w') as f:
            f.create_dataset('positions', data=direct_encoded[np.newaxis, ...])
            f.create_dataset('game_ids', data=np.array([0], dtype=np.int32))
            f.create_dataset('move_numbers', data=np.array([4], dtype=np.int16))

        # Load from HDF5
        loaded_positions, _ = load_positions_hdf5(temp_hdf5)

        # Should match exactly
        assert loaded_positions.shape == (1, 18, 19, 19)
        assert np.allclose(loaded_positions[0], direct_encoded)

    def test_encoding_correctness_black_stones(self, temp_hdf5):
        """Verify black stones appear in correct planes."""
        encoder = PositionEncoder(board_size=19)

        board_state = BoardState(size=19)
        # Play some black stones (black to move = black is "current player")
        board_state.play(3, 3, 'b')  # Move 1
        board_state.play(10, 10, 'w')  # Move 2
        board_state.play(5, 5, 'b')  # Move 3
        board_state.play(10, 11, 'w')  # Move 4
        # Now it's black to move

        encoded = encoder.encode(board_state).numpy()

        # Save and reload
        with h5py.File(temp_hdf5, 'w') as f:
            f.create_dataset('positions', data=encoded[np.newaxis, ...])
            f.create_dataset('game_ids', data=np.array([0], dtype=np.int32))
            f.create_dataset('move_numbers', data=np.array([4], dtype=np.int16))

        loaded, _ = load_positions_hdf5(temp_hdf5)
        loaded = loaded[0]

        # Black to move, so black stones are "current player" (plane 0)
        assert loaded[0, 3, 3] == 1.0, "Black stone at (3,3) should be in plane 0"
        assert loaded[0, 5, 5] == 1.0, "Black stone at (5,5) should be in plane 0"

        # White stones are "opponent" (plane 8)
        assert loaded[8, 10, 10] == 1.0, "White stone at (10,10) should be in plane 8"
        assert loaded[8, 10, 11] == 1.0, "White stone at (10,11) should be in plane 8"

        # Color planes
        assert loaded[16].sum() == 19 * 19, "Plane 16 should be all 1s (black to move)"
        assert loaded[17].sum() == 0, "Plane 17 should be all 0s (not white to move)"

    def test_encoding_correctness_white_to_move(self, temp_hdf5):
        """Verify encoding when white is to move."""
        encoder = PositionEncoder(board_size=19)

        board_state = BoardState(size=19)
        board_state.play(3, 3, 'b')  # Move 1 - now white to move

        encoded = encoder.encode(board_state).numpy()

        with h5py.File(temp_hdf5, 'w') as f:
            f.create_dataset('positions', data=encoded[np.newaxis, ...])
            f.create_dataset('game_ids', data=np.array([0], dtype=np.int32))
            f.create_dataset('move_numbers', data=np.array([1], dtype=np.int16))

        loaded, _ = load_positions_hdf5(temp_hdf5)
        loaded = loaded[0]

        # White to move, so white is "current player" (no white stones yet)
        assert loaded[0].sum() == 0, "No current player (white) stones"

        # Black is "opponent" (plane 8)
        assert loaded[8, 3, 3] == 1.0, "Black stone at (3,3) should be in plane 8 (opponent)"

        # Color planes
        assert loaded[16].sum() == 0, "Plane 16 should be all 0s (not black to move)"
        assert loaded[17].sum() == 19 * 19, "Plane 17 should be all 1s (white to move)"

    def test_history_planes_populated(self, temp_hdf5):
        """Verify history planes are populated correctly."""
        encoder = PositionEncoder(board_size=19)

        board_state = BoardState(size=19)
        # Play several moves to build history
        moves = [(3, 3, 'b'), (15, 15, 'w'), (4, 4, 'b'), (14, 14, 'w'), (5, 5, 'b')]
        for row, col, color in moves:
            board_state.play(row, col, color)

        encoded = encoder.encode(board_state).numpy()

        with h5py.File(temp_hdf5, 'w') as f:
            f.create_dataset('positions', data=encoded[np.newaxis, ...])
            f.create_dataset('game_ids', data=np.array([0], dtype=np.int32))
            f.create_dataset('move_numbers', data=np.array([5], dtype=np.int16))

        loaded, _ = load_positions_hdf5(temp_hdf5)
        loaded = loaded[0]

        # Plane 0 is current position, planes 1-7 are history
        # Check that history planes have some content (not all zeros)
        total_history = sum(loaded[i].sum() for i in range(1, 8))
        assert total_history > 0, "History planes should have some content"


# ============== HDF5 Creation Tests ==============

class TestHDF5Creation:
    """Tests for HDF5 file creation and write path."""

    def test_create_and_load_positions(self, temp_hdf5):
        """Test full pipeline: create encoded positions, save to HDF5, reload."""
        encoder = PositionEncoder(board_size=19)

        # Create multiple board states
        board_states = []
        for i in range(5):
            bs = BoardState(size=19)
            # Play some moves
            bs.play(i, i, 'b')
            bs.play(18 - i, 18 - i, 'w')
            board_states.append(bs)

        # Encode all positions
        encoded_positions = np.stack([encoder.encode(bs).numpy() for bs in board_states])
        game_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        move_numbers = np.array([0, 1, 0, 1, 0], dtype=np.int16)

        # Save to HDF5
        with h5py.File(temp_hdf5, 'w') as f:
            f.create_dataset('positions', data=encoded_positions.astype(np.float32))
            f.create_dataset('game_ids', data=game_ids)
            f.create_dataset('move_numbers', data=move_numbers)
            f.attrs['n_games'] = 3
            f.attrs['n_positions'] = 5
            f.attrs['source'] = 'test'

        # Load and verify
        loaded_positions, metadata = load_positions_hdf5(temp_hdf5)

        assert loaded_positions.shape == (5, 18, 19, 19)
        assert np.allclose(loaded_positions, encoded_positions)
        assert metadata['n_positions'] == 5
        assert metadata['n_games'] == 3
        assert np.array_equal(metadata['game_ids'], game_ids)
        assert np.array_equal(metadata['move_numbers'], move_numbers)

    def test_create_large_dataset(self, temp_hdf5):
        """Test creating and loading a larger dataset."""
        n_positions = 1000
        n_games = 50

        # Generate random encoded positions
        positions = np.random.randn(n_positions, 18, 19, 19).astype(np.float32)
        game_ids = np.repeat(np.arange(n_games), n_positions // n_games).astype(np.int32)
        move_numbers = np.tile(np.arange(n_positions // n_games), n_games).astype(np.int16)

        # Save
        with h5py.File(temp_hdf5, 'w') as f:
            f.create_dataset('positions', data=positions, compression='gzip')
            f.create_dataset('game_ids', data=game_ids)
            f.create_dataset('move_numbers', data=move_numbers)
            f.attrs['n_games'] = n_games
            f.attrs['n_positions'] = n_positions

        # Load and verify
        loaded_positions, metadata = load_positions_hdf5(temp_hdf5)

        assert loaded_positions.shape == (n_positions, 18, 19, 19)
        assert np.allclose(loaded_positions, positions)
        assert metadata['n_positions'] == n_positions

    def test_chunked_loading_consistency(self, temp_hdf5):
        """Test that loading with max_positions gives same data as full load."""
        n_positions = 100
        create_test_hdf5(temp_hdf5, n_positions=n_positions, n_games=10)

        # Load full dataset
        full_positions, full_meta = load_positions_hdf5(temp_hdf5)

        # Load first 50
        partial_positions, partial_meta = load_positions_hdf5(temp_hdf5, max_positions=50)

        # Verify partial is prefix of full
        assert np.allclose(partial_positions, full_positions[:50])
        assert np.array_equal(partial_meta['game_ids'], full_meta['game_ids'][:50])
        assert np.array_equal(partial_meta['move_numbers'], full_meta['move_numbers'][:50])


# ============== Streaming Tests ==============

class TestHDF5Streaming:
    """Tests for streaming/chunked HDF5 loading."""

    def test_streaming_basic(self, temp_hdf5):
        """Test basic streaming iteration."""
        n_positions = 100
        create_test_hdf5(temp_hdf5, n_positions=n_positions, n_games=10)

        # Collect all batches
        all_positions = []
        all_game_ids = []
        all_move_numbers = []

        for positions, game_ids, move_numbers in load_positions_hdf5_streaming(
            temp_hdf5, batch_size=25
        ):
            all_positions.append(positions)
            all_game_ids.append(game_ids)
            all_move_numbers.append(move_numbers)

        # Verify total
        total_positions = np.concatenate(all_positions)
        assert total_positions.shape == (n_positions, 18, 19, 19)
        assert len(all_positions) == 4  # 100 / 25 = 4 batches

    def test_streaming_batch_size(self, temp_hdf5):
        """Test that batches have correct sizes."""
        n_positions = 100
        batch_size = 30
        create_test_hdf5(temp_hdf5, n_positions=n_positions, n_games=10)

        batch_sizes = []
        for positions, _, _ in load_positions_hdf5_streaming(
            temp_hdf5, batch_size=batch_size
        ):
            batch_sizes.append(len(positions))

        # Should be [30, 30, 30, 10]
        assert batch_sizes[:-1] == [batch_size] * 3
        assert batch_sizes[-1] == 10

    def test_streaming_max_positions(self, temp_hdf5):
        """Test max_positions limits total streamed."""
        n_positions = 100
        create_test_hdf5(temp_hdf5, n_positions=n_positions, n_games=10)

        total = 0
        for positions, _, _ in load_positions_hdf5_streaming(
            temp_hdf5, batch_size=25, max_positions=50
        ):
            total += len(positions)

        assert total == 50

    def test_streaming_matches_full_load(self, temp_hdf5):
        """Streaming should produce same data as full load."""
        n_positions = 100
        create_test_hdf5(temp_hdf5, n_positions=n_positions, n_games=10)

        # Full load
        full_positions, full_meta = load_positions_hdf5(temp_hdf5)

        # Streaming
        streamed_positions = []
        streamed_game_ids = []
        streamed_move_numbers = []
        for pos, gids, mnums in load_positions_hdf5_streaming(temp_hdf5, batch_size=30):
            streamed_positions.append(pos)
            streamed_game_ids.append(gids)
            streamed_move_numbers.append(mnums)

        streamed_positions = np.concatenate(streamed_positions)
        streamed_game_ids = np.concatenate(streamed_game_ids)
        streamed_move_numbers = np.concatenate(streamed_move_numbers)

        assert np.allclose(full_positions, streamed_positions)
        assert np.array_equal(full_meta['game_ids'], streamed_game_ids)
        assert np.array_equal(full_meta['move_numbers'], streamed_move_numbers)

    def test_streaming_dtypes(self, temp_hdf5):
        """Streamed batches should have correct dtypes."""
        create_test_hdf5(temp_hdf5, n_positions=50, n_games=5)

        for positions, game_ids, move_numbers in load_positions_hdf5_streaming(
            temp_hdf5, batch_size=20
        ):
            assert positions.dtype == np.float32
            assert game_ids.dtype == np.int32
            assert move_numbers.dtype == np.int16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
