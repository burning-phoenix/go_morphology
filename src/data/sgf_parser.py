"""
SGF (Smart Game Format) parser for Go games.

Uses sgfmill library for all SGF parsing and Go logic.

Reference: docs/go_concepts.md
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Generator
from dataclasses import dataclass

from sgfmill import sgf, boards

from .position_encoder import BoardState, PositionEncoder


@dataclass
class GameInfo:
    """Metadata about a parsed game."""
    filename: str
    black_player: str
    white_player: str
    result: str
    board_size: int
    komi: float
    handicap: int
    num_moves: int


@dataclass
class Position:
    """A single board position with metadata."""
    board_state: BoardState
    move_number: int
    next_move: Optional[Tuple[int, int]]  # None for pass
    next_color: str  # 'b' or 'w'
    game_result: str


class SGFParser:
    """
    Parser for SGF files using sgfmill library.

    Provides two parsing modes:
    - Standard: Returns Position objects with BoardState (for analysis)
    - Direct encoding: Returns encoded tensors directly (for training, faster)
    """

    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self._encoder = None  # Lazy init

    @property
    def encoder(self) -> PositionEncoder:
        """Lazy-initialized encoder to avoid import overhead."""
        if self._encoder is None:
            self._encoder = PositionEncoder(self.board_size)
        return self._encoder

    def parse_file(self, filepath: str) -> Tuple[List[Position], GameInfo]:
        """Parse an SGF file and generate all positions."""
        with open(filepath, 'rb') as f:
            game = sgf.Sgf_game.from_bytes(f.read())
        return self._parse_game(game, str(filepath))

    def parse_string(self, sgf_string: str, filename: str = "unknown") -> Tuple[List[Position], GameInfo]:
        """Parse an SGF string and generate all positions."""
        game = sgf.Sgf_game.from_string(sgf_string)
        return self._parse_game(game, filename)

    def parse_file_to_tensors(
        self,
        filepath: str,
        min_move: int = 0,
        max_move: Optional[int] = None,
    ) -> Tuple[np.ndarray, GameInfo]:
        """
        Parse SGF file and return encoded tensors directly.

        This is faster than parse_file() when you don't need BoardState objects,
        as it avoids creating Position objects and deep copies.

        Args:
            filepath: Path to SGF file
            min_move: Minimum move number to include
            max_move: Maximum move number to include (None = all)

        Returns:
            Tuple of (encoded_positions, game_info)
            - encoded_positions: np.ndarray of shape (n_moves, 18, 19, 19)
            - game_info: GameInfo with metadata
        """
        with open(filepath, 'rb') as f:
            game = sgf.Sgf_game.from_bytes(f.read())
        return self._parse_game_to_tensors(game, str(filepath), min_move, max_move)

    def _parse_game_to_tensors(
        self,
        game: sgf.Sgf_game,
        filename: str,
        min_move: int = 0,
        max_move: Optional[int] = None,
    ) -> Tuple[np.ndarray, GameInfo]:
        """
        Parse game directly to encoded tensors without intermediate Position objects.

        Optimized path that:
        - Doesn't create Position objects
        - Doesn't deep-copy BoardState for each move
        - Encodes in-place to pre-allocated output
        """
        root = game.get_root()
        board_size = game.get_size()

        if board_size != self.board_size:
            raise ValueError(f"Board size {board_size} != expected {self.board_size}")

        # Get metadata
        def get_prop(prop, default=''):
            try:
                val = root.get(prop)
                return val if val else default
            except KeyError:
                return default

        black_player = get_prop('PB', 'Unknown')
        white_player = get_prop('PW', 'Unknown')
        result = get_prop('RE', 'Unknown')
        komi = float(get_prop('KM', '7.5') or '7.5')
        handicap = int(get_prop('HA', '0') or '0')

        # Count moves first to pre-allocate
        node = root
        move_count = 0
        while True:
            color, _ = node.get_move()
            if color is not None:
                move_count += 1
            children = node.get_children()
            if not children:
                break
            node = children[0]

        # Determine output size
        effective_max = max_move if max_move is not None else move_count
        n_positions = min(effective_max, move_count) - min_move
        n_positions = max(0, n_positions)

        if n_positions == 0:
            game_info = GameInfo(
                filename=filename, black_player=black_player,
                white_player=white_player, result=result,
                board_size=board_size, komi=komi,
                handicap=handicap, num_moves=0
            )
            return np.zeros((0, 18, self.board_size, self.board_size), dtype=np.float32), game_info

        # Pre-allocate output
        output = np.zeros((n_positions, 18, self.board_size, self.board_size), dtype=np.float32)

        # Replay game and encode directly
        board_state = BoardState(size=board_size)

        # Handle setup stones
        for color_prop, color in [('AB', 'b'), ('AW', 'w')]:
            setup_points = root.get(color_prop)
            if setup_points:
                for row, col in setup_points:
                    board_state._board.play(row, col, color)
                    board_state._cache_valid = False
                if color == 'b' and color_prop == 'AB':
                    board_state.current_player = 'w'

        # Traverse and encode
        node = root
        current_move = 0
        output_idx = 0

        while True:
            color, point = node.get_move()

            if color is not None:
                # Only encode positions in the desired range
                if min_move <= current_move < effective_max:
                    output[output_idx] = self.encoder.encode_to_array(board_state)
                    output_idx += 1

                # Play move
                if point is None:
                    board_state.pass_move()
                else:
                    row, col = point
                    try:
                        board_state.play(row, col, color)
                    except Exception:
                        break

                current_move += 1
                if current_move >= effective_max:
                    break

            children = node.get_children()
            if not children:
                break
            node = children[0]

        game_info = GameInfo(
            filename=filename,
            black_player=black_player,
            white_player=white_player,
            result=result,
            board_size=board_size,
            komi=komi,
            handicap=handicap,
            num_moves=output_idx
        )

        return output[:output_idx], game_info

    def _parse_game(self, game: sgf.Sgf_game, filename: str) -> Tuple[List[Position], GameInfo]:
        """Parse game using sgfmill."""
        root = game.get_root()
        board_size = game.get_size()

        if board_size != self.board_size:
            raise ValueError(f"Board size {board_size} != expected {self.board_size}")

        # Get metadata using sgfmill
        def get_prop(prop, default=''):
            try:
                val = root.get(prop)
                return val if val else default
            except KeyError:
                return default

        black_player = get_prop('PB', 'Unknown')
        white_player = get_prop('PW', 'Unknown')
        result = get_prop('RE', 'Unknown')
        komi = float(get_prop('KM', '7.5') or '7.5')
        handicap = int(get_prop('HA', '0') or '0')

        # Use sgfmill's board for replay
        sgf_board = boards.Board(board_size)
        board_state = BoardState(size=board_size)

        # Handle setup stones (handicap, etc.)
        for color_prop, color in [('AB', 'b'), ('AW', 'w')]:
            setup_points = root.get(color_prop)
            if setup_points:
                for row, col in setup_points:
                    sgf_board.play(row, col, color)
                    board_state._board.play(row, col, color)
                if color == 'b' and color_prop == 'AB':
                    board_state.current_player = 'w'  # White moves after handicap

        # Traverse main variation
        positions = []
        node = root

        while True:
            color, point = node.get_move()

            if color is not None:
                # Save position before move
                positions.append(Position(
                    board_state=board_state.copy(),
                    move_number=board_state.move_count,
                    next_move=point,  # (row, col) or None for pass
                    next_color=color,
                    game_result=result
                ))

                # Play move using sgfmill
                if point is None:
                    board_state.pass_move()
                else:
                    row, col = point
                    try:
                        board_state.play(row, col, color)
                    except Exception as e:
                        print(f"Warning: Invalid move in {filename}: {e}")
                        break

            # Next node in main variation
            children = node.get_children()
            if not children:
                break
            node = children[0]

        game_info = GameInfo(
            filename=filename,
            black_player=black_player,
            white_player=white_player,
            result=result,
            board_size=board_size,
            komi=komi,
            handicap=handicap,
            num_moves=len(positions)
        )

        return positions, game_info


def parse_sgf_file(filepath: str, board_size: int = 19) -> Tuple[List[Position], GameInfo]:
    """Convenience function to parse a single SGF file."""
    return SGFParser(board_size).parse_file(filepath)


def parse_sgf_directory(
    directory: str,
    board_size: int = 19,
    max_games: Optional[int] = None,
    positions_per_game: Optional[int] = None
) -> Generator[Tuple[Position, GameInfo], None, None]:
    """Parse all SGF files in a directory."""
    directory = Path(directory)
    sgf_files = list(directory.glob('**/*.sgf'))

    if max_games:
        sgf_files = sgf_files[:max_games]

    parser = SGFParser(board_size)

    for sgf_file in sgf_files:
        try:
            positions, game_info = parser.parse_file(str(sgf_file))

            if positions_per_game and len(positions) > positions_per_game:
                indices = np.linspace(0, len(positions) - 1, positions_per_game, dtype=int)
                positions = [positions[i] for i in indices]

            for pos in positions:
                yield pos, game_info

        except Exception as e:
            print(f"Error parsing {sgf_file}: {e}")
            continue


def create_position_dataset(
    sgf_paths: List[str],
    output_path: str,
    board_size: int = 19,
    max_positions: Optional[int] = None,
    positions_per_game: Optional[int] = None,
    save_encodings: bool = True
) -> Dict[str, Any]:
    """
    Create a dataset of positions from SGF files.

    Returns dict with dataset statistics.
    """
    parser = SGFParser(board_size)
    encoder = PositionEncoder(board_size)

    all_boards = []
    all_encodings = []
    game_count = 0
    skipped_count = 0

    for path_str in sgf_paths:
        path = Path(path_str)
        files = [path] if path.is_file() else list(path.glob('**/*.sgf'))

        for sgf_file in files:
            try:
                positions, _ = parser.parse_file(str(sgf_file))

                if positions_per_game and len(positions) > positions_per_game:
                    indices = np.linspace(0, len(positions) - 1, positions_per_game, dtype=int)
                    positions = [positions[i] for i in indices]

                for pos in positions:
                    all_boards.append(pos.board_state.board)

                    if save_encodings:
                        all_encodings.append(encoder.encode(pos.board_state).numpy())

                    if max_positions and len(all_boards) >= max_positions:
                        break

                game_count += 1
                if max_positions and len(all_boards) >= max_positions:
                    break

            except Exception as e:
                print(f"Error parsing {sgf_file}: {e}")
                skipped_count += 1

        if max_positions and len(all_boards) >= max_positions:
            break

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path.with_suffix('.npy'), np.stack(all_boards))

    if save_encodings and all_encodings:
        np.save(str(output_path) + '_encoded.npy', np.stack(all_encodings))

    return {
        'num_positions': len(all_boards),
        'num_games': game_count,
        'num_skipped': skipped_count,
        'board_size': board_size,
        'output_path': str(output_path)
    }


def load_positions_hdf5(
    hdf5_path: str,
    max_positions: Optional[int] = None,
    validate: bool = True,
    mmap_mode: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load pre-encoded positions from HDF5 file.

    This loads positions that were pre-encoded by sample_games.py,
    eliminating the need to replay move sequences.

    HDF5 Schema:
        Datasets (required):
            - positions: float32 array of shape (n_positions, 18, 19, 19)
                18-plane Leela Zero encoding (8 history planes per color + 2 color planes)
            - game_ids: int32 array of shape (n_positions,)
                Maps each position to its source game
            - move_numbers: int16 array of shape (n_positions,)
                Move number within the game for each position

        Attributes (optional but recommended):
            - n_games: int, number of unique games
            - n_positions: int, total positions (should match dataset length)
            - min_move: int, minimum move number sampled
            - max_move: int, maximum move number sampled
            - seed: int, random seed used for sampling
            - source: str, source file name

    Args:
        hdf5_path: Path to .hdf5 file with encoded positions
        max_positions: Limit number of positions loaded (None = all)
        validate: If True, validate dataset shapes and dtypes
        mmap_mode: Memory-map mode for very large files. Options:
            - None: Load into memory (default, fastest for files that fit in RAM)
            - 'r': Read-only memory-mapped (for files larger than RAM)
            - 'r+': Read-write memory-mapped
            Note: mmap_mode requires converting HDF5 to numpy format first

    Returns:
        Tuple of (positions, metadata)
        - positions: np.ndarray of shape (n_positions, 18, 19, 19), dtype float32
        - metadata: dict with:
            - n_positions: int, number of positions loaded
            - n_games: int, number of unique games
            - config: dict of HDF5 attributes
            - source_path: str, path to source file
            - game_ids: np.ndarray of shape (n_positions,), dtype int32
            - move_numbers: np.ndarray of shape (n_positions,), dtype int16

    Raises:
        ImportError: If h5py is not installed
        FileNotFoundError: If hdf5_path doesn't exist
        ValueError: If required datasets are missing or have invalid shape/dtype
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 loading: pip install h5py")

    hdf5_path = str(hdf5_path)  # Handle Path objects

    with h5py.File(hdf5_path, 'r') as f:
        # Validate required datasets exist
        required_datasets = {'positions', 'game_ids', 'move_numbers'}
        available_datasets = set(f.keys())
        missing = required_datasets - available_datasets
        if missing:
            raise ValueError(
                f"HDF5 file missing required datasets: {missing}. "
                f"Available: {available_datasets}"
            )

        # Validate shapes and dtypes if requested
        if validate:
            pos_shape = f['positions'].shape
            if len(pos_shape) != 4:
                raise ValueError(
                    f"positions must be 4D (n, 18, 19, 19), got shape {pos_shape}"
                )
            if pos_shape[1:] != (18, 19, 19):
                raise ValueError(
                    f"positions must have shape (n, 18, 19, 19), got {pos_shape}"
                )

            n_positions_file = pos_shape[0]
            if f['game_ids'].shape[0] != n_positions_file:
                raise ValueError(
                    f"game_ids length ({f['game_ids'].shape[0]}) doesn't match "
                    f"positions count ({n_positions_file})"
                )
            if f['move_numbers'].shape[0] != n_positions_file:
                raise ValueError(
                    f"move_numbers length ({f['move_numbers'].shape[0]}) doesn't match "
                    f"positions count ({n_positions_file})"
                )

        # Determine how many positions to load
        n_available = f['positions'].shape[0]
        n_to_load = n_available if max_positions is None else min(max_positions, n_available)

        # Load positions (with optional limit)
        if n_to_load < n_available:
            positions = f['positions'][:n_to_load]
            game_ids = f['game_ids'][:n_to_load]
            move_numbers = f['move_numbers'][:n_to_load]
        else:
            positions = f['positions'][:]
            game_ids = f['game_ids'][:]
            move_numbers = f['move_numbers'][:]

        # Ensure consistent dtypes (copy=False avoids copy if already correct dtype)
        positions = positions.astype(np.float32, copy=False)
        game_ids = game_ids.astype(np.int32, copy=False)
        move_numbers = move_numbers.astype(np.int16, copy=False)

        # Extract config from attributes
        config = {key: f.attrs[key] for key in f.attrs.keys()}

        # Lazy evaluation: only compute unique game_ids if n_games not in config
        n_games = config.get('n_games')
        if n_games is None:
            # O(n log n) operation - only when necessary
            n_games = len(np.unique(game_ids))

        metadata = {
            'n_positions': len(positions),
            'n_games': n_games,
            'config': config,
            'source_path': hdf5_path,
            'game_ids': game_ids,
            'move_numbers': move_numbers,
        }

    return positions, metadata


def load_positions_hdf5_streaming(
    hdf5_path: str,
    batch_size: int = 1000,
    max_positions: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Stream positions from HDF5 file in batches without loading all into memory.

    This is useful for very large files (>10GB) that don't fit in RAM.

    Args:
        hdf5_path: Path to .hdf5 file
        batch_size: Number of positions per batch
        max_positions: Maximum total positions to load (None = all)

    Yields:
        Tuple of (positions_batch, game_ids_batch, move_numbers_batch)
        - positions_batch: shape (batch_size, 18, 19, 19)
        - game_ids_batch: shape (batch_size,)
        - move_numbers_batch: shape (batch_size,)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 loading: pip install h5py")

    with h5py.File(hdf5_path, 'r') as f:
        n_positions = f['positions'].shape[0]
        if max_positions is not None:
            n_positions = min(n_positions, max_positions)

        for start in range(0, n_positions, batch_size):
            end = min(start + batch_size, n_positions)

            positions = f['positions'][start:end].astype(np.float32, copy=False)
            game_ids = f['game_ids'][start:end].astype(np.int32, copy=False)
            move_numbers = f['move_numbers'][start:end].astype(np.int16, copy=False)

            yield positions, game_ids, move_numbers

