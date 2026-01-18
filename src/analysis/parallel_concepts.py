"""
Parallel concept labeling for Go positions.

Uses multiprocessing to speed up concept label computation.
Designed for macOS compatibility with proper 'spawn' context.

Key Design Decisions:
1. Uses 'spawn' start method (not 'fork') for macOS safety
2. Each worker creates its own ConceptLabeler (no shared state)
3. Processes positions in batches to control memory
4. Uses imap_unordered for efficient memory-bounded processing

Usage:
    from src.analysis.parallel_concepts import compute_labels_parallel
    labels = compute_labels_parallel(positions_tensor, n_workers=4)
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import get_context
from typing import Dict, List, Tuple, Optional
from functools import partial
import sys


# Define concepts to compute (must match notebook 04)
CONCEPTS = [
    'is_edge', 'is_corner_region', 'is_center_region', 'is_star_point',
    'stone_black', 'stone_white', 'stone_empty',
    'is_atari', 'is_low_liberties',
    'is_eye_black', 'is_eye_white',
    'is_cutting_point',
    'has_adjacent_enemy_black', 'has_adjacent_enemy_white',
    'territory_black', 'territory_white',
]


def encoded_to_board(encoded_position: np.ndarray) -> np.ndarray:
    """
    Convert 18-plane encoded position back to board array.
    
    Leela Zero encoding (from leela-zero-pytorch/dataset.py):
    - Planes 0-7:   Current player's stones (T=0 to T=-7)
    - Planes 8-15:  Opponent's stones (T=0 to T=-7)
    - Plane 16:     All 1s if BLACK to move
    - Plane 17:     All 1s if WHITE to move
    
    Args:
        encoded_position: (18, 19, 19) array
        
    Returns:
        (19, 19) array with 0=empty, 1=black, 2=white
    """
    current_stones = encoded_position[0]  # Current player's stones (T=0)
    opponent_stones = encoded_position[8]  # Opponent's stones (T=0) <-- FIXED
    
    # Check who is to move
    black_to_play = encoded_position[16, 0, 0] > 0.5
    
    board = np.zeros((19, 19), dtype=np.int32)
    
    if black_to_play:
        # Current player = Black, Opponent = White
        board[current_stones > 0.5] = 1   # Black
        board[opponent_stones > 0.5] = 2  # White
    else:
        # Current player = White, Opponent = Black  
        board[current_stones > 0.5] = 2   # White
        board[opponent_stones > 0.5] = 1  # Black
    
    return board


def process_single_position(
    encoded_position: np.ndarray,
    board_size: int = 19
) -> Dict[str, np.ndarray]:
    """
    Process a single position and return flattened concept labels.
    
    This function is designed to be called by worker processes.
    Each call creates its own ConceptLabeler to avoid shared state.
    
    Args:
        encoded_position: (18, 19, 19) encoded position
        board_size: Size of Go board
        
    Returns:
        Dict mapping concept name -> (361,) flattened boolean array
    """
    # Import inside function to ensure clean worker state
    from .concepts import ConceptLabeler
    
    # Create labeler fresh in each call (no shared state)
    labeler = ConceptLabeler(board_size=board_size)
    
    # Convert encoded position to board
    board = encoded_to_board(encoded_position)
    
    # Compute labels
    labels = {}
    
    # Geometric (precomputed by labeler)
    labels['is_edge'] = labeler.is_edge()
    labels['is_corner_region'] = labeler.is_corner_region()
    labels['is_center_region'] = labeler.is_center_region()
    labels['is_star_point'] = labeler.is_star_point()
    
    # Stone colors
    stone_colors = labeler.stone_color(board)
    labels['stone_black'] = stone_colors['black']
    labels['stone_white'] = stone_colors['white']
    labels['stone_empty'] = stone_colors['empty']
    
    # Tactical
    labels['is_atari'] = labeler.is_atari(board)
    labels['is_low_liberties'] = labeler.is_low_liberties(board, threshold=3)
    
    eye_labels = labeler.is_potential_eye(board)
    labels['is_eye_black'] = eye_labels['black']
    labels['is_eye_white'] = eye_labels['white']
    
    labels['is_cutting_point'] = labeler.is_cutting_point(board)
    
    adjacent_enemy = labeler.has_adjacent_enemy(board)
    labels['has_adjacent_enemy_black'] = adjacent_enemy['black']
    labels['has_adjacent_enemy_white'] = adjacent_enemy['white']
    
    # Strategic
    territory = labeler.estimate_territory(board)
    labels['territory_black'] = territory['black_territory'] > 0.6
    labels['territory_white'] = territory['white_territory'] > 0.6
    
    # Flatten all labels
    return {name: arr.flatten().astype(np.bool_) for name, arr in labels.items()}


def _worker_initializer():
    """
    Worker initializer for multiprocessing pool.
    
    Called once when each worker process starts.
    Useful for setting up process-local state.
    """
    # No shared state needed - each task creates its own labeler
    pass


def _process_batch(
    batch_data: Tuple[int, np.ndarray],
    board_size: int = 19
) -> Tuple[int, Dict[str, np.ndarray]]:
    """
    Process a batch of positions.
    
    Args:
        batch_data: (batch_idx, positions_array) where positions_array is (batch_size, 18, 19, 19)
        board_size: Size of Go board
        
    Returns:
        (batch_idx, labels_dict) where labels_dict maps concept -> (batch_size * 361,) array
    """
    batch_idx, positions = batch_data
    batch_size = len(positions)
    
    # Initialize storage for this batch
    batch_labels = {concept: [] for concept in CONCEPTS}
    
    for pos in positions:
        pos_labels = process_single_position(pos, board_size)
        for concept in CONCEPTS:
            batch_labels[concept].append(pos_labels[concept])
    
    # Concatenate within batch
    return batch_idx, {
        concept: np.concatenate(labels)
        for concept, labels in batch_labels.items()
    }


def compute_labels_parallel(
    positions: np.ndarray,
    n_workers: Optional[int] = None,
    batch_size: int = 100,
    board_size: int = 19,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute concept labels for all positions using parallel processing.
    
    This function is safe to call from Jupyter notebooks on macOS.
    Uses 'spawn' context to avoid fork-related issues.
    
    Args:
        positions: (n_positions, 18, 19, 19) tensor/array of encoded positions
        n_workers: Number of worker processes (default: CPU count - 2)
        batch_size: Number of positions per batch (controls memory usage)
        board_size: Size of Go board
        verbose: Whether to print progress
        
    Returns:
        Dict mapping concept name -> (n_positions * 361,) boolean array
        
    Safety Features:
        - Uses 'spawn' start method (macOS compatible)
        - No shared mutable state between workers
        - Proper pool cleanup via context manager
        - Memory-bounded via batching
    """
    # Convert to numpy if needed
    if hasattr(positions, 'numpy'):
        positions = positions.numpy()
    
    n_positions = len(positions)
    n_points = board_size * board_size  # 361
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 2)
    n_workers = min(n_workers, mp.cpu_count())
    
    if verbose:
        print(f"Parallel concept labeling:")
        print(f"  Positions: {n_positions:,}")
        print(f"  Workers: {n_workers}")
        print(f"  Batch size: {batch_size}")
    
    # Create batches
    n_batches = (n_positions + batch_size - 1) // batch_size
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_positions)
        batches.append((i, positions[start_idx:end_idx]))
    
    if verbose:
        print(f"  Batches: {n_batches}")
    
    # Initialize storage for all labels
    all_labels = {
        concept: np.zeros((n_positions * n_points,), dtype=np.bool_)
        for concept in CONCEPTS
    }
    
    # Use 'spawn' context for macOS safety
    # This prevents issues with fork() and macOS system libraries
    ctx = get_context('spawn')
    
    # Create process pool with context manager (ensures cleanup)
    try:
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_initializer,
        ) as pool:
            
            # Use imap_unordered for memory efficiency
            # Results come back as they complete, not in order
            process_func = partial(_process_batch, board_size=board_size)
            
            completed = 0
            for batch_idx, batch_labels in pool.imap_unordered(process_func, batches):
                # Calculate where this batch's results go
                start_pos = batch_idx * batch_size
                actual_batch_size = len(batches[batch_idx][1])
                start_flat = start_pos * n_points
                end_flat = start_flat + actual_batch_size * n_points
                
                # Store results
                for concept in CONCEPTS:
                    all_labels[concept][start_flat:end_flat] = batch_labels[concept]
                
                completed += 1
                if verbose and completed % 10 == 0:
                    pct = 100 * completed / n_batches
                    print(f"  Progress: {completed}/{n_batches} batches ({pct:.1f}%)")
    
    except KeyboardInterrupt:
        print("\nInterrupted! Cleaning up...")
        raise
    
    if verbose:
        print(f"  Complete! Labeled {n_positions:,} positions")
    
    return all_labels


def compute_labels_sequential(
    positions: np.ndarray,
    board_size: int = 19,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Sequential fallback for environments where multiprocessing doesn't work.
    
    Same interface as compute_labels_parallel.
    """
    from tqdm import tqdm
    
    if hasattr(positions, 'numpy'):
        positions = positions.numpy()
    
    n_positions = len(positions)
    n_points = board_size * board_size
    
    all_labels = {concept: [] for concept in CONCEPTS}
    
    iterator = tqdm(range(n_positions)) if verbose else range(n_positions)
    
    for i in iterator:
        pos_labels = process_single_position(positions[i], board_size)
        for concept in CONCEPTS:
            all_labels[concept].append(pos_labels[concept])
    
    # Stack all labels
    return {
        concept: np.concatenate(labels)
        for concept, labels in all_labels.items()
    }


# Convenience function to auto-select best method
def compute_labels(
    positions: np.ndarray,
    n_workers: Optional[int] = None,
    board_size: int = 19,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute concept labels, automatically selecting parallel or sequential.
    
    Uses parallel processing if available and n_workers > 1.
    Falls back to sequential if multiprocessing fails.
    
    Args:
        positions: (n_positions, 18, 19, 19) encoded positions
        n_workers: Number of workers (None = auto, 1 = sequential)
        board_size: Size of Go board
        verbose: Print progress
        
    Returns:
        Dict mapping concept name -> flattened boolean array
    """
    # Determine if we should use parallel
    if n_workers == 1:
        return compute_labels_sequential(positions, board_size, verbose)
    
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 2)
    
    if n_workers <= 1:
        return compute_labels_sequential(positions, board_size, verbose)
    
    # Try parallel, fall back to sequential on error
    try:
        return compute_labels_parallel(
            positions,
            n_workers=n_workers,
            board_size=board_size,
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"Parallel processing failed ({e}), falling back to sequential...")
        return compute_labels_sequential(positions, board_size, verbose)
