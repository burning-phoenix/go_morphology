"""
Metadata export utilities for NB01.

Saves position-level and game-level metadata for trajectory analysis in downstream notebooks.
"""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch


def save_extraction_outputs(
    positions_tensor: torch.Tensor,
    metadata: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, str]:
    """
    Save all NB01 outputs: positions tensor + metadata files.
    
    Creates:
        outputs/data/positions.pt          - Encoded board positions
        outputs/data/metadata/positions.parquet - Position-level metadata
        outputs/data/games/game_index.json     - Game-level metadata
    
    Args:
        positions_tensor: Shape (n_positions, 18, 19, 19)
        metadata: Dict from load_sampled_games() with position_metadata, game_index
        output_dir: Base output directory
        
    Returns:
        Dict of saved file paths
    """
    output_dir = Path(output_dir)
    data_dir = output_dir / 'data'
    saved_files = {}
    
    # 1. Save positions tensor
    positions_path = data_dir / 'positions.pt'
    torch.save(positions_tensor, positions_path)
    saved_files['positions'] = str(positions_path)
    print(f"Saved positions to {positions_path}")
    print(f"  Shape: {positions_tensor.shape}")
    print(f"  Size: {positions_path.stat().st_size / 1e6:.1f} MB")
    
    # 2. Save position-level metadata (for trajectory reconstruction)
    if 'position_metadata' in metadata and metadata['position_metadata']:
        meta_dir = data_dir / 'metadata'
        meta_dir.mkdir(exist_ok=True)
        
        pos_meta_df = pd.DataFrame(metadata['position_metadata'])
        parquet_path = meta_dir / 'positions.parquet'
        pos_meta_df.to_parquet(parquet_path, index=False)
        saved_files['position_metadata'] = str(parquet_path)
        
        print(f"\nSaved position metadata to {parquet_path}")
        print(f"  Columns: {list(pos_meta_df.columns)}")
        print(f"  Shape: {pos_meta_df.shape}")
        
        # Show sample
        if len(pos_meta_df) > 0:
            print(f"\n  Sample (first 3 rows):")
            print(pos_meta_df.head(3).to_string(index=False))
    
    # 3. Save game-level index (for trajectory boundaries)
    if 'game_index' in metadata and metadata['game_index']:
        games_dir = data_dir / 'games'
        games_dir.mkdir(exist_ok=True)
        
        game_index_path = games_dir / 'game_index.json'
        with open(game_index_path, 'w') as f:
            # Convert keys to strings for JSON
            json_compatible = {str(k): v for k, v in metadata['game_index'].items()}
            json.dump(json_compatible, f, indent=2)
        saved_files['game_index'] = str(game_index_path)
        
        print(f"\nSaved game index to {game_index_path}")
        print(f"  Games: {len(metadata['game_index'])}")
        
        # Show sample
        sample_id = list(metadata['game_index'].keys())[0]
        print(f"  Sample game {sample_id}: {metadata['game_index'][sample_id]}")
    
    # 4. Validate trajectory continuity
    print("\n" + "="*50)
    if metadata.get('is_consecutive', False):
        print("✓ CONSECUTIVE sampling - trajectories preserved")
        print("  NB05/06/08 will work correctly")
    else:
        print("⚠ NON-CONSECUTIVE sampling detected")
        print("  Markov analysis may not work correctly")
        print("  Consider re-running sample_games.py with consecutive=True")
    print("="*50)
    
    return saved_files


def load_position_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Load position metadata from parquet file."""
    parquet_path = Path(data_dir) / 'metadata' / 'positions.parquet'
    if not parquet_path.exists():
        raise FileNotFoundError(f"Position metadata not found at {parquet_path}")
    return pd.read_parquet(parquet_path)


def load_game_index(data_dir: str | Path) -> Dict[int, Dict]:
    """Load game index from JSON file."""
    json_path = Path(data_dir) / 'games' / 'game_index.json'
    if not json_path.exists():
        raise FileNotFoundError(f"Game index not found at {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Convert string keys back to int
    return {int(k): v for k, v in data.items()}


def get_game_trajectory(
    game_id: int,
    activations: np.ndarray,
    position_metadata: pd.DataFrame,
) -> np.ndarray:
    """
    Extract activation trajectory for a single game.
    
    Args:
        game_id: Game ID to extract
        activations: Full activation array (n_positions, feature_dim)
        position_metadata: DataFrame with game_id, move_number columns
        
    Returns:
        Array of shape (n_moves_in_game, feature_dim)
    """
    game_positions = position_metadata[position_metadata['game_id'] == game_id]
    game_positions = game_positions.sort_values('move_number')
    indices = game_positions['position_idx'].values
    return activations[indices]
