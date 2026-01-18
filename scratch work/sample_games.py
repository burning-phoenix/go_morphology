#!/usr/bin/env python3
"""
Sample games from Leela Zero compressed SGF archive.

Sampling Strategy:
1. Reservoir sampling for uniform random selection without loading entire file
2. Skip first 30 moves (opening is memorized, less interesting for interpretability)
3. Sample middle-game positions where tactical/strategic patterns are richer
4. Save as compressed .pt file for fast loading in notebook 01

Usage:
    python sample_games.py [--input INPUT] [--output OUTPUT] [--n_games N]
    
Example:
    python sample_games.py --input all_5M.sgf.xz --output ../outputs/data/sampled_games.pt --n_games 10000
"""

import lzma
import gzip
import random
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Iterator
from dataclasses import dataclass
import pickle

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@dataclass
class SampledGame:
    """Lightweight game representation."""
    moves: List[str]  # List of move strings like 'B[dd]', 'W[pq]'
    result: str       # 'B+Resign', 'W+2.5', etc.
    date: str         # Game date
    

def parse_sgf_moves(sgf_content: str) -> Optional[SampledGame]:
    """
    Parse SGF content to extract moves and metadata.
    
    Args:
        sgf_content: Single game SGF string
        
    Returns:
        SampledGame or None if parsing fails
    """
    try:
        # Extract result
        result = ""
        if "RE[" in sgf_content:
            start = sgf_content.index("RE[") + 3
            end = sgf_content.index("]", start)
            result = sgf_content[start:end]
        
        # Extract date
        date = ""
        if "DT[" in sgf_content:
            start = sgf_content.index("DT[") + 3
            end = sgf_content.index("]", start)
            date = sgf_content[start:end]
        
        # Extract moves - find all ;B[..] and ;W[..] patterns
        moves = []
        i = 0
        while i < len(sgf_content):
            if i < len(sgf_content) - 4:
                if sgf_content[i:i+2] == ";B" or sgf_content[i:i+2] == ";W":
                    # Find the move
                    bracket_start = sgf_content.find("[", i)
                    bracket_end = sgf_content.find("]", bracket_start)
                    if bracket_start != -1 and bracket_end != -1:
                        color = sgf_content[i+1]
                        move = sgf_content[bracket_start+1:bracket_end]
                        if len(move) == 2 or move == "tt":  # Valid move or pass
                            moves.append(f"{color}[{move}]")
                        i = bracket_end
            i += 1
        
        if len(moves) < 50:  # Skip very short games
            return None
            
        return SampledGame(moves=moves, result=result, date=date)
        
    except Exception:
        return None


def reservoir_sample(iterator: Iterator[str], k: int, seed: int = 42) -> List[str]:
    """
    Reservoir sampling: Select k items uniformly at random from a stream.
    
    This allows sampling without loading entire file into memory.
    
    Args:
        iterator: Stream of items
        k: Number of items to select
        seed: Random seed for reproducibility
        
    Returns:
        List of k randomly selected items
    """
    random.seed(seed)
    reservoir = []
    
    for i, item in enumerate(iterator):
        if i < k:
            reservoir.append(item)
        else:
            # Replace with decreasing probability
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
                
    return reservoir


def stream_games_from_xz(filepath: Path) -> Iterator[str]:
    """
    Stream individual games from compressed SGF file.
    
    Yields one complete game SGF at a time.
    """
    with lzma.open(filepath, 'rt', encoding='utf-8') as f:
        current_game = []
        paren_depth = 0
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            for char in line:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                    
            current_game.append(line)
            
            # Game is complete when parentheses are balanced
            if paren_depth == 0 and current_game:
                game_str = ''.join(current_game)
                if game_str.startswith("(;GM[1]"):
                    yield game_str
                current_game = []


def sample_positions_from_game(
    game: SampledGame,
    min_move: int = 30,
    max_move: int = 200,
    positions_per_game: int = 5
) -> List[List[str]]:
    """
    Sample position snapshots from a game.
    
    Strategy:
    - Skip opening (first 30 moves) - too standardized
    - Sample from middle/late game where patterns are interesting
    - Return move sequences up to sampled positions
    
    Args:
        game: Parsed game
        min_move: Minimum move number to sample from
        max_move: Maximum move number
        positions_per_game: Number of positions to sample per game
        
    Returns:
        List of move sequences (each sequence is moves up to that point)
    """
    n_moves = len(game.moves)
    
    # Determine valid sampling range
    start = min(min_move, n_moves - 1)
    end = min(max_move, n_moves)
    
    if end <= start:
        return []
    
    # Sample move indices
    n_samples = min(positions_per_game, end - start)
    indices = sorted(random.sample(range(start, end), n_samples))
    
    # Return move prefixes up to each sampled position
    return [game.moves[:idx+1] for idx in indices]


def main():
    parser = argparse.ArgumentParser(description="Sample games from LZ SGF archive")
    parser.add_argument("--input", "-i", type=str, default="all_5M.sgf.xz",
                        help="Input .xz compressed SGF file")
    parser.add_argument("--output", "-o", type=str, default="../outputs/data/sampled_games.pkl.gz",
                        help="Output compressed pickle file")
    parser.add_argument("--n_games", "-n", type=int, default=10000,
                        help="Number of games to sample")
    parser.add_argument("--positions_per_game", "-p", type=int, default=5,
                        help="Positions to sample per game")
    parser.add_argument("--min_move", type=int, default=30,
                        help="Minimum move number to sample from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Sampling {args.n_games} games from {input_path}")
    print(f"  Positions per game: {args.positions_per_game}")
    print(f"  Min move: {args.min_move}")
    print(f"  Seed: {args.seed}")
    
    # Stream and sample games
    print("\nPhase 1: Reservoir sampling games...")
    game_strings = reservoir_sample(
        stream_games_from_xz(input_path),
        args.n_games,
        seed=args.seed
    )
    print(f"  Sampled {len(game_strings)} games")
    
    # Parse games
    print("\nPhase 2: Parsing games...")
    games = []
    for gs in game_strings:
        game = parse_sgf_moves(gs)
        if game is not None:
            games.append(game)
    print(f"  Parsed {len(games)} valid games")
    
    # Sample positions
    print("\nPhase 3: Sampling positions...")
    random.seed(args.seed)
    all_positions = []
    for game in games:
        positions = sample_positions_from_game(
            game,
            min_move=args.min_move,
            positions_per_game=args.positions_per_game
        )
        all_positions.extend(positions)
    print(f"  Total positions: {len(all_positions)}")
    
    # Save
    print(f"\nSaving to {output_path}...")
    output_data = {
        'positions': all_positions,  # List of move sequences
        'n_games': len(games),
        'n_positions': len(all_positions),
        'config': {
            'min_move': args.min_move,
            'positions_per_game': args.positions_per_game,
            'seed': args.seed,
            'source': str(input_path.name),
        }
    }
    
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    file_size = output_path.stat().st_size / 1e6
    print(f"  File size: {file_size:.1f} MB")
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Games sampled: {len(games)}")
    print(f"  Positions: {len(all_positions)}")
    print(f"  Estimated activations: {len(all_positions) * 361:,}")
    print(f"  Output: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
