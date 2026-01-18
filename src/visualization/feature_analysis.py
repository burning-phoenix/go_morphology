"""
Feature liveness analysis and visualization for MSAE.

This module provides tools to:
1. Compute feature liveness across the full dataset
2. Identify dead, rare, and healthy features
3. Generate visualizations only for interpretable features

Usage:
    from src.visualization.feature_analysis import (
        compute_feature_liveness,
        generate_feature_examples_with_liveness
    )
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class LivenessReport:
    """Summary of feature liveness analysis."""
    total_samples: int
    total_features: int
    activation_counts: np.ndarray  # Shape: (n_features,)
    
    @property
    def dead_features(self) -> np.ndarray:
        """Features with 0 activations."""
        return np.where(self.activation_counts == 0)[0]
    
    @property
    def dead_rate(self) -> float:
        """Percentage of dead features."""
        return len(self.dead_features) / self.total_features * 100
    
    def get_live_features(self, min_rate: float = 0.0001) -> np.ndarray:
        """Features with activation rate >= min_rate."""
        min_count = self.total_samples * min_rate
        return np.where(self.activation_counts >= min_count)[0]
    
    def get_interpretable_features(
        self, 
        min_rate: float = 0.001, 
        max_rate: float = 0.05
    ) -> np.ndarray:
        """
        'Goldilocks' features: not too rare, not too common.
        
        Args:
            min_rate: Minimum activation rate (default 0.1%)
            max_rate: Maximum activation rate (default 5%)
        """
        min_count = self.total_samples * min_rate
        max_count = self.total_samples * max_rate
        return np.where(
            (self.activation_counts > min_count) & 
            (self.activation_counts < max_count)
        )[0]
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 50,
            "FEATURE LIVENESS REPORT",
            "=" * 50,
            f"Total Samples: {self.total_samples:,}",
            f"Total Features: {self.total_features}",
            f"Dead Features: {len(self.dead_features)} ({self.dead_rate:.2f}%)",
            f"Live Features (>0.01%): {len(self.get_live_features(0.0001))}",
            f"Interpretable (0.1%-5%): {len(self.get_interpretable_features())}",
            f"Max activation count: {self.activation_counts.max():,}",
            "=" * 50,
        ]
        return "\n".join(lines)


def compute_feature_liveness(
    model_path: str,
    activations_dir: str,
    k: int = 64,
    device: Optional[str] = None,
    max_chunks: Optional[int] = None,
    batch_size: int = 4096,
    save_path: Optional[str] = None,
) -> LivenessReport:
    """
    Compute feature activation counts across the full dataset.
    
    Args:
        model_path: Path to MSAE checkpoint
        activations_dir: Directory containing chunk_*.npy files
        k: Sparsity level for TopK
        device: Device to use (auto-detected if None)
        max_chunks: Limit chunks for faster testing
        batch_size: Batch size for inference
        save_path: Optional path to save counts
        
    Returns:
        LivenessReport with activation statistics
    """
    # Import MSAE
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.msae import MatryoshkaSAE
    
    # Setup device
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"Computing feature liveness on {device}...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    msae = MatryoshkaSAE(input_dim=256, hidden_dim=4096, k_levels=[16, 32, 64, 128])
    msae.load_state_dict(checkpoint['model_state_dict'])
    msae = msae.to(device).eval()
    
    # Load normalization
    norm_mean = checkpoint['normalization']['mean']
    norm_std = checkpoint['normalization']['std']
    if isinstance(norm_mean, np.ndarray):
        norm_mean = torch.from_numpy(norm_mean)
        norm_std = torch.from_numpy(norm_std)
    norm_mean = norm_mean.to(device).float()
    norm_std = norm_std.to(device).float()
    
    # Initialize counters
    n_features = 4096
    feature_counts = torch.zeros(n_features, dtype=torch.long, device=device)
    total_samples = 0
    
    # Process chunks
    act_dir = Path(activations_dir)
    chunks = sorted(act_dir.glob('chunk_*.npy'))
    if max_chunks:
        chunks = chunks[:max_chunks]
    
    print(f"Processing {len(chunks)} chunks...")
    
    with torch.no_grad():
        for chunk_path in tqdm(chunks, desc="Liveness scan"):
            raw_acts = np.load(chunk_path)
            raw_tensor = torch.from_numpy(raw_acts).to(device).float()
            
            # Normalize
            x = (raw_tensor - norm_mean) / (norm_std + 1e-8)
            
            # Batch processing
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                _, _, z_sparse = msae(batch, k=k)
                
                # Count non-zeros
                active_mask = (z_sparse > 0).float()
                feature_counts += active_mask.sum(dim=0).long()
                total_samples += batch.shape[0]
    
    counts = feature_counts.cpu().numpy()
    
    report = LivenessReport(
        total_samples=total_samples,
        total_features=n_features,
        activation_counts=counts,
    )
    
    print(report.summary())
    
    if save_path:
        np.savez(
            save_path,
            counts=counts,
            total_samples=total_samples,
        )
        print(f"Saved to {save_path}")
    
    return report


def generate_feature_examples_with_liveness(
    model_path: str,
    activations_path: str,
    positions_path: str,
    output_dir: str,
    n_features: int = 10,
    n_examples: int = 5,
    k: int = 64,
    liveness_report: Optional[LivenessReport] = None,
    feature_selection: str = 'interpretable',  # 'interpretable', 'top', 'random_live'
    device: Optional[str] = None,
):
    """
    Generate feature examples with liveness-aware feature selection.
    
    Args:
        model_path: Path to MSAE checkpoint
        activations_path: Directory with activation chunks
        positions_path: Path to positions.pt
        output_dir: Where to save figures
        n_features: Number of features to visualize
        n_examples: Examples per feature
        k: Sparsity level
        liveness_report: Pre-computed liveness (computes if None)
        feature_selection: How to select features
        device: Device to use
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.msae import MatryoshkaSAE
    from visualization.go_board import GoBoardRenderer, BoardPosition
    
    # Setup device
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"Generating feature examples on {device}...")
    
    # Compute liveness if not provided
    if liveness_report is None:
        print("Computing feature liveness first...")
        liveness_report = compute_feature_liveness(
            model_path=model_path,
            activations_dir=activations_path,
            k=k,
            device=device,
            max_chunks=50,  # Use subset for speed
        )
    
    # Select features based on strategy
    if feature_selection == 'interpretable':
        candidate_features = liveness_report.get_interpretable_features()
        print(f"Using {len(candidate_features)} interpretable features (0.1%-5% activation rate)")
    elif feature_selection == 'top':
        # Top by activation count, but exclude dead
        live = liveness_report.get_live_features(0.0001)
        counts = liveness_report.activation_counts[live]
        top_indices = np.argsort(counts)[-n_features*2:][::-1]
        candidate_features = live[top_indices]
    else:  # random_live
        candidate_features = liveness_report.get_live_features(0.0001)
    
    if len(candidate_features) == 0:
        print("ERROR: No live features found! Model may have collapsed.")
        return
    
    # Select features
    if len(candidate_features) <= n_features:
        selected_features = candidate_features
    else:
        # Mix: top half by activation, random half
        counts = liveness_report.activation_counts[candidate_features]
        n_top = n_features // 2
        n_random = n_features - n_top
        
        top_indices = np.argsort(counts)[-n_top:][::-1]
        remaining = np.setdiff1d(np.arange(len(candidate_features)), top_indices)
        random_indices = np.random.choice(remaining, min(n_random, len(remaining)), replace=False)
        
        selected_features = np.concatenate([
            candidate_features[top_indices],
            candidate_features[random_indices]
        ])
    
    print(f"Selected features: {selected_features.tolist()}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    msae = MatryoshkaSAE(input_dim=256, hidden_dim=4096, k_levels=[16, 32, 64, 128])
    msae.load_state_dict(checkpoint['model_state_dict'])
    msae = msae.to(device).eval()
    
    # Load normalization
    norm = checkpoint['normalization']
    act_mean = norm['mean']
    act_std = norm['std']
    if isinstance(act_mean, torch.Tensor):
        act_mean = act_mean.cpu().numpy()
        act_std = act_std.cpu().numpy()
    
    # Load positions
    positions_data = torch.load(positions_path, map_location='cpu', weights_only=False)
    if isinstance(positions_data, torch.Tensor):
        positions = positions_data.numpy()
    else:
        positions = np.array(positions_data)
    
    n_positions = len(positions)
    print(f"Loaded {n_positions} positions")
    
    # Load activations
    act_dir = Path(activations_path)
    chunks = sorted(act_dir.glob('chunk_*.npy'))
    
    samples_per_pos = 361
    max_samples = 200000  # Use more for better examples
    
    all_activations = []
    all_position_ids = []
    all_spatial_ids = []
    cumulative_positions = 0
    
    for chunk_path in chunks:
        chunk = np.load(chunk_path)
        n_samples = len(chunk)
        n_pos_in_chunk = n_samples // samples_per_pos
        
        pos_ids = np.repeat(
            np.arange(cumulative_positions, cumulative_positions + n_pos_in_chunk),
            samples_per_pos
        )[:n_samples]
        
        spatial_ids = np.tile(np.arange(samples_per_pos), n_pos_in_chunk)[:n_samples]
        
        all_activations.append(chunk)
        all_position_ids.append(pos_ids)
        all_spatial_ids.append(spatial_ids)
        
        cumulative_positions += n_pos_in_chunk
        
        if sum(len(a) for a in all_activations) >= max_samples:
            break
    
    activations = np.concatenate(all_activations)[:max_samples]
    position_ids = np.concatenate(all_position_ids)[:max_samples]
    spatial_ids = np.concatenate(all_spatial_ids)[:max_samples]
    
    print(f"Using {len(activations)} samples from {cumulative_positions} positions")
    
    # Normalize
    activations = (activations - act_mean) / (act_std + 1e-8)
    
    # Extract features
    print(f"Extracting sparse features at k={k}...")
    with torch.no_grad():
        act_tensor = torch.from_numpy(activations).float().to(device)
        features_list = []
        batch_size = 4096
        for i in range(0, len(act_tensor), batch_size):
            batch = act_tensor[i:i+batch_size]
            _, _, z_sparse = msae(batch, k=k)
            features_list.append(z_sparse.cpu().numpy())
        features = np.concatenate(features_list)
    
    # Generate visualizations
    examples_dir = Path(output_dir) / 'feature_examples'
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    for feat_idx in selected_features:
        feat_idx = int(feat_idx)
        feat_activations = features[:, feat_idx]
        
        # Check if feature is active in this subset
        n_active = (feat_activations > 0).sum()
        if n_active == 0:
            print(f"Feature {feat_idx}: 0 activations in subset, skipping")
            continue
        
        print(f"Feature {feat_idx}: {n_active} activations ({100*n_active/len(feat_activations):.3f}%)")
        
        top_sample_indices = np.argsort(feat_activations)[-n_examples*20:][::-1]
        
        # Get unique positions
        seen_positions = set()
        examples = []
        
        for sample_idx in top_sample_indices:
            pos_id = position_ids[sample_idx]
            if pos_id not in seen_positions and pos_id < n_positions:
                seen_positions.add(pos_id)
                examples.append({
                    'sample_idx': sample_idx,
                    'position_id': pos_id,
                    'spatial_id': spatial_ids[sample_idx],
                    'activation': feat_activations[sample_idx]
                })
                if len(examples) >= n_examples:
                    break
        
        if len(examples) == 0:
            print(f"  No valid examples found")
            continue
        
        # Create figure
        fig, axes = plt.subplots(1, len(examples), figsize=(4*len(examples), 4))
        if len(examples) == 1:
            axes = [axes]
        
        for ex_idx, example in enumerate(examples):
            ax = axes[ex_idx]
            pos_id = int(example['position_id'])
            
            # Convert position planes to stones
            position = positions[pos_id]
            stones = np.zeros((19, 19), dtype=np.int8)
            
            if position.ndim == 3 and position.shape[0] >= 17:
                black_to_move = position[16].mean() > 0.5
                current_stones = position[0] > 0.5
                opponent_stones = position[8] > 0.5
                
                if black_to_move:
                    stones[current_stones] = 1  # Black
                    stones[opponent_stones] = 2  # White
                else:
                    stones[current_stones] = 2  # White
                    stones[opponent_stones] = 1  # Black
            
            # Render
            try:
                board_pos = BoardPosition(stones=stones)
                renderer = GoBoardRenderer(figsize=(4, 4))
                renderer.render_board(board_pos, ax=ax, show_coordinates=False)
                
                # Mark top activating point
                spatial_idx = int(example['spatial_id'])
                top_y, top_x = divmod(spatial_idx, 19)
                renderer.add_markers([(top_x, top_y)], marker='*', color='cyan', size=300)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title(f'Act: {example["activation"]:.2f}', fontsize=10)
            ax.axis('off')
        
        # Add liveness info to title
        act_count = liveness_report.activation_counts[feat_idx]
        act_rate = act_count / liveness_report.total_samples * 100
        fig.suptitle(
            f'Feature {feat_idx} - Top {len(examples)} Examples\n'
            f'(Global: {act_count:,} activations, {act_rate:.2f}%)',
            fontsize=12
        )
        plt.tight_layout()
        
        save_path = examples_dir / f'feature_{feat_idx:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {save_path}")
    
    print(f"\nFeature examples saved to {examples_dir}")
    return examples_dir


def plot_liveness_histogram(
    report: LivenessReport,
    output_path: str,
):
    """Create a histogram of feature liveness."""
    counts = report.activation_counts
    
    plt.figure(figsize=(10, 5))
    plt.hist(np.log10(counts + 1), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Log10(Activation Count + 1)')
    plt.ylabel('Number of Features')
    plt.title(f'Feature Liveness Distribution\n(Dead: {report.dead_rate:.1f}%, '
              f'Interpretable: {len(report.get_interpretable_features())})')
    
    # Add reference lines
    min_count = report.total_samples * 0.001
    max_count = report.total_samples * 0.05
    plt.axvline(np.log10(min_count), color='green', linestyle='--', 
                label=f'0.1% threshold ({min_count:.0f})')
    plt.axvline(np.log10(max_count), color='orange', linestyle='--',
                label=f'5% threshold ({max_count:.0f})')
    plt.legend()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {output_path}")
