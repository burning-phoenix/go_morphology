"""
Hierarchy analysis for Matryoshka SAE.

Analyzes the hierarchical structure of MSAE features:
- Nestedness: Are k=16 features a subset of k=32?
- Reconstruction gap: R²(k=128) - R²(k=16)
- Concept specificity: Which k level best predicts each concept?
- Feature overlap: Which features appear at multiple k levels?

Based on:
- docs/msae_paper.md (Wooldridge et al.)
- docs/multi_budget_sae.md (Balagansky et al.)

Optimized for 1M+ sample analysis on T4 GPU.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class HierarchyMetrics:
    """Hierarchy analysis metrics for an MSAE."""
    layer: str
    k_levels: List[int]
    nestedness: Dict[str, float]  # 'k16_in_k32' -> ratio
    reconstruction_r2: Dict[int, float]  # k -> R²
    reconstruction_gap: float  # R²(k_max) - R²(k_min)
    feature_stability: Dict[str, float]  # How stable features are across k


def compute_nestedness(
    model: nn.Module,
    activations: torch.Tensor,
    k_levels: List[int],
    z_cached: Optional[torch.Tensor] = None,
    batch_size: int = 4096,
) -> Dict[str, float]:
    """
    Compute nestedness: fraction of lower-k features that appear in higher-k.

    For perfect Matryoshka structure, k=16 features should be strict subset
    of k=32 features, etc.

    Vectorized implementation for GPU efficiency.

    Args:
        model: MSAE model
        activations: (batch, input_dim) normalized activations
        k_levels: List of k values to analyze
        z_cached: Optional pre-computed encoder output
        batch_size: Batch size for processing large inputs

    Returns:
        Dict mapping 'k{lower}_in_k{higher}' to overlap ratio
    """
    model.eval()
    device = next(model.parameters()).device
    
    k_levels = sorted(k_levels)
    nestedness = {}
    
    n_samples = len(activations)
    all_overlaps = {f'k{k_levels[i]}_in_k{k_levels[i+1]}': [] for i in range(len(k_levels) - 1)}
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = activations[start_idx:end_idx].to(device)
            
            # Get pre-activation latents (use cached if provided for this batch)
            if z_cached is not None:
                z = z_cached[start_idx:end_idx]
            else:
                z = model.encode(batch)
            
            # For each pair of adjacent k levels
            for i in range(len(k_levels) - 1):
                k_low = k_levels[i]
                k_high = k_levels[i + 1]
                
                # Get top-k indices for each level
                _, indices_low = torch.topk(z, k_low, dim=-1)  # (batch, k_low)
                _, indices_high = torch.topk(z, k_high, dim=-1)  # (batch, k_high)
                
                # Vectorized overlap computation
                # Expand dims for broadcasting: check if each low index appears in high indices
                low_expanded = indices_low.unsqueeze(2)  # (batch, k_low, 1)
                high_expanded = indices_high.unsqueeze(1)  # (batch, 1, k_high)
                matches = (low_expanded == high_expanded).any(dim=2)  # (batch, k_low)
                overlap = matches.float().mean(dim=1)  # (batch,) - fraction of low in high
                
                all_overlaps[f'k{k_low}_in_k{k_high}'].append(overlap.cpu())
    
    # Aggregate across batches
    for key in all_overlaps:
        nestedness[key] = torch.cat(all_overlaps[key]).mean().item()
    
    return nestedness


def compute_reconstruction_r2(
    model: nn.Module,
    activations: torch.Tensor,
    k_levels: List[int],
    batch_size: int = 4096,
) -> Dict[int, float]:
    """
    Compute reconstruction R² at each k level.

    R² = 1 - MSE(x, x_hat) / Var(x)

    Batched for memory efficiency.

    Args:
        model: MSAE model
        activations: (batch, input_dim) normalized activations
        k_levels: List of k values
        batch_size: Batch size for processing

    Returns:
        Dict mapping k -> R²
    """
    model.eval()
    device = next(model.parameters()).device
    
    n_samples = len(activations)
    
    # Compute variance on full dataset (streaming)
    var_sum = 0.0
    mean_sum = 0.0
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = activations[start_idx:end_idx].to(device)
        mean_sum += batch.sum().item()
    mean = mean_sum / (n_samples * activations.shape[1])
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = activations[start_idx:end_idx].to(device)
        var_sum += ((batch - mean) ** 2).sum().item()
    var_x = var_sum / (n_samples * activations.shape[1])
    
    r2_scores = {}
    
    with torch.no_grad():
        for k in k_levels:
            mse_sum = 0.0
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch = activations[start_idx:end_idx].to(device)
                
                x_hat, _, _ = model.forward(batch, k)
                mse_sum += ((x_hat - batch) ** 2).sum().item()
            
            mse = mse_sum / (n_samples * activations.shape[1])
            r2 = 1.0 - mse / (var_x + 1e-8)
            r2_scores[k] = r2
    
    return r2_scores


def compute_feature_stability(
    model: nn.Module,
    activations: torch.Tensor,
    k_levels: List[int],
    z_cached: Optional[torch.Tensor] = None,
    batch_size: int = 4096,
) -> Dict[str, float]:
    """
    Compute feature stability: how consistently features appear across k levels.

    A stable feature appears at all k levels where it's active.
    An unstable feature appears at some k but not others.

    Batched implementation for efficiency.

    Args:
        model: MSAE model
        activations: (batch, input_dim) normalized activations
        k_levels: List of k values
        z_cached: Optional pre-computed encoder output
        batch_size: Batch size for processing

    Returns:
        Dict with stability metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    k_levels = sorted(k_levels)
    k_max = max(k_levels)
    hidden_dim = model.hidden_dim
    
    # Track feature appearances across all samples
    # feature_k_counts[feat_idx][k] = count of samples where feature is active at k
    feature_k_counts = np.zeros((hidden_dim, len(k_levels)), dtype=np.int64)
    
    n_samples = len(activations)
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = activations[start_idx:end_idx].to(device)
            
            if z_cached is not None:
                z = z_cached[start_idx:end_idx]
            else:
                z = model.encode(batch)
            
            # Get active features at each k level (batched topk)
            for k_idx, k in enumerate(k_levels):
                _, indices = torch.topk(z, k, dim=-1)  # (batch, k)
                
                # Count occurrences using bincount
                for j in range(len(batch)):
                    active = indices[j].cpu().numpy()
                    feature_k_counts[active, k_idx] += 1
    
    # Compute stability: features active at lower k should also be active at higher k
    # A feature is "stable" if its appearance pattern is monotonic
    stable_count = 0
    total_features_with_activity = 0
    
    for feat_idx in range(hidden_dim):
        counts = feature_k_counts[feat_idx]
        if counts.sum() == 0:
            continue  # Never active
        
        total_features_with_activity += 1
        
        # Check monotonicity: count should increase or stay same as k increases
        is_monotonic = all(counts[i] <= counts[i+1] for i in range(len(k_levels) - 1))
        if is_monotonic:
            stable_count += 1
    
    stability_ratio = stable_count / max(total_features_with_activity, 1)
    
    return {
        'stability_ratio': stability_ratio,
        'total_features_analyzed': total_features_with_activity,
        'stable_features': stable_count,
    }


def compute_feature_importance_by_k(
    model: nn.Module,
    activations: torch.Tensor,
    k_levels: List[int],
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Compute which features are most important at each k level.

    Returns a (hidden_dim, len(k_levels)) matrix showing feature
    activation frequency at each k.

    Batched and vectorized for efficiency.

    Args:
        model: MSAE model
        activations: (batch, input_dim) activations
        k_levels: List of k values
        batch_size: Batch size for processing

    Returns:
        (hidden_dim, len(k_levels)) array of activation frequencies
    """
    model.eval()
    device = next(model.parameters()).device
    
    hidden_dim = model.hidden_dim
    importance = np.zeros((hidden_dim, len(k_levels)), dtype=np.float64)
    n_samples = len(activations)
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = activations[start_idx:end_idx].to(device)
            
            z = model.encode(batch)
            
            for k_idx, k in enumerate(k_levels):
                _, indices = torch.topk(z, k, dim=-1)  # (batch, k)
                
                # Flatten and count with bincount
                flat_indices = indices.flatten().cpu().numpy()
                counts = np.bincount(flat_indices, minlength=hidden_dim)
                importance[:, k_idx] += counts
    
    # Normalize by number of samples
    importance /= n_samples
    
    return importance


def analyze_hierarchy(
    model: nn.Module,
    activations: np.ndarray,
    layer_name: str,
    batch_size: int = 4096,
    device: Optional[torch.device] = None,
    n_samples: Optional[int] = None,
    stability_samples: Optional[int] = None,
) -> HierarchyMetrics:
    """
    Full hierarchy analysis for an MSAE.

    Args:
        model: MSAE model
        activations: (n_samples, input_dim) normalized activations
        layer_name: Name of the layer (e.g., 'block5')
        batch_size: Batch size for processing
        device: Device to use
        n_samples: Max samples for main analysis (default: use all)
        stability_samples: Max samples for stability analysis (default: n_samples // 10)

    Returns:
        HierarchyMetrics with all analysis results
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    k_levels = model.k_levels
    
    # Sample size configuration
    total_available = len(activations)
    if n_samples is None:
        n_samples = total_available
    n_samples = min(n_samples, total_available)
    
    if stability_samples is None:
        stability_samples = min(n_samples // 10, 50000)  # 10% or 50K max
    stability_samples = min(stability_samples, n_samples)
    
    # Random subsample if needed
    if n_samples < total_available:
        idx = np.random.choice(total_available, n_samples, replace=False)
        sample_activations = torch.from_numpy(activations[idx]).float()
    else:
        sample_activations = torch.from_numpy(activations).float()
    
    print(f"Analyzing hierarchy for {layer_name}...")
    print(f"  Using {n_samples:,} samples (stability: {stability_samples:,})")
    
    # Pre-compute encoder output once (cache for reuse)
    print("  Pre-computing encoder outputs...")
    with torch.no_grad():
        z_chunks = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = sample_activations[start_idx:end_idx].to(device)
            z_chunks.append(model.encode(batch))
        z_cached = torch.cat(z_chunks, dim=0)
    
    print("  Computing nestedness...")
    nestedness = compute_nestedness(
        model, sample_activations, k_levels,
        z_cached=z_cached, batch_size=batch_size
    )
    
    print("  Computing reconstruction R²...")
    r2_scores = compute_reconstruction_r2(
        model, sample_activations, k_levels,
        batch_size=batch_size
    )
    
    print("  Computing feature stability...")
    stability = compute_feature_stability(
        model, sample_activations[:stability_samples], k_levels,
        z_cached=z_cached[:stability_samples], batch_size=batch_size
    )
    
    # Clean up cached encoder output
    del z_cached
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Reconstruction gap
    k_min, k_max = min(k_levels), max(k_levels)
    rec_gap = r2_scores[k_max] - r2_scores[k_min]
    
    return HierarchyMetrics(
        layer=layer_name,
        k_levels=k_levels,
        nestedness=nestedness,
        reconstruction_r2=r2_scores,
        reconstruction_gap=rec_gap,
        feature_stability=stability,
    )


def save_hierarchy_results(
    metrics: List[HierarchyMetrics],
    output_dir: str = 'outputs',
    filename: str = 'hierarchy.json',
):
    """
    Save hierarchy analysis results to JSON.

    Output format matches CLAUDE.md specification.
    """
    output_path = Path(output_dir) / 'results' / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        'nestedness': {},
        'reconstruction_gap': {},
        'reconstruction_r2': {},
        'feature_stability': {},
    }

    for m in metrics:
        # Nestedness
        result['nestedness'][m.layer] = m.nestedness

        # Reconstruction gap
        result['reconstruction_gap'][m.layer] = m.reconstruction_gap

        # R² by k
        result['reconstruction_r2'][m.layer] = {
            f'k{k}': r2 for k, r2 in m.reconstruction_r2.items()
        }

        # Stability
        result['feature_stability'][m.layer] = m.feature_stability

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved hierarchy results to {output_path}")


def print_hierarchy_summary(metrics: HierarchyMetrics):
    """Print a summary of hierarchy analysis."""
    print(f"\n{'='*50}")
    print(f"Hierarchy Analysis: {metrics.layer}")
    print(f"{'='*50}")

    print(f"\nReconstruction R² by k:")
    for k in sorted(metrics.reconstruction_r2.keys()):
        print(f"  k={k:3d}: {metrics.reconstruction_r2[k]:.4f}")

    print(f"\nReconstruction gap (k_max - k_min): {metrics.reconstruction_gap:.4f}")

    print(f"\nNestedness (fraction of lower-k in higher-k):")
    for key in sorted(metrics.nestedness.keys()):
        print(f"  {key}: {metrics.nestedness[key]:.4f}")

    print(f"\nFeature stability:")
    for key, value in metrics.feature_stability.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
