"""
Unit tests for hierarchy analysis (nestedness, R² computation).

These tests validate the hierarchical metrics for MSAE.
Priority: 🟠 High - wrong metrics invalidate hierarchy claims.

References:
- Wooldridge et al. "Matryoshka SAE" (nestedness definition)
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.hierarchy import (
    compute_nestedness,
    compute_reconstruction_r2,
    compute_feature_stability,
    compute_feature_importance_by_k,
    HierarchyMetrics,
)
from models.msae import create_msae


# =============================================================================
# Nestedness Tests
# =============================================================================

class TestComputeNestedness:
    """Tests for nestedness computation."""

    def test_nestedness_output_keys(self):
        """
        INVARIANT: Output should have keys for all consecutive k pairs.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32, 64])
        model.eval()
        activations = torch.randn(100, 64)
        
        nestedness = compute_nestedness(model, activations, [8, 16, 32, 64])
        
        assert 'k8_in_k16' in nestedness
        assert 'k16_in_k32' in nestedness
        assert 'k32_in_k64' in nestedness

    def test_nestedness_range(self):
        """
        INVARIANT: Nestedness values must be in [0, 1].
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32])
        model.eval()
        activations = torch.randn(200, 64)
        
        nestedness = compute_nestedness(model, activations, [8, 16, 32])
        
        for key, value in nestedness.items():
            assert 0 <= value <= 1, f"{key} = {value} out of [0, 1] range"

    def test_nestedness_perfect_case(self):
        """
        VALIDATION: TopK naturally nests since top-8 ⊂ top-16.
        
        By definition of how torch.topk works, the top-k features
        at a lower k should be a subset of top-k features at higher k.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16])
        model.eval()
        x = torch.randn(100, 64)
        
        # Get pre-sparsity activations
        z = model.encode(x)
        
        # Verify subset property manually
        _, indices_8 = torch.topk(z, 8, dim=-1)
        _, indices_16 = torch.topk(z, 16, dim=-1)
        
        # All top-8 indices should be in top-16
        all_nested = True
        for i in range(len(x)):
            set_8 = set(indices_8[i].tolist())
            set_16 = set(indices_16[i].tolist())
            if not set_8.issubset(set_16):
                all_nested = False
                break
        
        assert all_nested, "TopK should produce nested feature sets"


# =============================================================================
# Reconstruction R² Tests
# =============================================================================

class TestComputeReconstructionR2:
    """Tests for reconstruction R² computation."""

    def test_r2_output_keys(self):
        """
        INVARIANT: Output should have R² for each k level.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32])
        model.eval()
        activations = torch.randn(200, 64)
        
        r2_scores = compute_reconstruction_r2(model, activations, [8, 16, 32])
        
        assert 8 in r2_scores
        assert 16 in r2_scores
        assert 32 in r2_scores

    def test_r2_bounded_above(self):
        """
        INVARIANT: R² ≤ 1 (cannot explain more than total variance).
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32])
        model.eval()
        activations = torch.randn(200, 64)
        
        r2_scores = compute_reconstruction_r2(model, activations, [8, 16, 32])
        
        for k, r2 in r2_scores.items():
            assert r2 <= 1.0, f"R² for k={k} is {r2} > 1.0"

    def test_r2_finite(self):
        """
        INVARIANT: R² should never be NaN or inf.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[16, 32])
        model.eval()
        activations = torch.randn(500, 64)
        
        r2_scores = compute_reconstruction_r2(model, activations, [16, 32])
        
        for k, r2 in r2_scores.items():
            assert not np.isnan(r2), f"R² for k={k} is NaN"
            assert not np.isinf(r2), f"R² for k={k} is inf"


# =============================================================================
# Feature Stability Tests
# =============================================================================

class TestComputeFeatureStability:
    """Tests for feature stability computation."""

    def test_stability_output_keys(self):
        """
        INVARIANT: Output should have expected keys.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32])
        model.eval()
        activations = torch.randn(200, 64)
        
        stability = compute_feature_stability(model, activations, [8, 16, 32])
        
        assert 'stability_ratio' in stability
        assert 'total_features_analyzed' in stability
        assert 'stable_features' in stability

    def test_stability_ratio_range(self):
        """
        INVARIANT: Stability ratio must be in [0, 1].
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32])
        model.eval()
        activations = torch.randn(200, 64)
        
        stability = compute_feature_stability(model, activations, [8, 16, 32])
        
        assert 0 <= stability['stability_ratio'] <= 1

    def test_stable_count_consistency(self):
        """
        INVARIANT: stable_features ≤ total_features_analyzed.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32])
        model.eval()
        activations = torch.randn(500, 64)
        
        stability = compute_feature_stability(model, activations, [8, 16, 32])
        
        assert stability['stable_features'] <= stability['total_features_analyzed']


# =============================================================================
# Feature Importance Tests
# =============================================================================

class TestComputeFeatureImportance:
    """Tests for feature importance by k level."""

    def test_importance_output_shape(self):
        """
        INVARIANT: Output shape = (hidden_dim, n_k_levels).
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32])
        model.eval()
        activations = torch.randn(200, 64)
        
        importance = compute_feature_importance_by_k(model, activations, [8, 16, 32])
        
        assert importance.shape == (256, 3), f"Expected (256, 3), got {importance.shape}"

    def test_importance_non_negative(self):
        """
        INVARIANT: Importance (activation counts) must be non-negative.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32])
        model.eval()
        activations = torch.randn(200, 64)
        
        importance = compute_feature_importance_by_k(model, activations, [8, 16, 32])
        
        assert (importance >= 0).all()


# =============================================================================
# HierarchyMetrics Dataclass
# =============================================================================

class TestHierarchyMetrics:
    """Tests for HierarchyMetrics dataclass."""

    def test_hierarchy_metrics_creation(self):
        """HierarchyMetrics should store all fields."""
        metrics = HierarchyMetrics(
            layer='block5',
            k_levels=[16, 32, 64, 128],
            nestedness={'k16_in_k32': 0.9, 'k32_in_k64': 0.85},
            reconstruction_r2={16: 0.5, 32: 0.7, 64: 0.85, 128: 0.92},
            reconstruction_gap=0.42,
            feature_stability={'stability_ratio': 0.75},
        )
        
        assert metrics.layer == 'block5'
        assert metrics.reconstruction_gap == 0.42
        assert metrics.nestedness['k16_in_k32'] == 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
