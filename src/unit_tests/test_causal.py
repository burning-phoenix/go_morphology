"""
Unit tests for causal analysis (ablation and steering experiments).

These tests validate the causal intervention methodology for feature effects.
Priority: 🔴 Critical - causal claims are core scientific results.

References:
- Gao et al. "Scaling and Evaluating Sparse Autoencoders" (ablation methodology)
- CLAUDE.md (causal intervention protocol)
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.causal import (
    compute_policy_kl_divergence,
    compute_ablation_sparsity,
    AblationResult,
    DifferentialImpactResult,
)


# =============================================================================
# KL Divergence Computation
# =============================================================================

class TestComputePolicyKLDivergence:
    """Tests for KL divergence computation."""

    def test_kl_identical_policies_is_zero(self):
        """
        INVARIANT: KL(P||P) = 0 for any distribution P.
        """
        policy = torch.randn(10, 362)
        
        kl = compute_policy_kl_divergence(policy, policy)
        
        assert torch.allclose(kl, torch.zeros(10), atol=1e-5), \
            f"KL(P||P) should be 0, got {kl}"

    def test_kl_different_policies_is_positive(self):
        """
        INVARIANT: KL(P||Q) > 0 when P != Q.
        """
        torch.manual_seed(42)
        policy1 = torch.randn(10, 362)
        policy2 = torch.randn(10, 362)
        
        kl = compute_policy_kl_divergence(policy1, policy2)
        
        assert (kl > 0).all(), "KL divergence should be positive for different policies"

    def test_kl_output_shape(self):
        """
        INVARIANT: KL returns one value per sample.
        """
        policy1 = torch.randn(32, 362)
        policy2 = torch.randn(32, 362)
        
        kl = compute_policy_kl_divergence(policy1, policy2)
        
        assert kl.shape == (32,)

    def test_kl_numerical_stability_extreme(self):
        """
        INVARIANT: No NaN/inf for extreme logits.
        """
        # Very high logit in one position
        policy1 = torch.zeros(10, 362)
        policy1[:, 0] = 100  # Very high
        
        policy2 = torch.zeros(10, 362)
        policy2[:, 1] = 100  # Different position
        
        kl = compute_policy_kl_divergence(policy1, policy2)
        
        assert not torch.isnan(kl).any(), "KL produced NaN"
        assert not torch.isinf(kl).any(), "KL produced inf"

    def test_kl_asymmetric(self):
        """
        PROPERTY: KL(P||Q) != KL(Q||P) in general.
        """
        torch.manual_seed(42)
        policy1 = torch.randn(10, 362)
        policy2 = torch.randn(10, 362)
        
        kl_12 = compute_policy_kl_divergence(policy1, policy2)
        kl_21 = compute_policy_kl_divergence(policy2, policy1)
        
        # Generally not equal (asymmetric)
        assert not torch.allclose(kl_12, kl_21), \
            "KL should be asymmetric"


# =============================================================================
# Ablation Sparsity Metric
# =============================================================================

class TestComputeAblationSparsity:
    """Tests for ablation sparsity metric."""

    def test_sparsity_single_nonzero(self):
        """
        INVARIANT: Single non-zero element should have high sparsity score.
        
        The formula: (L2² / L1²) * n
        For single element: (x² / x²) * n = n
        """
        effects = torch.zeros(362)
        effects[0] = 1.0
        
        sparsity = compute_ablation_sparsity(effects)
        
        # For sparse (concentrated) effects, ratio * n should equal n
        assert abs(sparsity - 362.0) < 1.0, \
            f"Single element sparsity should be ~n=362, got {sparsity}"

    def test_sparsity_uniform(self):
        """
        INVARIANT: Uniform effects should have sparsity ratio ≈ 1.
        
        For uniform: L2²/L1² * n = (n*v²)/(n²*v²) * n = 1
        """
        n = 100
        effects = torch.ones(n) / n
        
        sparsity = compute_ablation_sparsity(effects)
        
        assert abs(sparsity - 1.0) < 0.1, \
            f"Uniform effects should have sparsity ~1, got {sparsity}"

    def test_sparsity_zero_effects(self):
        """
        INVARIANT: Zero effects should return 0.
        """
        effects = torch.zeros(362)
        
        sparsity = compute_ablation_sparsity(effects)
        
        assert sparsity == 0.0

    def test_sparsity_range(self):
        """
        PROPERTY: Sparsity should be non-negative.
        """
        effects = torch.randn(100).abs()
        
        sparsity = compute_ablation_sparsity(effects)
        
        assert sparsity >= 0


# =============================================================================
# Ablation Result Structure
# =============================================================================

class TestAblationResult:
    """Tests for AblationResult dataclass."""

    def test_ablation_result_creation(self):
        """AblationResult should store all fields."""
        result = AblationResult(
            feature_idx=42,
            layer='block5',
            policy_kl_divergence=0.05,
            value_change=0.02,
            mean_activation=1.5,
            activation_rate=0.3,
            p_value=0.01,
            significant=True,
        )
        
        assert result.feature_idx == 42
        assert result.policy_kl_divergence == 0.05
        assert result.significant == True


class TestDifferentialImpactResult:
    """Tests for DifferentialImpactResult dataclass."""

    def test_differential_impact_creation(self):
        """DifferentialImpactResult should store all fields."""
        result = DifferentialImpactResult(
            feature_idx=42,
            concept='is_atari',
            layer='block5',
            impact_when_present=0.08,
            impact_when_absent=0.02,
            differential=0.06,
            p_value=0.03,
            significant=True,
            n_present=500,
            n_absent=4500,
        )
        
        assert result.differential == 0.06
        assert result.n_present == 500


# =============================================================================
# Ablation Operation Tests
# =============================================================================

class TestAblationOperation:
    """Tests for the core ablation operation."""

    def test_ablation_zeros_feature(self):
        """
        INVARIANT: Ablating feature i sets z[:, i] = 0.
        """
        z_sparse = torch.randn(100, 256)
        feature_idx = 42
        
        # Ablation operation
        z_ablated = z_sparse.clone()
        z_ablated[:, feature_idx] = 0.0
        
        assert (z_ablated[:, feature_idx] == 0).all()
        
        # Other features unchanged
        for i in range(256):
            if i != feature_idx:
                assert torch.equal(z_ablated[:, i], z_sparse[:, i])

    def test_ablation_preserves_sparsity(self):
        """
        INVARIANT: Ablating a feature should not increase sparsity.
        """
        # Create sparse tensor
        z_sparse = torch.zeros(100, 256)
        for i in range(100):
            indices = torch.randperm(256)[:32]
            z_sparse[i, indices] = torch.randn(32)
        
        # Count non-zeros before
        nnz_before = (z_sparse != 0).sum(dim=1)
        
        # Ablate
        feature_idx = 10
        z_ablated = z_sparse.clone()
        z_ablated[:, feature_idx] = 0.0
        
        # Count non-zeros after
        nnz_after = (z_ablated != 0).sum(dim=1)
        
        # Should be same or less
        assert (nnz_after <= nnz_before).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
