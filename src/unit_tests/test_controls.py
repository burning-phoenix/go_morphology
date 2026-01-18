"""
Unit tests for negative control experiments.

These tests validate that the control experiments properly break
the feature-label relationship to produce chance-level AUC.

Priority: 🟠 High - validates that positive results aren't artifacts.

References:
- Bricken et al. "Monosemantic Features" (negative controls methodology)
- CLAUDE.md (control specification)
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.controls import (
    shuffle_labels,
    permute_features,
    create_random_network_activations,
    ControlEvaluator,
)


# =============================================================================
# Shuffle Labels Control
# =============================================================================

class TestShuffleLabels:
    """Tests for label shuffling."""

    def test_shuffle_changes_order(self):
        """
        INVARIANT: Shuffled labels should differ from original.
        """
        np.random.seed(42)
        labels = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0])
        
        shuffled = shuffle_labels(labels)
        
        assert not np.array_equal(labels, shuffled), \
            "Shuffled labels should differ from original"

    def test_shuffle_preserves_distribution(self):
        """
        INVARIANT: Shuffling must preserve class balance.
        """
        labels = np.array([0] * 30 + [1] * 70)
        
        shuffled = shuffle_labels(labels)
        
        assert shuffled.sum() == 70, "Class balance must be preserved"
        assert (shuffled == 0).sum() == 30, "Class balance must be preserved"

    def test_shuffle_deterministic_with_seed(self):
        """
        INVARIANT: Same seed produces same shuffle.
        """
        labels = np.arange(100)
        
        shuf1 = shuffle_labels(labels, random_state=42)
        shuf2 = shuffle_labels(labels, random_state=42)
        
        assert np.array_equal(shuf1, shuf2), "Same seed must produce same shuffle"

    def test_shuffle_different_seeds_differ(self):
        """
        INVARIANT: Different seeds produce different shuffles.
        """
        labels = np.arange(100)
        
        shuf1 = shuffle_labels(labels, random_state=42)
        shuf2 = shuffle_labels(labels, random_state=43)
        
        assert not np.array_equal(shuf1, shuf2), "Different seeds should differ"


# =============================================================================
# Permute Features Control
# =============================================================================

class TestPermuteFeatures:
    """Tests for feature permutation."""

    def test_permute_changes_features(self):
        """
        INVARIANT: Permuted features must differ from original.
        """
        np.random.seed(42)
        features = np.random.randn(100, 50)
        
        permuted = permute_features(features)
        
        assert not np.allclose(features, permuted), \
            "Permuted features should differ from original"

    def test_permute_preserves_values(self):
        """
        INVARIANT: Permutation only reorders, doesn't change values.
        
        For each row, the set of values should be unchanged.
        """
        np.random.seed(42)
        features = np.random.randn(10, 20)
        
        permuted = permute_features(features)
        
        for i in range(10):
            orig_sorted = np.sort(features[i])
            perm_sorted = np.sort(permuted[i])
            assert np.allclose(orig_sorted, perm_sorted), \
                f"Row {i}: values changed, not just reordered"

    def test_permute_independent_per_sample(self):
        """
        INVARIANT: Each sample gets independent random permutation.
        
        This is critical - if all samples share the same permutation,
        the geometric structure is preserved and the control is invalid.
        """
        np.random.seed(42)
        # Create features where structure is in column 0
        features = np.zeros((100, 50))
        features[:, 0] = np.arange(100)  # Unique identifier in col 0
        
        permuted = permute_features(features)
        
        # Find where the unique value ended up in each row
        positions = []
        for i in range(100):
            pos = np.where(permuted[i] == float(i))[0][0]
            positions.append(pos)
        
        # Positions should vary (not all the same column)
        unique_positions = len(set(positions))
        assert unique_positions > 10, \
            f"Permutations not independent: only {unique_positions} unique positions"


# =============================================================================
# Random Network Activations
# =============================================================================

class TestRandomNetworkActivations:
    """Tests for random network baseline."""

    def test_random_activations_shape(self):
        """
        INVARIANT: Random activations must have correct shape.
        """
        shape = (1000, 256)
        activations = create_random_network_activations(shape)
        
        assert activations.shape == shape

    def test_random_activations_distribution(self):
        """
        INVARIANT: Random activations should be normally distributed.
        """
        shape = (10000, 128)
        activations = create_random_network_activations(shape)
        
        # Mean should be near 0
        assert abs(activations.mean()) < 0.1
        # Std should be near 1
        assert 0.9 < activations.std() < 1.1


# =============================================================================
# Control Evaluator
# =============================================================================

class TestControlEvaluator:
    """Tests for ControlEvaluator class."""

    def test_evaluator_init(self):
        """
        INVARIANT: Evaluator should initialize with correct defaults.
        """
        evaluator = ControlEvaluator()
        
        assert evaluator.tolerance == 0.05
        assert evaluator.n_shuffles >= 1

    def test_shuffled_control_runs(self):
        """
        VALIDATION: Shuffled labels control should run and return result.
        """
        evaluator = ControlEvaluator()
        
        np.random.seed(42)
        features = np.random.randn(500, 50).astype(np.float32)
        labels = np.random.randint(0, 2, 500).astype(np.float32)
        
        result = evaluator.run_shuffled_labels_control(
            features=features,
            labels=labels,
            concept_name='test',
            layer_name='block5',
            verbose=False,
        )
        
        assert result is not None
        assert hasattr(result, 'auc')
        assert 0 <= result.auc <= 1


# =============================================================================
# Scientific Validation Tests
# =============================================================================

class TestControlScientificValidity:
    """
    Tests that control experiments behave as expected scientifically.
    
    These tests are less strict about exact values because control
    experiment outcomes can vary. We just verify the controls are
    working in the right direction.
    """

    def test_shuffled_labels_reduces_auc(self):
        """
        VALIDATION: Shuffling labels should reduce AUC compared to real labels.
        """
        from analysis.probes import train_probe, evaluate_probe
        
        np.random.seed(42)
        # Create learnable data
        features = np.random.randn(1000, 50).astype(np.float32)
        real_labels = (features[:, 0] > 0).astype(np.float32)
        
        # Train on real labels
        probe_real = train_probe(features, real_labels, n_epochs=50, verbose=False)
        auc_real, _, _ = evaluate_probe(probe_real, features, real_labels, n_bootstrap=10)
        
        # Shuffle labels
        shuffled = shuffle_labels(real_labels)
        probe_shuffled = train_probe(features, shuffled, n_epochs=50, verbose=False)
        auc_shuffled, _, _ = evaluate_probe(probe_shuffled, features, shuffled, n_bootstrap=10)
        
        # Real AUC should be high, shuffled should be lower (toward 0.5)
        assert auc_real > 0.7, f"Real labels should be learnable, got AUC={auc_real}"
        assert auc_shuffled < auc_real, \
            f"Shuffled AUC ({auc_shuffled}) should be lower than real ({auc_real})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
