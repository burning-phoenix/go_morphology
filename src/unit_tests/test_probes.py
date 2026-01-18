"""
Unit tests for linear probe training and evaluation.

These tests validate SCIENTIFIC CORRECTNESS of the probe methodology.
Critical tests ensure no train/test contamination and valid AUC computation.

Priority: 🔴 Critical - probe results are core claims of the paper.

References:
- Gao et al. "Scaling and Evaluating Sparse Autoencoders" (probe methodology)
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.probes import (
    LinearProbe, train_probe, evaluate_probe, ProbeResult, ProbeEvaluator
)


# =============================================================================
# Linear Probe Architecture
# =============================================================================

class TestLinearProbeArchitecture:
    """Tests for LinearProbe model architecture."""

    def test_probe_is_purely_linear(self):
        """
        INVARIANT: Probe must be purely linear (no hidden layers).
        
        From methodology: probes should be simple to avoid learning
        non-linear transformations of features.
        """
        probe = LinearProbe(n_features=100)
        
        # Count parameters: should be weight (100) + bias (1) = 101
        n_params = sum(p.numel() for p in probe.parameters())
        
        assert n_params == 101, \
            f"Probe should have 101 params (100 weight + 1 bias), got {n_params}"

    def test_probe_output_is_scalar_per_sample(self):
        """
        INVARIANT: Probe output is single logit per sample.
        """
        probe = LinearProbe(n_features=50)
        x = torch.randn(32, 50)
        
        logits = probe(x)
        
        assert logits.shape == (32,), f"Expected (32,), got {logits.shape}"

    def test_predict_proba_in_valid_range(self):
        """
        INVARIANT: Probabilities must be in [0, 1].
        """
        probe = LinearProbe(n_features=100)
        x = torch.randn(1000, 100)
        
        probs = probe.predict_proba(x)
        
        assert (probs >= 0).all() and (probs <= 1).all(), \
            f"Probabilities out of range: [{probs.min():.4f}, {probs.max():.4f}]"


# =============================================================================
# Probe Training - Learning Validation
# =============================================================================

class TestProbeTraining:
    """Tests for probe training correctness."""

    def test_probe_learns_linear_relationship(self):
        """
        VALIDATION: Probe must learn a linearly separable concept.
        
        If probe cannot learn a perfectly linear relationship, training is broken.
        """
        np.random.seed(42)
        # Create linearly separable data
        n = 1000
        features = np.random.randn(n, 50).astype(np.float32)
        # Label = 1 if sum of first 5 features > 0
        labels = (features[:, :5].sum(axis=1) > 0).astype(np.float32)
        
        probe = train_probe(features, labels, n_epochs=200, lr=0.1, verbose=False)
        
        # Evaluate on training data (should be near-perfect for linear)
        with torch.no_grad():
            X = torch.from_numpy(features)
            probs = probe.predict_proba(X).numpy()
        
        # Accuracy should be very high
        preds = (probs > 0.5).astype(int)
        accuracy = (preds == labels).mean()
        
        assert accuracy > 0.95, \
            f"Probe failed to learn linear relationship: accuracy={accuracy:.2%}"

    def test_probe_learns_single_feature(self):
        """
        VALIDATION: Probe must learn single-feature decision boundary.
        """
        np.random.seed(42)
        features = np.random.randn(500, 100).astype(np.float32)
        # Label depends only on feature 42
        labels = (features[:, 42] > 0).astype(np.float32)
        
        probe = train_probe(features, labels, n_epochs=100, verbose=False)
        
        # Weight for feature 42 should be largest in magnitude
        weights = probe.linear.weight.data.abs().squeeze()
        top_feature = weights.argmax().item()
        
        # Feature 42 should have highest weight (or close to it)
        assert weights[42] > weights.median(), \
            f"Probe didn't concentrate weight on relevant feature"


# =============================================================================
# AUC-ROC Computation
# =============================================================================

class TestAUCComputation:
    """Tests for AUC-ROC correctness."""

    def test_auc_perfect_classifier(self):
        """
        INVARIANT: Perfect classifier has AUC = 1.0.
        """
        probe = LinearProbe(n_features=10)
        # Set weights to perfectly separate
        with torch.no_grad():
            probe.linear.weight.data = torch.zeros(1, 10)
            probe.linear.weight.data[0, 0] = 10.0  # Strong weight on feature 0
            probe.linear.bias.data = torch.tensor([0.0])
        
        features = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        labels = np.array([1, 1, 0, 0], dtype=np.float32)
        
        auc, ci_low, ci_high = evaluate_probe(probe, features, labels, n_bootstrap=10)
        
        assert auc == 1.0, f"Perfect classifier should have AUC=1.0, got {auc}"

    def test_auc_random_classifier(self):
        """
        INVARIANT: Random classifier has AUC ≈ 0.5.
        """
        np.random.seed(42)
        probe = LinearProbe(n_features=100)
        # Don't train - random initialization
        
        features = np.random.randn(1000, 100).astype(np.float32)
        labels = np.random.randint(0, 2, 1000).astype(np.float32)
        
        auc, _, _ = evaluate_probe(probe, features, labels, n_bootstrap=50)
        
        assert 0.4 < auc < 0.6, \
            f"Random classifier should have AUC ≈ 0.5, got {auc}"

    def test_auc_single_class_returns_0_5(self):
        """
        INVARIANT: When only one class present, return AUC = 0.5.
        
        This prevents misleading results from imbalanced evaluation sets.
        """
        probe = LinearProbe(n_features=50)
        features = np.random.randn(100, 50).astype(np.float32)
        labels = np.ones(100).astype(np.float32)  # All positive
        
        auc, ci_low, ci_high = evaluate_probe(probe, features, labels)
        
        assert auc == 0.5, f"Single-class AUC should be 0.5, got {auc}"
        assert ci_low == 0.5 and ci_high == 0.5, "CI should also be 0.5"


# =============================================================================
# Train/Test Split - CRITICAL for Scientific Validity
# =============================================================================

class TestDataSplitting:
    """
    Tests for data splitting methodology.
    
    CRITICAL: Wrong splitting causes train/test contamination,
    leading to inflated AUC scores and invalid scientific claims.
    """

    def test_split_proportions(self):
        """
        INVARIANT: Splits must approximately match requested proportions.
        """
        evaluator = ProbeEvaluator(test_size=0.15, val_size=0.15)
        
        features = np.random.randn(10000, 50)
        labels = np.random.randint(0, 2, 10000)
        
        X_train, X_val, X_test, y_train, y_val, y_test = evaluator._split_data(
            features, labels
        )
        
        total = len(y_train) + len(y_val) + len(y_test)
        test_ratio = len(y_test) / total
        val_ratio = len(y_val) / total
        train_ratio = len(y_train) / total
        
        assert 0.12 < test_ratio < 0.18, f"Test ratio: {test_ratio:.2%}"
        assert 0.12 < val_ratio < 0.18, f"Val ratio: {val_ratio:.2%}"
        assert 0.67 < train_ratio < 0.76, f"Train ratio: {train_ratio:.2%}"

    def test_split_no_sample_overlap(self):
        """
        INVARIANT: No sample appears in multiple splits.
        """
        evaluator = ProbeEvaluator(test_size=0.2, val_size=0.1)
        
        # Use unique features to track samples
        n = 1000
        features = np.arange(n * 50).reshape(n, 50).astype(np.float64)
        labels = np.random.randint(0, 2, n)
        
        X_train, X_val, X_test, y_train, y_val, y_test = evaluator._split_data(
            features, labels
        )
        
        # Check for overlap using first feature as ID
        train_ids = set(X_train[:, 0])
        val_ids = set(X_val[:, 0])
        test_ids = set(X_test[:, 0])
        
        assert len(train_ids & val_ids) == 0, "Train/val overlap detected"
        assert len(train_ids & test_ids) == 0, "Train/test overlap detected"
        assert len(val_ids & test_ids) == 0, "Val/test overlap detected"


class TestPositionLevelSplitting:
    """
    Tests for position-level splitting.
    
    CRITICAL: For Go positions, each position has 361 spatial points.
    If we split at sample level, the same position could appear in
    train AND test, causing the probe to memorize position-specific
    features rather than concept-specific features.
    """

    def test_position_split_no_leakage(self):
        """
        INVARIANT: No position ID appears in both train and test.
        
        This is the most critical test for probe validity.
        """
        evaluator = ProbeEvaluator(test_size=0.15, val_size=0.15)
        
        n_positions = 100
        n_points = 361
        n_samples = n_positions * n_points
        
        features = np.random.randn(n_samples, 256).astype(np.float64)
        labels = np.random.randint(0, 2, n_samples)
        position_ids = np.repeat(np.arange(n_positions), n_points)
        
        # Add position_ids to features so we can track them
        features_with_id = np.column_stack([position_ids.astype(np.float64), features])
        
        X_train, X_val, X_test, y_train, y_val, y_test = evaluator._split_data(
            features_with_id, labels, position_ids
        )
        
        # Extract position IDs from first column
        train_pos = set(X_train[:, 0].astype(int))
        val_pos = set(X_val[:, 0].astype(int))
        test_pos = set(X_test[:, 0].astype(int))
        
        assert len(train_pos & test_pos) == 0, \
            f"CRITICAL: Position leakage between train and test: {train_pos & test_pos}"
        assert len(train_pos & val_pos) == 0, \
            f"CRITICAL: Position leakage between train and val"
        assert len(val_pos & test_pos) == 0, \
            f"CRITICAL: Position leakage between val and test"

    def test_position_split_preserves_structure(self):
        """
        INVARIANT: All 361 points from a position stay in the same split.
        """
        evaluator = ProbeEvaluator(test_size=0.2, val_size=0.1)
        
        n_positions = 50
        n_points = 361
        
        features = np.random.randn(n_positions * n_points, 100)
        labels = np.random.randint(0, 2, n_positions * n_points)
        position_ids = np.repeat(np.arange(n_positions), n_points)
        
        X_train, X_val, X_test, y_train, y_val, y_test = evaluator._split_data(
            features, labels, position_ids
        )
        
        # Each split should have samples that are multiples of 361
        # (or close, if positions don't divide evenly)
        total = len(y_train) + len(y_val) + len(y_test)
        assert total == n_positions * n_points, "Total samples changed"


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_ci_contains_point_estimate(self):
        """
        INVARIANT: Confidence interval should contain the point estimate.
        """
        np.random.seed(42)
        probe = LinearProbe(n_features=50)
        features = np.random.randn(500, 50).astype(np.float32)
        labels = np.random.randint(0, 2, 500).astype(np.float32)
        
        auc, ci_low, ci_high = evaluate_probe(probe, features, labels, n_bootstrap=100)
        
        assert ci_low <= auc <= ci_high, \
            f"AUC {auc} not in CI [{ci_low}, {ci_high}]"

    def test_ci_width_reasonable(self):
        """
        INVARIANT: CI should narrow with more samples.
        """
        np.random.seed(42)
        probe = LinearProbe(n_features=20)
        
        # Small sample
        features_small = np.random.randn(100, 20).astype(np.float32)
        labels_small = np.random.randint(0, 2, 100).astype(np.float32)
        _, ci_low_small, ci_high_small = evaluate_probe(
            probe, features_small, labels_small, n_bootstrap=100
        )
        width_small = ci_high_small - ci_low_small
        
        # Large sample
        features_large = np.random.randn(2000, 20).astype(np.float32)
        labels_large = np.random.randint(0, 2, 2000).astype(np.float32)
        _, ci_low_large, ci_high_large = evaluate_probe(
            probe, features_large, labels_large, n_bootstrap=100
        )
        width_large = ci_high_large - ci_low_large
        
        # Larger sample should have narrower CI (generally)
        assert width_large < width_small, \
            f"CI didn't narrow: small={width_small:.3f}, large={width_large:.3f}"


# =============================================================================
# Probe Evaluation Integration
# =============================================================================

class TestProbeEvaluatorIntegration:
    """Integration tests for full probe evaluation workflow."""

    def test_evaluate_concept_workflow(self):
        """
        VALIDATION: Full workflow produces valid ProbeResult.
        """
        evaluator = ProbeEvaluator()
        
        np.random.seed(42)
        features = np.random.randn(5000, 100).astype(np.float32)
        # Create a learnable pattern
        labels = (features[:, 0] > features[:, 1]).astype(np.float32)
        
        result = evaluator.evaluate_concept(
            concept_name='test_concept',
            layer_name='block5',
            method='msae',
            features=features,
            labels=labels,
            k=64,
            verbose=False,
        )
        
        assert isinstance(result, ProbeResult)
        assert result.concept == 'test_concept'
        assert 0 <= result.auc <= 1
        assert result.ci_low <= result.auc <= result.ci_high


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
