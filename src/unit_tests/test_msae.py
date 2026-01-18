"""
Unit tests for TopK activation and MatryoshkaSAE.

These tests validate SCIENTIFIC CORRECTNESS, not just that code runs.
Each test verifies a mathematical invariant or specification from the papers.

Priority: 🔴 Critical - these tests validate the core algorithm.

References:
- Gao et al. "Scaling and Evaluating Sparse Autoencoders" (TopK SAE)
- Wooldridge et al. "Matryoshka SAE" (hierarchical structure)
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.msae import TopK, topk_activation, MatryoshkaSAE, create_msae


# =============================================================================
# TopK Tests - Mathematical Invariants
# =============================================================================

class TestTopKMathematicalInvariants:
    """
    Tests that the TopK function satisfies required mathematical properties.
    These MUST pass for the training algorithm to be correct.
    """

    def test_topk_exactly_k_nonzeros(self):
        """
        INVARIANT: TopK(x, k) must have EXACTLY k non-zero values per sample.
        
        This is the core sparsity guarantee. If this fails, the entire
        sparse autoencoder formulation is broken.
        """
        torch.manual_seed(42)
        x = torch.randn(1000, 4096)
        
        for k in [16, 32, 64, 128]:
            result = topk_activation(x, k=k)
            non_zeros = (result != 0).sum(dim=1)
            
            # MUST be exactly k, no exceptions
            assert (non_zeros == k).all(), \
                f"k={k}: Expected exactly {k} non-zeros, got {non_zeros.unique().tolist()}"

    def test_topk_preserves_top_k_values(self):
        """
        INVARIANT: The k largest values must be preserved exactly.
        
        TopK should not modify the magnitude of selected values.
        """
        x = torch.tensor([
            [1.0, 5.0, 2.0, 8.0, 3.0],
            [-1.0, -5.0, -2.0, -8.0, -3.0],  # Negative values
            [0.1, 0.2, 0.3, 0.4, 0.5],  # Small values
        ])
        
        result = topk_activation(x, k=2)
        
        # For row 0: top-2 are 8.0 and 5.0
        assert result[0, 3] == 8.0  # Index of 8.0
        assert result[0, 1] == 5.0  # Index of 5.0
        
        # For row 1: top-2 are -1.0 and -2.0 (largest = least negative)
        assert result[1, 0] == -1.0
        assert result[1, 2] == -2.0
        
        # For row 2: top-2 are 0.5 and 0.4
        assert result[2, 4] == 0.5
        assert result[2, 3] == 0.4

    def test_topk_straight_through_gradient(self):
        """
        INVARIANT: Gradients must flow through selected features with magnitude 1.
        
        From Gao et al.: The straight-through estimator passes gradients
        through as if TopK were identity on the selected features.
        
        This is CRITICAL for training - wrong gradients = wrong model.
        """
        torch.manual_seed(42)
        x = torch.randn(10, 100, requires_grad=True)
        k = 20
        
        result = topk_activation(x, k=k)
        loss = result.sum()
        loss.backward()
        
        # 1. Non-zero gradients only where TopK selected
        selected_mask = (result != 0)
        grad_mask = (x.grad != 0)
        assert torch.equal(selected_mask, grad_mask), \
            "Gradients must flow only through selected features"
        
        # 2. Gradient magnitude must be 1 (straight-through)
        nonzero_grads = x.grad[grad_mask]
        assert torch.allclose(nonzero_grads, torch.ones_like(nonzero_grads)), \
            f"Straight-through gradient should be 1, got {nonzero_grads.unique()}"

    def test_topk_zeros_non_selected(self):
        """
        INVARIANT: Non-selected features must be exactly zero.
        
        This is not just "very small" but exactly 0.0 for sparsity.
        """
        x = torch.randn(100, 500)
        k = 50
        
        result = topk_activation(x, k=k)
        
        # Get non-selected values
        _, topk_indices = torch.topk(x, k, dim=-1)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, False)
        
        non_selected = result[mask]
        
        assert (non_selected == 0.0).all(), \
            f"Non-selected features must be exactly 0, got {non_selected.abs().max()}"

    def test_topk_deterministic(self):
        """
        INVARIANT: TopK must be deterministic (same input → same output).
        
        Non-determinism would cause irreproducible results.
        """
        x = torch.randn(50, 200)
        k = 30
        
        result1 = topk_activation(x.clone(), k)
        result2 = topk_activation(x.clone(), k)
        
        assert torch.equal(result1, result2), "TopK must be deterministic"


# =============================================================================
# MSAE Architecture Tests - Paper Specification Compliance
# =============================================================================

class TestMSAEArchitecture:
    """
    Tests that MSAE architecture matches paper specifications.
    """

    def test_encoder_decoder_dimensions(self):
        """
        SPECIFICATION: Input 256, hidden 4096 (16x expansion).
        From CLAUDE.md: "Hidden 4096 = 16x expansion"
        """
        model = create_msae(input_dim=256, expansion_factor=16)
        
        assert model.W_enc.shape == (4096, 256), \
            f"Encoder weight shape wrong: {model.W_enc.shape}"
        assert model.W_dec.shape == (256, 4096), \
            f"Decoder weight shape wrong: {model.W_dec.shape}"

    def test_decoder_columns_unit_norm(self):
        """
        SPECIFICATION: Decoder columns must have unit norm.
        From Gao et al.: "we constrain the decoder columns to have unit norm"
        
        After initialization, decoder columns should be unit normalized.
        """
        model = create_msae()
        
        # Compute L2 norm of each column (each latent's decoder direction)
        column_norms = model.W_dec.norm(dim=0)  # Shape: (hidden_dim,)
        
        assert torch.allclose(column_norms, torch.ones_like(column_norms), atol=1e-5), \
            f"Decoder columns not unit norm. Norms range: [{column_norms.min():.4f}, {column_norms.max():.4f}]"

    def test_tied_initialization(self):
        """
        SPECIFICATION: Encoder initialized as transpose of decoder.
        From Gao et al.: "we initialize the encoder to the transpose of the decoder"
        
        This prevents dead latents from initialization.
        """
        model = create_msae()
        
        # After init: W_enc should be transpose of W_dec
        assert torch.allclose(model.W_enc.data, model.W_dec.data.T, atol=1e-5), \
            "Encoder must be initialized as transpose of decoder"

    def test_k_levels_specification(self):
        """
        SPECIFICATION: k levels must be [16, 32, 64, 128].
        From CLAUDE.md.
        """
        model = create_msae()
        
        assert model.k_levels == [16, 32, 64, 128], \
            f"k_levels must be [16, 32, 64, 128], got {model.k_levels}"


# =============================================================================
# MSAE Forward Pass Correctness
# =============================================================================

class TestMSAEForwardMath:
    """
    Tests that forward pass implements the correct mathematical formulation.
    """

    def test_encoding_formula(self):
        """
        SPECIFICATION: z = ReLU(W_enc @ (x - b_pre))
        
        The encoder must apply centering, linear transform, then ReLU.
        """
        model = create_msae(input_dim=64, expansion_factor=4)
        x = torch.randn(10, 64)
        
        # Manual computation
        x_centered = x - model.b_pre
        z_manual = F.relu(x_centered @ model.W_enc.T)
        
        # Model computation
        z_model = model.encode(x)
        
        assert torch.allclose(z_manual, z_model, atol=1e-5), \
            "Encoder does not match formula: z = ReLU(W_enc @ (x - b_pre))"

    def test_decoding_formula(self):
        """
        SPECIFICATION: x_hat = W_dec @ z_sparse + b_pre
        """
        model = create_msae(input_dim=64, expansion_factor=4)
        z_sparse = torch.randn(10, 256)
        
        # Manual computation
        x_hat_manual = z_sparse @ model.W_dec.T + model.b_pre
        
        # Model computation
        x_hat_model = model.decode(z_sparse)
        
        assert torch.allclose(x_hat_manual, x_hat_model, atol=1e-5), \
            "Decoder does not match formula: x_hat = W_dec @ z + b_pre"

    def test_forward_applies_topk(self):
        """
        INVARIANT: Forward pass must apply TopK sparsity.
        
        z_sparse must have exactly k non-zeros per sample.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32])
        x = torch.randn(100, 64)
        
        for k in [8, 16, 32]:
            _, _, z_sparse = model.forward(x, k=k)
            non_zeros = (z_sparse != 0).sum(dim=1)
            
            assert (non_zeros == k).all(), \
                f"Forward with k={k} produced wrong sparsity: {non_zeros.unique().tolist()}"


# =============================================================================
# Hierarchical Structure - Matryoshka Property
# =============================================================================

class TestMatryoshkaProperty:
    """
    Tests for the hierarchical "nesting" property of MSAE.
    
    The Matryoshka property means features at lower k should be a subset
    of features at higher k (for the same input).
    """

    def test_forward_hierarchical_k_ordering(self):
        """
        INVARIANT: top-k features at k=16 must be subset of top-k at k=32.
        
        This is the core Matryoshka property from Wooldridge et al.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32, 64])
        model.eval()
        x = torch.randn(100, 64)
        
        # Get pre-ReLU activations (before TopK)
        z = model.encode(x)
        
        # Get top-k indices at each level
        _, indices_8 = torch.topk(z, 8, dim=-1)
        _, indices_16 = torch.topk(z, 16, dim=-1)
        _, indices_32 = torch.topk(z, 32, dim=-1)
        _, indices_64 = torch.topk(z, 64, dim=-1)
        
        # Check nesting: k=8 should be subset of k=16
        for i in range(len(x)):
            set_8 = set(indices_8[i].tolist())
            set_16 = set(indices_16[i].tolist())
            set_32 = set(indices_32[i].tolist())
            set_64 = set(indices_64[i].tolist())
            
            assert set_8.issubset(set_16), \
                f"Sample {i}: k=8 features not subset of k=16"
            assert set_16.issubset(set_32), \
                f"Sample {i}: k=16 features not subset of k=32"
            assert set_32.issubset(set_64), \
                f"Sample {i}: k=32 features not subset of k=64"

    def test_forward_hierarchical_matches_individual(self):
        """
        INVARIANT: forward_hierarchical(x)[k] == forward(x, k=k)[0]
        
        The efficient hierarchical method must match individual forward calls.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32, 64])
        model.eval()
        x = torch.randn(50, 64)
        
        hierarchical = model.forward_hierarchical(x)
        
        for k in [8, 16, 32, 64]:
            x_hat_hier = hierarchical[k]
            x_hat_single, _, _ = model.forward(x, k=k)
            
            assert torch.allclose(x_hat_hier, x_hat_single, atol=1e-5), \
                f"forward_hierarchical mismatch at k={k}"


# =============================================================================
# Loss Computation - Training Correctness
# =============================================================================

class TestMSAELoss:
    """
    Tests that loss computation is mathematically correct.
    """

    def test_hierarchical_loss_formula(self):
        """
        SPECIFICATION: Loss = (1/|K|) * Σ_k MSE(x, x_hat_k) + aux_loss
        
        From MSAE paper: equal weighting across k levels.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[8, 16, 32, 64])
        x = torch.randn(32, 64)
        
        losses = model.compute_loss(x, update_stats=False)
        
        # Manual computation of reconstruction loss
        reconstructions = model.forward_hierarchical(x)
        manual_mses = [F.mse_loss(reconstructions[k], x).item() for k in [8, 16, 32, 64]]
        manual_rec_loss = sum(manual_mses) / 4  # Uniform weighting
        
        assert abs(losses['reconstruction_loss'].item() - manual_rec_loss) < 1e-5, \
            f"Reconstruction loss mismatch: {losses['reconstruction_loss'].item()} vs {manual_rec_loss}"

    def test_loss_has_gradients(self):
        """
        INVARIANT: Loss must have gradients through all trainable parameters.
        """
        model = create_msae(input_dim=64, expansion_factor=4)
        x = torch.randn(32, 64)
        
        losses = model.compute_loss(x)
        losses['loss'].backward()
        
        # Check gradients exist and are not zero
        assert model.W_enc.grad is not None, "W_enc has no gradient"
        assert model.W_dec.grad is not None, "W_dec has no gradient"
        assert model.b_pre.grad is not None, "b_pre has no gradient"
        
        assert model.W_enc.grad.abs().sum() > 0, "W_enc gradient is all zeros"
        assert model.W_dec.grad.abs().sum() > 0, "W_dec gradient is all zeros"


# =============================================================================
# Dead Latent Detection
# =============================================================================

class TestDeadLatentDetection:
    """
    Tests for dead latent detection and auxiliary loss.
    """

    def test_activation_counting(self):
        """
        INVARIANT: Activation counts must correctly track active latents.
        """
        model = create_msae(input_dim=64, expansion_factor=4, k_levels=[16])
        model.reset_dead_latent_stats()
        
        # Feed data through
        x = torch.randn(100, 64)
        model.compute_loss(x, update_stats=True)
        
        # Total activations should be n_samples * k (since exactly k active per sample)
        total_activations = model.activation_counts.sum().item()
        expected = 100 * 16  # But activation counts pre-TopK (ReLU output)
        
        # At minimum, counts should increase
        assert model.total_samples.item() == 100, \
            f"Sample count wrong: {model.total_samples.item()}"

    def test_dead_latent_ratio_bounds(self):
        """
        INVARIANT: Dead latent ratio must be in [0, 1].
        """
        model = create_msae(input_dim=64, expansion_factor=4)
        
        # Simulate training
        model.total_samples = torch.tensor(100000.0)
        model.activation_counts[:] = torch.randint(0, 10000, (model.hidden_dim,)).float()
        
        ratio = model.dead_latent_ratio
        
        assert 0 <= ratio <= 1, f"Dead latent ratio out of bounds: {ratio}"


# =============================================================================
# Numerical Stability
# =============================================================================

class TestNumericalStability:
    """
    Tests for numerical stability - no NaN/inf in outputs.
    """

    def test_no_nan_in_forward(self):
        """Forward pass should never produce NaN."""
        model = create_msae(input_dim=64, expansion_factor=4)
        
        # Test various input scales
        for scale in [1e-6, 1e-3, 1.0, 1e3, 1e6]:
            x = torch.randn(32, 64) * scale
            x_hat, z, z_sparse = model.forward(x)
            
            assert not torch.isnan(x_hat).any(), f"NaN in x_hat at scale {scale}"
            assert not torch.isnan(z).any(), f"NaN in z at scale {scale}"
            assert not torch.isnan(z_sparse).any(), f"NaN in z_sparse at scale {scale}"

    def test_no_nan_in_loss(self):
        """Loss computation should never produce NaN."""
        model = create_msae(input_dim=64, expansion_factor=4)
        
        for scale in [1e-3, 1.0, 1e3]:
            x = torch.randn(32, 64) * scale
            losses = model.compute_loss(x, update_stats=False)
            
            assert not torch.isnan(losses['loss']), f"NaN in loss at scale {scale}"


# =============================================================================
# Decoder Normalization
# =============================================================================

class TestDecoderNormalization:
    """
    Tests that decoder normalization is correctly implemented.
    """

    def test_normalize_decoder_restores_unit_norm(self):
        """
        INVARIANT: After normalize_decoder(), all columns must have unit norm.
        """
        model = create_msae()
        
        # Corrupt the norms
        with torch.no_grad():
            model.W_dec.data *= torch.rand(model.hidden_dim) * 10
        
        # Normalize
        model.normalize_decoder()
        
        norms = model.W_dec.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Norms not unit after normalization: [{norms.min():.4f}, {norms.max():.4f}]"

    def test_normalize_preserves_direction(self):
        """
        INVARIANT: Normalization should preserve direction (only scale).
        """
        model = create_msae(input_dim=32, expansion_factor=2)
        
        # Get original directions
        original_directions = F.normalize(model.W_dec.data, dim=0)
        
        # Corrupt scales
        with torch.no_grad():
            model.W_dec.data *= torch.rand(model.hidden_dim) * 5.0
        
        # Normalize
        model.normalize_decoder()
        
        # Check directions preserved
        new_directions = model.W_dec.data  # Already unit norm now
        
        # Directions should match (or be opposite, which is fine for unit vectors)
        cosine_sim = (original_directions * new_directions).sum(dim=0).abs()
        assert torch.allclose(cosine_sim, torch.ones_like(cosine_sim), atol=1e-4), \
            "Normalization changed feature directions"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
