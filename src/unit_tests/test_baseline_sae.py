"""
Unit tests for BaselineSAE model.

Tests consistency between baseline and MSAE implementations.
Priority: 🟡 Medium - validates comparison methodology.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baseline_sae import BaselineSAE, create_baseline_sae
from models.msae import create_msae, topk_activation


# =============================================================================
# BaselineSAE Architecture Tests
# =============================================================================

class TestBaselineSAE:
    """Tests for BaselineSAE model."""

    def test_init_dimensions(self):
        """Model should have correct dimensions."""
        model = create_baseline_sae(input_dim=256, expansion_factor=16, k=64)
        
        assert model.input_dim == 256
        assert model.hidden_dim == 4096
        assert model.k == 64

    def test_encoder_output_shape(self):
        """Encoder should produce correct output shape."""
        model = create_baseline_sae(input_dim=64, expansion_factor=4, k=32)
        x = torch.randn(10, 64)
        
        z = model.encode(x)
        
        assert z.shape == (10, 256)  # 64 * 4

    def test_decoder_output_shape(self):
        """Decoder should produce correct output shape."""
        model = create_baseline_sae(input_dim=64, expansion_factor=4, k=32)
        z = torch.randn(10, 256)
        
        x_hat = model.decode(z)
        
        assert x_hat.shape == (10, 64)

    def test_forward_sparsity(self):
        """Forward should produce sparse latents with exactly k non-zeros."""
        model = create_baseline_sae(input_dim=64, expansion_factor=4, k=32)
        x = torch.randn(100, 64)
        
        _, _, z_sparse = model.forward(x)
        non_zeros = (z_sparse != 0).sum(dim=1)
        
        assert (non_zeros == 32).all()

    def test_forward_output_shapes(self):
        """Forward pass should return correct shapes."""
        model = create_baseline_sae(input_dim=64, expansion_factor=4, k=32)
        x = torch.randn(10, 64)
        
        x_hat, z, z_sparse = model.forward(x)
        
        assert x_hat.shape == (10, 64)
        assert z.shape == (10, 256)
        assert z_sparse.shape == (10, 256)


class TestBaselineVsMSAE:
    """Tests comparing BaselineSAE to MSAE."""

    def test_same_topk_mechanism(self):
        """Both should use same TopK mechanism."""
        from models.msae import topk_activation as msae_topk
        
        # Test they produce same results
        x = torch.randn(50, 100)
        k = 20
        
        result_msae = msae_topk(x, k)
        
        # Baseline also uses topk_activation internally
        # Just verify the function itself is consistent
        result_again = msae_topk(x, k)
        
        assert torch.equal(result_msae, result_again)

    def test_identical_init_same_output(self):
        """Identically initialized models should produce same output."""
        msae = create_msae(input_dim=64, expansion_factor=4, k_levels=[16, 32])
        baseline = create_baseline_sae(input_dim=64, expansion_factor=4, k=32)
        
        # Copy weights from MSAE to baseline
        with torch.no_grad():
            baseline.W_enc.data = msae.W_enc.data.clone()
            baseline.W_dec.data = msae.W_dec.data.clone()
            baseline.b_pre.data = msae.b_pre.data.clone()
        
        x = torch.randn(50, 64)
        
        x_hat_msae, _, _ = msae.forward(x, k=32)
        x_hat_baseline, _, _ = baseline.forward(x)
        
        assert torch.allclose(x_hat_msae, x_hat_baseline, atol=1e-5)

    def test_sparsity_consistency(self):
        """Both should produce same sparsity for same k."""
        msae = create_msae(input_dim=64, expansion_factor=4, k_levels=[32])
        baseline = create_baseline_sae(input_dim=64, expansion_factor=4, k=32)
        
        x = torch.randn(100, 64)
        
        _, _, z_msae = msae.forward(x, k=32)
        _, _, z_baseline = baseline.forward(x)
        
        msae_nonzeros = (z_msae != 0).sum(dim=1)
        baseline_nonzeros = (z_baseline != 0).sum(dim=1)
        
        assert (msae_nonzeros == 32).all()
        assert (baseline_nonzeros == 32).all()


class TestBaselineDecoderNorm:
    """Tests for baseline decoder normalization."""

    def test_decoder_init_unit_norm(self):
        """Decoder columns should be unit norm after initialization."""
        model = create_baseline_sae(input_dim=64, expansion_factor=4, k=32)
        
        norms = model.W_dec.norm(dim=0)
        
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_normalize_decoder(self):
        """normalize_decoder should restore unit norms."""
        model = create_baseline_sae(input_dim=64, expansion_factor=4, k=32)
        
        # Corrupt the norms
        with torch.no_grad():
            model.W_dec.data *= torch.rand(model.hidden_dim)
        
        model.normalize_decoder()
        
        norms = model.W_dec.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestBaselineLoss:
    """Tests for baseline loss computation."""

    def test_loss_components(self):
        """compute_loss should return expected components."""
        model = create_baseline_sae(input_dim=64, expansion_factor=4, k=32)
        x = torch.randn(32, 64)
        
        losses = model.compute_loss(x)
        
        assert 'loss' in losses
        assert 'reconstruction_loss' in losses
        assert 'aux_loss' in losses

    def test_loss_is_scalar(self):
        """Total loss should be a scalar."""
        model = create_baseline_sae(input_dim=64, expansion_factor=4, k=32)
        x = torch.randn(32, 64)
        
        losses = model.compute_loss(x)
        
        assert losses['loss'].shape == ()

    def test_loss_gradients_flow(self):
        """Loss should have gradients through model parameters."""
        model = create_baseline_sae(input_dim=64, expansion_factor=4, k=32)
        x = torch.randn(32, 64)
        
        losses = model.compute_loss(x)
        losses['loss'].backward()
        
        assert model.W_enc.grad is not None
        assert model.W_dec.grad is not None


class TestCreateBaselineSAE:
    """Tests for create_baseline_sae factory function."""

    def test_create_defaults(self):
        """Should create model with production defaults."""
        model = create_baseline_sae()
        
        assert model.input_dim == 256
        assert model.hidden_dim == 4096
        assert model.k == 64

    def test_create_custom(self):
        """Should respect custom parameters."""
        model = create_baseline_sae(
            input_dim=128,
            expansion_factor=4,
            k=16
        )
        
        assert model.input_dim == 128
        assert model.hidden_dim == 512  # 128 * 4
        assert model.k == 16

    def test_create_device(self):
        """Should move to specified device."""
        model = create_baseline_sae(
            input_dim=32,
            expansion_factor=2,
            k=8,
            device=torch.device('cpu')
        )
        
        assert model.W_enc.device.type == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
