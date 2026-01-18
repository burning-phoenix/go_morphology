"""
Baseline Single-k Sparse Autoencoder for comparison with MSAE.

Based on "Scaling and Evaluating Sparse Autoencoders" (Gao et al.)
- docs/topk_sae_paper.md

Key differences from MSAE:
- Fixed single k value (k=64) instead of hierarchical k levels
- No multi-level reconstruction loss
- Simpler forward pass

Shared with MSAE:
- TopK sparsity (not L1 penalty)
- Tied initialization (encoder = decoder.T)
- Unit-norm decoder columns
- Auxiliary loss for dead latent prevention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

# Import TopK from msae module to avoid duplication
from .msae import topk_activation


class BaselineSAE(nn.Module):
    """
    Single-k Sparse Autoencoder baseline.

    Architecture (from Gao et al.):
        z = TopK(W_enc @ (x - b_pre))
        x_hat = W_dec @ z + b_pre

    Training:
        Loss = MSE(x, x_hat) + aux_loss

    Args:
        input_dim: Dimension of input activations (e.g., 256 for Leela Zero)
        hidden_dim: Number of latent features (e.g., 4096 = 16x expansion)
        k: Fixed sparsity level (default: 64)
        aux_k: Number of dead latents to use in auxiliary loss
        dead_threshold: Activation frequency below this = dead
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 4096,
        k: int = 64,
        aux_k: int = 512,
        dead_threshold: float = 1e-3,  # 0.1% activation rate per protocol
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.aux_k = aux_k
        self.dead_threshold = dead_threshold

        # Pre-encoder bias (for centering)
        self.b_pre = nn.Parameter(torch.zeros(input_dim))

        # Encoder: Linear (ReLU applied via TopK which ensures positive values)
        self.W_enc = nn.Parameter(torch.empty(hidden_dim, input_dim))

        # Decoder: Linear (no bias, uses b_pre)
        self.W_dec = nn.Parameter(torch.empty(input_dim, hidden_dim))

        # Initialize with tied weights (encoder = decoder.T)
        self._init_weights()

        # Track activation frequencies for dead latent detection
        self.register_buffer('activation_counts', torch.zeros(hidden_dim))
        self.register_buffer('total_samples', torch.tensor(0.0))

    def _init_weights(self):
        """
        Initialize with tied weights and unit-norm decoder columns.

        From Gao et al.: "we initialize the encoder to the transpose of the decoder"
        """
        # Initialize decoder with random unit-norm columns
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

        # Initialize encoder as transpose of decoder (tied init)
        with torch.no_grad():
            self.W_enc.data = self.W_dec.data.T.clone()

    def normalize_decoder(self):
        """Normalize decoder columns to unit norm after each update."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to pre-activation latent space.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Pre-activation latents (batch, hidden_dim)
        """
        # Center input
        x_centered = x - self.b_pre

        # Encode with ReLU to ensure non-negative activations
        # (TopK will select from these positive values)
        z = F.relu(x_centered @ self.W_enc.T)

        return z

    def decode(self, z_sparse: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse latents to reconstruction.

        Args:
            z_sparse: Sparse latent tensor (batch, hidden_dim)

        Returns:
            Reconstruction (batch, input_dim)
        """
        return z_sparse @ self.W_dec.T + self.b_pre

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with TopK sparsity.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            (reconstruction, latents, sparse_latents)
        """
        # Encode
        z = self.encode(x)

        # Apply TopK sparsity
        z_sparse = topk_activation(z, self.k)

        # Decode
        x_hat = self.decode(z_sparse)

        return x_hat, z, z_sparse

    def compute_loss(
        self,
        x: torch.Tensor,
        update_stats: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SAE loss.

        Loss = MSE(x, x_hat) + aux_loss

        Args:
            x: Input tensor (batch, input_dim)
            update_stats: Whether to update activation statistics

        Returns:
            Dict with 'loss', 'reconstruction_loss', 'aux_loss'
        """
        # Forward pass
        x_hat, z, z_sparse = self.forward(x)

        # Update activation statistics
        if update_stats and self.training:
            with torch.no_grad():
                # Count activations > 0
                active = (z > 0).float().sum(dim=0)
                self.activation_counts += active
                self.total_samples += x.shape[0]

        # Reconstruction loss
        rec_loss = F.mse_loss(x_hat, x)

        # Auxiliary loss for dead latents
        aux_loss = self._compute_aux_loss(x, z)

        # Total loss (1/32 coefficient for aux_loss per Gao et al.)
        total_loss = rec_loss + 0.03125 * aux_loss

        return {
            'loss': total_loss,
            'reconstruction_loss': rec_loss,
            'aux_loss': aux_loss,
        }

    def _compute_aux_loss(
        self,
        x: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Auxiliary loss using dead latents to reconstruct residual error.

        From Gao et al.: "auxiliary loss that models reconstruction error
        using the top-k_aux dead latents"
        """
        if self.total_samples < 1000:
            # Not enough samples to identify dead latents
            return torch.tensor(0.0, device=x.device)

        # Identify dead latents (activation frequency below threshold)
        activation_freq = self.activation_counts / (self.total_samples + 1e-8)
        dead_mask = activation_freq < self.dead_threshold

        if not dead_mask.any():
            return torch.tensor(0.0, device=x.device)

        # Get activations of dead latents only
        z_dead = z.clone()
        z_dead[:, ~dead_mask] = -float('inf')  # Mask out alive latents

        # Select top-k_aux from dead latents
        k_aux = min(self.aux_k, dead_mask.sum().item())
        if k_aux == 0:
            return torch.tensor(0.0, device=x.device)

        z_dead_sparse = topk_activation(z_dead, k_aux)

        # Compute residual (what wasn't reconstructed by main path)
        with torch.no_grad():
            z_main = topk_activation(z, self.k)
            x_hat_main = self.decode(z_main)
            residual = x - x_hat_main  # This is the error e = x - x_hat

        # Try to reconstruct residual using dead latents
        # e_hat = decode(z_dead_sparse)
        x_hat_aux = self.decode(z_dead_sparse)

        # Aux loss per Gao et al.: L_aux = ||e - e_hat||^2
        # Train dead latents to predict the residual error
        aux_loss = F.mse_loss(x_hat_aux, residual)

        return aux_loss

    @property
    def dead_latent_ratio(self) -> float:
        """Fraction of latents that are dead."""
        if self.total_samples < 1000:
            return 0.0
        activation_freq = self.activation_counts / (self.total_samples + 1e-8)
        return (activation_freq < self.dead_threshold).float().mean().item()

    def get_reconstruction_r2(self, x: torch.Tensor) -> float:
        """
        Compute R² (explained variance) for reconstruction.

        R² = 1 - MSE(x, x_hat) / Var(x)
        """
        with torch.no_grad():
            x_hat, _, _ = self.forward(x)
            mse = F.mse_loss(x_hat, x)
            var = x.var()
            r2 = 1.0 - mse / (var + 1e-8)
            return r2.item()

    def reset_dead_latent_stats(self):
        """Reset activation tracking statistics."""
        self.activation_counts.zero_()
        self.total_samples.zero_()


def create_baseline_sae(
    input_dim: int = 256,
    expansion_factor: int = 16,
    k: int = 64,
    device: Optional[torch.device] = None
) -> BaselineSAE:
    """
    Create baseline SAE with standard configuration.

    Args:
        input_dim: Input dimension (256 for Leela Zero)
        expansion_factor: Hidden dim = input_dim * expansion_factor
        k: Fixed sparsity level
        device: Device to create model on

    Returns:
        BaselineSAE instance
    """
    hidden_dim = input_dim * expansion_factor

    model = BaselineSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        k=k,
    )

    if device is not None:
        model = model.to(device)

    return model
