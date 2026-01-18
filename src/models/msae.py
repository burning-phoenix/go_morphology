"""
Matryoshka Sparse Autoencoder (MSAE) implementation.

Based on:
- "Matryoshka SAE" (Wooldridge et al.) - docs/msae_paper.md
- "Scaling and Evaluating Sparse Autoencoders" (Gao et al.) - docs/topk_sae_paper.md
- "HierarchicalTopK" (Balagansky et al.) - docs/multi_budget_sae.md

Key features:
- TopK sparsity (not L1 penalty) to avoid activation shrinkage
- Hierarchical loss across multiple k levels [16, 32, 64, 128]
- Dead latent prevention via tied initialization + auxiliary loss
- Unit-norm decoder columns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class TopK(torch.autograd.Function):
    """
    TopK activation with straight-through estimator for gradients.

    Forward: Keep only top-k values, zero the rest
    Backward: Gradient flows through as if identity on selected features

    Reference: Gao et al. "Scaling and Evaluating Sparse Autoencoders"
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int) -> torch.Tensor:
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(x, k, dim=-1)

        # Create sparse output
        output = torch.zeros_like(x)
        output.scatter_(-1, topk_indices, topk_values)

        # Save mask for backward
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        ctx.save_for_backward(mask)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mask, = ctx.saved_tensors
        # Straight-through: gradient flows only through selected features
        grad_input = grad_output * mask.float()
        return grad_input, None


def topk_activation(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply TopK activation with straight-through gradient."""
    return TopK.apply(x, k)


class MatryoshkaSAE(nn.Module):
    """
    Matryoshka Sparse Autoencoder with hierarchical TopK sparsity.

    Architecture (from TopK SAE paper):
        z = ReLU(W_enc @ (x - b_pre))
        z_sparse = TopK(z, k)
        x_hat = W_dec @ z_sparse + b_pre

    Training (from MSAE/HierarchicalTopK papers):
        Loss = (1/|K|) * Σ_k MSE(x, decode(topk(z, k))) + aux_loss

    Args:
        input_dim: Dimension of input activations (e.g., 256 for Leela Zero)
        hidden_dim: Number of latent features (e.g., 4096 = 16x expansion)
        k_levels: List of k values for hierarchical training [16, 32, 64, 128]
        weighting: 'uniform' or 'reverse' for loss weighting
        aux_k: Number of dead latents to use in auxiliary loss
        dead_threshold: Activation frequency below this = dead
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 4096,
        k_levels: List[int] = [16, 32, 64, 128],
        weighting: str = 'uniform',
        aux_k: int = 512,
        dead_threshold: float = 1e-3,  # 0.1% activation rate per protocol
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k_levels = sorted(k_levels)
        self.k_max = max(k_levels)
        self.weighting = weighting
        self.aux_k = aux_k
        self.dead_threshold = dead_threshold

        # Pre-encoder bias (for centering)
        self.b_pre = nn.Parameter(torch.zeros(input_dim))

        # Encoder: Linear + ReLU (ReLU applied in forward)
        self.W_enc = nn.Parameter(torch.empty(hidden_dim, input_dim))

        # Decoder: Linear (no bias, uses b_pre)
        self.W_dec = nn.Parameter(torch.empty(input_dim, hidden_dim))

        # Initialize with tied weights (encoder = decoder.T)
        # This prevents dead latents (Gao et al., Conerly et al.)
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
        Encode input to latent space.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Pre-activation latents (batch, hidden_dim)
        """
        # Center input
        x_centered = x - self.b_pre

        # Encode with ReLU
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
        x: torch.Tensor,
        k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with specified sparsity level.

        Args:
            x: Input tensor (batch, input_dim)
            k: Sparsity level (default: k_max)

        Returns:
            (reconstruction, latents, sparse_latents)
        """
        if k is None:
            k = self.k_max

        # Encode
        z = self.encode(x)

        # Apply TopK sparsity
        z_sparse = topk_activation(z, k)

        # Decode
        x_hat = self.decode(z_sparse)

        return x_hat, z, z_sparse

    def forward_hierarchical(
        self,
        x: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Forward pass returning reconstructions at all k levels.

        Efficient implementation using cumulative sum (from HierarchicalTopK paper).

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Dict mapping k -> reconstruction
        """
        # Encode once
        z = self.encode(x)

        # Get indices sorted by activation magnitude
        _, sorted_indices = torch.sort(z, dim=-1, descending=True)

        # Get sorted values
        sorted_z = torch.gather(z, -1, sorted_indices)

        reconstructions = {}

        for k in self.k_levels:
            # Create mask for top-k
            mask = torch.zeros_like(z)
            topk_indices = sorted_indices[..., :k]
            topk_values = sorted_z[..., :k]
            mask.scatter_(-1, topk_indices, topk_values)

            # Decode
            x_hat = self.decode(mask)
            reconstructions[k] = x_hat

        return reconstructions

    def compute_loss(
        self,
        x: torch.Tensor,
        update_stats: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical MSAE loss.

        Loss = (1/|K|) * Σ_k w_k * MSE(x, x_hat_k) + aux_loss

        Args:
            x: Input tensor (batch, input_dim)
            update_stats: Whether to update activation statistics

        Returns:
            Dict with 'loss', 'reconstruction_losses', 'aux_loss'
        """
        # Get reconstructions at all k levels
        reconstructions = self.forward_hierarchical(x)

        # Also get full encoding for aux loss and stats
        z = self.encode(x)

        # Update activation statistics
        if update_stats and self.training:
            with torch.no_grad():
                # Count activations > 0
                active = (z > 0).float().sum(dim=0)
                self.activation_counts += active
                self.total_samples += x.shape[0]

        # Compute reconstruction losses at each k level
        rec_losses = {}
        for k in self.k_levels:
            mse = F.mse_loss(reconstructions[k], x)
            rec_losses[k] = mse

        # Compute weighted sum based on weighting strategy
        if self.weighting == 'uniform':
            # Equal weight for all k levels
            weights = {k: 1.0 / len(self.k_levels) for k in self.k_levels}
        elif self.weighting == 'reverse':
            # Weight sparser reconstructions more heavily
            # w_k ∝ 1/k, then normalize
            raw_weights = {k: 1.0 / k for k in self.k_levels}
            total = sum(raw_weights.values())
            weights = {k: w / total for k, w in raw_weights.items()}
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")

        total_rec_loss = sum(weights[k] * rec_losses[k] for k in self.k_levels)

        # Auxiliary loss for dead latents (from Gao et al.)
        aux_loss = self._compute_aux_loss(x, z)

        # Total loss
        total_loss = total_rec_loss + 0.03125 * aux_loss  # 1/32 coefficient

        return {
            'loss': total_loss,
            'reconstruction_loss': total_rec_loss,
            'aux_loss': aux_loss,
            'reconstruction_losses': rec_losses,
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
            z_main = topk_activation(z, self.k_max)
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

    def get_reconstruction_r2(
        self,
        x: torch.Tensor,
        k: Optional[int] = None
    ) -> float:
        """
        Compute R² (explained variance) for reconstruction.

        R² = 1 - MSE(x, x_hat) / Var(x)
        """
        if k is None:
            k = self.k_max

        with torch.no_grad():
            x_hat, _, _ = self.forward(x, k)
            mse = F.mse_loss(x_hat, x)
            var = x.var()
            r2 = 1.0 - mse / (var + 1e-8)
            return r2.item()

    def reset_dead_latent_stats(self):
        """Reset activation tracking statistics."""
        self.activation_counts.zero_()
        self.total_samples.zero_()


def create_msae(
    input_dim: int = 256,
    expansion_factor: int = 16,
    k_levels: List[int] = [16, 32, 64, 128],
    weighting: str = 'uniform',
    device: Optional[torch.device] = None
) -> MatryoshkaSAE:
    """
    Create MSAE with standard configuration.

    Args:
        input_dim: Input dimension (256 for Leela Zero)
        expansion_factor: Hidden dim = input_dim * expansion_factor
        k_levels: Sparsity levels for hierarchical training
        weighting: 'uniform' or 'reverse'
        device: Device to create model on

    Returns:
        MatryoshkaSAE instance
    """
    hidden_dim = input_dim * expansion_factor

    model = MatryoshkaSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        k_levels=k_levels,
        weighting=weighting,
    )

    if device is not None:
        model = model.to(device)

    return model
