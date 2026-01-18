"""
Leela Zero network implementation for activation extraction.

This module provides a PyTorch implementation of the Leela Zero neural network
architecture, designed for extracting intermediate activations for SAE training.

Architecture:
- 40 residual blocks with 256 channels
- Input: 18 planes × 19 × 19 (Go board encoding)
- Policy head: 362 outputs (361 board positions + pass)
- Value head: 1 output (win probability)

Reference: docs/leela_zero_README.md
"""

import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2d + BatchNorm + optional ReLU.

    Leela Zero uses a specific batch norm convention where gamma=1 (fixed)
    and beta is learned. The bias term encodes beta * sqrt(var + eps).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        relu: bool = True
    ):
        super().__init__()
        assert kernel_size in (1, 3), "Only 1x1 and 3x3 kernels supported"

        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, bias=False
        )
        # BatchNorm without affine (gamma/beta handled separately)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        # Learnable beta parameter for batch norm
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.relu = relu

        # Kaiming initialization
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        # Add beta (reshaped for broadcasting)
        x = x + self.beta.view(1, -1, 1, 1)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class ResBlock(nn.Module):
    """Residual block with two convolutional layers and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, relu=True)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return F.relu(out, inplace=True)


class LeelaZero(nn.Module):
    """
    Leela Zero neural network for Go.

    Args:
        board_size: Size of Go board (default 19)
        in_channels: Number of input planes (default 18)
        residual_channels: Number of channels in residual tower (default 256)
        residual_blocks: Number of residual blocks (default 40)
    """

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 18,
        residual_channels: int = 256,
        residual_blocks: int = 40
    ):
        super().__init__()
        self.board_size = board_size
        self.in_channels = in_channels
        self.residual_channels = residual_channels
        self.residual_blocks = residual_blocks

        # Input convolution
        self.conv_input = ConvBlock(in_channels, residual_channels, kernel_size=3)

        # Residual tower - use ModuleList for individual block access
        self.residual_tower = nn.ModuleList([
            ResBlock(residual_channels) for _ in range(residual_blocks)
        ])

        # Policy head
        self.policy_conv = ConvBlock(residual_channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

        # Value head
        self.value_conv = ConvBlock(residual_channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Activation storage for extraction
        self._activations: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 18, 19, 19)

        Returns:
            Tuple of (policy_logits, value) where:
                - policy_logits: (batch, 362) raw logits for move probabilities
                - value: (batch, 1) win probability in [-1, 1]
        """
        # Input convolution
        x = self.conv_input(x)

        # Residual tower
        for block in self.residual_tower:
            x = block(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = policy.flatten(start_dim=1)
        policy = self.policy_fc(policy)

        # Value head
        value = self.value_conv(x)
        value = value.flatten(start_dim=1)
        value = F.relu(self.value_fc1(value), inplace=True)
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def forward_with_activations(
        self,
        x: torch.Tensor,
        block_indices: List[int]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Forward pass that also returns intermediate activations.

        Args:
            x: Input tensor of shape (batch, 18, 19, 19)
            block_indices: List of residual block indices to extract (0-indexed)

        Returns:
            Tuple of (outputs, activations) where:
                - outputs: (policy, value) tuple from forward()
                - activations: Dict mapping block_idx -> activation tensor
                  Each activation has shape (batch, 256, 19, 19)
        """
        activations = {}

        # Input convolution
        x = self.conv_input(x)

        # Residual tower with activation capture
        for idx, block in enumerate(self.residual_tower):
            x = block(x)
            if idx in block_indices:
                activations[idx] = x.clone()

        # Policy head
        policy = self.policy_conv(x)
        policy = policy.flatten(start_dim=1)
        policy = self.policy_fc(policy)

        # Value head
        value = self.value_conv(x)
        value = value.flatten(start_dim=1)
        value = F.relu(self.value_fc1(value), inplace=True)
        value = torch.tanh(self.value_fc2(value))

        return (policy, value), activations

    def register_activation_hooks(self, block_indices: List[int]) -> None:
        """
        Register forward hooks to capture activations at specified blocks.

        Args:
            block_indices: List of residual block indices to capture
        """
        self.clear_hooks()
        self._activations.clear()

        for idx in block_indices:
            if 0 <= idx < len(self.residual_tower):
                hook = self.residual_tower[idx].register_forward_hook(
                    self._make_hook(idx)
                )
                self._hooks.append(hook)

    def _make_hook(self, block_idx: int):
        """Create a hook function for a specific block."""
        def hook(module, input, output):
            self._activations[block_idx] = output.detach()
        return hook

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Get captured activations from last forward pass."""
        return self._activations.copy()

    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def load_leela_weights(
    model: LeelaZero,
    weights_path: str,
    device: Optional[torch.device] = None
) -> LeelaZero:
    """
    Load Leela Zero weights from a text file.

    The weights file format:
    - Line 1: Version number
    - Conv layers: 2 lines (weights, bias)
    - BatchNorm layers: 2 lines (means, variances)
    - FC layers: 2 lines (weights, bias)

    Args:
        model: LeelaZero model instance
        weights_path: Path to weights file (.txt or .txt.gz)
        device: Device to load weights to

    Returns:
        Model with loaded weights
    """
    weights_path = Path(weights_path)

    # Handle gzipped files
    if weights_path.suffix == '.gz':
        open_fn = lambda p: gzip.open(p, 'rt')
    else:
        open_fn = lambda p: open(p, 'r')

    with open_fn(weights_path) as f:
        lines = [line.strip() for line in f.readlines()]

    # Skip version line
    line_idx = 1

    def read_weights(shape: Tuple[int, ...]) -> torch.Tensor:
        nonlocal line_idx
        values = [float(x) for x in lines[line_idx].split()]
        line_idx += 1
        return torch.tensor(values, dtype=torch.float32).view(shape)

    def load_conv_block(conv_block: ConvBlock) -> None:
        nonlocal line_idx

        # Conv weights: [out, in, h, w]
        out_ch = conv_block.conv.out_channels
        in_ch = conv_block.conv.in_channels
        k = conv_block.conv.kernel_size[0]
        conv_block.conv.weight.data = read_weights((out_ch, in_ch, k, k))

        # Bias (encodes beta * sqrt(var + eps), but we store as bias)
        bias = read_weights((out_ch,))

        # BatchNorm running mean
        conv_block.bn.running_mean = read_weights((out_ch,))

        # BatchNorm running variance
        conv_block.bn.running_var = read_weights((out_ch,))

        # Recover beta from bias: beta = bias / sqrt(var + eps)
        conv_block.beta.data = bias / torch.sqrt(
            conv_block.bn.running_var + conv_block.bn.eps
        )

    def load_fc(fc: nn.Linear) -> None:
        nonlocal line_idx
        out_features = fc.out_features
        in_features = fc.in_features
        fc.weight.data = read_weights((out_features, in_features))
        fc.bias.data = read_weights((out_features,))

    # Load input convolution
    load_conv_block(model.conv_input)

    # Load residual tower
    for res_block in model.residual_tower:
        load_conv_block(res_block.conv1)
        load_conv_block(res_block.conv2)

    # Load policy head
    load_conv_block(model.policy_conv)
    load_fc(model.policy_fc)

    # Load value head
    load_conv_block(model.value_conv)
    load_fc(model.value_fc1)
    load_fc(model.value_fc2)

    if device is not None:
        model = model.to(device)

    model.eval()
    return model


def download_leela_weights(
    output_path: str = 'outputs/data/leela_weights.txt.gz',
    url: str = 'https://zero.sjeng.org/best-network'
) -> str:
    """
    Download Leela Zero weights from the official source.

    Args:
        output_path: Where to save the weights file
        url: URL to download from (redirects to actual weights)

    Returns:
        Path to downloaded file
    """
    import urllib.request

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Leela Zero weights from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")

    return str(output_path)


# Convenience function for creating and loading model
def create_leela_zero(
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> LeelaZero:
    """
    Create a Leela Zero model, optionally loading weights.

    Args:
        weights_path: Path to weights file (optional)
        device: Device to use
        **kwargs: Additional arguments for LeelaZero constructor

    Returns:
        LeelaZero model instance
    """
    model = LeelaZero(**kwargs)

    if weights_path is not None:
        model = load_leela_weights(model, weights_path, device)
    elif device is not None:
        model = model.to(device)

    return model
