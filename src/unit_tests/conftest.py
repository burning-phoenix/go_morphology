"""
Pytest fixtures and configuration for GO_MSAE unit tests.

Provides reusable fixtures for models, sample data, and test utilities.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))


# ============== Device Fixtures ==============

@pytest.fixture
def device():
    """Get available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@pytest.fixture
def cpu_device():
    """Force CPU device for deterministic tests."""
    return torch.device('cpu')


# ============== Model Fixtures ==============

@pytest.fixture
def msae_model():
    """Create a small MSAE for testing."""
    from models.msae import MatryoshkaSAE
    return MatryoshkaSAE(
        input_dim=64,  # Smaller for faster tests
        hidden_dim=256,
        k_levels=[8, 16, 32, 64],
    )


@pytest.fixture
def msae_model_full():
    """Create full-size MSAE matching production config."""
    from models.msae import MatryoshkaSAE
    return MatryoshkaSAE(
        input_dim=256,
        hidden_dim=4096,
        k_levels=[16, 32, 64, 128],
    )


@pytest.fixture
def baseline_sae_model():
    """Create a baseline SAE for testing."""
    from models.baseline_sae import BaselineSAE
    return BaselineSAE(
        input_dim=64,
        hidden_dim=256,
        k=32,
    )


# ============== Data Fixtures ==============

@pytest.fixture
def sample_activations():
    """Sample activations tensor."""
    torch.manual_seed(42)
    return torch.randn(100, 64)


@pytest.fixture
def sample_activations_large():
    """Larger sample for statistical tests."""
    torch.manual_seed(42)
    return torch.randn(1000, 256)


@pytest.fixture
def sample_board():
    """Sample Go board state."""
    board = np.zeros((19, 19), dtype=np.int32)
    # Add some stones
    board[3, 3] = 1  # Black
    board[3, 4] = 1
    board[4, 3] = 1
    board[15, 15] = 2  # White
    board[15, 16] = 2
    return board


@pytest.fixture
def sample_board_with_groups():
    """Board with multiple distinct groups."""
    board = np.zeros((19, 19), dtype=np.int32)
    # Black group 1 (top-left)
    board[0, 0] = 1
    board[0, 1] = 1
    board[1, 0] = 1
    
    # Black group 2 (center, disconnected)
    board[9, 9] = 1
    board[9, 10] = 1
    
    # White group (bottom-right)
    board[18, 18] = 2
    board[18, 17] = 2
    board[17, 18] = 2
    
    return board


@pytest.fixture
def sample_encoded_position():
    """Sample 18-plane encoded position."""
    torch.manual_seed(42)
    # Simulate encoded position
    planes = torch.zeros(18, 19, 19)
    planes[0, 3, 3] = 1  # Black stone
    planes[8, 15, 15] = 1  # White stone
    planes[16] = 1  # Black to play
    return planes


# ============== Label Fixtures ==============

@pytest.fixture
def sample_concept_labels():
    """Sample concept labels for probe testing."""
    np.random.seed(42)
    n_samples = 1000
    return {
        'is_edge': np.random.randint(0, 2, n_samples).astype(bool),
        'stone_black': np.random.randint(0, 2, n_samples).astype(bool),
        'is_atari': np.random.randint(0, 2, n_samples).astype(bool),
    }


@pytest.fixture
def sample_position_ids():
    """Sample position IDs for split testing."""
    # 100 positions, each with 361 points
    return np.repeat(np.arange(100), 361)


# ============== Utility Functions ==============

def assert_tensor_close(a, b, atol=1e-5, rtol=1e-5):
    """Assert two tensors are close, with helpful error message."""
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise AssertionError(
            f"Tensors not close. Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
        )


def count_non_zeros(tensor, dim=-1):
    """Count non-zero elements along dimension."""
    return (tensor != 0).sum(dim=dim)
