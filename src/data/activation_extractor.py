"""
Activation extractor for Leela Zero.

Extracts intermediate activations from residual blocks for SAE training.
Designed for memory-efficient streaming on Colab.

Reference: docs/msae_paper.md
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from tqdm import tqdm

from ..models.leela_zero import LeelaZero


class ActivationExtractor:
    """
    Extract and save activations from Leela Zero residual blocks.

    Designed for memory efficiency:
    - Processes in batches
    - Saves chunks immediately to disk
    - Flattens spatial dimensions for SAE training
    """

    def __init__(
        self,
        model: LeelaZero,
        block_indices: List[int] = [5, 20, 35],
        output_dir: str = 'outputs/data/activations',
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Loaded LeelaZero model
            block_indices: Which residual blocks to extract (0-indexed)
            output_dir: Where to save activation chunks
            device: Device to run on
        """
        self.model = model
        self.block_indices = block_indices
        self.output_dir = Path(output_dir)
        self.device = device or next(model.parameters()).device

        # Create output directories
        for idx in block_indices:
            (self.output_dir / f'block{idx}').mkdir(parents=True, exist_ok=True)

        self.model.eval()

    def extract_batch(
        self,
        inputs: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations for a single batch.

        Args:
            inputs: Tensor of shape (batch, 18, 19, 19)

        Returns:
            Dict mapping block_idx -> activations (batch, 256, 19, 19)
        """
        inputs = inputs.to(self.device)

        with torch.no_grad():
            _, activations = self.model.forward_with_activations(
                inputs, self.block_indices
            )

        return {k: v.cpu() for k, v in activations.items()}

    def flatten_spatial(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Flatten spatial dimensions for SAE training.

        (batch, 256, 19, 19) -> (batch * 361, 256)
        """
        batch, channels, h, w = activations.shape
        # Permute to (batch, h, w, channels) then flatten
        return activations.permute(0, 2, 3, 1).reshape(-1, channels)

    def pool_spatial(self, activations: torch.Tensor, mode: str = 'mean') -> torch.Tensor:
        """
        Pool spatial dimensions for board-level features.

        (batch, 256, 19, 19) -> (batch, 256)

        Args:
            activations: Tensor of shape (batch, 256, 19, 19)
            mode: Pooling mode - 'mean' or 'max'

        Returns:
            Tensor of shape (batch, 256)
        """
        if mode == 'mean':
            return activations.mean(dim=(-2, -1))
        elif mode == 'max':
            return activations.amax(dim=(-2, -1))
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    def extract_and_save(
        self,
        dataloader: Iterator[torch.Tensor],
        chunk_size: int = 10000,
        max_samples: Optional[int] = None,
        spatial_mode: str = 'pool',
        pool_mode: str = 'mean'
    ) -> Dict[int, Dict[str, any]]:
        """
        Extract activations from dataloader and save to disk in chunks.

        Args:
            dataloader: Iterator yielding input tensors (batch, 18, 19, 19)
            chunk_size: Number of samples per saved chunk
            max_samples: Maximum total samples to extract (None for all)
            spatial_mode: How to handle spatial dimensions:
                - 'pool': Pool to board-level features (batch, 256) - RECOMMENDED
                - 'flatten': Flatten to per-position features (batch*361, 256)
            pool_mode: Pooling mode if spatial_mode='pool' ('mean' or 'max')

        Returns:
            Stats dict per block
        """
        # Buffers for accumulating before save
        buffers = {idx: [] for idx in self.block_indices}
        # Track buffer sizes incrementally (optimization: avoid O(n) sum each batch)
        buffer_sizes = {idx: 0 for idx in self.block_indices}
        chunk_counts = {idx: 0 for idx in self.block_indices}
        total_samples = {idx: 0 for idx in self.block_indices}

        def save_chunk(block_idx: int):
            """Save accumulated buffer to disk."""
            if not buffers[block_idx]:
                return

            chunk_data = torch.cat(buffers[block_idx], dim=0).numpy()
            chunk_path = self.output_dir / f'block{block_idx}' / f'chunk_{chunk_counts[block_idx]:04d}.npy'
            np.save(chunk_path, chunk_data)

            total_samples[block_idx] += len(chunk_data)
            chunk_counts[block_idx] += 1
            buffers[block_idx] = []
            buffer_sizes[block_idx] = 0  # Reset buffer size counter

            # Clear memory
            del chunk_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar = tqdm(dataloader, desc="Extracting activations")

        for batch in pbar:
            # Handle different input formats
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            # Extract activations
            activations = self.extract_batch(inputs)

            for block_idx, acts in activations.items():
                # Process spatial dimensions based on mode
                if spatial_mode == 'pool':
                    # Pool to board-level: (batch, 256, 19, 19) -> (batch, 256)
                    processed = self.pool_spatial(acts, mode=pool_mode)
                elif spatial_mode == 'flatten':
                    # Flatten spatial: (batch, 256, 19, 19) -> (batch*361, 256)
                    processed = self.flatten_spatial(acts)
                else:
                    raise ValueError(f"Unknown spatial_mode: {spatial_mode}")

                buffers[block_idx].append(processed)
                # Incremental size tracking (O(1) instead of O(n))
                buffer_sizes[block_idx] += processed.shape[0]

                # Check if buffer is large enough to save
                if buffer_sizes[block_idx] >= chunk_size:
                    save_chunk(block_idx)

            # Check max samples (use tracked total, not min())
            if max_samples and total_samples.get(self.block_indices[0], 0) >= max_samples:
                break

            pbar.set_postfix({
                'samples': total_samples.get(self.block_indices[0], 0)
            })

        # Save remaining buffers
        for block_idx in self.block_indices:
            save_chunk(block_idx)

        return {
            idx: {
                'total_samples': total_samples[idx],
                'num_chunks': chunk_counts[idx],
                'output_dir': str(self.output_dir / f'block{idx}')
            }
            for idx in self.block_indices
        }


def load_activation_chunks(
    block_dir: str,
    max_chunks: Optional[int] = None
) -> Iterator[np.ndarray]:
    """
    Load activation chunks from disk as iterator.

    Args:
        block_dir: Directory containing chunk_*.npy files
        max_chunks: Maximum chunks to load

    Yields:
        Numpy arrays of shape (chunk_size, 256)
    """
    block_dir = Path(block_dir)
    chunk_files = sorted(block_dir.glob('chunk_*.npy'))

    if max_chunks:
        chunk_files = chunk_files[:max_chunks]

    for chunk_file in chunk_files:
        yield np.load(chunk_file)


def compute_activation_stats(
    block_dir: str,
    max_chunks: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compute mean and std of activations using streaming algorithm.

    Args:
        block_dir: Directory with activation chunks
        max_chunks: Max chunks to use

    Returns:
        Dict with 'mean' and 'std' arrays of shape (256,)
    """
    from ..utils.streaming_stats import StreamingStats

    stats = StreamingStats(n_features=256)

    for chunk in load_activation_chunks(block_dir, max_chunks):
        stats.update_batch(chunk)

    return {
        'mean': stats.mean,
        'std': stats.std,
        'count': stats.count
    }
