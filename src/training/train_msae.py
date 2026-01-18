"""
Training loop for Matryoshka SAE.

Based on:
- "Scaling and Evaluating Sparse Autoencoders" (Gao et al.)
- "Matryoshka SAE" (Wooldridge et al.)

Key training details from papers:
- Adam optimizer, lr=1e-4
- Large batch size (4096)
- Normalize decoder after each step
- Track dead latent ratio
- Checkpoint frequently for Colab sessions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, List, Optional, Callable
import json
import time
import numpy as np

from ..models.msae import MatryoshkaSAE
from ..models.baseline_sae import BaselineSAE
from ..utils.memory import clear_memory


class MSAETrainer:
    """
    Trainer for Matryoshka SAE.

    Handles:
    - Training loop with multi-level loss
    - Decoder normalization after each step
    - Dead latent tracking
    - Checkpointing for Colab
    - Validation and metrics
    """

    def __init__(
        self,
        model: MatryoshkaSAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 8e-4,  # Paper: 0.0008 from multi_budget_sae
        output_dir: str = 'outputs',
        device: Optional[torch.device] = None,
        checkpoint_every: int = 1,
        log_every: int = 100,
        # Optimizer config (from Gao et al. topk_sae_paper Appendix A)
        adam_betas: tuple = (0.9, 0.999),
        adam_eps: float = 6.25e-10,  # Very small for numerical stability at scale
    ):
        """
        Initialize trainer.

        Args:
            model: MatryoshkaSAE instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation (optional)
            lr: Learning rate (paper: 0.0008 for 4096 hidden dim)
            output_dir: Directory for checkpoints and logs
            device: Device to train on
            checkpoint_every: Save checkpoint every N epochs
            log_every: Log metrics every N batches
            adam_betas: Adam beta1, beta2 (paper: 0.9, 0.999)
            adam_eps: Adam epsilon (paper: 6.25e-10 for large scale)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.lr = lr  # Store for logging

        # Optimizer (Adam with settings from Gao et al. "Scaling and Evaluating SAEs")
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=adam_betas,
            eps=adam_eps,
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_rec_loss': [],
            'train_aux_loss': [],
            'val_loss': [],
            'val_r2': {},  # R² at each k level
            'dead_ratio': [],
        }

        # Ensure output directory exists
        (self.output_dir / 'models').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dict with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_aux_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            # Forward + loss
            self.optimizer.zero_grad()
            loss_dict = self.model.compute_loss(x, update_stats=True)

            # Backward
            loss_dict['loss'].backward()

            # Update
            self.optimizer.step()

            # Normalize decoder (critical for TopK SAE)
            self.model.normalize_decoder()

            # Accumulate metrics
            total_loss += loss_dict['loss'].item()
            total_rec_loss += loss_dict['reconstruction_loss'].item()
            total_aux_loss += loss_dict['aux_loss'].item()
            n_batches += 1

            # Log progress
            if (batch_idx + 1) % self.log_every == 0:
                avg_loss = total_loss / n_batches
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}: "
                      f"loss={avg_loss:.6f}, dead={self.model.dead_latent_ratio:.3f}")

        # Compute epoch averages
        metrics = {
            'loss': total_loss / n_batches,
            'rec_loss': total_rec_loss / n_batches,
            'aux_loss': total_aux_loss / n_batches,
            'dead_ratio': self.model.dead_latent_ratio,
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dict with validation metrics including R² at each k level
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Accumulators for R² calculation
        r2_numerators = {k: 0.0 for k in self.model.k_levels}
        r2_denominators = {k: 0.0 for k in self.model.k_levels}

        for batch in self.val_loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            # Compute loss
            loss_dict = self.model.compute_loss(x, update_stats=False)
            total_loss += loss_dict['loss'].item()

            # Compute R² at each k level
            reconstructions = self.model.forward_hierarchical(x)
            var_x = x.var() * x.numel()  # Total variance

            for k in self.model.k_levels:
                mse = ((reconstructions[k] - x) ** 2).sum()
                r2_numerators[k] += mse.item()
                r2_denominators[k] += var_x.item()

            n_batches += 1

        # Compute final metrics
        metrics = {
            'loss': total_loss / n_batches,
        }

        # R² for each k level
        for k in self.model.k_levels:
            r2 = 1.0 - r2_numerators[k] / (r2_denominators[k] + 1e-8)
            metrics[f'r2_k{k}'] = r2

        return metrics

    def train(
        self,
        epochs: int = 15,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, List]:
        """
        Full training loop.

        Args:
            epochs: Number of epochs
            early_stopping_patience: Stop if val loss doesn't improve for N epochs

        Returns:
            Training history
        """
        print(f"Training MSAE on {self.device}")
        print(f"  K levels: {self.model.k_levels}")
        print(f"  Hidden dim: {self.model.hidden_dim}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Epochs: {epochs}")
        print()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            print(f"Epoch {epoch}/{epochs}")
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_rec_loss'].append(train_metrics['rec_loss'])
            self.history['train_aux_loss'].append(train_metrics['aux_loss'])
            self.history['dead_ratio'].append(train_metrics['dead_ratio'])

            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                for k in self.model.k_levels:
                    key = f'r2_k{k}'
                    if key not in self.history['val_r2']:
                        self.history['val_r2'][key] = []
                    self.history['val_r2'][key].append(val_metrics[key])

            # Log epoch summary
            elapsed = time.time() - start_time
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Train loss: {train_metrics['loss']:.6f} "
                  f"(rec: {train_metrics['rec_loss']:.6f}, aux: {train_metrics['aux_loss']:.6f})")
            print(f"  Dead latents: {train_metrics['dead_ratio']:.3f}")

            if val_metrics:
                print(f"  Val loss: {val_metrics['loss']:.6f}")
                r2_str = ", ".join(f"k={k}: {val_metrics[f'r2_k{k}']:.4f}"
                                   for k in self.model.k_levels)
                print(f"  Val R²: {r2_str}")

                # Early stopping check
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    patience_counter += 1
                    if early_stopping_patience and patience_counter >= early_stopping_patience:
                        print(f"Early stopping after {epoch} epochs")
                        break

            print()

            # Checkpoint
            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(epoch)
                clear_memory(verbose=False)

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'k_levels': self.model.k_levels,
                'weighting': self.model.weighting,
            }
        }

        # Save epoch checkpoint
        path = self.output_dir / 'checkpoints' / f'msae_epoch{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

        # Save best model
        if is_best:
            best_path = self.output_dir / 'models' / 'msae_best.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path}")

    def save_final_model(self, name: str = 'msae_final.pt'):
        """Save final trained model."""
        path = self.output_dir / 'models' / name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'k_levels': self.model.k_levels,
                'weighting': self.model.weighting,
            },
            'history': self.history,
        }, path)
        print(f"Saved final model: {path}")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> 'MSAETrainer':
        """
        Load trainer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            device: Device to load onto

        Returns:
            MSAETrainer with restored state
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

        # Recreate model
        model = MatryoshkaSAE(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            k_levels=config['k_levels'],
            weighting=config['weighting'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        # Create trainer
        trainer = cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        # Restore optimizer state
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore history
        trainer.history = checkpoint['history']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        return trainer


class BaselineTrainer:
    """
    Trainer for baseline single-k SAE.

    Same structure as MSAETrainer but for BaselineSAE.
    """

    def __init__(
        self,
        model: BaselineSAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        output_dir: str = 'outputs',
        device: Optional[torch.device] = None,
        checkpoint_every: int = 1,
        log_every: int = 100,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=6.25e-10,
        )

        self.history = {
            'train_loss': [],
            'train_rec_loss': [],
            'train_aux_loss': [],
            'val_loss': [],
            'val_r2': [],
            'dead_ratio': [],
        }

        (self.output_dir / 'models').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_aux_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            self.optimizer.zero_grad()
            loss_dict = self.model.compute_loss(x, update_stats=True)
            loss_dict['loss'].backward()
            self.optimizer.step()
            self.model.normalize_decoder()

            total_loss += loss_dict['loss'].item()
            total_rec_loss += loss_dict['reconstruction_loss'].item()
            total_aux_loss += loss_dict['aux_loss'].item()
            n_batches += 1

            if (batch_idx + 1) % self.log_every == 0:
                avg_loss = total_loss / n_batches
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}: "
                      f"loss={avg_loss:.6f}, dead={self.model.dead_latent_ratio:.3f}")

        return {
            'loss': total_loss / n_batches,
            'rec_loss': total_rec_loss / n_batches,
            'aux_loss': total_aux_loss / n_batches,
            'dead_ratio': self.model.dead_latent_ratio,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_var = 0.0
        n_batches = 0

        for batch in self.val_loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            loss_dict = self.model.compute_loss(x, update_stats=False)
            total_loss += loss_dict['loss'].item()

            x_hat, _, _ = self.model(x)
            total_mse += ((x_hat - x) ** 2).sum().item()
            total_var += (x.var() * x.numel()).item()
            n_batches += 1

        r2 = 1.0 - total_mse / (total_var + 1e-8)

        return {
            'loss': total_loss / n_batches,
            'r2': r2,
        }

    def train(
        self,
        epochs: int = 15,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, List]:
        """Full training loop."""
        print(f"Training Baseline SAE on {self.device}")
        print(f"  K: {self.model.k}")
        print(f"  Hidden dim: {self.model.hidden_dim}")
        print(f"  Epochs: {epochs}")
        print()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            print(f"Epoch {epoch}/{epochs}")
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_rec_loss'].append(train_metrics['rec_loss'])
            self.history['train_aux_loss'].append(train_metrics['aux_loss'])
            self.history['dead_ratio'].append(train_metrics['dead_ratio'])

            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_r2'].append(val_metrics['r2'])

            elapsed = time.time() - start_time
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Train loss: {train_metrics['loss']:.6f}")
            print(f"  Dead latents: {train_metrics['dead_ratio']:.3f}")

            if val_metrics:
                print(f"  Val loss: {val_metrics['loss']:.6f}")
                print(f"  Val R²: {val_metrics['r2']:.4f}")

                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    patience_counter += 1
                    if early_stopping_patience and patience_counter >= early_stopping_patience:
                        print(f"Early stopping after {epoch} epochs")
                        break

            print()

            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(epoch)
                clear_memory(verbose=False)

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'k': self.model.k,
            }
        }

        path = self.output_dir / 'checkpoints' / f'baseline_sae_k{self.model.k}_epoch{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

        if is_best:
            best_path = self.output_dir / 'models' / f'baseline_sae_k{self.model.k}_best.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path}")

    def save_final_model(self, name: Optional[str] = None):
        """Save final trained model."""
        if name is None:
            name = f'baseline_sae_k{self.model.k}_final.pt'
        path = self.output_dir / 'models' / name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'k': self.model.k,
            },
            'history': self.history,
        }, path)
        print(f"Saved final model: {path}")


def create_activation_dataloader(
    activations_dir: str,
    block_idx: int,
    batch_size: int = 4096,
    normalize: bool = True,
    val_split: float = 0.1,
    num_workers: int = None,  # None = auto-detect
    max_chunks: Optional[int] = None,
    h5_path: Optional[str] = None,  # Optional: use HDF5 file instead of .npy chunks
) -> tuple:
    """
    Create train/val dataloaders from saved activation chunks.

    Supports two data sources:
    1. HDF5 file (h5_path) - memory-efficient chunked streaming
    2. .npy chunks (activations_dir) - legacy format, loads all into memory

    Args:
        activations_dir: Directory containing activation chunks
        block_idx: Which block's activations to load
        batch_size: Batch size (or None for auto-detect)
        normalize: Whether to normalize activations
        val_split: Fraction for validation
        num_workers: Number of dataloader workers (None = auto-detect)
        max_chunks: Maximum number of chunks to load (None for all, only for .npy)
        h5_path: Optional path to HDF5 file for memory-efficient loading

    Returns:
        (train_loader, val_loader, normalization_stats)

    Raises:
        FileNotFoundError: If the block directory doesn't exist
        ValueError: If no activation chunks are found
    """
    from ..data.activation_extractor import load_activation_chunks, compute_activation_stats
    
    # Auto-detect system capabilities
    try:
        from ..utils.system import get_system_capabilities
        caps = get_system_capabilities()
        print(caps.summary())
        
        # Auto-detect workers if not specified
        if num_workers is None:
            num_workers = caps.optimal_workers()
            print(f"Auto-detected optimal workers: {num_workers}")
    except ImportError:
        # Fallback if system module not available
        caps = None
        if num_workers is None:
            num_workers = 0
    
    # Try HDF5 path first (memory efficient)
    if h5_path is not None:
        try:
            from ..data.h5_dataset import create_h5_dataloaders
            print(f"Using HDF5 streaming from: {h5_path}")
            dataset_key = f'block{block_idx}'
            return create_h5_dataloaders(
                h5_path, dataset_key,
                batch_size=batch_size,
                val_split=val_split,
                normalize=normalize,
                num_workers=num_workers,
            )
        except Exception as e:
            print(f"Warning: HDF5 loading failed, falling back to .npy: {e}")
    
    # Legacy .npy chunk loading
    block_dir = Path(activations_dir) / f'block{block_idx}'

    if not block_dir.exists():
        raise FileNotFoundError(f"Activation directory not found: {block_dir}")

    # Load all chunks by consuming the generator
    print(f"Loading activations from {block_dir}...")
    chunk_list = list(load_activation_chunks(str(block_dir), max_chunks=max_chunks))

    if not chunk_list:
        raise ValueError(f"No activation chunks found in {block_dir}")

    # Concatenate all chunks into a single array
    activations = np.concatenate(chunk_list, axis=0)
    del chunk_list  # Free memory from list of separate arrays

    print(f"  Loaded {len(activations):,} samples with shape {activations.shape}")
    print(f"  Memory usage: {activations.nbytes / 1e9:.2f} GB")
    
    # Check if we should warn about memory
    if caps is not None:
        data_gb = activations.nbytes / 1e9
        if data_gb > caps.available_ram_gb * 0.5:
            print(f"  WARNING: Data uses {data_gb:.1f} GB, {data_gb/caps.available_ram_gb*100:.1f}% of available RAM")
            print(f"  Consider using HDF5 format with h5_path parameter for memory efficiency")

    # Load or compute normalization stats
    stats_path = block_dir / 'normalization_stats.npz'
    if stats_path.exists():
        print(f"  Loading precomputed normalization stats from {stats_path.name}")
        stats = np.load(stats_path)
        mean = torch.from_numpy(stats['mean'].astype(np.float32))
        std = torch.from_numpy(stats['std'].astype(np.float32))
    else:
        print("  Computing normalization stats (this may take a moment)...")
        stats = compute_activation_stats(str(block_dir), max_chunks=max_chunks)
        mean = torch.from_numpy(stats['mean'].astype(np.float32))
        std = torch.from_numpy(stats['std'].astype(np.float32))
        # Save for future use
        np.savez(stats_path, mean=stats['mean'], std=stats['std'])
        print(f"  Saved normalization stats to {stats_path.name}")

    # Convert to tensor
    activations = torch.from_numpy(activations).float()

    # Normalize
    if normalize:
        activations = (activations - mean) / (std + 1e-8)

    # Split into train/val
    n_samples = len(activations)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    # Shuffle before split (reproducible with generator)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_samples, generator=generator)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_data = activations[train_indices]
    val_data = activations[val_indices]

    # Free the full tensor now that we've split
    del activations
    clear_memory(verbose=False)

    print(f"  Train samples: {len(train_data):,}")
    print(f"  Val samples:   {len(val_data):,}")

    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # Ensures consistent batch sizes for training
    )

    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    norm_stats = {'mean': mean, 'std': std}

    return train_loader, val_loader, norm_stats

