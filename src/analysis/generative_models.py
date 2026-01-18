"""
Generative Models for Strategic Trajectory Prediction.

Trains autoregressive models on game trajectories in SAE feature space
to predict strategic evolution and generate synthetic trajectories.

Reference:
- MASTER_IMPLEMENTATION_PLAN.md: Lines 765-878 (NB08 spec)
- src/models/msae.py for PyTorch module patterns

Key concepts:
- Trajectory: Sequence of SAE feature vectors [T, D] for one game
- Context: Previous k moves used to predict next move
- Game boundaries: Never cross between games in training
- Temperature sampling: Control diversity of generated trajectories
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import h5py


@dataclass
class TrainingResult:
    """Results from training a trajectory predictor."""
    train_losses: List[float]
    val_losses: List[float]
    best_epoch: int
    best_val_loss: float
    final_model_state: dict


class TrajectoryPredictor(nn.Module):
    """
    Autoregressive trajectory prediction in SAE feature space.

    Uses GRU-based architecture to predict next feature vector
    given context of previous moves.

    Architecture:
        input (D) -> GRU (hidden_dim, n_layers) -> Linear (D)

    Usage:
        model = TrajectoryPredictor(input_dim=4096)
        next_pred, hidden = model(context_sequence)
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of SAE features
            hidden_dim: GRU hidden state dimension
            n_layers: Number of GRU layers
            dropout: Dropout rate between layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, input_dim] input sequence
            hidden: [n_layers, batch, hidden_dim] initial hidden state

        Returns:
            predictions: [batch, seq_len, input_dim] predicted next states
            hidden: [n_layers, batch, hidden_dim] final hidden state
        """
        # Project input
        h = self.input_proj(x)  # [batch, seq, hidden]
        h = F.relu(h)

        # GRU
        h, hidden = self.gru(h, hidden)

        # Layer norm
        h = self.layer_norm(h)

        # Output projection
        predictions = self.output_proj(h)

        return predictions, hidden

    def predict_next(
        self,
        context: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next feature vector given context.

        Args:
            context: [batch, context_len, input_dim] context sequence
            hidden: Optional hidden state from previous call

        Returns:
            next_pred: [batch, input_dim] predicted next state
            hidden: Updated hidden state
        """
        predictions, hidden = self(context, hidden)
        return predictions[:, -1, :], hidden

    def sample_trajectory(
        self,
        initial_context: torch.Tensor,
        n_steps: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate trajectory sample from trained model.

        Args:
            initial_context: [1, context_len, input_dim] starting context
            n_steps: Number of steps to generate
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            trajectory: [n_steps, input_dim] generated trajectory
        """
        device = next(self.parameters()).device
        initial_context = initial_context.to(device)

        trajectory = []
        context = initial_context
        hidden = None

        with torch.no_grad():
            for _ in range(n_steps):
                # Predict next
                next_pred, hidden = self.predict_next(context, hidden)

                # Add Gaussian noise scaled by temperature
                if temperature > 0:
                    noise = torch.randn_like(next_pred) * temperature
                    next_pred = next_pred + noise

                trajectory.append(next_pred.cpu())

                # Update context: shift and append new prediction
                context = torch.cat([
                    context[:, 1:, :],
                    next_pred.unsqueeze(1)
                ], dim=1)

        return torch.cat(trajectory, dim=0)


class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory sequences.

    Creates (context, target) pairs from game trajectories,
    respecting game boundaries.

    Usage:
        dataset = TrajectoryDataset(features, game_ids, context_length=10)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        features: np.ndarray,
        game_ids: np.ndarray,
        context_length: int = 10,
    ):
        """
        Args:
            features: [n_samples, n_features] feature matrix
            game_ids: [n_samples] game index for each sample
            context_length: Number of moves for context
        """
        self.features = features
        self.game_ids = game_ids
        self.context_length = context_length

        # Build valid (context, target) pairs
        self.pairs = self._build_pairs()
        print(f"TrajectoryDataset: {len(self.pairs)} valid pairs")

    def _build_pairs(self) -> List[Tuple[int, int]]:
        """Build list of valid (start_idx, end_idx) pairs."""
        pairs = []

        unique_games = np.unique(self.game_ids)

        for game_id in unique_games:
            mask = self.game_ids == game_id
            indices = np.where(mask)[0]

            # Need at least context_length + 1 moves
            if len(indices) < self.context_length + 1:
                continue

            # Create pairs within this game
            for i in range(len(indices) - self.context_length):
                start = indices[i]
                end = indices[i + self.context_length]
                pairs.append((start, end))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end = self.pairs[idx]

        # Context: positions [start, end)
        context = self.features[start:end]

        # Target: position end (next after context)
        target = self.features[end]

        return torch.from_numpy(context), torch.from_numpy(target)


class StreamingTrajectoryDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for large h5py files.

    Loads one game at a time to minimize memory usage.
    """

    def __init__(
        self,
        h5_path: str,
        dataset_key: str,
        game_ids: np.ndarray,
        context_length: int = 10,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            h5_path: Path to HDF5 file with features
            dataset_key: Dataset key in HDF5 file
            game_ids: [n_samples] game index for each sample
            context_length: Number of moves for context
            shuffle: Whether to shuffle games
            seed: Random seed
        """
        self.h5_path = h5_path
        self.dataset_key = dataset_key
        self.game_ids = game_ids
        self.context_length = context_length
        self.shuffle = shuffle
        self.seed = seed

        # Compute game boundaries
        self.unique_games = np.unique(game_ids)
        self.game_boundaries = {}
        for game_id in self.unique_games:
            mask = game_ids == game_id
            indices = np.where(mask)[0]
            if len(indices) >= context_length + 1:
                self.game_boundaries[game_id] = (indices[0], indices[-1] + 1)

        print(f"StreamingTrajectoryDataset: {len(self.game_boundaries)} valid games")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single process
            games = list(self.game_boundaries.keys())
            worker_seed = self.seed
        else:
            # Multi-process: split games among workers
            all_games = list(self.game_boundaries.keys())
            per_worker = (len(all_games) + worker_info.num_workers - 1) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(all_games))
            games = all_games[start:end]
            worker_seed = self.seed + worker_id

        rng = np.random.default_rng(worker_seed)

        if self.shuffle:
            rng.shuffle(games)

        with h5py.File(self.h5_path, 'r') as f:
            dset = f[self.dataset_key]

            for game_id in games:
                start, end = self.game_boundaries[game_id]

                # Load this game
                game_features = dset[start:end].astype(np.float32)

                # Generate pairs from this game
                n_pairs = len(game_features) - self.context_length

                pair_indices = list(range(n_pairs))
                if self.shuffle:
                    rng.shuffle(pair_indices)

                for i in pair_indices:
                    context = game_features[i:i + self.context_length]
                    target = game_features[i + self.context_length]

                    yield torch.from_numpy(context), torch.from_numpy(target)


def train_trajectory_predictor(
    model: TrajectoryPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = 'auto',
) -> TrainingResult:
    """
    Train trajectory predictor with early stopping.

    Args:
        model: TrajectoryPredictor to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Maximum training epochs
        lr: Learning rate
        patience: Early stopping patience
        device: 'cuda', 'cpu', or 'auto'

    Returns:
        TrainingResult with losses and best model state
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        for context, target in train_loader:
            context = context.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # Predict next step
            pred, _ = model.predict_next(context)

            loss = criterion(pred, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(1, n_batches)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for context, target in val_loader:
                context = context.to(device)
                target = target.to(device)

                pred, _ = model.predict_next(context)
                loss = criterion(pred, target)

                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= max(1, n_val_batches)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch {epoch+1}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best epoch: {best_epoch+1}, val_loss: {best_val_loss:.6f}")

    return TrainingResult(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        final_model_state=best_state or model.state_dict(),
    )


def sample_trajectories(
    model: TrajectoryPredictor,
    initial_contexts: torch.Tensor,
    n_steps: int = 50,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Generate multiple trajectory samples.

    Args:
        model: Trained TrajectoryPredictor
        initial_contexts: [n_samples, context_len, input_dim]
        n_steps: Steps per trajectory
        temperature: Sampling temperature

    Returns:
        [n_samples, n_steps, input_dim] generated trajectories
    """
    model.eval()
    trajectories = []

    with torch.no_grad():
        for i in range(len(initial_contexts)):
            context = initial_contexts[i:i+1]  # [1, context_len, D]
            traj = model.sample_trajectory(context, n_steps, temperature)
            trajectories.append(traj.numpy())

    return np.array(trajectories)


class GenerativeAnalyzer:
    """
    Complete generative analysis pipeline.

    Trains trajectory predictors, generates samples, and evaluates
    the quality of generated trajectories.

    Usage:
        analyzer = GenerativeAnalyzer(h5_path, 'block35')
        model, result = analyzer.train(game_ids, epochs=50)

        # Generate samples
        samples = analyzer.sample(model, initial_contexts, n_trajectories=10)

        # Evaluate
        metrics = analyzer.evaluate_diversity(samples)
    """

    def __init__(
        self,
        h5_path: Optional[str] = None,
        dataset_key: Optional[str] = None,
        context_length: int = 10,
        device: str = 'auto',
    ):
        """
        Args:
            h5_path: Path to HDF5 file (optional, for streaming)
            dataset_key: Dataset key in HDF5 file
            context_length: Context length for prediction
            device: 'cuda', 'cpu', or 'auto'
        """
        self.h5_path = h5_path
        self.dataset_key = dataset_key
        self.context_length = context_length

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def train(
        self,
        features: np.ndarray,
        game_ids: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        val_split: float = 0.1,
        hidden_dim: int = 512,
        n_layers: int = 2,
        lr: float = 1e-3,
    ) -> Tuple[TrajectoryPredictor, TrainingResult]:
        """
        Train trajectory predictor.

        Args:
            features: [n_samples, n_features] feature matrix
            game_ids: [n_samples] game indices
            epochs: Training epochs
            batch_size: Batch size
            val_split: Validation fraction
            hidden_dim: GRU hidden dimension
            n_layers: GRU layers
            lr: Learning rate

        Returns:
            (trained_model, training_result)
        """
        print("=" * 60)
        print("TRAINING TRAJECTORY PREDICTOR")
        print("=" * 60)

        # Create dataset
        dataset = TrajectoryDataset(
            features.astype(np.float32),
            game_ids,
            context_length=self.context_length
        )

        # Train/val split
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f"Train: {n_train}, Val: {n_val}")

        # Create model
        input_dim = features.shape[1]
        model = TrajectoryPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

        # Train
        result = train_trajectory_predictor(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, device=self.device
        )

        return model, result

    def sample(
        self,
        model: TrajectoryPredictor,
        features: np.ndarray,
        n_trajectories: int = 10,
        n_steps: int = 50,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate trajectory samples from random starting contexts.

        Args:
            model: Trained TrajectoryPredictor
            features: [n_samples, n_features] for selecting contexts
            n_trajectories: Number of trajectories to generate
            n_steps: Steps per trajectory
            temperature: Sampling temperature
            seed: Random seed

        Returns:
            [n_trajectories, n_steps, n_features] generated trajectories
        """
        rng = np.random.default_rng(seed)

        # Select random starting contexts
        valid_starts = len(features) - self.context_length
        start_indices = rng.choice(valid_starts, n_trajectories, replace=False)

        contexts = []
        for idx in start_indices:
            context = features[idx:idx + self.context_length]
            contexts.append(context)

        contexts = torch.from_numpy(np.array(contexts).astype(np.float32))

        return sample_trajectories(model, contexts, n_steps, temperature)

    def evaluate_diversity(
        self,
        samples: np.ndarray,
        real_features: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate diversity of generated samples.

        Args:
            samples: [n_samples, n_steps, n_features] generated trajectories
            real_features: [n_real, n_features] real data for comparison

        Returns:
            Dict with diversity metrics
        """
        n_samples, n_steps, n_features = samples.shape

        # Flatten to [n_samples * n_steps, n_features]
        flat_samples = samples.reshape(-1, n_features)

        metrics = {}

        # Mean pairwise distance between samples
        from scipy.spatial.distance import pdist
        if len(flat_samples) <= 5000:
            pairwise_dists = pdist(flat_samples)
            metrics['mean_pairwise_distance'] = float(np.mean(pairwise_dists))
            metrics['std_pairwise_distance'] = float(np.std(pairwise_dists))
        else:
            # Sample for large datasets
            sample_indices = np.random.choice(len(flat_samples), 5000, replace=False)
            pairwise_dists = pdist(flat_samples[sample_indices])
            metrics['mean_pairwise_distance'] = float(np.mean(pairwise_dists))
            metrics['std_pairwise_distance'] = float(np.std(pairwise_dists))

        # Feature-wise statistics
        metrics['mean_feature_std'] = float(np.mean(np.std(flat_samples, axis=0)))

        # Compare to real data if available
        if real_features is not None:
            # Compare means
            gen_mean = np.mean(flat_samples, axis=0)
            real_mean = np.mean(real_features, axis=0)
            metrics['mean_cosine_similarity'] = float(
                np.dot(gen_mean, real_mean) /
                (np.linalg.norm(gen_mean) * np.linalg.norm(real_mean) + 1e-8)
            )

            # Compare stds
            gen_std = np.std(flat_samples, axis=0)
            real_std = np.std(real_features, axis=0)
            metrics['std_correlation'] = float(np.corrcoef(gen_std, real_std)[0, 1])

        return metrics

    def visualize_samples(
        self,
        samples: np.ndarray,
        real_features: Optional[np.ndarray] = None,
        method: str = 'pca',
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize generated trajectories.

        Args:
            samples: [n_samples, n_steps, n_features] generated trajectories
            real_features: [n_real, n_features] real data for comparison
            method: 'pca' or 'umap' for dimensionality reduction
            save_path: If specified, save figure
        """
        import matplotlib.pyplot as plt

        n_samples, n_steps, n_features = samples.shape

        # Flatten samples
        flat_samples = samples.reshape(-1, n_features)

        # Reduce dimensionality
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)

            if real_features is not None:
                # Fit on real data
                reducer.fit(real_features)
                real_2d = reducer.transform(real_features)
            else:
                reducer.fit(flat_samples)
                real_2d = None

            samples_2d = reducer.transform(flat_samples)

        elif method == 'umap':
            import umap

            if real_features is not None:
                combined = np.vstack([real_features, flat_samples])
                reducer = umap.UMAP(n_components=2, random_state=42)
                combined_2d = reducer.fit_transform(combined)
                real_2d = combined_2d[:len(real_features)]
                samples_2d = combined_2d[len(real_features):]
            else:
                reducer = umap.UMAP(n_components=2, random_state=42)
                samples_2d = reducer.fit_transform(flat_samples)
                real_2d = None

        else:
            raise ValueError(f"Unknown method: {method}")

        # Reshape back to trajectories
        samples_2d = samples_2d.reshape(n_samples, n_steps, 2)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot real data background
        if real_2d is not None:
            subsample = min(5000, len(real_2d))
            indices = np.random.choice(len(real_2d), subsample, replace=False)
            ax.scatter(real_2d[indices, 0], real_2d[indices, 1],
                      c='lightgray', s=1, alpha=0.3, label='Real data')

        # Plot generated trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
        for i in range(n_samples):
            traj = samples_2d[i]
            ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[i], alpha=0.7, linewidth=1)
            ax.scatter(traj[0, 0], traj[0, 1], c='green', s=50, zorder=5, marker='o')
            ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=50, zorder=5, marker='x')

        ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
        ax.set_title('Generated Trajectories', fontsize=14)
        ax.legend(['Real data', 'Generated trajectory', 'Start', 'End'])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved trajectory visualization to {save_path}")

        plt.show()

    def save_model(self, model: TrajectoryPredictor, output_path: str) -> None:
        """Save trained model."""
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': model.input_dim,
            'hidden_dim': model.hidden_dim,
            'n_layers': model.n_layers,
            'context_length': self.context_length,
        }, output_path)
        print(f"Saved model to {output_path}")

    def load_model(self, input_path: str) -> TrajectoryPredictor:
        """Load trained model."""
        checkpoint = torch.load(input_path, map_location=self.device)

        model = TrajectoryPredictor(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            n_layers=checkpoint['n_layers'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        self.context_length = checkpoint['context_length']

        print(f"Loaded model from {input_path}")
        return model
