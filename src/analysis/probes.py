"""
Linear probes for evaluating feature quality.

Based on Gao et al. "Scaling and Evaluating Sparse Autoencoders":
- 1D logistic probe per latent
- AUC-ROC metric with bootstrap confidence intervals
- Train for each: (concept × layer × method × k_level)

Usage:
    from src.analysis.probes import LinearProbe, ProbeEvaluator
    probe = LinearProbe(n_features=4096)
    probe.fit(features, labels)
    auc = probe.evaluate(test_features, test_labels)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ProbeResult:
    """Result of a single probe evaluation."""
    concept: str
    layer: str
    method: str  # 'msae', 'baseline', 'raw'
    k: Optional[int]  # k level for MSAE, None for baseline/raw
    auc: float
    ci_low: float
    ci_high: float
    n_positive: int
    n_negative: int


class LinearProbe(nn.Module):
    """
    Binary linear classifier for concept detection.

    Architecture: Linear(n_features → 1) + Sigmoid

    From Gao et al.: "we train a 1d logistic probe on each latent"
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (before sigmoid)."""
        return self.linear(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities."""
        return torch.sigmoid(self.forward(x))


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 4096,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> LinearProbe:
    """
    Train a linear probe using logistic regression.

    Args:
        features: (n_samples, n_features) feature array
        labels: (n_samples,) binary label array
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device to train on
        verbose: Print progress

    Returns:
        Trained LinearProbe
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to tensors
    X = torch.from_numpy(features).float().to(device)
    y = torch.from_numpy(labels).float().to(device)

    n_samples, n_features = X.shape
    probe = LinearProbe(n_features).to(device)

    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    probe.train()
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        total_loss = 0.0

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i + batch_size]
            batch_X = X[batch_idx]
            batch_y = y[batch_idx]

            optimizer.zero_grad()
            logits = probe(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose and (epoch + 1) % 20 == 0:
            avg_loss = total_loss / (n_samples // batch_size + 1)
            print(f"Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.4f}")

    return probe


@torch.no_grad()
def evaluate_probe(
    probe: LinearProbe,
    features: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 100,
    device: Optional[torch.device] = None,
) -> Tuple[float, float, float]:
    """
    Evaluate probe using AUC-ROC with bootstrap confidence intervals.

    Args:
        probe: Trained LinearProbe
        features: (n_samples, n_features) test features
        labels: (n_samples,) test labels
        n_bootstrap: Number of bootstrap samples for CI
        device: Device to evaluate on

    Returns:
        (auc, ci_low, ci_high)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = torch.from_numpy(features).float().to(device)
    y_true = labels

    probe.eval()
    y_pred = probe.predict_proba(X).cpu().numpy()

    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.5

    # Compute AUC
    auc = roc_auc_score(y_true, y_pred)

    # Bootstrap for confidence interval
    n_samples = len(y_true)
    bootstrap_aucs = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        boot_auc = roc_auc_score(y_true_boot, y_pred_boot)
        bootstrap_aucs.append(boot_auc)

    if bootstrap_aucs:
        ci_low = np.percentile(bootstrap_aucs, 2.5)
        ci_high = np.percentile(bootstrap_aucs, 97.5)
    else:
        ci_low = ci_high = auc

    return auc, ci_low, ci_high


class ProbeEvaluator:
    """
    Run probe evaluation across concepts, layers, and methods.

    Handles train/val/test splits and saves results.
    """

    def __init__(
        self,
        output_dir: str = 'outputs',
        device: Optional[torch.device] = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.results: List[ProbeResult] = []

    def _split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        position_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test.

        Args:
            features: (n_samples, n_features) array
            labels: (n_samples,) binary labels
            position_ids: (n_samples,) position IDs for position-level splitting.
                         If provided, ensures entire positions stay in one split.
                         This prevents train/test leakage from spatial structure.

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test) tuple
        """
        if position_ids is not None:
            return self._split_data_by_position(features, labels, position_ids)

        # Original sample-level splitting (for backwards compatibility)
        # First split: separate test set
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            features, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels if len(np.unique(labels)) >= 2 else None
        )

        # Second split: separate validation from training
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=y_trainval if len(np.unique(y_trainval)) >= 2 else None
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _split_data_by_position(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        position_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data at position level to prevent train/test leakage.

        For Go positions, each position has 361 spatial points. This ensures
        all 361 points from a position stay in the same split, preventing
        the probe from learning position-specific (rather than concept-specific)
        features.

        Args:
            features: (n_samples, n_features) array
            labels: (n_samples,) binary labels
            position_ids: (n_samples,) integer position IDs

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test) tuple
        """
        unique_positions = np.unique(position_ids)
        n_positions = len(unique_positions)

        # Split position IDs (not samples)
        pos_trainval, pos_test = train_test_split(
            unique_positions,
            test_size=self.test_size,
            random_state=self.random_state
        )

        val_ratio = self.val_size / (1 - self.test_size)
        pos_train, pos_val = train_test_split(
            pos_trainval,
            test_size=val_ratio,
            random_state=self.random_state
        )

        # Create masks for each split
        train_mask = np.isin(position_ids, pos_train)
        val_mask = np.isin(position_ids, pos_val)
        test_mask = np.isin(position_ids, pos_test)

        X_train = features[train_mask]
        X_val = features[val_mask]
        X_test = features[test_mask]
        y_train = labels[train_mask]
        y_val = labels[val_mask]
        y_test = labels[test_mask]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def evaluate_concept(
        self,
        concept_name: str,
        layer_name: str,
        method: str,
        features: np.ndarray,
        labels: np.ndarray,
        k: Optional[int] = None,
        position_ids: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> ProbeResult:
        """
        Train and evaluate a probe for one concept/layer/method combination.

        Args:
            concept_name: Name of the concept
            layer_name: Name of the layer (e.g., 'block5')
            method: Feature extraction method ('msae', 'baseline', 'raw')
            features: (n_samples, n_features) feature array
            labels: (n_samples,) binary label array
            k: k level for MSAE (None for baseline/raw)
            position_ids: (n_samples,) position IDs for position-level splitting.
                         If provided, prevents train/test leakage from spatial structure.
            verbose: Print progress

        Returns:
            ProbeResult with AUC and confidence intervals
        """
        if verbose:
            print(f"  Evaluating: {concept_name} / {layer_name} / {method}" +
                  (f" / k={k}" if k else ""))

        # Count class balance
        n_positive = labels.sum()
        n_negative = len(labels) - n_positive

        # Need at least some samples of each class
        if n_positive < 10 or n_negative < 10:
            if verbose:
                print(f"    Skipping: too few samples (pos={n_positive}, neg={n_negative})")
            return ProbeResult(
                concept=concept_name,
                layer=layer_name,
                method=method,
                k=k,
                auc=0.5,
                ci_low=0.5,
                ci_high=0.5,
                n_positive=int(n_positive),
                n_negative=int(n_negative),
            )

        # Split data (use position-level split if position_ids provided)
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(
            features, labels, position_ids
        )

        # Train probe
        probe = train_probe(
            X_train, y_train,
            device=self.device,
            verbose=False,
        )

        # Evaluate on test set
        auc, ci_low, ci_high = evaluate_probe(
            probe, X_test, y_test,
            device=self.device,
        )

        result = ProbeResult(
            concept=concept_name,
            layer=layer_name,
            method=method,
            k=k,
            auc=auc,
            ci_low=ci_low,
            ci_high=ci_high,
            n_positive=int(n_positive),
            n_negative=int(n_negative),
        )

        self.results.append(result)

        if verbose:
            print(f"    AUC: {auc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

        return result

    def save_results(self, filename: str = 'probes.json'):
        """Save all results to JSON."""
        results_path = self.output_dir / 'results' / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to nested dict structure per CLAUDE.md format
        output = {}
        for result in self.results:
            if result.concept not in output:
                output[result.concept] = {}
            if result.layer not in output[result.concept]:
                output[result.concept][result.layer] = {}

            key = f"{result.method}_k{result.k}" if result.k else result.method
            output[result.concept][result.layer][key] = {
                'auc': result.auc,
                'ci_low': result.ci_low,
                'ci_high': result.ci_high,
                'n_positive': result.n_positive,
                'n_negative': result.n_negative,
            }

        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Saved probe results to {results_path}")

    def summarize(self) -> Dict[str, float]:
        """Summarize results by method."""
        summary = {}

        for method in ['msae', 'baseline', 'raw']:
            method_results = [r for r in self.results if r.method == method]
            if method_results:
                avg_auc = np.mean([r.auc for r in method_results])
                summary[f'{method}_avg_auc'] = avg_auc

        return summary


def extract_sae_features(
    model: nn.Module,
    activations: np.ndarray,
    k: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Extract SAE features from activations.

    Args:
        model: MSAE or BaselineSAE model
        activations: (n_samples, input_dim) normalized activations
        k: k level for MSAE (uses model's k for baseline)
        device: Device to use

    Returns:
        (n_samples, hidden_dim) sparse features
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    batch_size = 4096
    n_samples = len(activations)
    features_list = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.from_numpy(activations[i:i + batch_size]).float().to(device)

            if k is not None:
                # MSAE with specified k
                _, _, z_sparse = model.forward(batch, k)
            else:
                # Baseline SAE
                _, _, z_sparse = model.forward(batch)

            features_list.append(z_sparse.cpu().numpy())

    return np.concatenate(features_list, axis=0)
