"""
Negative controls for validating SAE feature analysis.

Implements three types of negative controls per CLAUDE.md specification:
1. Shuffled Labels: Randomly permute concept labels, retrain probe
2. Random Network: Extract activations from untrained Leela Zero
3. Permuted Features: Shuffle feature dimensions, retrain probe

Expected result for all controls: AUC ~0.50 (chance level)

Based on:
- docs/monosemantic_features.md (Bricken et al.) - random network control
- docs/topk_sae_paper.md (Gao et al.) - validation methodology

Usage:
    from src.analysis.controls import ControlEvaluator
    evaluator = ControlEvaluator()
    results = evaluator.run_all_controls(features, labels, concept_name, layer_name)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import json
from pathlib import Path

from .probes import LinearProbe, train_probe, evaluate_probe


@dataclass
class ControlResult:
    """Result of a single negative control evaluation."""
    control_type: str  # 'shuffled', 'random_network', 'permuted_features'
    concept: str
    layer: str
    method: str  # 'msae', 'baseline', 'raw'
    k: Optional[int]  # k level for MSAE
    auc: float
    ci_low: float
    ci_high: float
    expected_auc: float = 0.50  # All controls should be ~0.50


@dataclass
class ControlSummary:
    """Summary of all control results for a concept/layer/method combination."""
    concept: str
    layer: str
    method: str
    k: Optional[int]
    shuffled_auc: float
    permuted_auc: float
    random_network_auc: Optional[float]  # None if not run
    all_near_chance: bool  # True if all controls are ~0.50


def shuffle_labels(labels: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Randomly permute concept labels.

    This destroys the relationship between features and labels while
    preserving the marginal distribution of labels.

    Args:
        labels: (n_samples,) binary labels
        random_state: Random seed for reproducibility

    Returns:
        Shuffled labels array
    """
    rng = np.random.RandomState(random_state)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    return shuffled


def permute_features(features: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Randomly permute feature dimensions INDEPENDENTLY per sample.

    This destroys the geometric structure of features by applying a different
    random permutation to each sample, breaking the consistent relationship
    between feature dimensions and labels.

    Note: A same-permutation-for-all-samples approach would NOT work as a
    negative control because a linear probe could learn the inverse permutation.
    Per-sample independent permutation is required to actually destroy the
    feature-label correlation structure.

    Reference: Conceptually similar to randomized transformer baseline in
    Bricken et al. (2023) "Towards Monosemanticity" which tests if signal
    comes from learned structure vs spurious correlations.

    Args:
        features: (n_samples, n_features) feature array
        random_state: Random seed for reproducibility

    Returns:
        Features with independently permuted dimensions per sample
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = features.shape

    # Vectorized: generate all permutation indices at once
    # For each row, argsort of random values gives a random permutation
    random_keys = rng.random((n_samples, n_features))
    perm_indices = np.argsort(random_keys, axis=1)

    # Advanced indexing to apply per-row permutations
    row_indices = np.arange(n_samples)[:, np.newaxis]
    permuted = features[row_indices, perm_indices]

    return permuted


def create_random_network_activations(
    activations_shape: Tuple[int, ...],
    random_state: int = 42,
) -> np.ndarray:
    """
    Create activations as if from a random (untrained) network.

    From Bricken et al.: "To assess the effect of dataset correlations on
    the interpretability of feature activations, we run dictionary learning
    on a version of our one-layer model with random weights."

    We simulate this by generating random Gaussian activations with the
    same shape as real activations.

    Args:
        activations_shape: Shape of activations (n_samples, n_features)
        random_state: Random seed

    Returns:
        Random activations array
    """
    rng = np.random.RandomState(random_state)
    # Use standard normal as network activations are typically normalized
    return rng.randn(*activations_shape).astype(np.float32)


def randomize_network_weights(model: nn.Module, random_state: int = 42) -> nn.Module:
    """
    Randomize all weights in a network (in-place).

    From Bricken et al.: "We generate a model with random weights by
    randomly shuffling the entries of each weight matrix of the trained
    transformer."

    Args:
        model: Neural network model
        random_state: Random seed

    Returns:
        Same model with randomized weights
    """
    rng = np.random.RandomState(random_state)

    with torch.no_grad():
        for name, param in model.named_parameters():
            # Shuffle weights within each parameter tensor
            flat = param.data.view(-1).cpu().numpy()
            rng.shuffle(flat)
            param.data = torch.from_numpy(flat.reshape(param.shape)).to(param.device)

    return model


class ControlEvaluator:
    """
    Run negative control experiments for validating probe results.

    All controls should produce AUC ~0.50. If they don't, it indicates:
    - Shuffled labels > 0.50: Data leakage or probe overfitting
    - Permuted features > 0.50: Features have spurious structure
    - Random network > 0.50: Signal is from data, not model
    """

    def __init__(
        self,
        output_dir: str = 'outputs',
        device: Optional[torch.device] = None,
        n_shuffles: int = 5,  # Number of shuffle permutations to average
        tolerance: float = 0.05,  # How close to 0.50 is "chance level"
    ):
        self.output_dir = Path(output_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_shuffles = n_shuffles
        self.tolerance = tolerance

        self.results: List[ControlResult] = []

    def _split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/test."""
        from sklearn.model_selection import train_test_split

        return train_test_split(
            features, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels if len(np.unique(labels)) >= 2 else None
        )

    def run_shuffled_labels_control(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        concept_name: str,
        layer_name: str,
        method: str = 'msae',
        k: Optional[int] = None,
        verbose: bool = False,
    ) -> ControlResult:
        """
        Run shuffled labels control.

        Randomly permutes labels and retrains probe. Expected AUC ~0.50.
        Averages over multiple shuffles for stability.

        Args:
            features: (n_samples, n_features) feature array
            labels: (n_samples,) binary labels
            concept_name: Name of concept being tested
            layer_name: Layer name (e.g., 'block5')
            method: Method name ('msae', 'baseline', 'raw')
            k: k level for MSAE
            verbose: Print progress

        Returns:
            ControlResult with AUC and confidence interval
        """
        if verbose:
            print(f"  Running shuffled labels control: {concept_name}/{layer_name}/{method}")

        aucs = []

        for shuffle_idx in range(self.n_shuffles):
            # Shuffle labels with different seed each time
            shuffled_labels = shuffle_labels(labels, random_state=42 + shuffle_idx)

            # Split data
            X_train, X_test, y_train, y_test = self._split_data(
                features, shuffled_labels, random_state=42 + shuffle_idx
            )

            # Skip if not enough samples of each class
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            # Train probe
            probe = train_probe(
                X_train, y_train,
                device=self.device,
                n_epochs=50,  # Fewer epochs for controls
                verbose=False,
            )

            # Evaluate
            auc, _, _ = evaluate_probe(
                probe, X_test, y_test,
                device=self.device,
                n_bootstrap=50,
            )
            aucs.append(auc)

        # Compute mean and CI across shuffles
        if aucs:
            mean_auc = np.mean(aucs)
            ci_low = np.percentile(aucs, 2.5) if len(aucs) > 1 else mean_auc
            ci_high = np.percentile(aucs, 97.5) if len(aucs) > 1 else mean_auc
        else:
            mean_auc = ci_low = ci_high = 0.5

        result = ControlResult(
            control_type='shuffled',
            concept=concept_name,
            layer=layer_name,
            method=method,
            k=k,
            auc=mean_auc,
            ci_low=ci_low,
            ci_high=ci_high,
        )

        self.results.append(result)

        if verbose:
            print(f"    Shuffled AUC: {mean_auc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

        return result

    def run_permuted_features_control(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        concept_name: str,
        layer_name: str,
        method: str = 'msae',
        k: Optional[int] = None,
        verbose: bool = False,
    ) -> ControlResult:
        """
        Run permuted features control.

        Randomly permutes feature dimensions and retrains probe.
        Expected AUC ~0.50. This tests whether the geometric structure
        of features matters.

        Args:
            features: (n_samples, n_features) feature array
            labels: (n_samples,) binary labels
            concept_name: Name of concept being tested
            layer_name: Layer name (e.g., 'block5')
            method: Method name ('msae', 'baseline', 'raw')
            k: k level for MSAE
            verbose: Print progress

        Returns:
            ControlResult with AUC and confidence interval
        """
        if verbose:
            print(f"  Running permuted features control: {concept_name}/{layer_name}/{method}")

        aucs = []

        for perm_idx in range(self.n_shuffles):
            # Permute features with different seed each time
            permuted_features = permute_features(features, random_state=42 + perm_idx)

            # Split data
            X_train, X_test, y_train, y_test = self._split_data(
                permuted_features, labels, random_state=42 + perm_idx
            )

            # Skip if not enough samples
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            # Train probe
            probe = train_probe(
                X_train, y_train,
                device=self.device,
                n_epochs=50,
                verbose=False,
            )

            # Evaluate
            auc, _, _ = evaluate_probe(
                probe, X_test, y_test,
                device=self.device,
                n_bootstrap=50,
            )
            aucs.append(auc)

        # Compute mean and CI
        if aucs:
            mean_auc = np.mean(aucs)
            ci_low = np.percentile(aucs, 2.5) if len(aucs) > 1 else mean_auc
            ci_high = np.percentile(aucs, 97.5) if len(aucs) > 1 else mean_auc
        else:
            mean_auc = ci_low = ci_high = 0.5

        result = ControlResult(
            control_type='permuted_features',
            concept=concept_name,
            layer=layer_name,
            method=method,
            k=k,
            auc=mean_auc,
            ci_low=ci_low,
            ci_high=ci_high,
        )

        self.results.append(result)

        if verbose:
            print(f"    Permuted AUC: {mean_auc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

        return result

    def run_random_network_control(
        self,
        activations_shape: Tuple[int, int],
        labels: np.ndarray,
        concept_name: str,
        layer_name: str,
        sae_model: Optional[nn.Module] = None,
        k: Optional[int] = None,
        verbose: bool = False,
    ) -> ControlResult:
        """
        Run random network control.

        Uses random activations (simulating an untrained network) and
        trains probes. Expected AUC ~0.50. This tests whether signal
        comes from the model or just data correlations.

        Args:
            activations_shape: (n_samples, activation_dim) shape
            labels: (n_samples,) binary labels
            concept_name: Name of concept being tested
            layer_name: Layer name
            sae_model: Optional SAE to encode random activations
            k: k level for MSAE
            verbose: Print progress

        Returns:
            ControlResult with AUC and confidence interval
        """
        if verbose:
            print(f"  Running random network control: {concept_name}/{layer_name}")

        aucs = []

        for rand_idx in range(self.n_shuffles):
            # Generate random activations
            random_activations = create_random_network_activations(
                activations_shape, random_state=42 + rand_idx
            )

            # If SAE model provided, encode through it
            if sae_model is not None:
                sae_model.eval()
                device = next(sae_model.parameters()).device

                with torch.no_grad():
                    batch_size = 4096
                    features_list = []

                    for i in range(0, len(random_activations), batch_size):
                        batch = torch.from_numpy(
                            random_activations[i:i + batch_size]
                        ).float().to(device)

                        # Check if model accepts k parameter (MSAE) or not (BaselineSAE)
                        # BaselineSAE has fixed k internally, MSAE accepts k as argument
                        if k is not None and hasattr(sae_model, 'k_levels'):
                            # MSAE with specified k
                            _, _, z = sae_model.forward(batch, k)
                        else:
                            # BaselineSAE or MSAE without k
                            _, _, z = sae_model.forward(batch)

                        features_list.append(z.cpu().numpy())

                    features = np.concatenate(features_list, axis=0)
            else:
                features = random_activations

            # Split data
            X_train, X_test, y_train, y_test = self._split_data(
                features, labels, random_state=42 + rand_idx
            )

            # Skip if not enough samples
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            # Train probe
            probe = train_probe(
                X_train, y_train,
                device=self.device,
                n_epochs=50,
                verbose=False,
            )

            # Evaluate
            auc, _, _ = evaluate_probe(
                probe, X_test, y_test,
                device=self.device,
                n_bootstrap=50,
            )
            aucs.append(auc)

        # Compute mean and CI
        if aucs:
            mean_auc = np.mean(aucs)
            ci_low = np.percentile(aucs, 2.5) if len(aucs) > 1 else mean_auc
            ci_high = np.percentile(aucs, 97.5) if len(aucs) > 1 else mean_auc
        else:
            mean_auc = ci_low = ci_high = 0.5

        result = ControlResult(
            control_type='random_network',
            concept=concept_name,
            layer=layer_name,
            method='random',
            k=k,
            auc=mean_auc,
            ci_low=ci_low,
            ci_high=ci_high,
        )

        self.results.append(result)

        if verbose:
            print(f"    Random network AUC: {mean_auc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

        return result

    def run_all_controls(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        concept_name: str,
        layer_name: str,
        method: str = 'msae',
        k: Optional[int] = None,
        include_random_network: bool = True,
        sae_model: Optional[nn.Module] = None,
        activation_dim: int = 256,
        verbose: bool = True,
    ) -> ControlSummary:
        """
        Run all negative controls for a concept/layer/method combination.

        Args:
            features: (n_samples, n_features) feature array
            labels: (n_samples,) binary labels
            concept_name: Name of concept
            layer_name: Layer name
            method: Method name
            k: k level for MSAE
            include_random_network: Whether to run random network control
            sae_model: SAE model for random network control
            activation_dim: Dimension of raw activations
            verbose: Print progress

        Returns:
            ControlSummary with all control results
        """
        if verbose:
            print(f"\nRunning controls for {concept_name}/{layer_name}/{method}" +
                  (f"/k={k}" if k else ""))

        # Shuffled labels
        shuffled_result = self.run_shuffled_labels_control(
            features, labels, concept_name, layer_name, method, k, verbose
        )

        # Permuted features
        permuted_result = self.run_permuted_features_control(
            features, labels, concept_name, layer_name, method, k, verbose
        )

        # Random network (optional)
        random_auc = None
        if include_random_network:
            random_result = self.run_random_network_control(
                activations_shape=(len(labels), activation_dim),
                labels=labels,
                concept_name=concept_name,
                layer_name=layer_name,
                sae_model=sae_model,
                k=k,
                verbose=verbose,
            )
            random_auc = random_result.auc

        # Check if all near chance
        all_near_chance = (
            abs(shuffled_result.auc - 0.5) <= self.tolerance and
            abs(permuted_result.auc - 0.5) <= self.tolerance
        )
        if random_auc is not None:
            all_near_chance = all_near_chance and abs(random_auc - 0.5) <= self.tolerance

        return ControlSummary(
            concept=concept_name,
            layer=layer_name,
            method=method,
            k=k,
            shuffled_auc=shuffled_result.auc,
            permuted_auc=permuted_result.auc,
            random_network_auc=random_auc,
            all_near_chance=all_near_chance,
        )

    def save_results(self, filename: str = 'controls.json'):
        """
        Save all control results to JSON.

        Output format matches CLAUDE.md specification.
        """
        results_path = self.output_dir / 'results' / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # Organize by control type
        output = {
            'shuffled': {},
            'permuted': {},
            'random_network': {},
        }

        for result in self.results:
            control_dict = output[result.control_type]

            if result.concept not in control_dict:
                control_dict[result.concept] = {}
            if result.layer not in control_dict[result.concept]:
                control_dict[result.concept][result.layer] = {}

            key = f"{result.method}_k{result.k}" if result.k else result.method
            control_dict[result.concept][result.layer][key] = {
                'auc': result.auc,
                'ci_low': result.ci_low,
                'ci_high': result.ci_high,
                'expected': result.expected_auc,
            }

        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Saved control results to {results_path}")

    def validate_results(self) -> Dict[str, bool]:
        """
        Validate that all controls are near chance level.

        Returns:
            Dict mapping control_type -> whether all results are valid
        """
        validation = {
            'shuffled': True,
            'permuted': True,
            'random_network': True,
        }

        for result in self.results:
            if abs(result.auc - 0.5) > self.tolerance:
                validation[result.control_type] = False

        return validation

    def print_summary(self):
        """Print a summary of control results."""
        print(f"\n{'='*60}")
        print("Negative Control Summary")
        print(f"{'='*60}")

        # Group by control type
        by_type = {'shuffled': [], 'permuted_features': [], 'random_network': []}
        for result in self.results:
            by_type.get(result.control_type, []).append(result)

        for control_type, results in by_type.items():
            if not results:
                continue

            aucs = [r.auc for r in results]
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)

            status = "PASS" if abs(mean_auc - 0.5) <= self.tolerance else "FAIL"
            print(f"\n{control_type}:")
            print(f"  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
            print(f"  Expected: 0.50")
            print(f"  Status: {status}")
