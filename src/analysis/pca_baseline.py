"""
PCA Baseline Analysis for SAE comparison.

Compares PCA to SAE at multiple component levels to justify
the use of sparse representations.

Reference: MASTER_IMPLEMENTATION_PLAN.md - Notebook 04
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle


@dataclass
class PCAResult:
    """Results from PCA at a single k level."""
    k: int
    pca: PCA
    variance_explained: float
    components: np.ndarray  # [k, input_dim]


@dataclass
class ComparisonResult:
    """Comparison between PCA and SAE at a single k level."""
    k: int
    pca_variance: float
    sae_r2: float
    pca_sparsity: float
    sae_sparsity: float


class PCABaseline:
    """
    Fit and analyze PCA at multiple component levels for comparison with MSAE.

    Usage:
        baseline = PCABaseline(input_dim=256)
        pca_results = baseline.fit_multiple_k(activations, k_levels=[16, 32, 64, 128])
        comparison = baseline.compare_to_sae(sae_model, activations, k_levels, device)
    """

    def __init__(self, input_dim: int = 256):
        """
        Args:
            input_dim: Dimension of input activations (256 for Leela Zero)
        """
        self.input_dim = input_dim
        self.pca_models: Dict[int, PCAResult] = {}

    def fit_multiple_k(
        self,
        activations: np.ndarray,
        k_levels: List[int] = [16, 32, 64, 128]
    ) -> Dict[int, PCAResult]:
        """
        Fit PCA at multiple component levels.

        Args:
            activations: [n_samples, input_dim] activation array (normalized)
            k_levels: List of component counts to try

        Returns:
            Dict mapping k -> PCAResult
        """
        results = {}

        for k in k_levels:
            print(f"Fitting PCA with k={k} components...")

            pca = PCA(n_components=k)
            pca.fit(activations)

            variance_explained = pca.explained_variance_ratio_.sum()

            result = PCAResult(
                k=k,
                pca=pca,
                variance_explained=variance_explained,
                components=pca.components_,
            )

            results[k] = result
            self.pca_models[k] = result

            print(f"  Variance explained: {variance_explained:.4f}")

        return results

    def compute_reconstruction_r2(
        self,
        activations: np.ndarray,
        k_levels: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute reconstruction R² for each k level.

        Args:
            activations: [n_samples, input_dim] activation array
            k_levels: List of k values (uses all fitted if None)

        Returns:
            Dict mapping k -> R² value
        """
        if k_levels is None:
            k_levels = list(self.pca_models.keys())

        results = {}

        # Total variance
        total_var = np.var(activations) * activations.size

        for k in k_levels:
            if k not in self.pca_models:
                raise ValueError(f"PCA not fitted for k={k}")

            pca = self.pca_models[k].pca

            # Transform and inverse transform
            transformed = pca.transform(activations)
            reconstructed = pca.inverse_transform(transformed)

            # Compute R²
            mse = np.sum((activations - reconstructed) ** 2)
            r2 = 1.0 - mse / total_var

            results[k] = r2

        return results

    def compare_to_sae(
        self,
        sae_model,
        activations: np.ndarray,
        k_levels: List[int],
        device: torch.device,
        batch_size: int = 4096
    ) -> List[ComparisonResult]:
        """
        Compare PCA variance explained vs SAE R² at each k level.

        Args:
            sae_model: Trained MatryoshkaSAE model
            activations: [n_samples, input_dim] normalized activations
            k_levels: [16, 32, 64, 128]
            device: Torch device
            batch_size: Batch size for SAE inference

        Returns:
            List of ComparisonResult for each k
        """
        results = []

        # Compute PCA R² if not already done
        pca_r2 = self.compute_reconstruction_r2(activations, k_levels)

        # Compute SAE R² at each k
        sae_model.eval()
        activations_tensor = torch.from_numpy(activations).float().to(device)

        for k in k_levels:
            print(f"Comparing at k={k}...")

            # SAE R²
            total_mse = 0.0
            total_var = 0.0

            with torch.no_grad():
                for i in range(0, len(activations_tensor), batch_size):
                    batch = activations_tensor[i:i+batch_size]

                    # Get reconstruction at this k
                    reconstructions = sae_model.forward_hierarchical(batch)
                    x_hat = reconstructions[k]

                    total_mse += ((batch - x_hat) ** 2).sum().item()
                    total_var += (batch.var() * batch.numel()).item()

            sae_r2 = 1.0 - total_mse / total_var

            # Compute sparsity
            pca_sparsity = self._compute_pca_sparsity(k)
            sae_sparsity = self._compute_sae_sparsity(sae_model, activations_tensor, k, batch_size)

            result = ComparisonResult(
                k=k,
                pca_variance=pca_r2[k],
                sae_r2=sae_r2,
                pca_sparsity=pca_sparsity,
                sae_sparsity=sae_sparsity,
            )

            results.append(result)

            print(f"  PCA R²: {pca_r2[k]:.4f}, SAE R²: {sae_r2:.4f}")
            print(f"  PCA sparsity: {pca_sparsity:.4f}, SAE sparsity: {sae_sparsity:.4f}")

        return results

    def _compute_pca_sparsity(self, k: int, threshold: float = 0.01) -> float:
        """
        Compute sparsity of PCA components.

        PCA is dense by nature - this measures fraction of near-zero loadings.

        Args:
            k: Number of components
            threshold: Values below this are considered "sparse"

        Returns:
            Fraction of loadings below threshold (typically near 0)
        """
        if k not in self.pca_models:
            return 0.0

        components = self.pca_models[k].components
        # Normalize each component to unit norm for fair comparison
        norms = np.linalg.norm(components, axis=1, keepdims=True)
        normalized = components / (norms + 1e-8)

        # Fraction of normalized values below threshold
        sparsity = (np.abs(normalized) < threshold).mean()

        return sparsity

    def _compute_sae_sparsity(
        self,
        sae_model,
        activations: torch.Tensor,
        k: int,
        batch_size: int
    ) -> float:
        """
        Compute sparsity of SAE features at given k.

        TopK SAE has exactly k non-zero features per sample.
        Sparsity = (hidden_dim - k) / hidden_dim

        Args:
            sae_model: MatryoshkaSAE model
            activations: [n_samples, input_dim] tensor
            k: Sparsity level
            batch_size: Batch size

        Returns:
            Fraction of zero activations
        """
        # For TopK SAE, sparsity is deterministic
        hidden_dim = sae_model.hidden_dim
        sparsity = 1.0 - (k / hidden_dim)
        return sparsity

    def save_models(self, output_dir: str):
        """
        Save fitted PCA models to disk.

        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir) / 'pca_models'
        output_path.mkdir(parents=True, exist_ok=True)

        for k, result in self.pca_models.items():
            model_path = output_path / f'pca_k{k}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(result, f)
            print(f"Saved PCA k={k} to {model_path}")

    def load_models(self, output_dir: str, k_levels: List[int]):
        """
        Load PCA models from disk.

        Args:
            output_dir: Directory containing models
            k_levels: Which k levels to load
        """
        output_path = Path(output_dir) / 'pca_models'

        for k in k_levels:
            model_path = output_path / f'pca_k{k}.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.pca_models[k] = pickle.load(f)
                print(f"Loaded PCA k={k} from {model_path}")


def compare_pca_sae_summary(
    comparison_results: List[ComparisonResult],
    k_levels: List[int]
) -> Dict:
    """
    Create summary dict of PCA vs SAE comparison.

    Args:
        comparison_results: List of ComparisonResult
        k_levels: List of k values

    Returns:
        Summary dict suitable for JSON serialization
    """
    summary = {}

    for result in comparison_results:
        summary[f'k{result.k}'] = {
            'pca_r2': result.pca_variance,
            'sae_r2': result.sae_r2,
            'r2_difference': result.sae_r2 - result.pca_variance,
            'pca_sparsity': result.pca_sparsity,
            'sae_sparsity': result.sae_sparsity,
        }

    # Add overall statistics
    pca_r2_max = max(r.pca_variance for r in comparison_results)
    sae_r2_max = max(r.sae_r2 for r in comparison_results)

    summary['overall'] = {
        'pca_r2_at_max_k': pca_r2_max,
        'sae_r2_at_max_k': sae_r2_max,
        'sae_advantage': sae_r2_max - pca_r2_max,
        'sae_wins_all_k': all(r.sae_r2 >= r.pca_variance for r in comparison_results),
    }

    return summary
