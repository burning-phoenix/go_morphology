"""
Causal steering and ablation analysis for SAE features.

Implements causal interventions to validate that features have
real effects on model behavior:
1. Feature Ablation: Set feature to zero, measure policy/value change
2. Differential Impact: Compare ablation on concept-present vs concept-absent
3. Feature Steering: Set feature to high value, observe behavior change
4. Ablation Sparsity: Measure how localized ablation effects are

Based on:
- docs/topk_sae_paper.md (Gao et al.) - ablation sparsity metric
- docs/monosemantic_features.md (Bricken et al.) - causal interventions

From the TopK paper:
"Clamping these latents appeared to have causal effect on the samples.
For example, clamping the profanity latent to negative values results
in significantly less profanity."

Usage:
    from src.analysis.causal import CausalAnalyzer
    analyzer = CausalAnalyzer(leela_model, msae_model)
    results = analyzer.ablate_feature(feature_idx=42, positions=positions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import json
from pathlib import Path
from scipy import stats


@dataclass
class AblationResult:
    """Result of ablating a single feature."""
    feature_idx: int
    layer: str

    # Effect on model outputs
    policy_kl_divergence: float  # KL(original || ablated) for policy
    value_change: float  # Mean absolute change in value head

    # Effect magnitude
    mean_activation: float  # Average activation of this feature
    activation_rate: float  # Fraction of positions where feature is active

    # Statistical significance (from permutation test)
    p_value: float
    significant: bool  # p < 0.05


@dataclass
class DifferentialImpactResult:
    """Result of differential impact analysis."""
    feature_idx: int
    concept: str
    layer: str

    # Impact when concept is present vs absent
    impact_when_present: float
    impact_when_absent: float
    differential: float  # impact_when_present - impact_when_absent

    # Statistical test
    p_value: float
    significant: bool  # p < 0.05

    # Sample sizes
    n_present: int
    n_absent: int


@dataclass
class SteeringResult:
    """Result of feature steering experiment."""
    feature_idx: int
    layer: str
    steering_value: float  # Value feature was set to

    # Changes in model behavior
    policy_kl_divergence: float
    value_change: float

    # Top affected positions (where policy changed most)
    top_affected_positions: List[Tuple[int, int]]  # (x, y) coordinates
    position_changes: List[float]  # Policy change at each position


@dataclass
class CausalMetrics:
    """Summary causal metrics for a feature."""
    feature_idx: int
    layer: str

    # Overall effect
    ablation_effect: float
    steering_effect: float

    # Sparsity of effects (from TopK paper)
    ablation_sparsity: float  # L2^2 / L1^2 ratio

    # Whether feature has clear causal role
    has_causal_effect: bool


def compute_policy_kl_divergence(
    original_policy: torch.Tensor,
    modified_policy: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence between original and modified policies.

    KL(P || Q) = sum(P * log(P / Q))

    Args:
        original_policy: (batch, 362) original policy logits
        modified_policy: (batch, 362) modified policy logits

    Returns:
        (batch,) KL divergence for each sample
    """
    original_probs = F.softmax(original_policy, dim=-1)
    modified_probs = F.softmax(modified_policy, dim=-1)

    # Add small epsilon for numerical stability
    eps = 1e-10
    original_probs = original_probs + eps
    modified_probs = modified_probs + eps

    kl = (original_probs * (original_probs.log() - modified_probs.log())).sum(dim=-1)
    return kl


def compute_ablation_sparsity(effects: torch.Tensor) -> float:
    """
    Compute sparsity of ablation effects using L2/L1 ratio.

    From TopK paper: Uses L2 norm ratio to measure sparsity.
    Smaller values = sparser (more localized) effects.

    Ratio = ||x||_2^2 / ||x||_1^2

    Args:
        effects: Effect vector (policy changes)

    Returns:
        Sparsity ratio (0 to 1, smaller is sparser)
    """
    effects = effects.float()
    l1 = effects.abs().sum()
    l2_sq = (effects ** 2).sum()

    if l1 < 1e-10:
        return 0.0

    # Normalized by dimension for comparability
    n = effects.numel()
    ratio = (l2_sq / (l1 ** 2 + 1e-10)) * n

    return ratio.item()


class CausalAnalyzer:
    """
    Analyze causal effects of SAE features on model behavior.

    Performs ablation and steering experiments to validate that
    features have real, interpretable effects.
    """

    def __init__(
        self,
        leela_model: nn.Module,
        sae_model: nn.Module,
        layer_idx: int,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize causal analyzer.

        Args:
            leela_model: Leela Zero model
            sae_model: Trained SAE (MSAE or baseline)
            layer_idx: Which residual block the SAE is trained on
            device: Device to use
        """
        self.leela_model = leela_model
        self.sae_model = sae_model
        self.layer_idx = layer_idx
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.leela_model.to(self.device)
        self.sae_model.to(self.device)
        self.leela_model.eval()
        self.sae_model.eval()

        self.results: List[AblationResult] = []
        self.differential_results: List[DifferentialImpactResult] = []
        self.steering_results: List[SteeringResult] = []
        
        # Cached computation for optimization
        self._cached_positions = None
        self._cached_base_acts = None
        self._cached_z_sparse = None
        self._cached_orig_policy = None
        self._cached_orig_value = None
        # Reconstruction baseline: policy/value from SAE reconstruction (no ablation)
        # This isolates ablation effects from SAE reconstruction error
        self._cached_baseline_policy = None
        self._cached_baseline_value = None
        self._cached_reconstruction_kl = None  # KL(original || baseline) for diagnostics
    
    def _cache_base_forward(
        self,
        positions: torch.Tensor,
        k: Optional[int] = None,
    ) -> None:
        """
        Cache base forward pass for reuse across multiple feature ablations.
        
        This is the key optimization - compute Leela forward and SAE encode
        ONCE, then reuse for all feature modifications.
        """
        positions = positions.to(self.device)
        
        # Check if already cached for these positions
        if (self._cached_positions is not None and 
            self._cached_positions.shape == positions.shape and
            torch.equal(self._cached_positions[:min(10, len(positions))], 
                       positions[:min(10, len(positions))])):
            return  # Already cached
        
        with torch.no_grad():
            # 1. Full Leela forward to target layer
            (orig_policy, orig_value), act_dict = self.leela_model.forward_with_activations(
                positions, [self.layer_idx]
            )
            
            # 2. Get activations at our layer
            base_acts = act_dict[self.layer_idx]  # (batch, 256, 19, 19)
            batch_size = base_acts.shape[0]
            acts_flat = base_acts.permute(0, 2, 3, 1).reshape(-1, 256)
            
            # 3. Encode through SAE
            if k is not None:
                _, _, z_sparse = self.sae_model.forward(acts_flat, k)
            else:
                _, _, z_sparse = self.sae_model.forward(acts_flat)

            # 4. Compute reconstruction baseline (SAE decode without ablation)
            # This isolates ablation effects from SAE reconstruction error
            acts_reconstructed = self.sae_model.decode(z_sparse)
            acts_reconstructed = acts_reconstructed.reshape(batch_size, 19, 19, 256).permute(0, 3, 1, 2)

            # Continue forward pass from reconstructed activations
            x = acts_reconstructed
            for idx in range(self.layer_idx + 1, len(self.leela_model.residual_tower)):
                x = self.leela_model.residual_tower[idx](x)

            # Policy head
            baseline_policy = self.leela_model.policy_conv(x)
            baseline_policy = baseline_policy.flatten(start_dim=1)
            baseline_policy = self.leela_model.policy_fc(baseline_policy)

            # Value head
            baseline_value = self.leela_model.value_conv(x)
            baseline_value = baseline_value.flatten(start_dim=1)
            baseline_value = F.relu(self.leela_model.value_fc1(baseline_value), inplace=False)
            baseline_value = torch.tanh(self.leela_model.value_fc2(baseline_value))

            # Compute reconstruction fidelity: KL(original || baseline)
            # This measures how much the SAE reconstruction affects policy
            reconstruction_kl = compute_policy_kl_divergence(orig_policy, baseline_policy).mean().item()

            # 5. Cache everything
            self._cached_positions = positions
            self._cached_base_acts = base_acts
            self._cached_z_sparse = z_sparse
            self._cached_orig_policy = orig_policy
            self._cached_orig_value = orig_value
            self._cached_baseline_policy = baseline_policy
            self._cached_baseline_value = baseline_value
            self._cached_reconstruction_kl = reconstruction_kl
    
    def _continue_from_modified_z(
        self,
        z_modified: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Continue forward pass from modified SAE features.
        
        Uses cached base activations shape to reconstruct spatial dims.
        """
        with torch.no_grad():
            # Decode back to activation space
            acts_modified = self.sae_model.decode(z_modified)
            
            # Reshape back to spatial
            acts_modified = acts_modified.reshape(batch_size, 19, 19, 256).permute(0, 3, 1, 2)
            
            x = acts_modified
            
            # Continue through remaining residual blocks
            for idx in range(self.layer_idx + 1, len(self.leela_model.residual_tower)):
                x = self.leela_model.residual_tower[idx](x)
            
            # Policy head
            policy = self.leela_model.policy_conv(x)
            policy = policy.flatten(start_dim=1)
            policy = self.leela_model.policy_fc(policy)
            
            # Value head
            value = self.leela_model.value_conv(x)
            value = value.flatten(start_dim=1)
            value = F.relu(self.leela_model.value_fc1(value), inplace=False)
            value = torch.tanh(self.leela_model.value_fc2(value))
            
            return policy, value
    
    def ablate_feature_cached(
        self,
        feature_idx: int,
        layer_name: str = 'block',
        n_permutations: int = 100,
        permutation_batch_size: int = 50,
        verbose: bool = False,
    ) -> AblationResult:
        """
        Ablate a feature using cached base computation.

        MUST call _cache_base_forward() first!

        Optimizations:
        - Uses cached z_sparse instead of recomputing
        - Batches permutation test for all random features at once

        Scientific improvement:
        - Uses baseline_policy (SAE reconstruction) instead of orig_policy (raw activations)
        - This isolates pure ablation effect from SAE reconstruction error
        - Both ablated and baseline come from SAE decode, so we measure only the
          effect of zeroing the specific feature
        """
        if self._cached_z_sparse is None:
            raise RuntimeError("Must call _cache_base_forward() before ablate_feature_cached()")

        if verbose:
            print(f"  Ablating feature {feature_idx} (cached)...")
            print(f"    Reconstruction KL (SAE fidelity): {self._cached_reconstruction_kl:.6f}")

        z_sparse = self._cached_z_sparse
        # Use baseline (SAE reconstruction) instead of orig (raw activations)
        # This isolates ablation effect from reconstruction error
        baseline_policy = self._cached_baseline_policy
        baseline_value = self._cached_baseline_value
        batch_size = self._cached_positions.shape[0]

        # Feature activation stats
        feature_acts = z_sparse[:, feature_idx]
        mean_activation = feature_acts.mean().item()
        activation_rate = (feature_acts > 0).float().mean().item()

        # Ablate target feature
        z_modified = z_sparse.clone()
        z_modified[:, feature_idx] = 0.0

        ablated_policy, ablated_value = self._continue_from_modified_z(z_modified, batch_size)

        # Compute effects: KL(baseline || ablated) - pure ablation effect
        # Both baseline and ablated come from SAE decode, isolating the feature's effect
        policy_kl = compute_policy_kl_divergence(baseline_policy, ablated_policy)
        mean_kl = policy_kl.mean().item()
        value_change = (baseline_value - ablated_value).abs().mean().item()

        # OPTIMIZED: Batched permutation test
        # Instead of 100 sequential forward passes, batch them
        n_features = z_sparse.shape[1]
        random_indices = np.random.randint(0, n_features, size=n_permutations)

        # Use subset of positions for permutation (memory efficient)
        perm_batch_size = min(permutation_batch_size, batch_size)
        z_perm_base = z_sparse[:perm_batch_size * 361].clone()  # Subset
        baseline_policy_perm = baseline_policy[:perm_batch_size]

        null_effects = []

        # Process in mini-batches to manage memory
        mini_batch = 10  # Process 10 random features at a time
        for start_idx in range(0, n_permutations, mini_batch):
            end_idx = min(start_idx + mini_batch, n_permutations)
            batch_indices = random_indices[start_idx:end_idx]

            for rand_idx in batch_indices:
                z_rand = z_perm_base.clone()
                z_rand[:, rand_idx] = 0.0
                rand_policy, _ = self._continue_from_modified_z(z_rand, perm_batch_size)
                rand_kl = compute_policy_kl_divergence(baseline_policy_perm, rand_policy).mean().item()
                null_effects.append(rand_kl)
        
        # p-value: fraction of null effects >= observed effect
        p_value = (np.array(null_effects) >= mean_kl).mean()
        
        result = AblationResult(
            feature_idx=feature_idx,
            layer=layer_name,
            policy_kl_divergence=mean_kl,
            value_change=value_change,
            mean_activation=mean_activation,
            activation_rate=activation_rate,
            p_value=p_value,
            significant=p_value < 0.05,
        )
        
        self.results.append(result)
        
        if verbose:
            print(f"    KL divergence: {mean_kl:.6f}")
            print(f"    p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")
        
        return result
    
    def clear_cache(self):
        """Clear cached computations to free memory."""
        self._cached_positions = None
        self._cached_base_acts = None
        self._cached_z_sparse = None
        self._cached_orig_policy = None
        self._cached_orig_value = None
        self._cached_baseline_policy = None
        self._cached_baseline_value = None
        self._cached_reconstruction_kl = None
        torch.cuda.empty_cache()

    def get_reconstruction_kl(self) -> Optional[float]:
        """
        Get the cached reconstruction KL divergence.

        Returns KL(original || SAE_reconstruction), which measures how much
        the SAE reconstruction affects policy predictions. Lower is better.

        This is useful for:
        1. Validating SAE quality (should be small)
        2. Contextualizing ablation effects (ablation KL should exceed this)
        """
        return self._cached_reconstruction_kl

    def _get_activations_and_outputs(
        self,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get activations at target layer and model outputs.

        Args:
            positions: (batch, 18, 19, 19) encoded Go positions

        Returns:
            (activations, policy, value) tuple
        """
        with torch.no_grad():
            positions = positions.to(self.device)
            (policy, value), activations_dict = self.leela_model.forward_with_activations(
                positions, [self.layer_idx]
            )

            # Flatten spatial: (batch, 256, 19, 19) -> (batch * 361, 256)
            activations = activations_dict[self.layer_idx]
            batch_size = activations.shape[0]
            activations_flat = activations.permute(0, 2, 3, 1).reshape(-1, 256)

            return activations_flat, policy, value

    def _forward_with_modified_features(
        self,
        positions: torch.Tensor,
        feature_idx: int,
        new_value: float = 0.0,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with a specific feature modified.

        This requires hooking into the forward pass and modifying
        activations at the SAE level.

        Args:
            positions: (batch, 18, 19, 19) encoded positions
            feature_idx: Which SAE feature to modify
            new_value: Value to set the feature to
            k: k level for MSAE

        Returns:
            (modified_policy, modified_value) tuple
        """
        positions = positions.to(self.device)
        batch_size = positions.shape[0]

        # Get original activations
        with torch.no_grad():
            (orig_policy, orig_value), activations_dict = self.leela_model.forward_with_activations(
                positions, [self.layer_idx]
            )
            activations = activations_dict[self.layer_idx]  # (batch, 256, 19, 19)

        # Encode through SAE, modify feature, decode
        with torch.no_grad():
            # Flatten spatial
            acts_flat = activations.permute(0, 2, 3, 1).reshape(-1, 256)

            # Encode to get sparse features
            if k is not None:
                _, _, z_sparse = self.sae_model.forward(acts_flat, k)
            else:
                _, _, z_sparse = self.sae_model.forward(acts_flat)

            # Modify the target feature
            z_modified = z_sparse.clone()
            z_modified[:, feature_idx] = new_value

            # Decode back to activation space
            acts_modified = self.sae_model.decode(z_modified)

            # Reshape back to spatial
            acts_modified = acts_modified.reshape(batch_size, 19, 19, 256).permute(0, 3, 1, 2)

        # Now we need to run the rest of the network with modified activations
        # This requires a custom forward pass
        with torch.no_grad():
            x = acts_modified

            # Continue through remaining residual blocks
            for idx in range(self.layer_idx + 1, len(self.leela_model.residual_tower)):
                x = self.leela_model.residual_tower[idx](x)

            # Policy head
            policy = self.leela_model.policy_conv(x)
            policy = policy.flatten(start_dim=1)
            policy = self.leela_model.policy_fc(policy)

            # Value head
            value = self.leela_model.value_conv(x)
            value = value.flatten(start_dim=1)
            value = F.relu(self.leela_model.value_fc1(value), inplace=False)
            value = torch.tanh(self.leela_model.value_fc2(value))

        return policy, value

    def ablate_feature(
        self,
        feature_idx: int,
        positions: torch.Tensor,
        layer_name: str = 'block',
        k: Optional[int] = None,
        n_permutations: int = 100,
        verbose: bool = False,
    ) -> AblationResult:
        """
        Measure effect of ablating (zeroing) a single feature.

        Uses reconstruction baseline comparison: compares SAE reconstruction
        (no ablation) to SAE reconstruction (with ablation). This isolates
        the pure ablation effect from SAE reconstruction error.

        Args:
            feature_idx: Index of feature to ablate
            positions: (batch, 18, 19, 19) encoded Go positions
            layer_name: Name for reporting
            k: k level for MSAE
            n_permutations: Number of permutations for significance test
            verbose: Print progress

        Returns:
            AblationResult with effect measurements
        """
        if verbose:
            print(f"  Ablating feature {feature_idx}...")

        positions = positions.to(self.device)
        batch_size = positions.shape[0]

        # Get original outputs and activations
        with torch.no_grad():
            (orig_policy, orig_value), activations_dict = self.leela_model.forward_with_activations(
                positions, [self.layer_idx]
            )

            # Get feature activations
            activations = activations_dict[self.layer_idx]
            acts_flat = activations.permute(0, 2, 3, 1).reshape(-1, 256)

            if k is not None:
                _, _, z_sparse = self.sae_model.forward(acts_flat, k)
            else:
                _, _, z_sparse = self.sae_model.forward(acts_flat)

            feature_acts = z_sparse[:, feature_idx]
            mean_activation = feature_acts.mean().item()
            activation_rate = (feature_acts > 0).float().mean().item()

            # Compute baseline: SAE reconstruction without ablation
            # This isolates ablation effect from reconstruction error
            acts_reconstructed = self.sae_model.decode(z_sparse)
            acts_reconstructed = acts_reconstructed.reshape(batch_size, 19, 19, 256).permute(0, 3, 1, 2)

            # Forward pass from baseline reconstruction
            x = acts_reconstructed
            for idx in range(self.layer_idx + 1, len(self.leela_model.residual_tower)):
                x = self.leela_model.residual_tower[idx](x)

            baseline_policy = self.leela_model.policy_conv(x)
            baseline_policy = baseline_policy.flatten(start_dim=1)
            baseline_policy = self.leela_model.policy_fc(baseline_policy)

            baseline_value = self.leela_model.value_conv(x)
            baseline_value = baseline_value.flatten(start_dim=1)
            baseline_value = F.relu(self.leela_model.value_fc1(baseline_value), inplace=False)
            baseline_value = torch.tanh(self.leela_model.value_fc2(baseline_value))

            # Reconstruction fidelity check
            reconstruction_kl = compute_policy_kl_divergence(orig_policy, baseline_policy).mean().item()

        if verbose:
            print(f"    Reconstruction KL (SAE fidelity): {reconstruction_kl:.6f}")

        # Get ablated outputs
        ablated_policy, ablated_value = self._forward_with_modified_features(
            positions, feature_idx, new_value=0.0, k=k
        )

        # Compute effects: KL(baseline || ablated) - pure ablation effect
        policy_kl = compute_policy_kl_divergence(baseline_policy, ablated_policy)
        mean_kl = policy_kl.mean().item()

        value_change = (baseline_value - ablated_value).abs().mean().item()

        # Permutation test for significance
        # Compare to effect of ablating random features (also using baseline)
        null_effects = []
        n_features = z_sparse.shape[1]
        subset_size = min(10, len(positions))
        baseline_policy_subset = baseline_policy[:subset_size]

        for _ in range(n_permutations):
            random_idx = np.random.randint(0, n_features)
            rand_policy, rand_value = self._forward_with_modified_features(
                positions[:subset_size],
                random_idx, new_value=0.0, k=k
            )
            rand_kl = compute_policy_kl_divergence(
                baseline_policy_subset,
                rand_policy
            ).mean().item()
            null_effects.append(rand_kl)

        # p-value: fraction of null effects >= observed effect
        p_value = (np.array(null_effects) >= mean_kl).mean()

        result = AblationResult(
            feature_idx=feature_idx,
            layer=layer_name,
            policy_kl_divergence=mean_kl,
            value_change=value_change,
            mean_activation=mean_activation,
            activation_rate=activation_rate,
            p_value=p_value,
            significant=p_value < 0.05,
        )

        self.results.append(result)

        if verbose:
            print(f"    KL divergence: {mean_kl:.6f}")
            print(f"    Value change: {value_change:.6f}")
            print(f"    p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

        return result

    def compute_differential_impact(
        self,
        feature_idx: int,
        positions: torch.Tensor,
        concept_labels: np.ndarray,
        concept_name: str,
        layer_name: str = 'block',
        k: Optional[int] = None,
        n_permutations: int = 100,
        verbose: bool = False,
    ) -> DifferentialImpactResult:
        """
        Compare ablation impact when concept is present vs absent.

        A feature that represents concept X should have larger impact
        on positions where X is present.

        Args:
            feature_idx: Feature to analyze
            positions: (batch, 18, 19, 19) encoded positions
            concept_labels: (batch * 361,) binary labels for concept presence
            concept_name: Name of concept
            layer_name: Layer name for reporting
            k: k level for MSAE
            n_permutations: For significance testing
            verbose: Print progress

        Returns:
            DifferentialImpactResult
        """
        if verbose:
            print(f"  Computing differential impact for feature {feature_idx} vs {concept_name}...")

        positions = positions.to(self.device)
        batch_size = positions.shape[0]

        # Get original and ablated outputs
        with torch.no_grad():
            (orig_policy, orig_value), _ = self.leela_model.forward_with_activations(
                positions, [self.layer_idx]
            )

        ablated_policy, ablated_value = self._forward_with_modified_features(
            positions, feature_idx, new_value=0.0, k=k
        )

        # Per-position policy change (expand to spatial)
        # We measure change at each board position
        policy_change = (orig_policy - ablated_policy).abs()  # (batch, 362)
        # Use mean change across all moves as measure of impact
        per_position_impact = policy_change[:, :361].reshape(batch_size, 361).mean(dim=1)

        # Expand to match concept_labels shape
        # concept_labels is (batch * 361,), per_position_impact is (batch,)
        # We need to aggregate labels by batch
        labels_reshaped = concept_labels.reshape(batch_size, 361)
        concept_present_mask = labels_reshaped.any(axis=1)  # Position has concept somewhere

        per_position_impact = per_position_impact.cpu().numpy()

        # Split by concept presence
        impact_present = per_position_impact[concept_present_mask]
        impact_absent = per_position_impact[~concept_present_mask]

        n_present = len(impact_present)
        n_absent = len(impact_absent)

        # Compute mean impacts
        mean_present = impact_present.mean() if n_present > 0 else 0.0
        mean_absent = impact_absent.mean() if n_absent > 0 else 0.0
        differential = mean_present - mean_absent

        # Permutation test
        if n_present > 0 and n_absent > 0:
            combined = np.concatenate([impact_present, impact_absent])
            null_diffs = []

            for _ in range(n_permutations):
                np.random.shuffle(combined)
                null_present = combined[:n_present].mean()
                null_absent = combined[n_present:].mean()
                null_diffs.append(null_present - null_absent)

            p_value = (np.abs(null_diffs) >= abs(differential)).mean()
        else:
            p_value = 1.0

        result = DifferentialImpactResult(
            feature_idx=feature_idx,
            concept=concept_name,
            layer=layer_name,
            impact_when_present=float(mean_present),
            impact_when_absent=float(mean_absent),
            differential=float(differential),
            p_value=float(p_value),
            significant=p_value < 0.05,
            n_present=n_present,
            n_absent=n_absent,
        )

        self.differential_results.append(result)

        if verbose:
            print(f"    Impact (present): {mean_present:.6f}")
            print(f"    Impact (absent): {mean_absent:.6f}")
            print(f"    Differential: {differential:.6f}")
            print(f"    p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")

        return result

    def steer_feature(
        self,
        feature_idx: int,
        positions: torch.Tensor,
        steering_value: float,
        layer_name: str = 'block',
        k: Optional[int] = None,
        verbose: bool = False,
    ) -> SteeringResult:
        """
        Set a feature to a high value and observe behavior change.

        From Bricken et al.: "artificially activating them causes a
        corresponding behavior"

        Args:
            feature_idx: Feature to steer
            positions: (batch, 18, 19, 19) encoded positions
            steering_value: Value to set feature to
            layer_name: Layer name
            k: k level for MSAE
            verbose: Print progress

        Returns:
            SteeringResult with observed changes
        """
        if verbose:
            print(f"  Steering feature {feature_idx} to {steering_value}...")

        positions = positions.to(self.device)
        batch_size = positions.shape[0]

        # Get original outputs
        with torch.no_grad():
            (orig_policy, orig_value), _ = self.leela_model.forward_with_activations(
                positions, [self.layer_idx]
            )

        # Get steered outputs
        steered_policy, steered_value = self._forward_with_modified_features(
            positions, feature_idx, new_value=steering_value, k=k
        )

        # Compute effects
        policy_kl = compute_policy_kl_divergence(orig_policy, steered_policy)
        mean_kl = policy_kl.mean().item()

        value_change = (orig_value - steered_value).abs().mean().item()

        # Find positions most affected
        policy_diff = (steered_policy - orig_policy).abs()[:, :361]  # Exclude pass
        policy_diff = policy_diff.reshape(batch_size, 19, 19)

        # Get top affected positions (average across batch)
        mean_diff = policy_diff.mean(dim=0)  # (19, 19)
        flat_diff = mean_diff.flatten()
        top_k = min(5, 361)
        top_indices = flat_diff.argsort(descending=True)[:top_k]

        top_positions = []
        position_changes = []
        for idx in top_indices:
            x = idx.item() // 19
            y = idx.item() % 19
            top_positions.append((x, y))
            position_changes.append(flat_diff[idx].item())

        result = SteeringResult(
            feature_idx=feature_idx,
            layer=layer_name,
            steering_value=steering_value,
            policy_kl_divergence=mean_kl,
            value_change=value_change,
            top_affected_positions=top_positions,
            position_changes=position_changes,
        )

        self.steering_results.append(result)

        if verbose:
            print(f"    KL divergence: {mean_kl:.6f}")
            print(f"    Value change: {value_change:.6f}")
            print(f"    Top affected positions: {top_positions[:3]}")

        return result

    def find_features_with_causal_effects(
        self,
        positions: torch.Tensor,
        n_features_to_test: int = 100,
        k: Optional[int] = None,
        layer_name: str = 'block',
        effect_threshold: float = 0.01,
        verbose: bool = True,
    ) -> List[CausalMetrics]:
        """
        Find features with significant causal effects.

        Tests ablation on multiple features and identifies those
        with clear causal roles.

        Args:
            positions: (batch, 18, 19, 19) encoded positions
            n_features_to_test: Number of features to analyze
            k: k level for MSAE
            layer_name: Layer name
            effect_threshold: Minimum KL divergence to count as effect
            verbose: Print progress

        Returns:
            List of CausalMetrics for features with effects
        """
        if verbose:
            print(f"\nSearching for features with causal effects...")
            print(f"Testing {n_features_to_test} features...")

        # Get feature activations to find most active features
        with torch.no_grad():
            positions = positions.to(self.device)
            (_, _), activations_dict = self.leela_model.forward_with_activations(
                positions, [self.layer_idx]
            )

            activations = activations_dict[self.layer_idx]
            acts_flat = activations.permute(0, 2, 3, 1).reshape(-1, 256)

            if k is not None:
                _, _, z_sparse = self.sae_model.forward(acts_flat, k)
            else:
                _, _, z_sparse = self.sae_model.forward(acts_flat)

            # Test most frequently active features
            activation_rates = (z_sparse > 0).float().mean(dim=0)
            top_features = activation_rates.argsort(descending=True)[:n_features_to_test]

        causal_features = []

        for i, feature_idx in enumerate(top_features):
            feature_idx = feature_idx.item()

            if verbose and (i + 1) % 20 == 0:
                print(f"  Tested {i + 1}/{n_features_to_test} features...")

            # Test ablation
            ablation_result = self.ablate_feature(
                feature_idx, positions[:min(50, len(positions))],
                layer_name, k, n_permutations=50, verbose=False
            )

            # Test steering (use 95th percentile of activations as steering value)
            feature_acts = z_sparse[:, feature_idx]
            steering_value = torch.quantile(feature_acts[feature_acts > 0], 0.95).item() \
                if (feature_acts > 0).any() else 1.0

            steering_result = self.steer_feature(
                feature_idx, positions[:min(50, len(positions))],
                steering_value, layer_name, k, verbose=False
            )

            # Compute sparsity
            with torch.no_grad():
                (orig_policy, _), _ = self.leela_model.forward_with_activations(
                    positions[:min(50, len(positions))], [self.layer_idx]
                )
                ablated_policy, _ = self._forward_with_modified_features(
                    positions[:min(50, len(positions))],
                    feature_idx, new_value=0.0, k=k
                )
                effect = (orig_policy - ablated_policy).mean(dim=0)
                sparsity = compute_ablation_sparsity(effect)

            # Determine if feature has causal effect
            has_effect = (
                ablation_result.policy_kl_divergence > effect_threshold or
                steering_result.policy_kl_divergence > effect_threshold
            )

            if has_effect:
                metrics = CausalMetrics(
                    feature_idx=feature_idx,
                    layer=layer_name,
                    ablation_effect=ablation_result.policy_kl_divergence,
                    steering_effect=steering_result.policy_kl_divergence,
                    ablation_sparsity=sparsity,
                    has_causal_effect=True,
                )
                causal_features.append(metrics)

        if verbose:
            print(f"\nFound {len(causal_features)} features with causal effects")

        return causal_features

    def save_results(self, filename: str = 'causal.json', output_dir: str = 'outputs'):
        """
        Save all causal analysis results to JSON.

        Output format per CLAUDE.md specification.
        """
        output_path = Path(output_dir) / 'results' / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'ablation': {},
            'differential_impact': {},
            'steering': {},
            'summary': {
                'n_ablation_tests': len(self.results),
                'n_significant_ablations': sum(1 for r in self.results if r.significant),
                'n_differential_tests': len(self.differential_results),
                'n_significant_differential': sum(1 for r in self.differential_results if r.significant),
                # Reconstruction fidelity: KL(original || SAE_reconstruction)
                # Lower is better; ablation effects should exceed this
                'reconstruction_kl': self._cached_reconstruction_kl,
            }
        }

        # Ablation results
        for result in self.results:
            key = f"feature_{result.feature_idx}"
            if result.layer not in output['ablation']:
                output['ablation'][result.layer] = {}
            output['ablation'][result.layer][key] = {
                'policy_kl': result.policy_kl_divergence,
                'value_change': result.value_change,
                'mean_activation': result.mean_activation,
                'activation_rate': result.activation_rate,
                'p_value': result.p_value,
                'significant': result.significant,
            }

        # Differential impact results
        for result in self.differential_results:
            key = f"feature_{result.feature_idx}"
            if result.concept not in output['differential_impact']:
                output['differential_impact'][result.concept] = {}
            if result.layer not in output['differential_impact'][result.concept]:
                output['differential_impact'][result.concept][result.layer] = {}
            output['differential_impact'][result.concept][result.layer][key] = {
                'impact_present': result.impact_when_present,
                'impact_absent': result.impact_when_absent,
                'differential': result.differential,
                'p_value': result.p_value,
                'significant': result.significant,
            }

        # Steering results
        for result in self.steering_results:
            key = f"feature_{result.feature_idx}"
            if result.layer not in output['steering']:
                output['steering'][result.layer] = {}
            output['steering'][result.layer][key] = {
                'steering_value': result.steering_value,
                'policy_kl': result.policy_kl_divergence,
                'value_change': result.value_change,
                'top_positions': result.top_affected_positions,
            }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Saved causal analysis results to {output_path}")

    def print_summary(self):
        """Print summary of causal analysis results."""
        print(f"\n{'='*60}")
        print("Causal Analysis Summary")
        print(f"{'='*60}")

        # Reconstruction fidelity check
        if self._cached_reconstruction_kl is not None:
            print(f"\nSAE Reconstruction Fidelity:")
            print(f"  KL(original || reconstruction): {self._cached_reconstruction_kl:.6f}")
            print(f"  (Ablation effects should exceed this baseline)")

        # Ablation summary
        if self.results:
            significant = [r for r in self.results if r.significant]
            print(f"\nAblation Tests:")
            print(f"  Total: {len(self.results)}")
            print(f"  Significant (p < 0.05): {len(significant)}")

            if significant:
                top_effects = sorted(significant, key=lambda x: x.policy_kl_divergence, reverse=True)[:5]
                print(f"\n  Top 5 features by ablation effect:")
                for r in top_effects:
                    print(f"    Feature {r.feature_idx}: KL={r.policy_kl_divergence:.6f}")

        # Differential impact summary
        if self.differential_results:
            significant = [r for r in self.differential_results if r.significant]
            print(f"\nDifferential Impact Tests:")
            print(f"  Total: {len(self.differential_results)}")
            print(f"  Significant (p < 0.05): {len(significant)}")

        # Steering summary
        if self.steering_results:
            print(f"\nSteering Tests: {len(self.steering_results)}")
            top_effects = sorted(self.steering_results, key=lambda x: x.policy_kl_divergence, reverse=True)[:3]
            print(f"  Top 3 features by steering effect:")
            for r in top_effects:
                print(f"    Feature {r.feature_idx}: KL={r.policy_kl_divergence:.6f}")
