"""
Transition Path Theory (TPT) for analyzing rare events and pathways.

Uses deeptime library for:
- Forward/backward committors
- Reactive flux
- Transition rates
- Dominant pathway decomposition

Reference:
- docs/dynamical_systems_chaos/Transition_path_theory.md
- docs/dynamical_systems_chaos/rates_from_transition_paths.md
- Metzner et al. (2009) "Transition Path Theory for Markov Jump Processes"

Example:
    >>> from src.analysis.tpt import TPTAnalyzer
    >>> tpt = TPTAnalyzer(msm)
    >>> result = tpt.compute_flux(source_states=[0, 1], target_states=[9])
    >>> pathways = tpt.find_pathways(fraction=0.9)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Check for deeptime availability
HAS_DEEPTIME = False
try:
    from deeptime.markov import reactive_flux
    from deeptime.markov.msm import MarkovStateModel
    HAS_DEEPTIME = True
except ImportError:
    logger.warning("deeptime not installed. Install with: pip install deeptime")


@dataclass
class TPTResult:
    """Result of Transition Path Theory analysis."""
    forward_committor: np.ndarray   # [n_states] P(reach target before source)
    backward_committor: np.ndarray  # [n_states] P(came from source)
    stationary_distribution: np.ndarray
    net_flux: np.ndarray            # [n_states, n_states] net reactive flux
    gross_flux: np.ndarray          # [n_states, n_states] gross reactive flux
    total_flux: float               # Total A→B flux
    rate: float                     # Transition rate k_AB
    source_states: List[int]
    target_states: List[int]


@dataclass
class PathwayResult:
    """Result of pathway decomposition."""
    paths: List[List[int]]          # List of state sequences
    fluxes: List[float]             # Flux along each pathway
    fraction_covered: float         # Total flux fraction covered


class TPTAnalyzer:
    """
    Transition Path Theory analyzer for MSM dynamics.
    
    Computes committors, reactive flux, and dominant transition pathways
    between defined source and target state sets.
    
    Requires deeptime library.
    """
    
    def __init__(self, transition_matrix: np.ndarray):
        """
        Initialize TPT analyzer.
        
        Args:
            transition_matrix: Row-stochastic [n, n] transition matrix
        """
        if not HAS_DEEPTIME:
            raise ImportError(
                "deeptime required for TPT. Install with: pip install deeptime"
            )
        
        self.transition_matrix = np.asarray(transition_matrix)
        self.n_states = self.transition_matrix.shape[0]
        
        # Create deeptime MSM
        self._msm = MarkovStateModel(self.transition_matrix)
        
        self._result: Optional[TPTResult] = None
        self._flux_object = None
    
    def compute_flux(
        self,
        source_states: List[int],
        target_states: List[int]
    ) -> TPTResult:
        """
        Compute reactive flux and committors for A→B transition.
        
        Args:
            source_states: Indices of source state set A
            target_states: Indices of target state set B
            
        Returns:
            TPTResult with committors, flux, and rate
        """
        # Convert to Python native int (deeptime rejects numpy int32/int64)
        source_states = [int(s) for s in source_states]
        target_states = [int(s) for s in target_states]

        logger.info(
            f"Computing TPT: {len(source_states)} source → {len(target_states)} target"
        )

        # Validate non-empty state sets
        if len(source_states) == 0:
            raise ValueError("source_states cannot be empty")
        if len(target_states) == 0:
            raise ValueError("target_states cannot be empty")

        # Compute reactive flux via deeptime MSM method
        self._flux_object = self._msm.reactive_flux(
            source_states,
            target_states
        )
        
        self._result = TPTResult(
            forward_committor=self._flux_object.forward_committor,
            backward_committor=self._flux_object.backward_committor,
            stationary_distribution=self._flux_object.stationary_distribution,
            net_flux=self._flux_object.net_flux,
            gross_flux=self._flux_object.gross_flux,
            total_flux=self._flux_object.total_flux,
            rate=self._flux_object.rate,
            source_states=list(source_states),
            target_states=list(target_states),
        )
        
        logger.info(
            f"TPT complete: total_flux={self._result.total_flux:.6f}, "
            f"rate={self._result.rate:.6f}"
        )
        
        return self._result
    
    def find_pathways(
        self,
        fraction: float = 0.99,
        max_pathways: int = 100
    ) -> PathwayResult:
        """
        Decompose reactive flux into dominant pathways.
        
        Uses iterative max-flow algorithm to find paths capturing
        the specified fraction of total flux.
        
        Args:
            fraction: Fraction of flux to capture (0-1)
            max_pathways: Maximum number of pathways to find
            
        Returns:
            PathwayResult with paths and their fluxes
        """
        if self._flux_object is None:
            raise ValueError("Must call compute_flux() first")
        
        from deeptime.markov.tools.flux import pathways
        
        paths_array, fluxes = pathways(
            self._flux_object.net_flux,
            self._result.source_states,
            self._result.target_states,
            fraction=fraction,
            maxiter=max_pathways
        )
        
        # Convert to list of lists
        paths_list = [p.tolist() for p in paths_array]
        fluxes_list = [float(f) for f in fluxes]
        
        total_captured = sum(fluxes_list)
        fraction_covered = total_captured / self._result.total_flux
        
        logger.info(
            f"Found {len(paths_list)} pathways covering {fraction_covered:.1%} of flux"
        )
        
        return PathwayResult(
            paths=paths_list,
            fluxes=fluxes_list,
            fraction_covered=fraction_covered,
        )
    
    def get_bottleneck_states(self, threshold: float = 0.1) -> List[int]:
        """
        Find transition state / bottleneck states.
        
        Bottleneck states have committor values near 0.5, meaning
        equal probability of reaching source or target next.
        
        Args:
            threshold: How close to 0.5 to consider a bottleneck
            
        Returns:
            List of bottleneck state indices
        """
        if self._result is None:
            raise ValueError("Must call compute_flux() first")
        
        q = self._result.forward_committor
        
        # Find states with committor near 0.5
        bottlenecks = np.where(np.abs(q - 0.5) < threshold)[0]
        
        return bottlenecks.tolist()
    
    def get_mfpt_ab(self) -> float:
        """
        Compute mean first passage time from A to B.
        
        MFPT_AB = 1 / (sum_i pi_i * q+_i * flux_out_i / total_flux)
        
        Returns:
            Mean first passage time in lag units
        """
        if self._result is None:
            raise ValueError("Must call compute_flux() first")
        
        if self._result.rate > 0:
            # MFPT ≈ 1 / rate for equilibrium processes
            return 1.0 / self._result.rate
        else:
            return np.inf
    
    @property
    def result(self) -> Optional[TPTResult]:
        """Get the TPT result."""
        return self._result


def compute_tpt(
    transition_matrix: np.ndarray,
    source_states: List[int],
    target_states: List[int],
    pathway_fraction: float = 0.9
) -> Tuple[TPTResult, PathwayResult]:
    """
    Convenience function for full TPT analysis.
    
    Args:
        transition_matrix: Row-stochastic MSM
        source_states: Source state indices
        target_states: Target state indices
        pathway_fraction: Fraction of flux to capture in pathways
        
    Returns:
        Tuple of (TPTResult, PathwayResult)
    """
    tpt = TPTAnalyzer(transition_matrix)
    result = tpt.compute_flux(source_states, target_states)
    pathways = tpt.find_pathways(fraction=pathway_fraction)
    
    return result, pathways


def save_tpt_result(result: TPTResult, filepath: str) -> None:
    """Save TPT result to npz file."""
    np.savez(
        filepath,
        forward_committor=result.forward_committor,
        backward_committor=result.backward_committor,
        stationary_distribution=result.stationary_distribution,
        net_flux=result.net_flux,
        gross_flux=result.gross_flux,
        total_flux=result.total_flux,
        rate=result.rate,
        source_states=np.array(result.source_states),
        target_states=np.array(result.target_states),
    )
    logger.info(f"Saved TPT result to {filepath}")


def load_tpt_result(filepath: str) -> TPTResult:
    """Load TPT result from npz file."""
    data = np.load(filepath)
    
    return TPTResult(
        forward_committor=data['forward_committor'],
        backward_committor=data['backward_committor'],
        stationary_distribution=data['stationary_distribution'],
        net_flux=data['net_flux'],
        gross_flux=data['gross_flux'],
        total_flux=float(data['total_flux']),
        rate=float(data['rate']),
        source_states=data['source_states'].tolist(),
        target_states=data['target_states'].tolist(),
    )
