"""
configuration.py
================
Simulation configuration dataclass.
Keeps simulation parameters separate from UI code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SimulationMode(str, Enum):
    SINGLE_WALK = "single_walk"
    MONTE_CARLO = "monte_carlo"


class VisualizationMode(str, Enum):
    STATIC     = "static"
    ANIMATED   = "animated"


@dataclass
class WalkConfig:
    """Complete configuration for a random walk experiment."""

    # Core parameters
    dimension: int = 2
    n_steps: int = 1000
    n_walkers: int = 500
    seed: Optional[int] = 42

    # Mode
    mode: SimulationMode = SimulationMode.SINGLE_WALK
    viz_mode: VisualizationMode = VisualizationMode.STATIC

    # Storage options
    store_trajectories: bool = False
    compute_returns: bool = True
    compute_unique_sites: bool = True
    compute_first_passage: bool = False

    # First-passage target
    first_passage_target: float = 10.0

    # Biased walk
    bias_p: float = 0.5
    bias_axis: int = 0

    # Convergence experiment
    convergence_walker_counts: list = field(
        default_factory=lambda: [10, 100, 1_000, 10_000]
    )

    # Thinning (for large simulations: save statistics every k steps)
    thinning: int = 1

    def __post_init__(self):
        if self.dimension < 1 or self.dimension > 5:
            raise ValueError(f"dimension must be 1–5, got {self.dimension}")
        if self.n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {self.n_steps}")
        if self.n_walkers < 1:
            raise ValueError(f"n_walkers must be >= 1, got {self.n_walkers}")
        if not 0.0 <= self.bias_p <= 1.0:
            raise ValueError(f"bias_p must be in [0,1], got {self.bias_p}")

    @property
    def is_high_dimension(self) -> bool:
        return self.dimension >= 4

    @property
    def is_computationally_expensive(self) -> bool:
        """Flag large simulations for user warning."""
        return self.n_walkers * self.n_steps > 10_000_000

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "n_steps": self.n_steps,
            "n_walkers": self.n_walkers,
            "seed": self.seed,
            "mode": self.mode.value,
            "viz_mode": self.viz_mode.value,
            "store_trajectories": self.store_trajectories,
            "compute_returns": self.compute_returns,
            "compute_unique_sites": self.compute_unique_sites,
            "bias_p": self.bias_p,
        }
