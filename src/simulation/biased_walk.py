"""
biased_walk.py
==============
Extension: biased random walk simulation.

Model definition
----------------
For each step the walker still moves in exactly ONE coordinate (chosen
uniformly), but the direction probabilities are:

    P(+1 in chosen coordinate) = p
    P(−1 in chosen coordinate) = 1 − p

For p = 0.5 this reduces to the standard unbiased walk.
For p ≠ 0.5 the walk has a drift.

Theoretical results for the biased model (1D for simplicity):
    E[X_n]   = n * (2p − 1)
    Var(X_n) = 4 * n * p * (1 − p)
    E[R²]    = n (this holds only for the unbiased case)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def simulate_biased_walk(
    dimension: int,
    n_steps: int,
    p: float = 0.5,
    seed: int | None = None,
    bias_axis: int = 0,
) -> NDArray[np.int32]:
    """
    Simulate a biased single random walk.

    Parameters
    ----------
    dimension : int
        Spatial dimension (1–5).
    n_steps : int
        Number of steps.
    p : float
        Probability of +1 in the chosen coordinate. p=0.5 → unbiased.
    seed : int or None
        Random seed.
    bias_axis : int
        The coordinate along which the bias is applied.  In all other
        coordinates the walk remains unbiased (p=0.5).

    Returns
    -------
    trajectory : ndarray of shape (n_steps + 1, dimension)
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")

    rng = np.random.default_rng(seed)
    trajectory = np.zeros((n_steps + 1, dimension), dtype=np.int32)

    if n_steps == 0:
        return trajectory

    coord_idx  = rng.integers(0, dimension, size=n_steps)
    # For the biased axis, use p; for all others, use 0.5
    # Generate all directions with p=0.5 first, then override for biased axis
    raw = rng.random(n_steps)
    directions = np.where(raw < 0.5, 1, -1).astype(np.int8)

    # Override direction for steps along bias_axis
    biased_mask = (coord_idx == bias_axis)
    raw_biased  = rng.random(int(biased_mask.sum()))
    directions[biased_mask] = np.where(raw_biased < p, 1, -1)

    steps = np.zeros((n_steps, dimension), dtype=np.int8)
    steps[np.arange(n_steps), coord_idx] = directions

    trajectory[1:] = np.cumsum(steps, axis=0)
    return trajectory


def biased_ensemble_final(
    dimension: int,
    n_steps: int,
    n_walkers: int,
    p: float = 0.5,
    seed: int | None = None,
) -> NDArray[np.int32]:
    """
    Fast biased ensemble: only returns final positions.
    """
    rng = np.random.default_rng(seed)
    positions = np.zeros((n_walkers, dimension), dtype=np.int32)

    CHUNK = min(2000, n_steps)
    step = 0
    while step < n_steps:
        chunk_size = min(CHUNK, n_steps - step)
        coord_idx  = rng.integers(0, dimension, size=(chunk_size, n_walkers))
        raw        = rng.random((chunk_size, n_walkers))
        directions = np.where(raw < p, 1, -1).astype(np.int32)
        for local_t in range(chunk_size):
            positions[np.arange(n_walkers), coord_idx[local_t]] += directions[local_t]
        step += chunk_size

    return positions


def theoretical_biased_mean(n_steps: int, p: float, dimension: int) -> float:
    """
    Theoretical mean displacement along the biased axis.
    E[X_bias] = n_steps * (1/dimension) * (2p - 1)
    (Each step uses the biased axis with prob 1/d.)
    """
    return n_steps * (1.0 / dimension) * (2 * p - 1)


def theoretical_biased_msd_1d(n_steps: int, p: float) -> float:
    """
    Theoretical E[X_n²] for a purely 1D biased walk.
    E[X²] = n*(2p-1)² + 4np(1-p) = n*(1 - 4p + 4p²) + 4np - 4np² = n
    Wait: E[X²] = n[1 - (2p-1)²] + n*(2p-1)² = n, only if unbiased!

    Actually for biased 1D:
        E[X_n]   = n*(2p-1)
        Var(X_n) = 4np(1-p)
        E[X_n²]  = Var(X_n) + E[X_n]² = 4np(1-p) + n²*(2p-1)²
    """
    mean = n_steps * (2 * p - 1)
    var  = 4 * n_steps * p * (1 - p)
    return var + mean ** 2
