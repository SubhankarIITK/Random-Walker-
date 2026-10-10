"""
random_walk.py
==============
Core simulation engine for the standard nearest-neighbor unbiased lattice
random walk in arbitrary dimension d.

Model definition
----------------
At every step the walker selects exactly ONE coordinate uniformly at random
from {1, …, d} and increments it by +1 or −1, each with probability 1/2.

This gives 2d equally likely moves per step, so the step probability is 1/(2d).

IMPORTANT: Only one coordinate changes per step. This is NOT the model where
all coordinates change simultaneously.

Theoretical properties (see src/theory/moments.py):
    E[X_n] = 0  (no drift)
    Var(X_i_n) = n / d
    E[|X_n|²]  = n   (for all dimensions!)
    RMS ~ sqrt(n)     (diffusive scaling)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Step generation helpers
# ---------------------------------------------------------------------------

def generate_steps(
    n_steps: int,
    dimension: int,
    rng: np.random.Generator,
) -> NDArray[np.int8]:
    """
    Generate n_steps random steps for a single walker in `dimension` dimensions.

    Returns
    -------
    steps : ndarray of shape (n_steps, dimension), dtype int8
        Each row is a step vector with exactly one non-zero entry (±1).
    """
    if n_steps == 0:
        return np.zeros((0, dimension), dtype=np.int8)

    # Choose which coordinate to change: uniform in {0, …, d-1}
    coord_idx = rng.integers(0, dimension, size=n_steps)

    # Choose direction: +1 or −1
    directions = rng.choice(np.array([1, -1], dtype=np.int8), size=n_steps)

    # Build step array
    steps = np.zeros((n_steps, dimension), dtype=np.int8)
    steps[np.arange(n_steps), coord_idx] = directions
    return steps


def generate_steps_batch(
    n_steps: int,
    dimension: int,
    n_walkers: int,
    rng: np.random.Generator,
) -> NDArray[np.int8]:
    """
    Generate steps for n_walkers simultaneously (vectorized).

    Returns
    -------
    steps : ndarray of shape (n_steps, n_walkers, dimension), dtype int8
    """
    if n_steps == 0:
        return np.zeros((0, n_walkers, dimension), dtype=np.int8)

    coord_idx = rng.integers(0, dimension, size=(n_steps, n_walkers))
    directions = rng.choice(np.array([1, -1], dtype=np.int8), size=(n_steps, n_walkers))

    steps = np.zeros((n_steps, n_walkers, dimension), dtype=np.int8)
    # Vectorized fancy indexing
    step_idx = np.arange(n_steps)[:, None] * np.ones(n_walkers, dtype=int)[None, :]
    walker_idx = np.ones(n_steps, dtype=int)[:, None] * np.arange(n_walkers)[None, :]
    steps[step_idx, walker_idx, coord_idx] = directions
    return steps


# ---------------------------------------------------------------------------
# Single-walker trajectory simulation
# ---------------------------------------------------------------------------

def simulate_walk(
    dimension: int,
    n_steps: int,
    seed: int | None = None,
) -> NDArray[np.int32]:
    """
    Simulate a single random walk and return the full trajectory.

    Parameters
    ----------
    dimension : int
        Spatial dimension (1–5).
    n_steps : int
        Number of steps.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    trajectory : ndarray of shape (n_steps + 1, dimension), dtype int32
        trajectory[0] = origin = (0, …, 0)
        trajectory[t] = position after t steps.
    """
    if dimension < 1:
        raise ValueError(f"dimension must be >= 1, got {dimension}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")

    rng = np.random.default_rng(seed)

    # Allocate trajectory including starting position
    trajectory = np.zeros((n_steps + 1, dimension), dtype=np.int32)

    if n_steps == 0:
        return trajectory

    steps = generate_steps(n_steps, dimension, rng)
    trajectory[1:] = np.cumsum(steps, axis=0)
    return trajectory


def simulate_walk_positions(
    dimension: int,
    n_steps: int,
    seed: int | None = None,
) -> NDArray[np.int32]:
    """
    Convenience alias – identical to simulate_walk.
    Returns positions at every time step (including t=0).
    """
    return simulate_walk(dimension, n_steps, seed)


# ---------------------------------------------------------------------------
# Squared displacement along a trajectory
# ---------------------------------------------------------------------------

def squared_displacement(trajectory: NDArray) -> NDArray[np.float64]:
    """
    Compute |X_t|² for each time step in a trajectory.

    Parameters
    ----------
    trajectory : ndarray of shape (T, d)

    Returns
    -------
    r2 : ndarray of shape (T,)
    """
    return np.sum(trajectory.astype(np.float64) ** 2, axis=1)


def displacement(trajectory: NDArray) -> NDArray[np.float64]:
    """Compute |X_t| (Euclidean distance from origin) for each time step."""
    return np.sqrt(squared_displacement(trajectory))


# ---------------------------------------------------------------------------
# Distinct sites
# ---------------------------------------------------------------------------

def count_distinct_sites(trajectory: NDArray) -> int:
    """
    Count the number of distinct lattice sites visited in a trajectory.

    Parameters
    ----------
    trajectory : ndarray of shape (T, d)

    Returns
    -------
    n_distinct : int
    """
    # Convert each position to a tuple for hashing
    tuples = set(map(tuple, trajectory))
    return len(tuples)


def distinct_sites_over_time(trajectory: NDArray) -> NDArray[np.int64]:
    """
    Compute the cumulative count of distinct sites visited at each time step.

    Returns
    -------
    counts : ndarray of shape (T,)
    """
    visited = set()
    counts = np.zeros(len(trajectory), dtype=np.int64)
    for t, pos in enumerate(trajectory):
        visited.add(tuple(pos))
        counts[t] = len(visited)
    return counts


# ---------------------------------------------------------------------------
# Return-to-origin statistics (single trajectory)
# ---------------------------------------------------------------------------

def returns_to_origin(trajectory: NDArray) -> dict:
    """
    Analyse return-to-origin events in a single trajectory.

    Returns a dict with:
        n_returns        : total number of returns to origin (t > 0)
        first_return_time: step of first return (None if never returned)
        return_times     : list of all steps where position == origin
    """
    origin = np.zeros(trajectory.shape[1], dtype=trajectory.dtype)
    # Check each row after t=0
    at_origin = np.all(trajectory[1:] == origin, axis=1)
    return_steps = np.where(at_origin)[0] + 1  # +1 because we skipped t=0

    return {
        "n_returns": int(len(return_steps)),
        "first_return_time": int(return_steps[0]) if len(return_steps) > 0 else None,
        "return_times": return_steps.tolist(),
    }


# ---------------------------------------------------------------------------
# First-passage time to a target
# ---------------------------------------------------------------------------

def first_passage_time(
    dimension: int,
    n_steps: int,
    target: int | float = 10,
    seed: int | None = None,
) -> int | None:
    """
    Simulate a walk and return the first step at which |X_t| >= target.

    For 1D, this is first time |x| >= target.
    For higher dimensions, it is first time Euclidean distance >= target.

    Returns None if the walk never reaches the target within n_steps.
    """
    rng = np.random.default_rng(seed)
    pos = np.zeros(dimension, dtype=np.int32)

    for t in range(1, n_steps + 1):
        coord = rng.integers(0, dimension)
        direction = rng.choice([1, -1])
        pos[coord] += direction

        dist = float(np.sqrt(np.sum(pos.astype(np.float64) ** 2)))
        if dist >= target:
            return t

    return None


def first_passage_times_ensemble(
    dimension: int,
    n_steps: int,
    target: float,
    n_walkers: int,
    seed: int | None = None,
) -> list[int | None]:
    """
    Run first-passage-time experiment for n_walkers.

    Returns list of first-passage times (None = did not reach target).
    """
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_walkers):
        child_seed = int(rng.integers(0, 2**31))
        fpt = first_passage_time(dimension, n_steps, target, seed=child_seed)
        results.append(fpt)
    return results
