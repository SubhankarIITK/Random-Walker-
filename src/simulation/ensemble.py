"""
ensemble.py
===========
Vectorized Monte Carlo ensemble simulation for random walks.

The ensemble engine simulates n_walkers simultaneously using NumPy
broadcasting, avoiding Python-level loops over walkers.

Key design choices:
- Full trajectory storage is optional (controlled by store_trajectories).
- When store_trajectories=False, only running statistics are accumulated,
  drastically reducing memory for large ensembles.
- The MSD is accumulated incrementally for each time slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class EnsembleResult:
    """
    Container for ensemble simulation results.

    All arrays are indexed by time step (length = n_steps + 1).
    """

    dimension: int
    n_steps: int
    n_walkers: int
    seed: int | None

    # --- Time-averaged statistics (shape: n_steps+1) ---
    time_steps: NDArray[np.int64]        # [0, 1, …, n_steps]
    mean_msd: NDArray[np.float64]        # <|X_t|²>
    std_msd: NDArray[np.float64]         # std of |X_t|² across walkers
    mean_displacement: NDArray[np.float64]  # <|X_t|>
    mean_coord: NDArray[np.float64]      # shape (n_steps+1, d): <X_i_t>
    var_coord: NDArray[np.float64]       # shape (n_steps+1, d): Var(X_i_t)

    # --- Final-step statistics ---
    final_positions: NDArray[np.int32]   # shape (n_walkers, d)

    # --- Return statistics ---
    n_returns: NDArray[np.int64]         # returns per walker (shape: n_walkers)
    ever_returned: NDArray[np.bool_]     # shape (n_walkers,)

    # --- Optional full trajectories ---
    trajectories: Optional[NDArray[np.int32]] = None  # (n_walkers, n_steps+1, d)

    @property
    def theoretical_msd(self) -> NDArray[np.float64]:
        """Theoretical E[|X_t|²] = t (independent of dimension)."""
        return self.time_steps.astype(np.float64)

    @property
    def fraction_returned(self) -> float:
        """Fraction of walkers that returned to origin at least once."""
        return float(np.mean(self.ever_returned))

    @property
    def final_msd_sim(self) -> float:
        return float(self.mean_msd[-1])

    @property
    def final_msd_theory(self) -> float:
        return float(self.n_steps)

    @property
    def final_msd_abs_error(self) -> float:
        return abs(self.final_msd_sim - self.final_msd_theory)

    @property
    def final_msd_rel_error(self) -> float:
        if self.final_msd_theory == 0:
            return 0.0
        return self.final_msd_abs_error / self.final_msd_theory


def simulate_ensemble(
    dimension: int,
    n_steps: int,
    n_walkers: int,
    seed: int | None = None,
    store_trajectories: bool = False,
    compute_returns: bool = True,
    thinning: int = 1,
    on_step=None,
) -> EnsembleResult:
    """
    Simulate an ensemble of n_walkers independent random walks.

    Parameters
    ----------
    dimension : int
        Spatial dimension (1–5).
    n_steps : int
        Number of steps per walker.
    n_walkers : int
        Number of independent walkers.
    seed : int or None
        Random seed.
    store_trajectories : bool
        If True, store every position for every walker. Requires
        O(n_walkers × n_steps × d) memory. Use only for small simulations.
    compute_returns : bool
        If True, count returns to origin (slightly slower).
    thinning : int
        Save statistics every `thinning` steps to reduce memory/compute for
        very long walks. thinning=1 means every step.

    Returns
    -------
    EnsembleResult
    """
    if dimension < 1:
        raise ValueError(f"dimension must be >= 1, got {dimension}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    if n_walkers < 1:
        raise ValueError(f"n_walkers must be >= 1, got {n_walkers}")

    # on_step(t, positions) receives the live positions at t=0 and every step.
    # Consumers must copy positions if retaining them beyond the callback.
    rng = np.random.default_rng(seed)

    # --- Allocate position array for current step ---
    positions = np.zeros((n_walkers, dimension), dtype=np.int32)

    # --- Determine time checkpoints ---
    checkpoint_steps = np.arange(0, n_steps + 1, thinning, dtype=np.int64)
    if checkpoint_steps[-1] != n_steps:
        checkpoint_steps = np.append(checkpoint_steps, n_steps)
    n_checkpoints = len(checkpoint_steps)

    # --- Accumulators ---
    mean_msd   = np.zeros(n_checkpoints, dtype=np.float64)
    std_msd    = np.zeros(n_checkpoints, dtype=np.float64)
    mean_disp  = np.zeros(n_checkpoints, dtype=np.float64)
    mean_coord = np.zeros((n_checkpoints, dimension), dtype=np.float64)
    var_coord  = np.zeros((n_checkpoints, dimension), dtype=np.float64)

    if store_trajectories:
        traj_store = np.zeros((n_walkers, n_checkpoints, dimension), dtype=np.int32)

    if compute_returns:
        return_counts = np.zeros(n_walkers, dtype=np.int64)

    def _record(ci: int):
        r2 = np.sum(positions.astype(np.float64) ** 2, axis=1)
        mean_msd[ci]  = np.mean(r2)
        std_msd[ci]   = np.std(r2, ddof=1) if n_walkers > 1 else 0.0
        mean_disp[ci] = np.mean(np.sqrt(r2))
        mean_coord[ci] = np.mean(positions, axis=0)
        var_coord[ci]  = np.var(positions, axis=0, ddof=1) if n_walkers > 1 else np.zeros(dimension)
        if store_trajectories:
            traj_store[:, ci, :] = positions

    # Record t=0
    checkpoint_set = set(checkpoint_steps.tolist())
    ci = 0
    if 0 in checkpoint_set:
        _record(ci)
        ci += 1

    if on_step is not None:
        on_step(0, positions)

    origin = np.zeros(dimension, dtype=np.int32)

    # --- Main simulation loop ---
    # Chunk generation to balance memory and speed
    CHUNK = min(1000, n_steps)
    step = 0

    while step < n_steps:
        remaining = n_steps - step
        chunk_size = min(CHUNK, remaining)

        # Generate chunk of steps: (chunk_size, n_walkers)
        coord_idx  = rng.integers(0, dimension, size=(chunk_size, n_walkers))
        directions = rng.integers(0, 2, size=(chunk_size, n_walkers)) * 2 - 1  # ±1

        for local_t in range(chunk_size):
            global_t = step + local_t + 1  # t=0 already recorded

            # Apply one step to each walker
            positions[np.arange(n_walkers), coord_idx[local_t]] += directions[local_t]

            if on_step is not None:
                on_step(global_t, positions)

            # Count returns to origin
            if compute_returns:
                at_origin = np.all(positions == origin, axis=1)
                return_counts[at_origin] += 1

            # Record checkpoint
            if global_t in checkpoint_set:
                _record(ci)
                ci += 1

        step += chunk_size

    final_positions = positions.copy()

    if compute_returns:
        ever_returned = return_counts > 0
    else:
        return_counts = np.zeros(n_walkers, dtype=np.int64)
        ever_returned = np.zeros(n_walkers, dtype=bool)

    return EnsembleResult(
        dimension=dimension,
        n_steps=n_steps,
        n_walkers=n_walkers,
        seed=seed,
        time_steps=checkpoint_steps,
        mean_msd=mean_msd,
        std_msd=std_msd,
        mean_displacement=mean_disp,
        mean_coord=mean_coord,
        var_coord=var_coord,
        final_positions=final_positions,
        n_returns=return_counts,
        ever_returned=ever_returned,
        trajectories=traj_store if store_trajectories else None,
    )


def ensemble_final_positions(
    dimension: int,
    n_steps: int,
    n_walkers: int,
    seed: int | None = None,
) -> NDArray[np.int32]:
    """
    Fast variant: return only final positions (no trajectory storage).
    O(n_walkers × dimension) memory.
    """
    rng = np.random.default_rng(seed)
    positions = np.zeros((n_walkers, dimension), dtype=np.int32)

    CHUNK = min(5000, n_steps)
    step = 0
    while step < n_steps:
        chunk_size = min(CHUNK, n_steps - step)
        coord_idx  = rng.integers(0, dimension, size=(chunk_size, n_walkers))
        directions = rng.integers(0, 2, size=(chunk_size, n_walkers)) * 2 - 1
        for local_t in range(chunk_size):
            positions[np.arange(n_walkers), coord_idx[local_t]] += directions[local_t]
        step += chunk_size

    return positions


def replay_ensemble_walker(dimension, n_steps, n_walkers, seed, walker_id):
    """Recover a numbered member of the same ensemble without storing all paths.

    Preserve the ensemble engine's RNG calls and chunk boundaries exactly.
    Walker IDs are one-based; only the selected column is accumulated.
    """
    if seed is None or not 1 <= walker_id <= n_walkers:
        raise ValueError("A recorded seed and valid walker ID are required")
    rng = np.random.default_rng(seed)
    path = np.zeros((n_steps + 1, dimension), dtype=np.int32)
    chunk = min(1000, n_steps)
    for start in range(0, n_steps, chunk):
        count = min(chunk, n_steps - start)
        axes = rng.integers(0, dimension, size=(count, n_walkers))[:, walker_id - 1].copy()
        signs = (rng.integers(0, 2, size=(count, n_walkers)) * 2 - 1)[:, walker_id - 1].copy()
        moves = np.zeros((count, dimension), dtype=np.int32)
        moves[np.arange(count), axes] = signs
        path[start + 1:start + count + 1] = path[start] + np.cumsum(moves, axis=0)
    return path
