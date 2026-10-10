"""
convergence.py
==============
Monte Carlo convergence experiment.

Demonstrates that as the number of walkers increases, the empirical MSD
converges toward the theoretical value E[|X_n|²] = n.

All values come from actual simulation — nothing is fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..simulation.ensemble import ensemble_final_positions
from ..theory.moments import msd_theory


def run_convergence_experiment(
    dimension: int,
    n_steps: int,
    walker_counts: list[int],
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Run convergence experiment: simulate with increasing walker counts.

    Parameters
    ----------
    dimension : int
    n_steps : int
    walker_counts : list of int
        Ensemble sizes to test, e.g. [10, 100, 1000, 10000]
    seed : int or None

    Returns
    -------
    df : pd.DataFrame with columns:
        n_walkers, msd_sim, msd_theory, abs_error, rel_error, pct_error, se
    """
    rng = np.random.default_rng(seed)
    theory = float(msd_theory(n_steps))

    rows = []
    for n_w in walker_counts:
        child_seed = int(rng.integers(0, 2**31))
        final_pos = ensemble_final_positions(dimension, n_steps, n_w, seed=child_seed)
        r2 = np.sum(final_pos.astype(np.float64) ** 2, axis=1)

        msd_sim = float(np.mean(r2))
        se      = float(np.std(r2, ddof=1) / np.sqrt(n_w)) if n_w > 1 else 0.0
        abs_err = abs(msd_sim - theory)
        rel_err = abs_err / theory if theory > 0 else 0.0

        rows.append({
            "n_walkers":  n_w,
            "msd_sim":    msd_sim,
            "msd_theory": theory,
            "abs_error":  abs_err,
            "rel_error":  rel_err,
            "pct_error":  rel_err * 100.0,
            "se":         se,
            "ci95_low":   msd_sim - 1.96 * se,
            "ci95_high":  msd_sim + 1.96 * se,
        })

    return pd.DataFrame(rows)


def run_multi_dim_convergence(
    dimensions: list[int],
    n_steps: int,
    walker_counts: list[int],
    seed: int | None = None,
) -> dict[int, pd.DataFrame]:
    """
    Run convergence experiment for multiple dimensions.

    Returns
    -------
    dict mapping dimension -> convergence DataFrame
    """
    rng = np.random.default_rng(seed)
    results = {}
    for d in dimensions:
        child_seed = int(rng.integers(0, 2**31))
        results[d] = run_convergence_experiment(d, n_steps, walker_counts, seed=child_seed)
    return results
