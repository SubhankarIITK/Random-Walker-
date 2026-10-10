"""
statistics.py
=============
Statistical analysis functions for random walk simulations.
All functions work on actual simulation output — no fabricated values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any

from ..theory.moments import msd_theory, coordinate_variance


def compute_walk_statistics(
    trajectory: NDArray[np.int32],
    dimension: int,
) -> dict[str, Any]:
    """
    Compute comprehensive statistics for a single-walker trajectory.

    Parameters
    ----------
    trajectory : ndarray of shape (n_steps+1, dimension)
    dimension : int

    Returns
    -------
    stats : dict containing all computed quantities
    """
    n_steps = len(trajectory) - 1
    positions = trajectory.astype(np.float64)

    # Squared displacement at each time
    r2 = np.sum(positions ** 2, axis=1)
    r  = np.sqrt(r2)

    # Final position
    final_pos = trajectory[-1]
    final_r2  = float(r2[-1])
    final_r   = float(r[-1])

    # Mean displacement (should be ≈ 0)
    mean_disp_vec = np.mean(positions, axis=0)

    # MSD of the trajectory (average over time, not ensemble)
    mean_r2_over_time = float(np.mean(r2[1:])) if n_steps > 0 else 0.0

    # Coordinate statistics
    coord_means = np.mean(positions[1:], axis=0) if n_steps > 0 else np.zeros(dimension)
    coord_vars  = np.var(positions[1:], axis=0, ddof=1) if n_steps > 1 else np.zeros(dimension)
    coord_stds  = np.sqrt(coord_vars)

    # Theoretical values at final step
    msd_th   = float(msd_theory(n_steps))
    var_th_val = coordinate_variance(n_steps, dimension)
    var_th   = float(var_th_val if isinstance(var_th_val, (int, float)) else var_th_val[0])

    # Error
    abs_err = abs(final_r2 - msd_th)
    rel_err = abs_err / msd_th if msd_th > 0 else 0.0

    # Max / min displacement
    max_r = float(np.max(r))
    min_r = float(np.min(r[1:])) if n_steps > 0 else 0.0

    return {
        "n_steps": n_steps,
        "dimension": dimension,
        "final_position": final_pos.tolist(),
        "final_r": final_r,
        "final_r2": final_r2,
        "msd_theory": msd_th,
        "abs_error_msd": abs_err,
        "rel_error_msd": rel_err,
        "pct_error_msd": rel_err * 100.0,
        "mean_displacement_vec": mean_disp_vec.tolist(),
        "mean_r_over_time": float(np.mean(r[1:])) if n_steps > 0 else 0.0,
        "max_r": max_r,
        "min_r": min_r,
        "coord_means": coord_means.tolist(),
        "coord_vars_sim": coord_vars.tolist(),
        "coord_vars_theory": [var_th] * dimension,
        "coord_stds_sim": coord_stds.tolist(),
        "r2_series": r2,
        "r_series": r,
    }


def compute_ensemble_statistics(
    final_positions: NDArray[np.int32],
    n_steps: int,
    dimension: int,
) -> dict[str, Any]:
    """
    Compute statistics from final positions of an ensemble.

    Parameters
    ----------
    final_positions : ndarray of shape (n_walkers, dimension)
    n_steps : int
    dimension : int
    """
    n_walkers = len(final_positions)
    pos = final_positions.astype(np.float64)

    r2 = np.sum(pos ** 2, axis=1)
    r  = np.sqrt(r2)

    msd_sim   = float(np.mean(r2))
    msd_std   = float(np.std(r2, ddof=1)) if n_walkers > 1 else 0.0
    msd_se    = msd_std / np.sqrt(n_walkers)
    msd_th    = float(msd_theory(n_steps))

    abs_err = abs(msd_sim - msd_th)
    rel_err = abs_err / msd_th if msd_th > 0 else 0.0

    coord_means = np.mean(pos, axis=0)
    coord_vars  = np.var(pos, axis=0, ddof=1) if n_walkers > 1 else np.zeros(dimension)
    var_th      = n_steps / dimension

    return {
        "n_walkers": n_walkers,
        "n_steps": n_steps,
        "dimension": dimension,
        "msd_sim": msd_sim,
        "msd_std": msd_std,
        "msd_se": msd_se,
        "msd_theory": msd_th,
        "abs_error_msd": abs_err,
        "rel_error_msd": rel_err,
        "pct_error_msd": rel_err * 100.0,
        "rms_sim": float(np.sqrt(msd_sim)),
        "rms_theory": float(np.sqrt(msd_th)),
        "mean_r": float(np.mean(r)),
        "max_r": float(np.max(r)),
        "min_r": float(np.min(r)),
        "coord_means": coord_means.tolist(),
        "coord_vars_sim": coord_vars.tolist(),
        "coord_var_theory": var_th,
        "r2_values": r2,
        "r_values": r,
    }


def compute_return_statistics(
    return_counts: NDArray[np.int64],
    ever_returned: NDArray[np.bool_],
    dimension: int,
) -> dict[str, Any]:
    """
    Compute return-to-origin statistics from ensemble data.
    """
    from ..theory.recurrence import (
        is_recurrent,
        recurrence_classification,
        theoretical_return_probability,
    )

    n_walkers = len(return_counts)
    frac_returned = float(np.mean(ever_returned))
    mean_returns  = float(np.mean(return_counts))
    max_returns   = int(np.max(return_counts))
    theory_prob   = theoretical_return_probability(dimension)

    result = {
        "n_walkers": n_walkers,
        "dimension": dimension,
        "is_recurrent_theory": is_recurrent(dimension),
        "recurrence_class": recurrence_classification(dimension),
        "fraction_returned_sim": frac_returned,
        "theory_return_probability": theory_prob,
        "mean_return_count": mean_returns,
        "max_return_count": max_returns,
        "n_walkers_returned": int(np.sum(ever_returned)),
        "return_counts": return_counts,
    }

    if theory_prob is not None:
        result["abs_error_return"] = abs(frac_returned - theory_prob)
    return result


def unique_sites_statistics(trajectory: NDArray[np.int32]) -> dict[str, Any]:
    """
    Compute unique sites statistics from a trajectory.
    """
    from ..simulation.random_walk import distinct_sites_over_time

    counts = distinct_sites_over_time(trajectory)
    n_steps = len(trajectory) - 1

    return {
        "n_distinct_sites": int(counts[-1]),
        "n_steps": n_steps,
        "revisit_count": int(n_steps + 1 - counts[-1]),  # steps − unique
        "coverage_fraction": float(counts[-1]) / (n_steps + 1) if n_steps > 0 else 1.0,
        "distinct_over_time": counts,
    }


def build_comparison_dataframe(
    results_by_dim: dict[int, dict],
) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing statistics across dimensions.

    Parameters
    ----------
    results_by_dim : dict mapping dimension -> stats dict

    Returns
    -------
    df : pd.DataFrame
    """
    rows = []
    for dim, stats in sorted(results_by_dim.items()):
        row = {
            "Dimension": dim,
            "Walkers": stats.get("n_walkers", "—"),
            "Steps": stats.get("n_steps", "—"),
            "MSD (Sim)": round(stats.get("msd_sim", float("nan")), 3),
            "MSD (Theory)": round(stats.get("msd_theory", float("nan")), 3),
            "Rel. Error (%)": round(stats.get("pct_error_msd", float("nan")), 4),
            "RMS (Sim)": round(stats.get("rms_sim", float("nan")), 3),
            "Frac. Returned": round(stats.get("fraction_returned", float("nan")), 4),
        }
        rows.append(row)
    return pd.DataFrame(rows)
