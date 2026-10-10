"""
uncertainty.py
==============
Statistical uncertainty and confidence interval calculations.
All uncertainty estimates are genuine — no fake intervals.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def standard_error(values: NDArray[np.float64]) -> float:
    """
    Standard error of the mean: std / sqrt(n).
    Returns 0 if n <= 1.
    """
    n = len(values)
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def confidence_interval_95(
    values: NDArray[np.float64],
    use_t_distribution: bool = True,
) -> tuple[float, float]:
    """
    95% confidence interval for the mean of `values`.

    Uses t-distribution for small n (< 30) and normal for large n.

    Returns
    -------
    (lower, upper) bounds
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        v = float(values[0])
        return (v, v)

    mean = float(np.mean(values))
    se   = standard_error(values)

    if use_t_distribution and n < 30:
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
    else:
        t_crit = 1.96  # normal approximation

    return (mean - t_crit * se, mean + t_crit * se)


def bootstrap_confidence_interval(
    values: NDArray[np.float64],
    statistic_fn=np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    values : 1-D array
    statistic_fn : callable
    n_bootstrap : int
    confidence : float
    seed : int or None

    Returns
    -------
    (lower, upper)
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    stats_boot = np.array([
        statistic_fn(rng.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - confidence
    lower = float(np.percentile(stats_boot, 100 * alpha / 2))
    upper = float(np.percentile(stats_boot, 100 * (1 - alpha / 2)))
    return lower, upper


def msd_standard_error(r2_values: NDArray[np.float64]) -> float:
    """Standard error of the MSD estimate from an ensemble."""
    return standard_error(r2_values)
