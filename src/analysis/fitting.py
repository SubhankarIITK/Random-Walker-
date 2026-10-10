"""
fitting.py
==========
Curve fitting utilities for scaling analysis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit


def fit_power_law(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> dict:
    """
    Fit y = A * x^alpha via log-log linear regression.

    Only uses points where x > 0 and y > 0.

    Returns
    -------
    dict with keys: alpha, A, r_squared, alpha_se, log_A_se
    """
    mask = (x > 0) & (y > 0)
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])

    if len(log_x) < 2:
        return {"alpha": float("nan"), "A": float("nan"), "r_squared": float("nan")}

    slope, intercept, r_value, p_value, stderr = scipy_stats.linregress(log_x, log_y)

    return {
        "alpha": float(slope),
        "A": float(np.exp(intercept)),
        "r_squared": float(r_value ** 2),
        "alpha_se": float(stderr),
        "log_A": float(intercept),
        "p_value": float(p_value),
    }


def fit_linear(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    force_zero_intercept: bool = False,
) -> dict:
    """
    Fit y = slope * x + intercept (or y = slope * x if force_zero_intercept).

    Returns
    -------
    dict with: slope, intercept, r_squared, slope_se
    """
    if force_zero_intercept:
        # Least squares through origin: slope = (x·y) / (x·x)
        if np.sum(x ** 2) == 0:
            return {"slope": float("nan"), "intercept": 0.0, "r_squared": float("nan")}
        slope = float(np.dot(x, y) / np.dot(x, x))
        residuals = y - slope * x
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return {"slope": slope, "intercept": 0.0, "r_squared": r2, "slope_se": float("nan")}
    else:
        slope, intercept, r_value, p_value, stderr = scipy_stats.linregress(x, y)
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "slope_se": float(stderr),
            "p_value": float(p_value),
        }


def fit_msd_slope(
    time_steps: NDArray[np.float64],
    msd_values: NDArray[np.float64],
) -> dict:
    """
    Fit MSD = 2*D*t (or just slope*t for 1D lattice walk).
    Returns slope (should be ≈ 1 for the standard model).
    """
    # Use t > 0 only
    mask = time_steps > 0
    return fit_linear(time_steps[mask], msd_values[mask], force_zero_intercept=False)
