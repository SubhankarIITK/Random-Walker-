"""
scaling.py
==========
Diffusive scaling theory: R_typical ~ n^alpha, alpha = 0.5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


THEORETICAL_SCALING_EXPONENT: float = 0.5


def theoretical_scaling_exponent() -> float:
    """Return the theoretical diffusive scaling exponent alpha = 0.5."""
    return THEORETICAL_SCALING_EXPONENT


def theoretical_rms_at_time(n: int | NDArray, prefactor: float = 1.0) -> float | NDArray:
    """
    Theoretical RMS displacement: prefactor * sqrt(n).
    For the standard unbiased walk, prefactor = 1.
    """
    return prefactor * np.sqrt(np.asarray(n, dtype=np.float64))


def log_log_fit_alpha(
    time_steps: NDArray[np.float64],
    rms_values: NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Fit log(R) = alpha * log(t) + const via linear regression.

    Parameters
    ----------
    time_steps : 1-D array of time values (must be > 0)
    rms_values : 1-D array of RMS displacement values (must be > 0)

    Returns
    -------
    alpha : float   — measured scaling exponent
    const : float   — intercept in log-log space (log of prefactor)
    r2    : float   — coefficient of determination of the log-log fit
    """
    mask = (time_steps > 0) & (rms_values > 0)
    log_t = np.log(time_steps[mask])
    log_r = np.log(rms_values[mask])

    # Linear fit: log_r = alpha * log_t + const
    coeffs = np.polyfit(log_t, log_r, 1)
    alpha = float(coeffs[0])
    const = float(coeffs[1])

    # R² of the fit
    residuals = log_r - (alpha * log_t + const)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_r - np.mean(log_r)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    return alpha, const, r2
