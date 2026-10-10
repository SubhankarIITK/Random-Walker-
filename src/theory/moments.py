"""
moments.py
==========
Theoretical moments for the standard nearest-neighbor unbiased lattice
random walk in d dimensions.

Model recap
-----------
    X_0 = 0
    ΔX_t ∈ {±e_1, …, ±e_d}  with prob 1/(2d) each
    Only ONE coordinate changes per step.

Key results
-----------
A.  E[X_t] = 0   (no drift)

B.  Cov(X_i_t, X_j_t) = (n/d) * δ_{ij}
    => each coordinate is uncorrelated and has variance n/d.

C.  E[|X_t|²] = sum_i Var(X_i_t) = d * (n/d) = n
    => MSD = n   for ALL dimensions!

D.  Diffusive scaling: RMS ~ sqrt(n), exponent alpha = 0.5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def expected_position(n_steps: int, dimension: int) -> NDArray[np.float64]:
    """
    E[X_n] = (0, 0, …, 0) — vector of zeros.
    """
    return np.zeros(dimension, dtype=np.float64)


def coordinate_variance(n_steps: int, dimension: int, coord: int | None = None) -> float | NDArray[np.float64]:
    """
    Theoretical variance of each coordinate.

    Var(X_i_n) = n / d  (same for every i).

    Parameters
    ----------
    n_steps : int
    dimension : int
    coord : int or None
        If given, return scalar variance for that coordinate.
        If None, return array of length dimension.
    """
    var = n_steps / dimension
    if coord is not None:
        return float(var)
    return np.full(dimension, var, dtype=np.float64)


def msd_theory(n_steps: int | NDArray, dimension: int = 1) -> float | NDArray[np.float64]:
    """
    Theoretical mean-square displacement at step n.

    E[|X_n|²] = n   (for all d ∈ {1,2,3,4,5})

    This is independent of dimension for the standard model!

    Parameters
    ----------
    n_steps : int or array-like
    dimension : int  (unused — kept for API symmetry)
    """
    n = np.asarray(n_steps, dtype=np.float64)
    return n  # MSD = n, independent of d


def rms_displacement_theory(n_steps: int | NDArray, dimension: int = 1) -> float | NDArray[np.float64]:
    """
    Theoretical RMS displacement: sqrt(E[|X_n|²]) = sqrt(n).
    """
    n = np.asarray(n_steps, dtype=np.float64)
    return np.sqrt(n)


def coordinate_std_theory(n_steps: int, dimension: int) -> float:
    """
    Theoretical standard deviation of each coordinate: sqrt(n/d).
    """
    return float(np.sqrt(n_steps / dimension))


def msd_at_times(time_array: NDArray, dimension: int = 1) -> NDArray[np.float64]:
    """
    Compute theoretical MSD for an array of time steps.
    Returns array of same shape as time_array.
    """
    return np.asarray(time_array, dtype=np.float64)  # MSD = t


def coordinate_variance_at_times(
    time_array: NDArray,
    dimension: int,
) -> NDArray[np.float64]:
    """
    Var(X_i_t) = t / d   for array of time steps.
    """
    return np.asarray(time_array, dtype=np.float64) / dimension
