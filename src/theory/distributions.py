"""
distributions.py
================
Theoretical probability distributions for the random walk coordinates and
radial displacement.

By the Central Limit Theorem, for large n, each coordinate X_i_n
approaches a Gaussian:
    X_i_n ~ Normal(0, n/d)

The radial distribution depends on dimension.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def coordinate_gaussian_pdf(
    x: NDArray[np.float64],
    n_steps: int,
    dimension: int,
) -> NDArray[np.float64]:
    """
    Theoretical PDF of a single coordinate X_i after n steps.

    X_i ~ Normal(0, n/d) for large n (CLT).

    Parameters
    ----------
    x : array of x values
    n_steps : int
    dimension : int
    """
    sigma2 = n_steps / dimension
    sigma  = np.sqrt(sigma2)
    return stats.norm.pdf(x, loc=0.0, scale=sigma)


def radial_distribution_1d(
    r: NDArray[np.float64],
    n_steps: int,
) -> NDArray[np.float64]:
    """
    PDF of |X| in 1D.
    |X| = |X_1|, so P(|X| = r) = 2 * Normal(0, n).pdf(r) for r >= 0.
    """
    sigma = np.sqrt(float(n_steps))
    r = np.asarray(r, dtype=np.float64)
    pdf = 2.0 * stats.norm.pdf(r, loc=0.0, scale=sigma)
    pdf[r < 0] = 0.0
    return pdf


def radial_distribution_2d(
    r: NDArray[np.float64],
    n_steps: int,
) -> NDArray[np.float64]:
    """
    PDF of radial distance in 2D.
    With X_1, X_2 ~ N(0, n/2) independently,
    R = sqrt(X_1² + X_2²) follows a Rayleigh distribution with sigma=sqrt(n/2).

    p(r) = r/sigma² * exp(-r²/(2*sigma²))  for r >= 0.
    """
    sigma2 = n_steps / 2.0
    r = np.asarray(r, dtype=np.float64)
    pdf = (r / sigma2) * np.exp(-r ** 2 / (2 * sigma2))
    pdf[r < 0] = 0.0
    return pdf


def radial_distribution_3d(
    r: NDArray[np.float64],
    n_steps: int,
) -> NDArray[np.float64]:
    """
    PDF of radial distance in 3D.
    X_i ~ N(0, n/3) => R^2 / (n/3) ~ chi-squared(3).
    Maxwell-Boltzmann: p(r) = 4*pi*r^2 * (1/(2*pi*sigma^2))^(3/2) * exp(-r^2/(2*sigma^2))
    with sigma^2 = n/3.
    """
    sigma2 = n_steps / 3.0
    sigma  = np.sqrt(sigma2)
    r = np.asarray(r, dtype=np.float64)
    prefactor = 4.0 * np.pi * r ** 2 * (1.0 / (2.0 * np.pi * sigma2)) ** 1.5
    pdf = prefactor * np.exp(-r ** 2 / (2 * sigma2))
    pdf[r < 0] = 0.0
    return pdf


def radial_distribution_nd(
    r: NDArray[np.float64],
    n_steps: int,
    dimension: int,
) -> NDArray[np.float64]:
    """
    PDF of radial distance R = |X_n| in d dimensions.

    Each X_i ~ N(0, n/d). The d-dim Gaussian in spherical coordinates gives:
        p(r) = (2 * pi^(d/2) / Gamma(d/2)) * r^(d-1) *
               (1/(2*pi*sigma²))^(d/2) * exp(-r²/(2*sigma²))

    where sigma² = n/d.
    """
    from scipy.special import gamma
    d = dimension
    sigma2 = n_steps / d
    sigma  = np.sqrt(sigma2)
    r = np.asarray(r, dtype=np.float64)
    # chi distribution with d degrees of freedom, scaled
    # chi_pdf(r/sigma) / sigma
    # That is exactly: stats.chi.pdf(r, df=d, scale=sigma)
    pdf = stats.chi.pdf(r, df=d, scale=sigma)
    pdf[r < 0] = 0.0
    return pdf


def coordinate_cdf(
    x: NDArray[np.float64],
    n_steps: int,
    dimension: int,
) -> NDArray[np.float64]:
    """Theoretical CDF of a single coordinate."""
    sigma = np.sqrt(n_steps / dimension)
    return stats.norm.cdf(x, loc=0.0, scale=sigma)
