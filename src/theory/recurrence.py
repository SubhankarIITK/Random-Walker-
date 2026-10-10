"""
recurrence.py
=============
Recurrence and transience theory for the standard lattice random walk.

Classical result (Polya, 1921):
    d = 1, 2  → recurrent   (probability of eventual return to origin = 1)
    d >= 3    → transient    (positive probability of never returning)

IMPORTANT CAVEAT:
This module provides the theoretical classification only.
A finite simulation cannot prove or disprove infinite-time recurrence.
The empirical return statistics from simulations are:
  - finite-time estimates
  - subject to Monte Carlo noise
  - approaching but not equal to the infinite-time result

Do NOT interpret finite simulation return rates as proof of recurrence.
"""

from __future__ import annotations


RECURRENCE_CLASSIFICATION = {
    1: "Recurrent",
    2: "Recurrent",
    3: "Transient",
    4: "Transient",
    5: "Transient",
}

RECURRENCE_NOTES = {
    1: (
        "The 1D random walk is recurrent (Polya, 1921). "
        "The probability of eventual return to the origin is 1. "
        "The walk returns to the origin infinitely often with probability 1."
    ),
    2: (
        "The 2D random walk is recurrent (Polya, 1921). "
        "The probability of eventual return to the origin is 1. "
        "However, the mean return time is infinite."
    ),
    3: (
        "The 3D random walk is transient (Polya, 1921). "
        "The probability of eventual return to the origin is ≈ 0.3405. "
        "The walk escapes to infinity with probability ≈ 0.6595."
    ),
    4: (
        "The 4D random walk is transient. "
        "The probability of eventual return is smaller than in 3D. "
        "Transience becomes stronger as dimension increases."
    ),
    5: (
        "The 5D random walk is transient. "
        "In high dimensions the walk spends most of its time far from the origin."
    ),
}

# Approximate return probabilities (theoretical values where known)
# For d=1,2 the exact return probability = 1.
# For d=3 the return probability is approximately 0.3405 (Watson 1939).
# For d=4,5 exact values require numerical integration; approximate values given.
RETURN_PROBABILITY = {
    1: 1.0,
    2: 1.0,
    3: 0.3405,    # Watson (1939) constant
    4: 0.1932,    # approximate
    5: 0.1351,    # approximate
}


def is_recurrent(dimension: int) -> bool:
    """Return True if the walk in `dimension` is theoretically recurrent."""
    return dimension <= 2


def recurrence_classification(dimension: int) -> str:
    """Return 'Recurrent' or 'Transient' for the given dimension."""
    return RECURRENCE_CLASSIFICATION.get(dimension, "Unknown")


def theoretical_return_probability(dimension: int) -> float | None:
    """
    Return theoretical infinite-time return probability.
    Returns None for dimensions not in the lookup table.
    """
    return RETURN_PROBABILITY.get(dimension, None)


def recurrence_note(dimension: int) -> str:
    """Return a human-readable explanation of the recurrence classification."""
    return RECURRENCE_NOTES.get(dimension, "No information available.")


def finite_return_rate_caveat() -> str:
    """Return a standard scientific caveat about finite-time simulations."""
    return (
        "⚠️ SCIENTIFIC CAVEAT: The return frequencies shown above are "
        "empirical estimates from a finite simulation horizon. "
        "They approximate—but do NOT prove—the infinite-time theoretical "
        "recurrence/transience classification. "
        "For recurrent walks (d≤2) the finite-time return fraction will be "
        "less than 1 even though the theoretical eventual-return probability is 1. "
        "As simulation length → ∞ the empirical fraction converges to the "
        "theoretical value."
    )
