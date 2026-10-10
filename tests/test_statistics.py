"""
test_statistics.py
==================
Unit tests for statistical analysis functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.simulation.random_walk import simulate_walk
from src.simulation.ensemble import simulate_ensemble, ensemble_final_positions
from src.analysis.statistics import (
    compute_walk_statistics, compute_ensemble_statistics,
    compute_return_statistics,
)
from src.analysis.fitting import fit_power_law, fit_linear
from src.analysis.uncertainty import standard_error, confidence_interval_95


class TestWalkStatistics:

    def test_structure_keys_present(self):
        traj = simulate_walk(2, 100, seed=0)
        stats = compute_walk_statistics(traj, 2)
        required_keys = [
            "n_steps", "dimension", "final_position", "final_r", "final_r2",
            "msd_theory", "abs_error_msd", "rel_error_msd",
        ]
        for k in required_keys:
            assert k in stats, f"Missing key: {k}"

    def test_zero_steps_statistics(self):
        traj = simulate_walk(1, 0, seed=0)
        stats = compute_walk_statistics(traj, 1)
        assert stats["n_steps"] == 0
        assert stats["final_r2"] == 0.0

    def test_msd_theory_matches_module(self):
        from src.theory.moments import msd_theory
        for d in [1, 2, 3]:
            n = 100
            traj = simulate_walk(d, n, seed=d)
            stats = compute_walk_statistics(traj, d)
            assert abs(stats["msd_theory"] - float(msd_theory(n))) < 1e-10

    def test_error_non_negative(self):
        traj = simulate_walk(2, 500, seed=42)
        stats = compute_walk_statistics(traj, 2)
        assert stats["abs_error_msd"] >= 0
        assert stats["rel_error_msd"] >= 0


class TestEnsembleStatistics:

    def test_msd_approaches_theory_large_ensemble(self):
        """With many walkers, ensemble MSD should be close to theoretical."""
        d = 3
        n = 1000
        n_walk = 5000
        pos = ensemble_final_positions(d, n, n_walk, seed=0)
        stats = compute_ensemble_statistics(pos, n, d)
        assert stats["pct_error_msd"] < 5.0, (
            f"MSD error too large: {stats['pct_error_msd']:.2f}%"
        )

    def test_coord_means_near_zero(self):
        """E[X_i] = 0: coordinate means should be small."""
        d = 2
        n = 500
        pos = ensemble_final_positions(d, n, 2000, seed=7)
        stats = compute_ensemble_statistics(pos, n, d)
        for m in stats["coord_means"]:
            sigma = np.sqrt(n / d)
            # Mean should be within 3 sigma / sqrt(2000)
            bound = 3 * sigma / np.sqrt(2000)
            assert abs(m) < bound * 2, f"coord mean {m} unexpectedly large"

    def test_rms_close_to_sqrt_n(self):
        d = 2
        n = 1000
        pos = ensemble_final_positions(d, n, 3000, seed=99)
        stats = compute_ensemble_statistics(pos, n, d)
        rms_th = np.sqrt(n)
        assert abs(stats["rms_sim"] - rms_th) / rms_th < 0.1


class TestFitting:

    def test_fit_power_law_exact(self):
        t = np.arange(1, 101, dtype=float)
        r = 3.0 * t ** 0.5
        result = fit_power_law(t, r)
        assert abs(result["alpha"] - 0.5) < 1e-5
        assert abs(result["A"] - 3.0) < 1e-3
        assert result["r_squared"] > 0.9999

    def test_fit_linear_exact(self):
        x = np.arange(1, 101, dtype=float)
        y = 2.5 * x + 1.0
        result = fit_linear(x, y)
        assert abs(result["slope"] - 2.5) < 1e-8
        assert abs(result["intercept"] - 1.0) < 1e-8

    def test_fit_power_law_handles_zeros(self):
        t = np.array([0.0, 1.0, 4.0, 9.0])
        r = np.array([0.0, 1.0, 2.0, 3.0])
        result = fit_power_law(t, r)
        assert abs(result["alpha"] - 0.5) < 1e-5

    def test_fit_linear_force_zero(self):
        x = np.arange(1, 50, dtype=float)
        y = 1.5 * x + np.random.default_rng(0).normal(0, 0.01, len(x))
        result = fit_linear(x, y, force_zero_intercept=True)
        assert result["intercept"] == 0.0
        assert abs(result["slope"] - 1.5) < 0.05


class TestUncertainty:

    def test_standard_error_known_value(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # std (ddof=1) = sqrt(2.5), n=5 → SE = sqrt(2.5/5) = sqrt(0.5) ≈ 0.7071
        se = standard_error(values)
        expected = np.std(values, ddof=1) / np.sqrt(5)
        assert abs(se - expected) < 1e-12

    def test_standard_error_single_value(self):
        assert standard_error(np.array([42.0])) == 0.0

    def test_ci_contains_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(5.0, 1.0, 30)
        lo, hi = confidence_interval_95(values)
        mean = float(np.mean(values))
        assert lo <= mean <= hi

    def test_ci_width_decreases_with_n(self):
        rng = np.random.default_rng(0)
        v_small = rng.normal(0, 1, 20)
        v_large = rng.normal(0, 1, 2000)
        lo_s, hi_s = confidence_interval_95(v_small)
        lo_l, hi_l = confidence_interval_95(v_large)
        assert (hi_s - lo_s) > (hi_l - lo_l)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
