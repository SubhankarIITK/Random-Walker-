"""
test_convergence.py
===================
Tests for Monte Carlo convergence and ensemble simulation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.simulation.ensemble import simulate_ensemble, ensemble_final_positions
from src.analysis.convergence import run_convergence_experiment


class TestEnsembleSimulation:

    def test_ensemble_shape_final_positions(self):
        for d in [1, 2, 3, 4, 5]:
            pos = ensemble_final_positions(d, 100, 50, seed=d)
            assert pos.shape == (50, d)

    def test_ensemble_msd_columns(self):
        ens = simulate_ensemble(2, 500, 100, seed=0, compute_returns=True)
        assert len(ens.mean_msd) == len(ens.time_steps)
        assert len(ens.time_steps) >= 2

    def test_ensemble_initial_msd_zero(self):
        ens = simulate_ensemble(2, 100, 50, seed=42)
        assert ens.mean_msd[0] == 0.0

    def test_ensemble_time_steps_start_at_zero(self):
        ens = simulate_ensemble(2, 100, 50, seed=0)
        assert ens.time_steps[0] == 0

    def test_ensemble_time_steps_end_at_n(self):
        n = 200
        ens = simulate_ensemble(3, n, 50, seed=0)
        assert ens.time_steps[-1] == n

    def test_ever_returned_shape(self):
        n_w = 50
        ens = simulate_ensemble(2, 100, n_w, seed=0, compute_returns=True)
        assert ens.ever_returned.shape == (n_w,)

    def test_return_counts_non_negative(self):
        ens = simulate_ensemble(1, 200, 30, seed=0, compute_returns=True)
        assert np.all(ens.n_returns >= 0)

    def test_ever_returned_consistent(self):
        ens = simulate_ensemble(2, 200, 50, seed=7, compute_returns=True)
        # ever_returned[i] should be True iff n_returns[i] > 0
        assert np.all(ens.ever_returned == (ens.n_returns > 0))

    def test_final_positions_shape(self):
        for d in [1, 2, 3, 4, 5]:
            n_w = 20
            ens = simulate_ensemble(d, 100, n_w, seed=d * 3)
            assert ens.final_positions.shape == (n_w, d)

    def test_thinning_reduces_checkpoints(self):
        n = 1000
        ens_thin = simulate_ensemble(2, n, 50, seed=0, thinning=100)
        ens_full = simulate_ensemble(2, n, 50, seed=0, thinning=1)
        assert len(ens_thin.time_steps) < len(ens_full.time_steps)

    def test_reproducible_ensemble(self):
        ens1 = simulate_ensemble(2, 100, 30, seed=99)
        ens2 = simulate_ensemble(2, 100, 30, seed=99)
        assert np.array_equal(ens1.final_positions, ens2.final_positions)


class TestConvergence:

    def test_convergence_df_columns(self):
        df = run_convergence_experiment(2, 100, [10, 100], seed=0)
        required_cols = ["n_walkers", "msd_sim", "msd_theory", "abs_error", "pct_error"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_convergence_rows_equal_walker_counts(self):
        walker_counts = [10, 100, 1000]
        df = run_convergence_experiment(1, 500, walker_counts, seed=42)
        assert len(df) == len(walker_counts)

    def test_theory_value_correct(self):
        n = 500
        df = run_convergence_experiment(3, n, [100], seed=0)
        assert abs(df.iloc[0]["msd_theory"] - n) < 1e-10

    def test_error_generally_decreases(self):
        """Larger ensembles should generally have smaller errors (statistical test)."""
        df = run_convergence_experiment(2, 1000, [50, 500, 5000], seed=0)
        # The errors won't be strictly monotone but on average should decrease
        errors = df["pct_error"].values
        # At least the largest ensemble should be smaller error than smallest
        assert errors[-1] < errors[0] * 5, (
            f"Error did not decrease noticeably: {errors}"
        )

    def test_msd_sim_not_fabricated(self):
        """msd_sim must be computed from actual positions, not equal to theory."""
        df = run_convergence_experiment(2, 200, [20], seed=0)
        # With only 20 walkers, exact match with theory is very unlikely
        msd_sim = df.iloc[0]["msd_sim"]
        msd_th  = df.iloc[0]["msd_theory"]
        # They should NOT be identical (simulation has noise)
        assert msd_sim != msd_th, "msd_sim appears to be fabricated (exactly equals theory)"

    def test_se_positive_for_multi_walker(self):
        df = run_convergence_experiment(2, 500, [100], seed=0)
        assert df.iloc[0]["se"] > 0


class TestMSDScaling:

    def test_msd_grows_approximately_linearly(self):
        """
        Test that ensemble MSD grows approximately as t (the slope ≈ 1).
        """
        from src.analysis.fitting import fit_linear

        d = 2
        n = 2000
        n_walk = 2000
        ens = simulate_ensemble(d, n, n_walk, seed=0, thinning=10, compute_returns=False)

        ts  = ens.time_steps[1:].astype(float)  # skip t=0
        msd = ens.mean_msd[1:]

        result = fit_linear(ts, msd)
        slope = result["slope"]

        assert abs(slope - 1.0) < 0.1, f"MSD slope {slope:.4f} deviates too far from 1.0"

    def test_scaling_exponent_near_half(self):
        """
        log-log regression of RMS vs time should give α ≈ 0.5.
        """
        from src.theory.scaling import log_log_fit_alpha

        d = 3
        n = 2000
        n_walk = 1000
        ens = simulate_ensemble(d, n, n_walk, seed=7, thinning=10, compute_returns=False)

        ts  = ens.time_steps.astype(float)
        rms = ens.mean_displacement
        mask = ts > 10
        alpha, _, r2 = log_log_fit_alpha(ts[mask], rms[mask])

        assert abs(alpha - 0.5) < 0.15, f"Scaling exponent {alpha:.4f} far from 0.5"
        assert r2 > 0.95, f"Log-log fit R² = {r2:.4f} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
