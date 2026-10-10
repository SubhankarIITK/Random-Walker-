"""
test_theory.py
==============
Unit tests for the theoretical predictions module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.theory.moments import (
    msd_theory, coordinate_variance, rms_displacement_theory,
    coordinate_std_theory, msd_at_times,
)
from src.theory.distributions import (
    coordinate_gaussian_pdf, radial_distribution_nd,
)
from src.theory.scaling import theoretical_scaling_exponent, log_log_fit_alpha
from src.theory.recurrence import (
    is_recurrent, recurrence_classification, theoretical_return_probability,
)


class TestMoments:

    def test_msd_theory_equals_n(self):
        """E[|X_n|²] = n for all dimensions."""
        for d in [1, 2, 3, 4, 5]:
            for n in [0, 1, 10, 100, 1000]:
                assert msd_theory(n, d) == float(n), f"d={d}, n={n}"

    def test_msd_theory_is_dimension_independent(self):
        """MSD must be the same for all dimensions."""
        n = 500
        values = [float(msd_theory(n, d)) for d in [1, 2, 3, 4, 5]]
        assert len(set(values)) == 1, f"MSD varies by dimension: {values}"

    def test_coordinate_variance_formula(self):
        """Var(X_i) = n/d."""
        for d in [1, 2, 3, 4, 5]:
            for n in [1, 100, 1000]:
                expected = n / d
                got_val = coordinate_variance(n, d)
                got = float(got_val if isinstance(got_val, (int, float)) else got_val[0])
                assert abs(got - expected) < 1e-10, f"d={d}, n={n}"

    def test_coordinate_variance_array_length(self):
        d = 4
        var_arr = coordinate_variance(100, d)
        assert len(var_arr) == d

    def test_coordinate_variance_all_equal(self):
        d = 5
        var_arr = coordinate_variance(1000, d)
        assert np.allclose(var_arr, var_arr[0])

    def test_rms_theory_sqrt_n(self):
        for n in [1, 4, 9, 25, 100]:
            assert abs(float(rms_displacement_theory(n)) - np.sqrt(n)) < 1e-10

    def test_coordinate_std_sqrt_n_over_d(self):
        for d in [1, 2, 3, 4, 5]:
            for n in [100, 1000]:
                expected = np.sqrt(n / d)
                got = coordinate_std_theory(n, d)
                assert abs(got - expected) < 1e-10

    def test_msd_at_times_array(self):
        ts = np.array([0, 1, 10, 100])
        result = msd_at_times(ts)
        assert np.allclose(result, ts.astype(float))

    def test_msd_zero_steps(self):
        assert float(msd_theory(0)) == 0.0

    def test_coordinate_variance_sums_to_msd(self):
        """Sum of Var(X_i) over all i should equal MSD = n."""
        for d in [1, 2, 3, 4, 5]:
            n = 1000
            var_val = coordinate_variance(n, d)
            scalar_var = float(var_val if isinstance(var_val, (int, float)) else var_val[0])
            total_var = d * scalar_var
            assert abs(total_var - n) < 1e-10, f"d={d}: sum of variances = {total_var}"


class TestDistributions:

    def test_gaussian_pdf_integrates_to_one(self):
        """Numerical integration of coordinate_gaussian_pdf should ≈ 1."""
        from scipy import integrate
        for d in [1, 2, 3, 5]:
            n = 500
            result, _ = integrate.quad(
                lambda x: coordinate_gaussian_pdf(np.array([x]), n, d)[0],
                -200, 200
            )
            assert abs(result - 1.0) < 0.01, f"d={d}: integral = {result}"

    def test_gaussian_pdf_symmetric(self):
        n, d = 100, 3
        x = np.array([5.0])
        assert abs(
            coordinate_gaussian_pdf(x, n, d)[0] - coordinate_gaussian_pdf(-x, n, d)[0]
        ) < 1e-12

    def test_gaussian_pdf_peaks_at_zero(self):
        n, d = 500, 2
        pdf_0  = coordinate_gaussian_pdf(np.array([0.0]), n, d)[0]
        pdf_10 = coordinate_gaussian_pdf(np.array([10.0]), n, d)[0]
        assert pdf_0 > pdf_10

    def test_radial_pdf_non_negative(self):
        r = np.linspace(0, 100, 200)
        for d in [1, 2, 3, 4, 5]:
            pdf = radial_distribution_nd(r, 500, d)
            assert np.all(pdf >= 0), f"d={d}: negative PDF values"

    def test_radial_pdf_zero_at_negative_r(self):
        r = np.array([-1.0, -5.0])
        for d in [1, 2, 3, 4, 5]:
            pdf = radial_distribution_nd(r, 500, d)
            assert np.all(pdf == 0)


class TestScaling:

    def test_theoretical_exponent_is_half(self):
        assert theoretical_scaling_exponent() == 0.5

    def test_log_log_fit_perfect_sqrt(self):
        """log-log fit of sqrt(t) should give alpha ≈ 0.5."""
        t = np.arange(1, 201, dtype=float)
        r = np.sqrt(t)
        alpha, const, r2 = log_log_fit_alpha(t, r)
        assert abs(alpha - 0.5) < 1e-6
        assert abs(r2 - 1.0) < 1e-10

    def test_log_log_fit_power_law(self):
        """Fit a known power law."""
        t = np.arange(1, 101, dtype=float)
        r = 2.0 * t ** 0.7
        alpha, _, r2 = log_log_fit_alpha(t, r)
        assert abs(alpha - 0.7) < 1e-6

    def test_log_log_fit_ignores_t_zero(self):
        t = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        alpha, _, _ = log_log_fit_alpha(t, r)
        assert abs(alpha - 0.5) < 1e-6


class TestRecurrence:

    def test_1d_is_recurrent(self):
        assert is_recurrent(1) is True

    def test_2d_is_recurrent(self):
        assert is_recurrent(2) is True

    def test_3d_is_transient(self):
        assert is_recurrent(3) is False

    def test_4d_is_transient(self):
        assert is_recurrent(4) is False

    def test_5d_is_transient(self):
        assert is_recurrent(5) is False

    def test_classifications(self):
        assert recurrence_classification(1) == "Recurrent"
        assert recurrence_classification(2) == "Recurrent"
        assert recurrence_classification(3) == "Transient"
        assert recurrence_classification(4) == "Transient"
        assert recurrence_classification(5) == "Transient"

    def test_1d_return_prob_is_one(self):
        assert theoretical_return_probability(1) == 1.0

    def test_2d_return_prob_is_one(self):
        assert theoretical_return_probability(2) == 1.0

    def test_3d_return_prob_less_than_one(self):
        p = theoretical_return_probability(3)
        assert p is not None
        assert 0 < p < 1

    def test_return_prob_decreases_with_dimension(self):
        probs = [theoretical_return_probability(d) for d in [3, 4, 5]]
        for i in range(len(probs) - 1):
            assert probs[i] > probs[i + 1], "Return probability should decrease with dimension"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
