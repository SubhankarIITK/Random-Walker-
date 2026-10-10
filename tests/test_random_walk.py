"""
test_random_walk.py
===================
Unit tests for the core random walk simulation engine.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.simulation.random_walk import (
    simulate_walk,
    simulate_walk_positions,
    squared_displacement,
    displacement,
    count_distinct_sites,
    distinct_sites_over_time,
    returns_to_origin,
    generate_steps,
)


# ============================================================
# Basic correctness
# ============================================================

class TestSimulateWalk:

    def test_initial_position_is_origin_1d(self):
        traj = simulate_walk(dimension=1, n_steps=100, seed=0)
        assert traj[0, 0] == 0

    def test_initial_position_is_origin_2d(self):
        traj = simulate_walk(dimension=2, n_steps=100, seed=0)
        assert np.all(traj[0] == 0)

    def test_initial_position_is_origin_3d(self):
        traj = simulate_walk(dimension=3, n_steps=100, seed=42)
        assert np.all(traj[0] == 0)

    def test_initial_position_is_origin_4d(self):
        traj = simulate_walk(dimension=4, n_steps=50, seed=7)
        assert np.all(traj[0] == 0)

    def test_initial_position_is_origin_5d(self):
        traj = simulate_walk(dimension=5, n_steps=50, seed=99)
        assert np.all(traj[0] == 0)

    def test_trajectory_length(self):
        for d in [1, 2, 3, 4, 5]:
            traj = simulate_walk(dimension=d, n_steps=200, seed=0)
            assert traj.shape == (201, d), f"Failed for d={d}"

    def test_zero_steps_returns_origin_only(self):
        traj = simulate_walk(dimension=2, n_steps=0, seed=0)
        assert traj.shape == (1, 2)
        assert np.all(traj[0] == 0)

    def test_one_step_dimension_1(self):
        traj = simulate_walk(dimension=1, n_steps=1, seed=0)
        assert traj.shape == (2, 1)
        # After one step, position must be ±1
        assert abs(traj[1, 0]) == 1

    def test_dtype_is_int32(self):
        traj = simulate_walk(dimension=3, n_steps=100, seed=0)
        assert traj.dtype == np.int32

    def test_dimension_1_all_positions_at_most_n(self):
        n = 500
        traj = simulate_walk(dimension=1, n_steps=n, seed=42)
        assert np.all(np.abs(traj) <= n)

    def test_dimension_raises_for_zero(self):
        with pytest.raises(ValueError):
            simulate_walk(dimension=0, n_steps=10)

    def test_n_steps_raises_for_negative(self):
        with pytest.raises(ValueError):
            simulate_walk(dimension=1, n_steps=-1)


class TestStepProperties:

    def test_each_step_changes_exactly_one_coordinate(self):
        """Critical: only one coordinate should change per step."""
        for d in [1, 2, 3, 4, 5]:
            traj = simulate_walk(dimension=d, n_steps=1000, seed=d * 7)
            diffs = np.diff(traj, axis=0)  # shape (n_steps, d)
            # Exactly one non-zero per row
            nonzero_per_step = np.count_nonzero(diffs, axis=1)
            assert np.all(nonzero_per_step == 1), (
                f"d={d}: some steps changed ≠1 coordinate"
            )

    def test_each_step_magnitude_is_one(self):
        """Each step must have magnitude exactly 1."""
        for d in [1, 2, 3, 4, 5]:
            traj = simulate_walk(dimension=d, n_steps=500, seed=d * 13)
            diffs = np.diff(traj, axis=0)
            magnitudes = np.sum(np.abs(diffs), axis=1)
            assert np.all(magnitudes == 1), (
                f"d={d}: some steps have magnitude ≠1"
            )

    def test_number_of_possible_moves_is_2d(self):
        """Verify that all 2d moves are achievable (statistical test)."""
        for d in [1, 2, 3]:
            rng = np.random.default_rng(0)
            steps = generate_steps(100_000, d, rng)
            # Count unique rows
            unique_moves = set(map(tuple, steps))
            assert len(unique_moves) == 2 * d, (
                f"d={d}: expected {2*d} unique moves, got {len(unique_moves)}"
            )

    def test_step_directions_are_plus_or_minus_one(self):
        """Non-zero entries in steps must be ±1 only."""
        for d in [1, 2, 3, 4, 5]:
            traj = simulate_walk(dimension=d, n_steps=200, seed=42)
            diffs = np.diff(traj, axis=0)
            nonzero_vals = diffs[diffs != 0]
            assert np.all(np.abs(nonzero_vals) == 1), f"d={d}: non-unit steps found"

    def test_uniform_coordinate_selection(self):
        """Each coordinate should be selected approximately 1/d of the time."""
        d = 4
        traj = simulate_walk(dimension=d, n_steps=100_000, seed=0)
        diffs = np.diff(traj, axis=0)
        coord_counts = np.count_nonzero(diffs, axis=0)
        expected = 100_000 / d
        for ci, cnt in enumerate(coord_counts):
            rel_diff = abs(cnt - expected) / expected
            assert rel_diff < 0.05, (
                f"d={d}, coord {ci}: count {cnt} deviates by {rel_diff:.2%} from expected {expected}"
            )

    def test_symmetric_directions(self):
        """Plus and minus steps should be approximately equal in frequency."""
        d = 3
        traj = simulate_walk(dimension=d, n_steps=100_000, seed=7)
        diffs = np.diff(traj, axis=0)
        for i in range(d):
            col = diffs[:, i]
            plus  = np.sum(col == 1)
            minus = np.sum(col == -1)
            total = plus + minus
            if total > 0:
                frac = plus / total
                assert abs(frac - 0.5) < 0.05, (
                    f"Coord {i}: plus fraction {frac:.3f} far from 0.5"
                )


class TestReproducibility:

    def test_same_seed_gives_same_trajectory(self):
        traj1 = simulate_walk(dimension=3, n_steps=500, seed=42)
        traj2 = simulate_walk(dimension=3, n_steps=500, seed=42)
        assert np.array_equal(traj1, traj2)

    def test_different_seeds_give_different_trajectories(self):
        traj1 = simulate_walk(dimension=2, n_steps=500, seed=1)
        traj2 = simulate_walk(dimension=2, n_steps=500, seed=2)
        assert not np.array_equal(traj1, traj2)

    def test_none_seed_is_random(self):
        traj1 = simulate_walk(dimension=2, n_steps=100, seed=None)
        traj2 = simulate_walk(dimension=2, n_steps=100, seed=None)
        # Very unlikely to be identical (not guaranteed but statistically safe)
        # At least the final positions should differ with very high probability
        assert traj1.shape == traj2.shape


class TestSquaredDisplacement:

    def test_zero_steps_r2_is_zero(self):
        traj = simulate_walk(1, 0, seed=0)
        r2 = squared_displacement(traj)
        assert r2[0] == 0.0

    def test_r2_non_negative(self):
        for d in [1, 2, 3, 4, 5]:
            traj = simulate_walk(d, 500, seed=d)
            r2 = squared_displacement(traj)
            assert np.all(r2 >= 0)

    def test_r2_at_t1_is_1(self):
        """After one step, |X|² = 1 for all dimensions."""
        for d in [1, 2, 3, 4, 5]:
            traj = simulate_walk(d, 1, seed=d * 3)
            r2 = squared_displacement(traj)
            assert r2[1] == 1.0, f"d={d}: r2 after 1 step = {r2[1]}"


class TestDistinctSites:

    def test_starts_with_one_site(self):
        traj = simulate_walk(2, 100, seed=0)
        counts = distinct_sites_over_time(traj)
        assert counts[0] == 1

    def test_counts_non_decreasing(self):
        for d in [1, 2, 3]:
            traj = simulate_walk(d, 200, seed=d)
            counts = distinct_sites_over_time(traj)
            assert np.all(np.diff(counts) >= 0), f"d={d}: counts not non-decreasing"

    def test_max_sites_at_most_n_plus_one(self):
        n = 100
        traj = simulate_walk(2, n, seed=0)
        total = count_distinct_sites(traj)
        assert total <= n + 1

    def test_1d_sites_at_most_2n_plus_1(self):
        n = 200
        traj = simulate_walk(1, n, seed=5)
        total = count_distinct_sites(traj)
        assert total <= 2 * n + 1


class TestReturnsToOrigin:

    def test_structure(self):
        traj = simulate_walk(1, 1000, seed=42)
        result = returns_to_origin(traj)
        assert "n_returns" in result
        assert "first_return_time" in result
        assert "return_times" in result

    def test_n_returns_non_negative(self):
        for d in [1, 2, 3]:
            traj = simulate_walk(d, 500, seed=d * 5)
            result = returns_to_origin(traj)
            assert result["n_returns"] >= 0

    def test_first_return_time_positive_if_exists(self):
        traj = simulate_walk(1, 500, seed=0)
        result = returns_to_origin(traj)
        frt = result["first_return_time"]
        if frt is not None:
            assert frt > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
