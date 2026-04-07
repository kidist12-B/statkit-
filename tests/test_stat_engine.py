"""
test_stat_engine.py
===================
Unit tests for DataAnalyzer and the server crash simulation.

Run from the project root:
    python -m unittest discover -s tests -v
or:
    python tests/test_stat_engine.py
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stat_engine import DataAnalyzer, DatasetEmptyError, NonNumericDataError
from src.monte_carlo import run_server_simulation, CRASH_PROBABILITY


# ===========================================================================
# Group 1 — Object creation and input validation
# ===========================================================================

class TestCreation(unittest.TestCase):

    def test_accepts_integer_list(self):
        da = DataAnalyzer([10, 20, 30])
        self.assertEqual(da.count, 3)

    def test_accepts_float_list(self):
        da = DataAnalyzer([1.5, 2.5, 3.5])
        self.assertAlmostEqual(da.values[0], 1.5)

    def test_accepts_tuple(self):
        da = DataAnalyzer((5, 10, 15))
        self.assertEqual(da.count, 3)

    def test_stores_values_as_floats(self):
        da = DataAnalyzer([1, 2, 3])
        for v in da.values:
            self.assertIsInstance(v, float)

    def test_rejects_string_value(self):
        with self.assertRaises(NonNumericDataError):
            DataAnalyzer([1, 2, "hello"])

    def test_rejects_numeric_string(self):
        """A string like '5' looks numeric but must still be rejected."""
        with self.assertRaises(NonNumericDataError):
            DataAnalyzer([1, "5", 3])

    def test_rejects_none(self):
        with self.assertRaises(NonNumericDataError):
            DataAnalyzer([1, None, 3])

    def test_rejects_boolean_true(self):
        with self.assertRaises(NonNumericDataError):
            DataAnalyzer([1, True, 3])

    def test_rejects_boolean_false(self):
        with self.assertRaises(NonNumericDataError):
            DataAnalyzer([False, 2, 3])

    def test_empty_list_raises(self):
        with self.assertRaises(DatasetEmptyError):
            DataAnalyzer([])

    def test_empty_tuple_raises(self):
        with self.assertRaises(DatasetEmptyError):
            DataAnalyzer(())

    def test_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            DataAnalyzer(42)

    def test_dict_raises_type_error(self):
        with self.assertRaises(TypeError):
            DataAnalyzer({"a": 1})

    def test_single_value_accepted(self):
        da = DataAnalyzer([99])
        self.assertEqual(da.count, 1)

    def test_error_message_names_bad_values(self):
        try:
            DataAnalyzer([1, "oops", None])
        except NonNumericDataError as e:
            msg = str(e)
            self.assertIn("oops", msg)
            self.assertIn("None", msg)


# ===========================================================================
# Group 2 — Mean
# ===========================================================================

class TestMean(unittest.TestCase):

    def test_basic_mean(self):
        self.assertAlmostEqual(DataAnalyzer([2, 4, 6]).get_mean(), 4.0)

    def test_mean_with_negatives(self):
        self.assertAlmostEqual(DataAnalyzer([-10, 0, 10]).get_mean(), 0.0)

    def test_mean_single_value(self):
        self.assertAlmostEqual(DataAnalyzer([42]).get_mean(), 42.0)

    def test_mean_all_same(self):
        self.assertAlmostEqual(DataAnalyzer([7, 7, 7, 7]).get_mean(), 7.0)

    def test_mean_one_to_hundred(self):
        da = DataAnalyzer(list(range(1, 101)))
        self.assertAlmostEqual(da.get_mean(), 50.5)

    def test_mean_floats(self):
        da = DataAnalyzer([0.1, 0.2, 0.3])
        self.assertAlmostEqual(da.get_mean(), 0.2, places=10)


# ===========================================================================
# Group 3 — Median
# ===========================================================================

class TestMedian(unittest.TestCase):

    def test_median_odd_length(self):
        """Odd n — single middle element after sorting."""
        da = DataAnalyzer([5, 1, 3])      # sorted: 1 3 5 → median = 3
        self.assertAlmostEqual(da.get_median(), 3.0)

    def test_median_even_length(self):
        """Even n — average of two central elements."""
        da = DataAnalyzer([1, 3, 5, 7])   # (3+5)/2 = 4.0
        self.assertAlmostEqual(da.get_median(), 4.0)

    def test_median_unsorted_input(self):
        """Input order must not affect the result."""
        da = DataAnalyzer([9, 1, 5, 3, 7])
        self.assertAlmostEqual(da.get_median(), 5.0)

    def test_median_two_values(self):
        da = DataAnalyzer([10, 20])
        self.assertAlmostEqual(da.get_median(), 15.0)

    def test_median_single_value(self):
        da = DataAnalyzer([55])
        self.assertAlmostEqual(da.get_median(), 55.0)

    def test_median_with_repeated_values(self):
        da = DataAnalyzer([4, 4, 4, 4, 4])
        self.assertAlmostEqual(da.get_median(), 4.0)

    def test_median_negative_values(self):
        da = DataAnalyzer([-6, -2, -4])   # sorted: -6 -4 -2 → median = -4
        self.assertAlmostEqual(da.get_median(), -4.0)


# ===========================================================================
# Group 4 — Mode
# ===========================================================================

class TestMode(unittest.TestCase):

    def test_single_mode(self):
        da = DataAnalyzer([3, 3, 1, 2])
        self.assertEqual(da.get_mode(), [3.0])

    def test_bimodal(self):
        da = DataAnalyzer([1, 1, 2, 2, 3])
        self.assertEqual(da.get_mode(), [1.0, 2.0])

    def test_trimodal(self):
        da = DataAnalyzer([4, 4, 6, 6, 8, 8, 1])
        self.assertEqual(da.get_mode(), [4.0, 6.0, 8.0])

    def test_all_unique_returns_string(self):
        result = DataAnalyzer([10, 20, 30, 40]).get_mode()
        self.assertIsInstance(result, str)

    def test_all_unique_message_content(self):
        result = DataAnalyzer([10, 20, 30, 40]).get_mode()
        self.assertIn("once", result.lower())

    def test_mode_sorted_ascending(self):
        da = DataAnalyzer([9, 9, 1, 1, 5, 5])
        self.assertEqual(da.get_mode(), [1.0, 5.0, 9.0])


# ===========================================================================
# Group 5 — Variance
# ===========================================================================

class TestVariance(unittest.TestCase):
    """
    Reference dataset: [2, 4, 4, 4, 5, 5, 7, 9]
      n = 8, mean = 5
      Σ(xᵢ−μ)² = 32
      Population variance = 32 / 8  = 4.0
      Sample variance     = 32 / 7  ≈ 4.5714…
    """
    DATA = [2, 4, 4, 4, 5, 5, 7, 9]

    def test_population_variance_known_value(self):
        da = DataAnalyzer(self.DATA)
        self.assertAlmostEqual(da.get_variance(population=True), 4.0, places=10)

    def test_sample_variance_known_value(self):
        da = DataAnalyzer(self.DATA)
        self.assertAlmostEqual(da.get_variance(population=False), 32 / 7, places=10)

    def test_sample_greater_than_population(self):
        da = DataAnalyzer(self.DATA)
        self.assertGreater(
            da.get_variance(population=False),
            da.get_variance(population=True)
        )

    def test_zero_variance_identical_values(self):
        da = DataAnalyzer([5, 5, 5, 5])
        self.assertAlmostEqual(da.get_variance(population=True), 0.0)
        self.assertAlmostEqual(da.get_variance(population=False), 0.0)

    def test_sample_variance_single_element_raises(self):
        da = DataAnalyzer([7])
        with self.assertRaises(DatasetEmptyError):
            da.get_variance(population=False)

    def test_population_variance_single_element_is_zero(self):
        da = DataAnalyzer([7])
        self.assertAlmostEqual(da.get_variance(population=True), 0.0)


# ===========================================================================
# Group 6 — Standard deviation
# ===========================================================================

class TestStandardDeviation(unittest.TestCase):

    def test_population_std_known_value(self):
        """σ² = 4.0, so σ = 2.0 exactly."""
        da = DataAnalyzer([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertAlmostEqual(da.get_standard_deviation(population=True), 2.0, places=10)

    def test_sample_std_equals_sqrt_of_sample_variance(self):
        da = DataAnalyzer([3, 7, 7, 19])
        expected = math.sqrt(da.get_variance(population=False))
        self.assertAlmostEqual(da.get_standard_deviation(population=False), expected, places=12)

    def test_population_std_equals_sqrt_of_population_variance(self):
        da = DataAnalyzer([3, 7, 7, 19])
        expected = math.sqrt(da.get_variance(population=True))
        self.assertAlmostEqual(da.get_standard_deviation(population=True), expected, places=12)

    def test_std_always_non_negative(self):
        da = DataAnalyzer([-100, -50, 0, 50, 100])
        self.assertGreaterEqual(da.get_standard_deviation(), 0.0)

    def test_std_zero_for_identical_values(self):
        da = DataAnalyzer([8, 8, 8])
        self.assertAlmostEqual(da.get_standard_deviation(), 0.0)


# ===========================================================================
# Group 7 — Outlier detection
# ===========================================================================

class TestOutliers(unittest.TestCase):

    def test_symmetric_outliers_both_detected(self):
        """Symmetric outliers well beyond 2 std on each side."""
        core = [50] * 20
        da = DataAnalyzer(core + [200, -100])
        outliers = da.get_outliers(threshold=2.0)
        self.assertIn(200.0, outliers)
        self.assertIn(-100.0, outliers)

    def test_no_outliers_in_tight_cluster(self):
        da = DataAnalyzer([10, 10, 11, 10, 9, 10, 11])
        self.assertEqual(da.get_outliers(threshold=2.0), [])

    def test_outlier_result_is_sorted(self):
        core = [50] * 20
        da = DataAnalyzer(core + [300, -200])
        outliers = da.get_outliers(threshold=2.0)
        self.assertEqual(outliers, sorted(outliers))

    def test_zero_threshold_raises(self):
        da = DataAnalyzer([1, 2, 3])
        with self.assertRaises(ValueError):
            da.get_outliers(threshold=0)

    def test_negative_threshold_raises(self):
        da = DataAnalyzer([1, 2, 3])
        with self.assertRaises(ValueError):
            da.get_outliers(threshold=-2)

    def test_identical_values_no_outliers(self):
        """std dev = 0 → z-score undefined → return empty list safely."""
        da = DataAnalyzer([6, 6, 6, 6, 6])
        self.assertEqual(da.get_outliers(), [])

    def test_tighter_threshold_catches_more(self):
        da = DataAnalyzer([2, 4, 4, 4, 5, 5, 7, 9])
        loose  = da.get_outliers(threshold=2.0)
        strict = da.get_outliers(threshold=0.5)
        self.assertGreaterEqual(len(strict), len(loose))


# ===========================================================================
# Group 8 — Monte Carlo simulation
# ===========================================================================

class TestMonteCarlo(unittest.TestCase):

    def test_result_is_dict(self):
        result = run_server_simulation(100, seed=0)
        self.assertIsInstance(result, dict)

    def test_result_has_expected_keys(self):
        result = run_server_simulation(100, seed=0)
        for key in ("days", "total_crashes", "observed_prob", "theoretical", "error"):
            self.assertIn(key, result)

    def test_crashes_non_negative(self):
        result = run_server_simulation(500, seed=1)
        self.assertGreaterEqual(result["total_crashes"], 0)

    def test_crashes_not_more_than_days(self):
        result = run_server_simulation(500, seed=2)
        self.assertLessEqual(result["total_crashes"], 500)

    def test_observed_prob_in_valid_range(self):
        result = run_server_simulation(500, seed=3)
        self.assertGreaterEqual(result["observed_prob"], 0.0)
        self.assertLessEqual(result["observed_prob"], 1.0)

    def test_lln_convergence_at_large_n(self):
        """At 100,000 days the observed probability must be within 0.5% of 4.5%."""
        result = run_server_simulation(100_000, seed=77)
        self.assertAlmostEqual(result["observed_prob"], CRASH_PROBABILITY, delta=0.005)

    def test_same_seed_reproducible(self):
        r1 = run_server_simulation(1000, seed=42)
        r2 = run_server_simulation(1000, seed=42)
        self.assertEqual(r1, r2)

    def test_different_seeds_differ(self):
        r1 = run_server_simulation(1000, seed=10)
        r2 = run_server_simulation(1000, seed=20)
        self.assertNotEqual(r1["total_crashes"], r2["total_crashes"])

    def test_invalid_days_raises(self):
        with self.assertRaises(ValueError):
            run_server_simulation(0)

    def test_theoretical_prob_correct(self):
        result = run_server_simulation(100, seed=0)
        self.assertAlmostEqual(result["theoretical"], 0.045)


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)

