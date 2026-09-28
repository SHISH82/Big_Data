"""Запуск: python3 -m unittest discover -s labs/02_01 -p 'test_*.py'."""

import math
import unittest

import numpy as np

from main import (
    generate_series, moving_trend, count_turning_points,
    turning_points_test, kendall_test,
)


class LaboratoryTests(unittest.TestCase):
    def test_model_and_reproducibility(self):
        k, exact, series = generate_series()
        self.assertEqual(len(k), 501)
        self.assertEqual((k[0], k[-1]), (0, 500))
        self.assertEqual((exact[0], exact[-1]), (0, 5))
        np.testing.assert_array_equal(series, generate_series()[2])
        self.assertFalse(np.array_equal(series, generate_series(7)[2]))

    def test_centering_and_boundaries(self):
        line = np.arange(501, dtype=float)
        for window in (21, 51, 111):
            m = window // 2
            for method in ('mean', 'median'):
                trend = moving_trend(line, window, method)
                np.testing.assert_allclose(trend[m:-m], line[m:-m])
                self.assertEqual(trend[0], m / 2)
                self.assertEqual(trend[-1], 500 - m / 2)
                self.assertEqual(np.isfinite(trend).sum(), 501)

    def test_median_rejects_isolated_outlier(self):
        series = np.zeros(31)
        series[15] = 21
        mean = moving_trend(series, 21, 'mean')
        median = moving_trend(series, 21, 'median')
        np.testing.assert_allclose(mean[10:-10], 1)
        np.testing.assert_array_equal(median[10:-10], 0)

    def test_turning_points_with_plateaus(self):
        self.assertEqual(count_turning_points([1, 3, 2, 4, 0]), 3)
        self.assertEqual(count_turning_points([1, 2, 2, 1]), 0)
        self.assertEqual(count_turning_points(np.arange(100)), 0)
        self.assertLess(turning_points_test(np.arange(100))['turning_p'], 0.05)

    def test_kendall_trend_and_reversal(self):
        ascending = kendall_test(np.arange(100))
        descending = kendall_test(np.arange(100)[::-1])
        self.assertEqual(ascending['kendall_tau'], 1)
        self.assertEqual(descending['kendall_tau'], -1)
        self.assertEqual(ascending['kendall_p'], descending['kendall_p'])
        self.assertLess(ascending['kendall_p'], 0.05)
        variance_tau = 2 * (2 * 100 + 5) / (9 * 100 * 99)
        self.assertAlmostEqual(ascending['kendall_z'], 1 / math.sqrt(variance_tau))

    def test_kendall_equal_pairs_are_not_decreasing(self):
        result = kendall_test([1, 1, 2, 2])
        self.assertEqual(result['kendall_P'], 4)
        self.assertEqual(result['kendall_Q'], 0)
        self.assertEqual(result['tied_pairs'], 2)
        self.assertAlmostEqual(result['kendall_tau_a'], 2 / 3)
        self.assertAlmostEqual(result['kendall_tau'], 4 / math.sqrt(24))
        self.assertAlmostEqual(result['kendall_z'], math.sqrt(2.4))

    def test_permutation_test_reproducibility(self):
        series = [0, 1, 0, -1, 0, 2, 0, -2]
        result = turning_points_test(series, permutations=199)
        self.assertEqual(result['turning_method'], 'permutation')
        self.assertEqual(result, turning_points_test(series, permutations=199))
        self.assertGreater(result['turning_p'], 0)
        self.assertLessEqual(result['turning_p'], 1)


if __name__ == '__main__':
    unittest.main()
