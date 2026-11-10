import unittest

import numpy as np

from belgium_abm.analysis.inequality import equivalised_size, p90_p10, poverty_rate, top_share, weighted_gini, weighted_quantile


class InequalityTests(unittest.TestCase):
    def test_gini_zero_for_identical_positive_incomes(self) -> None:
        self.assertAlmostEqual(weighted_gini(np.full(50, 3.0)), 0.0, places=12)

    def test_gini_known_toy_case(self) -> None:
        self.assertAlmostEqual(weighted_gini(np.array([0.0, 0.0, 0.0, 1.0])), 0.75, places=12)
        self.assertAlmostEqual(weighted_gini(np.array([1.0, 2.0, 3.0, 4.0])), 0.25, places=12)

    def test_gini_weights_equal_replication(self) -> None:
        self.assertAlmostEqual(weighted_gini(np.array([1.0, 2.0]), np.array([1.0, 3.0])), weighted_gini(np.array([1.0, 2.0, 2.0, 2.0])), places=12)

    def test_gini_degenerate_inputs(self) -> None:
        self.assertTrue(np.isnan(weighted_gini(np.array([]))))
        self.assertTrue(np.isnan(weighted_gini(np.zeros(4))))

    def test_gini_with_negative_values_can_exceed_one(self) -> None:
        self.assertGreater(weighted_gini(np.array([-5.0, 0.0, 1.0, 10.0])), 0.5)

    def test_gini_scale_invariant(self) -> None:
        x = np.array([1.0, 4.0, 9.0, 16.0])
        self.assertAlmostEqual(weighted_gini(x), weighted_gini(7 * x), places=12)

    def test_negative_weights_rejected(self) -> None:
        with self.assertRaises(ValueError):
            weighted_gini(np.array([1.0, 2.0]), np.array([1.0, -1.0]))

    def test_quantiles_and_p90_p10(self) -> None:
        x = np.arange(1.0, 101.0)
        self.assertAlmostEqual(float(weighted_quantile(x, 0.5)), 50.5)
        self.assertGreater(p90_p10(x), 1.0)
        self.assertTrue(np.isnan(p90_p10(np.array([0.0] * 9 + [5.0]))))

    def test_top_share(self) -> None:
        self.assertAlmostEqual(top_share(np.array([1.0, 1.0, 1.0, 1.0, 6.0]), 0.2), 0.6)
        self.assertAlmostEqual(top_share(np.array([1.0, 2.0, 3.0]), 1.0), 1.0)
        with self.assertRaises(ValueError):
            top_share(np.array([1.0]), 0.0)

    def test_top_share_fractional_unit(self) -> None:
        # 10 equal people: the top 15% = 1.5 people = 15% of the total
        self.assertAlmostEqual(top_share(np.ones(10), 0.15), 0.15)

    def test_poverty_rate(self) -> None:
        self.assertAlmostEqual(poverty_rate(np.arange(1.0, 11.0)), 0.3)

    def test_equivalised_size(self) -> None:
        self.assertAlmostEqual(equivalised_size(np.array([40, 38, 10, 5]), 1.0, 0.5, 0.3, 14), 2.1)
        self.assertAlmostEqual(equivalised_size(np.array([40]), 1.0, 0.5, 0.3, 14), 1.0)
        self.assertAlmostEqual(equivalised_size(np.array([14, 40]), 1.0, 0.5, 0.3, 14), 1.5)  # age 14 counts as an adult-size member


if __name__ == "__main__":
    unittest.main()
