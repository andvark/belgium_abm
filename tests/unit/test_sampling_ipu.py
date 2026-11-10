import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

from belgium_abm.engines.integerise import objective, refine_by_swaps, systematic_integerise
from belgium_abm.engines.ipu import kish_effective_size, run_ipu
from belgium_abm.engines.sampling import (
    age_pmf_from_bands,
    allocate_labels,
    assign_grouped_labels,
    band_index,
    refine_group_counts,
    sample_ages,
    sample_ages_in_ranges,
    systematic_counts,
)


class SamplingTests(unittest.TestCase):
    def test_systematic_counts_sum_and_closeness(self) -> None:
        rng = np.random.default_rng(0)
        for n in (0, 1, 7, 100, 1234):
            shares = np.array([0.1, 0.0, 0.25, 0.65])
            c = systematic_counts(rng, n, shares)
            self.assertEqual(int(c.sum()), n)
            self.assertTrue((c >= 0).all())
            self.assertTrue((np.abs(c - n * shares) < 1.0).all())

    def test_systematic_counts_unbiased(self) -> None:
        rng = np.random.default_rng(1)
        total = np.zeros(3)
        for _ in range(4000):
            total += systematic_counts(rng, 7, np.array([0.2, 0.3, 0.5]))
        np.testing.assert_allclose(total / 4000, 7 * np.array([0.2, 0.3, 0.5]), atol=0.05)

    def test_systematic_counts_rejects_bad_input(self) -> None:
        rng = np.random.default_rng(0)
        for bad in (np.array([]), np.array([-1.0, 2.0]), np.array([0.0, 0.0]), np.array([np.nan, 1.0])):
            with self.assertRaises(ValueError):
                systematic_counts(rng, 5, bad)
        with self.assertRaises(ValueError):
            systematic_counts(rng, -1, np.array([1.0]))

    def test_allocate_labels(self) -> None:
        out = allocate_labels(np.random.default_rng(0), 10, np.array([0.3, 0.7]), ["a", "b"])
        self.assertEqual(sorted(out), ["a"] * 3 + ["b"] * 7)
        with self.assertRaises(ValueError):
            allocate_labels(np.random.default_rng(0), 3, np.array([1.0]), ["a", "b"])

    def test_band_index(self) -> None:
        starts = np.array([0, 5, 10])
        np.testing.assert_array_equal(band_index(np.array([0, 4, 5, 9, 10, 99]), starts), [0, 0, 1, 1, 2, 2])
        self.assertEqual(int(band_index(np.array([2]), np.array([5, 10]))[0]), -1)

    def test_age_pmf_and_sampling(self) -> None:
        bands = pd.DataFrame({"age_min": [0, 5], "age_max": [4, 9], "count": [50, 100]})
        pmf = age_pmf_from_bands(bands, 9)
        self.assertAlmostEqual(pmf.sum(), 150)
        self.assertAlmostEqual(pmf[0], 10)
        self.assertAlmostEqual(pmf[9], 20)
        ages = sample_ages(np.random.default_rng(0), pmf, 3, 6, 1000)
        self.assertTrue(((ages >= 3) & (ages <= 6)).all())
        with self.assertRaises(ValueError):
            sample_ages(np.random.default_rng(0), pmf, 7, 3, 5)

    def test_sample_ages_in_ranges(self) -> None:
        pmf = np.zeros(20)
        pmf[5:10] = 1.0
        lo = np.array([0, 5, 7, 12, 8])
        hi = np.array([19, 6, 9, 15, 2])
        ages, ok = sample_ages_in_ranges(np.random.default_rng(0), pmf, lo, hi)
        np.testing.assert_array_equal(ok, [True, True, True, False, False])  # no mass / empty range
        self.assertTrue(((ages[ok] >= np.maximum(lo[ok], 5)) & (ages[ok] <= np.minimum(hi[ok], 9))).all())


class CountRefinementTests(unittest.TestCase):
    def _setup(self):
        rng = np.random.default_rng(5)
        sizes = rng.integers(20, 200, 24)
        shares = rng.dirichlet(np.ones(4), 24)
        margin = np.repeat(np.arange(4), 6)
        return sizes, shares, margin

    @staticmethod
    def _margin_err(counts, sizes, shares, margin):
        exp = sizes[:, None] * shares
        return sum(np.abs(counts[margin == m].sum(0) - exp[margin == m].sum(0)).sum() for m in np.unique(margin))

    def test_group_sizes_preserved_and_nonnegative(self) -> None:
        sizes, shares, margin = self._setup()
        c = refine_group_counts(np.random.default_rng(1), sizes, shares, [margin], 2000)
        np.testing.assert_array_equal(c.sum(axis=1), sizes)
        self.assertTrue((c >= 0).all())

    def test_refinement_improves_margin_error(self) -> None:
        sizes, shares, margin = self._setup()
        base = refine_group_counts(np.random.default_rng(1), sizes, shares, [margin], 0)
        ref = refine_group_counts(np.random.default_rng(1), sizes, shares, [margin], 3000)
        self.assertLessEqual(self._margin_err(ref, sizes, shares, margin), self._margin_err(base, sizes, shares, margin))

    def test_deterministic_given_seed(self) -> None:
        sizes, shares, margin = self._setup()
        a = refine_group_counts(np.random.default_rng(9), sizes, shares, [margin], 500)
        b = refine_group_counts(np.random.default_rng(9), sizes, shares, [margin], 500)
        np.testing.assert_array_equal(a, b)

    def test_empty_groups_are_allowed(self) -> None:
        c = refine_group_counts(np.random.default_rng(0), np.array([0, 10, 0]), np.full((3, 2), 0.5), [], 100)
        np.testing.assert_array_equal(c.sum(axis=1), [0, 10, 0])

    def test_assign_grouped_labels_writes_every_member(self) -> None:
        members = [np.arange(0, 7), np.arange(7, 12), np.array([], dtype=int)]
        shares = np.array([[0.5, 0.5], [0.2, 0.8], [0.5, 0.5]])
        out = np.empty(12, dtype=object)
        counts = assign_grouped_labels(np.random.default_rng(2), out, members, shares, ["a", "b"], [], 100)
        self.assertEqual(int(counts.sum()), 12)
        for g, idx in enumerate(members):
            self.assertEqual(int((out[idx] == "a").sum()), int(counts[g, 0]))


class IpuTests(unittest.TestCase):
    def test_toy_problem_converges(self) -> None:
        M = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=float))
        res = run_ipu(M, np.array([3.0, 5.0]), tol=1e-10)
        self.assertTrue(res.converged)
        np.testing.assert_allclose(M @ res.weights, [3.0, 5.0], atol=1e-8)

    def test_multiplicities_count(self) -> None:
        M = sp.csr_matrix(np.array([[2, 0], [0, 1]], dtype=float))  # household 0 holds two units of constraint 0
        res = run_ipu(M, np.array([10.0, 4.0]), tol=1e-10)
        np.testing.assert_allclose(res.weights, [5.0, 4.0], atol=1e-8)

    def test_unsupported_and_zero_targets(self) -> None:
        M = sp.csr_matrix(np.array([[1, 0], [0, 0], [0, 1]], dtype=float))
        res = run_ipu(M, np.array([4.0, 2.0, 0.0]))
        self.assertEqual(res.unsupported, [1])
        self.assertEqual(res.weights[1], 0.0)  # zero target removes the contributing household

    def test_bad_input(self) -> None:
        M = sp.csr_matrix(np.eye(2))
        with self.assertRaises(ValueError):
            run_ipu(M, np.array([1.0]))
        with self.assertRaises(ValueError):
            run_ipu(M, np.array([1.0, -1.0]))

    def test_kish_effective_size(self) -> None:
        self.assertAlmostEqual(kish_effective_size(np.ones(10)), 10.0)
        self.assertAlmostEqual(kish_effective_size(np.array([1.0, 0.0, 0.0])), 1.0)


class IntegeriseTests(unittest.TestCase):
    def test_systematic_integerise_sums_to_total(self) -> None:
        c = systematic_integerise(np.random.default_rng(0), np.array([0.2, 0.3, 0.5]), 11)
        self.assertEqual(int(c.sum()), 11)
        self.assertEqual(int(systematic_integerise(np.random.default_rng(0), np.zeros(3), 6).sum()), 6)

    def test_swaps_reduce_objective_and_keep_templates(self) -> None:
        rng = np.random.default_rng(3)
        n_cand, n_cells = 60, 8
        M = sp.csr_matrix(rng.integers(0, 3, (n_cells, n_cand)).astype(float))
        template = np.repeat([0, 1], 30)
        sel0 = np.concatenate([rng.integers(0, 30, 20), rng.integers(30, 60, 20)])
        planted = np.asarray(M[:, np.concatenate([rng.integers(0, 30, 20), rng.integers(30, 60, 20)])].sum(axis=1)).ravel()
        res = refine_by_swaps(np.random.default_rng(1), sel0, template, M, planted, 20000)
        self.assertLess(res.objective_after, res.objective_before)
        np.testing.assert_array_equal(np.sort(template[res.selection]), np.sort(template[sel0]))
        counts = np.asarray(M[:, res.selection].sum(axis=1)).ravel()
        self.assertAlmostEqual(objective(counts, planted), res.objective_after, places=6)

    def test_zero_proposals_is_identity(self) -> None:
        M = sp.csr_matrix(np.eye(3))
        res = refine_by_swaps(np.random.default_rng(0), np.array([0, 1]), np.zeros(3, dtype=int), M, np.ones(3), 0)
        np.testing.assert_array_equal(res.selection, [0, 1])


if __name__ == "__main__":
    unittest.main()
