"""Seeded property-style tests (no hypothesis available): loops over seeds and sizes."""

import re
import unittest
from pathlib import Path

import numpy as np

from belgium_abm.analysis.inequality import weighted_gini, weighted_quantile
from belgium_abm.analysis.validation import check_population
from belgium_abm.engines.population import generate_population
from belgium_abm.engines.sampling import systematic_counts
from belgium_abm.io.loaders import load_targets
from belgium_abm.policies import POLICY_COMPONENTS, PolicyConfig
from belgium_abm.run import evaluate
from tests.helpers import ROOT, modified_demo_config


class GeneratorProperties(unittest.TestCase):
    def test_invariants_hold_across_seeds_and_sizes(self) -> None:
        for seed, n in ((1, 1500), (2, 1800), (3, 2500), (11, 1200), (42, 2000)):
            cfg = modified_demo_config(n)
            targets = load_targets(cfg)
            pop = generate_population(cfg, targets, seed)
            checks = check_population(pop.persons, pop.households, cfg, targets.max_age)
            bad = [c.name for c in checks if not c.passed]
            self.assertEqual(bad, [], f"seed={seed} n={n}")
            verdict, _, comparisons, _ = evaluate(cfg, targets, pop, seed)
            self.assertNotEqual(verdict, "FAIL", f"seed={seed} n={n}")
            warn_fraction = float((comparisons["status"] == "warn").mean())
            self.assertLess(warn_fraction, 0.05, f"seed={seed} n={n}")


class MathProperties(unittest.TestCase):
    def test_systematic_counts_always_sum_and_stay_close(self) -> None:
        rng = np.random.default_rng(123)
        for _ in range(300):
            k = int(rng.integers(1, 12))
            shares = rng.random(k) * (rng.random(k) > 0.2)
            if shares.sum() == 0:
                shares[0] = 1.0
            n = int(rng.integers(0, 5000))
            c = systematic_counts(rng, n, shares)
            self.assertEqual(int(c.sum()), n)
            self.assertTrue((np.abs(c - n * shares / shares.sum()) < 1.0 + 1e-9).all())

    def test_gini_bounds_scale_and_permutation(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(100):
            x = rng.lognormal(0, 1, int(rng.integers(2, 200)))
            w = rng.random(x.size) + 0.1
            g = weighted_gini(x, w)
            self.assertTrue(0.0 <= g < 1.0)
            perm = rng.permutation(x.size)
            self.assertAlmostEqual(g, weighted_gini(x[perm], w[perm]), places=10)
            self.assertAlmostEqual(g, weighted_gini(3.5 * x, w), places=10)

    def test_quantile_is_monotone(self) -> None:
        rng = np.random.default_rng(5)
        x, w = rng.normal(size=300), rng.random(300) + 0.1
        qs = weighted_quantile(x, np.linspace(0.01, 0.99, 50), w)
        self.assertTrue((np.diff(qs) >= -1e-12).all())


class NoHiddenEmpiricalValues(unittest.TestCase):
    def test_policy_skeleton_is_empty(self) -> None:
        self.assertEqual(PolicyConfig().components, {})
        self.assertEqual(len(PolicyConfig().missing_components()), len(POLICY_COMPONENTS))

    def test_source_code_does_not_embed_statutory_or_demographic_numbers(self) -> None:
        """Crude scan: no percent-style literals or tax/benefit words next to numbers in src/."""
        banned = re.compile(r"(tax_rate|vat_rate|bracket|marginal_rate|contribution_rate|mortality_rate|fertility_rate)\s*[:=]\s*[0-9]", re.I)
        for path in (ROOT / "src").rglob("*.py"):
            self.assertIsNone(banned.search(path.read_text(encoding="utf-8")), path)

    def test_every_demo_parameter_that_looks_empirical_is_marked_synthetic(self) -> None:
        cfg = modified_demo_config(1000)
        empirical = {"adult_age", "working_age_min", "min_parent_age_at_birth", "max_parent_age_at_birth", "couple_age_difference_sd"}
        for name in empirical:
            self.assertTrue(cfg.parameters[name].synthetic_demo_only, name)

    def test_demo_data_files_all_declare_fictional_origin(self) -> None:
        for path in (Path(ROOT) / "data" / "demo").glob("*.csv"):
            head = "\n".join(path.read_text().splitlines()[:8])
            self.assertIn("synthetic_demo_only: true", head, path)
            self.assertIn("NOT Belgian data", head, path)


if __name__ == "__main__":
    unittest.main()
