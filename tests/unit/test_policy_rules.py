"""Pure tax/benefit functions, policy file validation, overrides, common random numbers."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import yaml

from belgium_abm.engines import crn
from belgium_abm.engines.economy import EconomyConfig
from belgium_abm.policies.config import MissingPolicyDataError
from belgium_abm.policies.rules import (
    PolicyRules,
    personal_income_tax,
    progressive_tax,
    self_employed_contribution,
    statutory_pension,
    unemployment_benefit,
)
from belgium_abm.scenarios import Scenario
from tests.helpers import ROOT

POLICY = ROOT / "configs" / "policy_illustrative.yaml"
BRACKETS = [[10, 0.2], [30, 0.4], [None, 0.5]]


class BracketTests(unittest.TestCase):
    def test_known_values(self) -> None:
        x = np.array([-5.0, 0, 10, 20, 30, 50])
        np.testing.assert_allclose(progressive_tax(x, BRACKETS), [0, 0, 2, 6, 10, 20])

    def test_properties_random(self) -> None:
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 200, 500))
        t = progressive_tax(x, BRACKETS)
        self.assertTrue((np.diff(t) >= -1e-12).all(), "monotone")
        self.assertTrue((t <= 0.5 * x + 1e-12).all(), "never above top marginal rate")
        self.assertTrue((np.diff(t) <= 0.5 * np.diff(x) + 1e-12).all(), "1-Lipschitz at the top rate (continuity)")
        self.assertTrue((t[x > 50] >= 0.2 * x[x > 50] - 1e-9).all())

    def test_threshold_edges_are_continuous(self) -> None:
        for edge in (10.0, 30.0):
            lo, hi = progressive_tax(np.array([edge - 1e-9, edge + 1e-9]), BRACKETS)
            self.assertAlmostEqual(lo, hi, places=6)


class BenefitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = PolicyRules.load(POLICY)

    def test_allowances_reduce_tax(self) -> None:
        base = np.array([40000.0])
        t0 = personal_income_tax(base, np.array([0.0]), np.array([0.0]), self.rules)
        t1 = personal_income_tax(base, np.array([1.0]), np.array([0.0]), self.rules)
        t2 = personal_income_tax(base, np.array([1.0]), np.array([2.0]), self.rules)
        self.assertGreater(t0[0], t1[0])
        self.assertGreater(t1[0], t2[0])

    def test_unemployment_benefit_bounds_and_eligibility(self) -> None:
        c = self.rules.components["unemployment_benefits"]
        b = unemployment_benefit(np.array([1e3, 3e4, 1e6, 3e4]), np.array([True, True, True, False]), self.rules)
        self.assertEqual(b[0], c["min_annual_eur"])
        self.assertEqual(b[2], c["max_annual_eur"])
        self.assertEqual(b[3], 0.0)
        self.assertAlmostEqual(b[1], 3e4 * c["replacement_rate"] if 3e4 * c["replacement_rate"] <= c["max_annual_eur"] else c["max_annual_eur"])

    def test_pension_bounds(self) -> None:
        c = self.rules.components["pensions"]
        p = statutory_pension(np.array([0.0, 20000.0, 1e9]), self.rules)
        self.assertEqual((p[0], p[2]), (c["min_annual_eur"], c["max_annual_eur"]))
        self.assertEqual(p[1], 20000.0)

    def test_self_employed_minimum_only_when_active(self) -> None:
        c = self.rules.components["self_employed_social_contributions"]
        out = self_employed_contribution(np.array([0.0, 1e5, 0.0]), self.rules, np.array([True, True, False]))
        self.assertEqual(out[0], c["min_annual_eur"])
        self.assertAlmostEqual(out[1], 1e5 * c["rate"])
        self.assertEqual(out[2], 0.0)


class PolicyFileTests(unittest.TestCase):
    def test_every_component_is_sourced_and_flagged_unverified(self) -> None:
        r = PolicyRules.load(POLICY)
        self.assertFalse(r.verified)
        self.assertIn("NOT BELGIAN LAW", r.banner)
        for name, block in r.components.items():
            self.assertEqual(block["status"], "scenario", name)
            self.assertIn("source", block)

    def test_missing_component_or_source_rejected(self) -> None:
        raw = yaml.safe_load(POLICY.read_text())
        bad = copy.deepcopy(raw)
        del bad["components"]["pensions"]
        with self.assertRaises(MissingPolicyDataError):
            PolicyRules.from_dict(bad)
        bad = copy.deepcopy(raw)
        del bad["components"]["corporate_income_tax"]["source"]
        with self.assertRaises(MissingPolicyDataError):
            PolicyRules.from_dict(bad)

    def test_overrides(self) -> None:
        r = PolicyRules.load(POLICY)
        same = r.with_overrides({}, "x")
        self.assertEqual(same.fingerprint(), r.fingerprint())
        changed = r.with_overrides({"corporate_income_tax.rate": 0.4})
        self.assertEqual(changed.g("corporate_income_tax", "rate"), 0.4)
        self.assertEqual(r.g("corporate_income_tax", "rate"), 0.25)  # original untouched
        self.assertIn("SCENARIO OVERRIDES", changed.banner)
        with self.assertRaises(KeyError):
            r.with_overrides({"corporate_income_tax.nope": 1})
        with self.assertRaises(KeyError):
            r.with_overrides({"nonsense.rate": 1})

    def test_scenario_deltas_and_validation(self) -> None:
        r = PolicyRules.load(POLICY)
        s = Scenario.load(ROOT / "scenarios" / "corporate_tax_plus_5pp.yaml")
        self.assertAlmostEqual(s.apply_policy(r).g("corporate_income_tax", "rate"), 0.30)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.yaml"
            p.write_text("name: a\nbogus: 1\n")
            with self.assertRaises(ValueError):
                Scenario.load(p)

    def test_all_economy_parameters_are_assumptions(self) -> None:
        e = EconomyConfig.load(ROOT / "configs" / "economy_stylised.yaml")
        self.assertTrue(all(p.status.value == "assumed" for p in e.parameters.values()))
        self.assertEqual(e.with_values(fertility_multiplier=0.5).parameters["fertility_multiplier"].status.value, "scenario")


class CrnTests(unittest.TestCase):
    def test_draw_depends_only_on_key(self) -> None:
        ids = np.arange(1, 5001)
        u = crn.uniform(7, "s", 3, ids)
        np.testing.assert_array_equal(crn.uniform(7, "s", 3, ids[::-1])[::-1], u)
        np.testing.assert_array_equal(crn.uniform(7, "s", 3, ids[:10]), u[:10])
        self.assertFalse(np.allclose(crn.uniform(7, "t", 3, ids), u))
        self.assertFalse(np.allclose(crn.uniform(7, "s", 4, ids), u))
        self.assertFalse(np.allclose(crn.uniform(8, "s", 3, ids), u))

    def test_distribution(self) -> None:
        u = crn.uniform(1, "d", 0, np.arange(200000))
        self.assertTrue(u.min() > 0 and u.max() < 1)
        self.assertAlmostEqual(u.mean(), 0.5, delta=0.005)
        self.assertAlmostEqual(u.std(), (1 / 12) ** 0.5, delta=0.005)
        z = crn.normal(1, "d", 0, np.arange(200000))
        self.assertAlmostEqual(z.mean(), 0.0, delta=0.01)
        self.assertAlmostEqual(z.std(), 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
