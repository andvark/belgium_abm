"""Dynamic model: invariants, accounting identities, determinism, common random numbers, scenarios, CLI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from belgium_abm.analysis.inequality import weighted_gini
from belgium_abm.engines.flows import REVENUE, SPENDING, compute_flows, net_fiscal_balance
from belgium_abm.engines.simulation import check_state, run_simulation
from belgium_abm.engines.state import net_worth
from belgium_abm.scenarios import Scenario
from tests.helpers import ROOT, real_sim_inputs
from tests.integration.test_cli import run_cli

YEARS = 4


class FlowsAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg, cls.inp, cls.targets, cls.st0, cls.info = real_sim_inputs()
        cls.fl = compute_flows(cls.st0.copy(), cls.inp.policy)

    def test_level_calibration_hits_published_median(self) -> None:
        self.assertAlmostEqual(self.info["achieved_median"] / self.info["target_median"], 1.0, delta=0.02)

    def test_initial_state_passes_structural_invariants(self) -> None:
        self.assertEqual(check_state(self.st0), [])

    def test_government_totals_equal_person_level_sums(self) -> None:
        w = self.st0.person_weight
        self.assertAlmostEqual(self.fl.gov["personal_income_tax"], self.fl.p["pit"].sum() * w, delta=1.0)
        self.assertAlmostEqual(self.fl.gov["employer_ssc"], self.fl.p["employer_ssc"].sum() * w, delta=1.0)
        self.assertAlmostEqual(self.fl.gov["employee_ssc"], self.fl.p["emp_ssc"].sum() * w, delta=1.0)

    def test_employer_contributions_are_not_household_income(self) -> None:
        p = self.fl.p
        np.testing.assert_allclose(p["market_a"], p["wage"] + p["profit_sp"] + p["dir_pay"] + p["dividends"])
        self.assertGreater(p["employer_ssc"].sum(), 0)

    def test_firm_flows_match_owner_income(self) -> None:
        """Firms' director pay and dividends paid = owners' received; BV retained profit is not household income (spec 11.4)."""
        F, P, st = self.fl.firms, self.fl.p, self.st0
        owners = st.firms["owner_person_id"].to_numpy()
        self.assertAlmostEqual(F["director_pay"].sum(), P.loc[owners, "dir_pay"].sum(), places=6)
        self.assertAlmostEqual(F["dividends"].sum(), P.loc[owners, "dividends"].sum(), places=6)
        self.assertAlmostEqual(self.fl.gov["corporate_tax"], F["corporate_tax"].sum() * st.person_weight, delta=1.0)
        self.assertTrue((F["retained_delta"] != 0).any())
        # retained profit appears nowhere in the owners' market income
        self.assertAlmostEqual(P.loc[owners, "market_a"].sum(), (P.loc[owners, "profit_sp"] + P.loc[owners, "wage"] + P.loc[owners, "dir_pay"] + P.loc[owners, "dividends"]).sum(), places=6)

    def test_bv_equity_counted_once_in_net_worth(self) -> None:
        st = self.st0.copy()
        st.firms["equity"] = 1_000_000.0
        before = net_worth(st)
        st.firms["equity"] = 0.0
        after = net_worth(st)
        owner_hh = st.persons.loc[st.firms["owner_person_id"], "household_id"].to_numpy()
        gain = (before - after).groupby(level=0).sum()
        counts = pd.Series(owner_hh).value_counts()
        for hid, n in counts.items():
            self.assertAlmostEqual(gain.loc[hid], 1_000_000.0 * n)
        self.assertEqual(float((before - after).drop(index=counts.index).abs().sum()), 0.0)

    def test_household_disposable_identity(self) -> None:
        H, P, st = self.fl.h, self.fl.p, self.st0
        pers = P["cash"].groupby(st.persons["household_id"]).sum().reindex(H.index, fill_value=0.0)
        np.testing.assert_allclose(H["disp"], pers + H["child_ben"] + H["cap_gross"] - H["cap_tax"] + H["assist"], atol=1e-6)

    def test_no_negative_consumption_or_vat(self) -> None:
        self.assertTrue((self.fl.h["consumption"] >= 0).all())
        self.assertTrue((self.fl.h["vat"] >= 0).all())


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg, cls.inp, cls.targets, cls.st0, cls.info = real_sim_inputs()
        cls.res = run_simulation(cls.st0, cls.inp.policy, YEARS, check_invariants=True)

    def test_invariants_hold_every_year(self) -> None:
        self.assertEqual(self.res.invariant_failures, [])

    def test_household_cash_identity_and_no_negative_stocks(self) -> None:
        self.assertLess(self.res.max_cash_identity_residual, 1e-6)
        st = self.res.final_state
        self.assertTrue((st.households["financial_assets"] >= -1e-9).all())
        self.assertTrue((st.firms["equity"] >= -1e-9).all())

    def test_ledger_reconciles_and_ties_to_flows(self) -> None:
        led = self.res.ledger
        self.assertTrue(led.reconcile())
        by_cat = led.total_cents_by_category()
        fiscal = self.res.fiscal[self.res.fiscal.scenario == "dynamic"]
        for cat in (*REVENUE, *SPENDING):
            self.assertAlmostEqual(by_cat.get(cat, 0) / 100.0, fiscal[(fiscal.year >= 1)][cat].sum(), delta=0.01 * YEARS + 1.0, msg=cat)

    def test_timing_convention_ages_increment_first(self) -> None:
        a0 = self.st0.persons["age"]
        a1 = self.res.final_state.persons["age"]
        survivors = a1.index.intersection(a0.index)
        self.assertTrue(((a1.loc[survivors] - a0.loc[survivors]) == YEARS).all())
        newborn = a1.index.difference(a0.index)
        self.assertTrue((a1.loc[newborn] < YEARS).all() and len(newborn) > 0)

    def test_state_evolves_plausibly(self) -> None:
        m = self.res.metrics[self.res.metrics.scenario == "dynamic"].set_index("year")
        self.assertEqual(list(m.index), list(range(YEARS + 1)))
        self.assertTrue(((m["gini_disposable"] > 0.1) & (m["gini_disposable"] < 0.6)).all())
        self.assertTrue((m["population"].diff().dropna().abs() / m["population"].iloc[0] < 0.02).all())
        ev = self.res.events.groupby("event")["value"].sum()
        for k in ("deaths", "births", "retirements", "job_losses", "job_findings"):
            self.assertGreater(ev.get(k, 0), 0, k)

    def test_group_cuts_flag_small_samples(self) -> None:
        g = self.res.groups
        self.assertTrue({"age_group", "region", "work_arrangement", "baseline_wealth_quintile"} <= set(g.group_var))
        self.assertTrue((g.loc[g.n_sim < 30, "small_sample_warning"]).all())

    def test_mobility_rows_sum_to_one(self) -> None:
        np.testing.assert_allclose(self.res.mobility.sum(axis=1).to_numpy(), 1.0, atol=1e-9)

    def test_deterministic_for_identical_input(self) -> None:
        again = run_simulation(self.st0, self.inp.policy, YEARS)
        pd.testing.assert_frame_equal(self.res.metrics, again.metrics)
        pd.testing.assert_frame_equal(self.res.final_state.persons, again.final_state.persons)

    def test_baseline_equals_identical_policy_override(self) -> None:
        same = self.inp.policy.with_overrides({"corporate_income_tax.rate": self.inp.policy.g("corporate_income_tax", "rate")}, "same")
        other = run_simulation(self.st0, same, YEARS)
        pd.testing.assert_frame_equal(self.res.metrics.drop(columns="scenario"), other.metrics.drop(columns="scenario"))

    def test_static_shadow_isolates_mechanical_effect(self) -> None:
        scen = Scenario.load(ROOT / "scenarios" / "corporate_tax_plus_5pp.yaml")
        pol = scen.apply_policy(self.inp.policy)
        shadowed = run_simulation(self.st0, self.inp.policy, 2, shadow={scen.name: pol})
        m = shadowed.metrics
        dyn0 = m[(m.scenario == "dynamic") & (m.year == 0)].iloc[0]
        stat0 = m[(m.scenario == f"static:{scen.name}") & (m.year == 0)].iloc[0]
        direct = compute_flows(self.st0.copy(), pol)
        self.assertAlmostEqual(stat0["gov_corporate_tax"], direct.gov["corporate_tax"], delta=1e-3)
        self.assertGreater(stat0["gov_corporate_tax"], dyn0["gov_corporate_tax"])
        # behaviour is untouched by the shadow: the dynamic trajectory is identical with and without it
        plain = run_simulation(self.st0, self.inp.policy, 2)
        pd.testing.assert_frame_equal(plain.metrics, m[m.scenario == "dynamic"].reset_index(drop=True))

    def test_common_random_numbers_pair_scenario_and_baseline(self) -> None:
        scen = Scenario.load(ROOT / "scenarios" / "corporate_tax_plus_5pp.yaml")
        s = run_simulation(self.st0, scen.apply_policy(self.inp.policy), YEARS, label="s")
        b = self.res.final_state.persons
        t = s.final_state.persons
        # demography does not respond to this tax: same people alive, same ages
        self.assertEqual(set(b.index), set(t.index))
        self.assertTrue((b["age"] == t.loc[b.index, "age"]).all())

    def test_higher_corporate_tax_raises_corporate_revenue(self) -> None:
        scen = Scenario.load(ROOT / "scenarios" / "corporate_tax_plus_5pp.yaml")
        s = run_simulation(self.st0, scen.apply_policy(self.inp.policy), YEARS, label="s")
        self.assertGreater(s.fiscal.corporate_tax.iloc[-1], self.res.fiscal.corporate_tax.iloc[-1])

    def test_different_seed_changes_outcome(self) -> None:
        cfg, inp, targets, st0, info = real_sim_inputs(5000, 43)
        other = run_simulation(st0, inp.policy, 2)
        self.assertNotEqual(other.metrics["gini_disposable"].iloc[1], self.res.metrics["gini_disposable"].iloc[1])


class CorruptionTests(unittest.TestCase):
    def test_check_state_detects_corruption(self) -> None:
        _, _, _, st0, _ = real_sim_inputs()
        st = st0.copy()
        st.persons.loc[st.persons.index[0], "household_id"] = 10**9
        self.assertTrue(any("non-existent" in m for m in check_state(st)))
        st = st0.copy()
        st.households.loc[st.households.index[0], "financial_assets"] = -5.0
        self.assertTrue(any("negative" in m for m in check_state(st)))
        st = st0.copy()
        emp = st.persons.index[(st.persons["work_state"] == "EMPLOYED")][0]
        st.persons.loc[emp, "arrangement"] = "NONE"
        self.assertTrue(any("without an arrangement" in m for m in check_state(st)))


class GiniToyTests(unittest.TestCase):
    def test_equal_incomes_give_zero(self) -> None:
        self.assertAlmostEqual(weighted_gini(np.full(10, 5.0)), 0.0)


class CliSimulationTests(unittest.TestCase):
    def test_simulate_compare_arrangements(self) -> None:
        cfg = str(ROOT / "configs" / "belgium_baseline.yaml")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sim"
            code, o, e = run_cli("simulate", "--config", cfg, "--years", "3", "--n-persons", "3000", "--replicates", "2", "--seed", "11",
                                 "--scenario", str(ROOT / "scenarios" / "corporate_tax_plus_5pp.yaml"), "--output-dir", str(out), "--check-invariants")
            self.assertEqual(code, 0, e)
            base, scen = out / "baseline", out / "scenario_corporate_tax_plus_5pp"
            for name in ("metrics_by_year.csv", "group_metrics.csv", "fiscal_by_year.csv", "events.csv", "income_mobility.csv", "uncertainty_summary.csv",
                         "ledger_summary.csv", "inequality_over_time.png", "run_manifest.json", "report.md", "validation_vs_published.csv"):
                self.assertTrue((base / name).is_file(), name)
            man = json.loads((base / "run_manifest.json").read_text())
            self.assertEqual(man["seeds"], [11, 12])
            self.assertFalse(man["policy"]["verified"])
            self.assertEqual(man["invariant_failures"], [])
            self.assertIn("NOT BELGIAN LAW", (base / "report.md").read_text())
            self.assertTrue((scen / "winners_losers_by_baseline_wealth_quintile.csv").is_file())
            code, o, e = run_cli("compare", "--baseline", str(base), "--scenario", str(scen))
            self.assertEqual(code, 0, e)
            comp = pd.read_csv(scen / "comparison_vs_baseline.csv")
            self.assertEqual(set(comp.effect), {"dynamic_total", "static_fixed_behaviour", "behavioural_residual_NOT_CAUSAL"})
            row = comp[(comp.metric == "gov_net_fiscal_balance") & (comp.effect == "static_fixed_behaviour") & (comp.year == 3)].iloc[0]
            self.assertGreater(row["mean_diff"], 0)  # higher corporate tax -> higher balance under fixed behaviour
            self.assertEqual(row["n_replicates"], 2)
            self.assertIn("NOT_CAUSAL", (scen / "comparison_report.md").read_text())
            # mismatched seeds are refused
            other = Path(d) / "sim2"
            run_cli("simulate", "--config", cfg, "--years", "1", "--n-persons", "2000", "--seed", "99", "--scenario",
                    str(ROOT / "scenarios" / "corporate_tax_plus_5pp.yaml"), "--output-dir", str(other))
            code, _, err = run_cli("compare", "--baseline", str(base), "--scenario", str(other / "scenario_corporate_tax_plus_5pp"))
            self.assertEqual(code, 2)
            self.assertIn("common random numbers", err)

            arr = Path(d) / "arr"
            code, o, e = run_cli("arrangements", "--config", cfg, "--years", "6", "--replicates", "40", "--output-dir", str(arr), "--children", "1", "--partner", "inactive")
            self.assertEqual(code, 0, e)
            s = pd.read_csv(arr / "arrangements_summary.csv")
            self.assertEqual(list(s.arrangement), ["EMPLOYEE", "SOLE_PROPRIETOR", "BV_OWNER_MANAGER"])
            self.assertTrue((arr / "arrangements_lifecycle.png").is_file())
            ts = pd.read_csv(arr / "arrangements_by_year.csv")
            self.assertTrue((ts.groupby("arrangement")["year"].nunique() == 6).all())

    def test_simulate_requires_simulation_section(self) -> None:
        code, _, err = run_cli("simulate", "--config", str(ROOT / "configs" / "demo_synthetic.yaml"), "--years", "1")
        self.assertEqual(code, 2)
        self.assertIn("simulation", err)


if __name__ == "__main__":
    unittest.main()
