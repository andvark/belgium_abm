"""Public-data behavioural parameters: application to the economy, derivations, validation function."""

from __future__ import annotations

import io
import unittest

import numpy as np
import pandas as pd

from belgium_abm.analysis.behaviour_validation import behaviour_validation
from belgium_abm.datasources.behaviour import lognormal_sd_from_low_wage_share
from belgium_abm.engines.economy import EconomyConfig
from belgium_abm.engines.state import calibrate_skill_sigma, labour_productivity
from tests.helpers import ROOT, real_sim_inputs

REAL = ROOT / "data" / "real" / "belgium_2025"


def _table(name: str) -> pd.DataFrame:
    lines = [ln for ln in (REAL / name).read_text().splitlines() if not ln.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(lines)))


class BehaviourTableTests(unittest.TestCase):
    def test_table_has_expected_public_values(self) -> None:
        t = _table("behaviour_observed.csv").set_index("name")
        self.assertTrue(0.01 < t.loc["job_separation_rate", "value"] < 0.06)
        self.assertGreater(t.loc["sole_prop_exit_rate", "value"], t.loc["bv_exit_rate", "value"])
        self.assertGreater(t.loc["parttime_share_f", "value"], t.loc["parttime_share_m", "value"])
        self.assertTrue(1.0 < t.loc["education_wage_ratio_medium", "value"] < t.loc["education_wage_ratio_high", "value"])
        self.assertTrue((t["status"].isin(["observed", "derived"])).all())
        self.assertTrue(t["source"].str.len().gt(10).all())

    def test_apply_observed_replaces_assumptions_and_sets_wage_ratios(self) -> None:
        econ = EconomyConfig.load(ROOT / "configs" / "economy_stylised.yaml")
        t = _table("behaviour_observed.csv")
        e2 = econ.apply_observed(t)
        self.assertEqual(e2.parameters["sole_prop_exit_rate"].status.value, "observed")
        self.assertAlmostEqual(e2.v("sole_prop_exit_rate"), float(t.set_index("name").loc["sole_prop_exit_rate", "value"]))
        self.assertAlmostEqual(e2.v("wage_base_medium") / e2.v("wage_base_low"), float(t.set_index("name").loc["education_wage_ratio_medium", "value"]))
        self.assertEqual(econ.parameters["sole_prop_exit_rate"].status.value, "assumed")  # original untouched
        self.assertIn("education_wage_ratio_high", e2.parameters)


class DerivationTests(unittest.TestCase):
    def test_lognormal_equivalent_sd_roundtrip(self) -> None:
        from scipy.stats import norm

        for share in (0.05, 0.1586, 0.3):
            sd = lognormal_sd_from_low_wage_share(share)
            self.assertAlmostEqual(float(norm.cdf(np.log(2 / 3) / sd)), share, places=9)

    def test_skill_sigma_calibration_hits_target_dispersion(self) -> None:
        _, inp, _, st0, _ = real_sim_inputs()
        p, e = st0.persons, st0.econ
        emp = ((p.work_state == "EMPLOYED") & (p.arrangement == "EMPLOYEE")).to_numpy()
        w = np.log(labour_productivity(p, e, 0, st0.level_scale))[emp]
        target = lognormal_sd_from_low_wage_share(float(e.v("low_wage_share_target")))
        self.assertAlmostEqual(float(np.std(w)), target, delta=0.01)
        self.assertEqual(e.parameters["skill_sigma"].status.value, "derived")
        # idempotent
        again = calibrate_skill_sigma(p, e, st0.level_scale)
        self.assertAlmostEqual(again.v("skill_sigma"), e.v("skill_sigma"), places=6)

    def test_labour_force_exit_is_derived_from_observed_flow_minus_retirement(self) -> None:
        _, inp, _, st0, info = real_sim_inputs()
        obs = float(inp.econ.v("labour_force_exit_total_observed"))
        self.assertAlmostEqual(info["labour_force_exit_rate_derived"] + info["retirement_flow_share"], obs, places=6)
        self.assertEqual(st0.econ.parameters["labour_force_exit_rate"].status.value, "derived")


class ValidationFunctionTests(unittest.TestCase):
    def test_rates_and_status(self) -> None:
        m = pd.DataFrame({"scenario": "baseline", "year": [0, 1], "n_employed_le74": [1000, 1000], "n_unemployed": [60, 60], "n_inactive_like_le74": [500, 500],
                          "n_sole_prop": [100, 100], "n_bv_owner_managers": [50, 50], "household_saving_rate": [0.1, 0.12]})
        ev = pd.DataFrame({"scenario": "baseline", "year": 1, "event": ["job_losses", "job_findings", "exits_SOLE_PROPRIETOR"], "value": [27.0, 30.0, 10.0]})
        beh = pd.DataFrame({"name": ["job_separation_rate", "observed_flow_unemployed_to_employed", "sole_prop_exit_rate", "observed_household_saving_rate"], "value": [0.027, 0.5, 0.10, 0.14]})
        v = behaviour_validation(m, ev, beh, {}).set_index("indicator")
        self.assertAlmostEqual(v.loc["employment_to_unemployment", "simulated"], 0.027)
        self.assertEqual(v.loc["employment_to_unemployment", "status"], "close (<=15%)")
        self.assertAlmostEqual(v.loc["unemployment_to_employment", "simulated"], 0.5)
        self.assertAlmostEqual(v.loc["sole_prop_exit_rate", "simulated"], 0.10)


if __name__ == "__main__":
    unittest.main()
