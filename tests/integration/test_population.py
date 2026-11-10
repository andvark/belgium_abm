import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from belgium_abm.agents.household import HOUSEHOLD_COLUMNS
from belgium_abm.agents.person import PERSON_COLUMNS
from belgium_abm.analysis.validation import check_population, summarize_checks
from belgium_abm.engines.population import generate_population
from belgium_abm.io.loaders import load_targets
from belgium_abm.io.outputs import read_table, write_table
from belgium_abm.rng import RngFactory
from belgium_abm.run import evaluate
from tests.helpers import demo_population, modified_demo_config


def failing(cfg, persons, households) -> set[str]:
    return {c.name for c in check_population(persons, households, cfg) if not c.passed}


class PopulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg, self.targets, self.pop = demo_population(2000, 42)

    def test_sizes_and_schema(self) -> None:
        p, h = self.pop.persons, self.pop.households
        self.assertAlmostEqual(len(p), 2000, delta=40)
        self.assertEqual(list(p.columns), [c.name for c in PERSON_COLUMNS])
        self.assertEqual(list(h.columns), [c.name for c in HOUSEHOLD_COLUMNS])
        self.assertEqual(int(p["weight"].iloc[0] * len(p)), int(self.targets.total_persons))

    def test_all_structural_invariants_pass(self) -> None:
        checks = check_population(self.pop.persons, self.pop.households, self.cfg, self.targets.max_age)
        self.assertEqual(summarize_checks(checks)["failed"], 0, [c for c in checks if not c.passed])
        self.assertGreaterEqual(len(checks), 20)

    def test_verdict_is_not_fail_and_has_no_failing_rows(self) -> None:
        verdict, report, comparisons, _ = evaluate(self.cfg, self.targets, self.pop, 42)
        self.assertNotEqual(verdict, "FAIL")
        self.assertEqual(int((comparisons["status"] == "fail").sum()), 0)
        self.assertIn("SYNTHETIC", report["banner"])
        self.assertEqual(set(comparisons["level"]), {"cell", "marginal", "total"})

    def test_same_seed_is_bit_identical_and_other_seed_differs(self) -> None:
        again = generate_population(self.cfg, self.targets, 42)
        pd.testing.assert_frame_equal(self.pop.persons, again.persons)
        pd.testing.assert_frame_equal(self.pop.households, again.households)
        other = generate_population(self.cfg, self.targets, 43)
        self.assertFalse(self.pop.persons["age"].equals(other.persons["age"]))

    def test_csv_roundtrip_preserves_the_tables(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            write_table(self.pop.persons, Path(d), "persons", parquet=False)
            write_table(self.pop.households, Path(d), "households", parquet=False)
            back_p = read_table(Path(d) / "persons.csv", PERSON_COLUMNS)
            back_h = read_table(Path(d) / "households.csv", HOUSEHOLD_COLUMNS)
        pd.testing.assert_frame_equal(back_p, self.pop.persons, check_dtype=False)
        pd.testing.assert_frame_equal(back_h, self.pop.households, check_dtype=False)

    def test_changing_one_module_stream_does_not_perturb_others(self) -> None:
        a = RngFactory(42).stream("education").random(3)
        RngFactory(42).stream("tenure").random(10)
        np.testing.assert_array_equal(a, RngFactory(42).stream("education").random(3))
        self.assertIn("integerise_objective_after_swaps", self.pop.diagnostics)

    def test_swap_refinement_does_not_make_things_worse(self) -> None:
        d = self.pop.diagnostics
        self.assertLessEqual(d["integerise_objective_after_swaps"], d["integerise_objective_after_rounding"])

    def test_no_income_or_wealth_values_exist(self) -> None:
        for col in ("gross_employee_wage", "dividends_received", "pension_income"):
            self.assertTrue(self.pop.persons[col].isna().all())
        for col in ("housing_value", "financial_assets"):
            self.assertTrue(self.pop.households[col].isna().all())

    def test_children_below_working_age_have_no_labour_state(self) -> None:
        kids = self.pop.persons[self.pop.persons["age"] < self.cfg.working_age_min]
        self.assertTrue(kids["work_state"].isna().all())
        self.assertTrue((kids["education_level"] == "NOT_ASSESSED").all())

    def test_tenure_can_be_omitted(self) -> None:
        import dataclasses

        t = dataclasses.replace(self.targets, tenure=None)
        pop = generate_population(self.cfg, t, 1)
        self.assertTrue(pop.households["housing_tenure"].isna().all())


class CorruptionTests(unittest.TestCase):
    """The validator must actually catch broken populations."""

    def setUp(self) -> None:
        self.cfg, _, pop = demo_population(2000, 42)
        self.p, self.h = pop.persons.copy(), pop.households.copy()

    def test_clean_population_has_no_failures(self) -> None:
        self.assertEqual(failing(self.cfg, self.p, self.h), set())

    def test_ghost_member_detected(self) -> None:
        self.h.loc[0, "member_ids"] = self.h.loc[0, "member_ids"] + ";999999"
        self.assertTrue(any("member_ids" in n for n in failing(self.cfg, self.p, self.h)))

    def test_orphan_person_detected(self) -> None:
        self.p.loc[0, "household_id"] = 10**9
        self.assertTrue(any("existing household" in n for n in failing(self.cfg, self.p, self.h)))

    def test_duplicate_ids_detected(self) -> None:
        self.p.loc[1, "person_id"] = self.p.loc[0, "person_id"]
        self.assertTrue(any("unique" in n for n in failing(self.cfg, self.p, self.h)))

    def test_invalid_parent_age_detected(self) -> None:
        i = self.p.index[self.p["parent1_id"].notna()][0]
        self.p.loc[i, "age"] = 60
        self.assertTrue(failing(self.cfg, self.p, self.h))

    def test_dangling_parent_reference_detected_without_crashing(self) -> None:
        i = self.p.index[self.p["parent1_id"].notna()][0]
        self.p.loc[i, "parent1_id"] = 10**9
        self.assertTrue(any("refer to existing" in n for n in failing(self.cfg, self.p, self.h)))

    def test_second_head_detected(self) -> None:
        i = self.p.index[self.p["role"] == "PARTNER"][0]
        self.p.loc[i, "role"] = "HEAD"
        self.assertTrue(any("HEAD" in n for n in failing(self.cfg, self.p, self.h)))

    def test_inconsistent_arrangement_detected(self) -> None:
        i = self.p.index[self.p["work_state"] == "RETIRED"][0]
        self.p.loc[i, "primary_work_arrangement"] = "EMPLOYEE"
        self.assertTrue(any("EMPLOYED" in n for n in failing(self.cfg, self.p, self.h)))

    def test_invented_income_detected(self) -> None:
        self.p.loc[0, "gross_employee_wage"] = 1234.0
        self.assertTrue(any("invented" in n for n in failing(self.cfg, self.p, self.h)))

    def test_household_composition_mismatch_detected(self) -> None:
        self.h.loc[0, "n_children"] = self.h.loc[0, "n_children"] + 1
        self.assertTrue(any("n_adults" in n for n in failing(self.cfg, self.p, self.h)))

    def test_nonuniform_weight_detected(self) -> None:
        self.p.loc[0, "weight"] = 1.0
        self.assertTrue(any("weights" in n for n in failing(self.cfg, self.p, self.h)))

    def test_dead_person_detected(self) -> None:
        self.p.loc[0, "alive"] = False
        self.assertTrue(any("alive" in n for n in failing(self.cfg, self.p, self.h)))

    def test_json_report_is_serialisable(self) -> None:
        cfg, targets, pop = demo_population(2000, 42)
        _, report, _, _ = evaluate(cfg, targets, pop, 42)
        json.dumps(report, default=str)


class OtherConfigTests(unittest.TestCase):
    def test_pool_too_small_still_runs_or_reports_unsupported(self) -> None:
        cfg = modified_demo_config(1500, generator={"n_persons": 1500, "pool_multiplier": 1.0, "min_pool_per_template": 20})
        pop = generate_population(cfg, load_targets(cfg), 3)
        self.assertGreater(len(pop.persons), 1000)

    def test_disabling_refinement_gives_worse_or_equal_fit(self) -> None:
        cfg = modified_demo_config(1500, generator={"n_persons": 1500, "refine_swaps_per_household": 0, "refine_counts_proposals_per_cell": 0})
        pop = generate_population(cfg, load_targets(cfg), 3)
        d = pop.diagnostics
        self.assertEqual(d["swaps_accepted"], 0)
        self.assertEqual(d["integerise_objective_after_swaps"], d["integerise_objective_after_rounding"])


if __name__ == "__main__":
    unittest.main()
