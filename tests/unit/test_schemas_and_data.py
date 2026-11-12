import unittest

import pandas as pd

from belgium_abm.agents.household import HOUSEHOLD_COLUMNS
from belgium_abm.agents.person import PERSON_COLUMNS
from belgium_abm.demo_data import DEMO_FILES, demo_tables
from belgium_abm.io.registry import load_registry
from belgium_abm.io.schemas import TABLE_SCHEMAS
from belgium_abm.io.templates import template_text
from tests.helpers import ROOT


class SchemaTests(unittest.TestCase):
    def test_person_schema_has_every_spec_field(self) -> None:
        spec_fields = """person_id household_id age sex_or_demographic_stratum region education_level student_status work_state
        primary_work_arrangement working_hours_fraction occupation_or_sector skill_score health_work_capacity gross_employee_wage
        self_employed_turnover self_employed_costs director_pay dividends_received other_capital_income unemployment_eligibility
        disability_eligibility accrued_statutory_pension_rights pension_income alive weight""".split()
        names = {c.name for c in PERSON_COLUMNS}
        self.assertTrue(set(spec_fields) <= names, set(spec_fields) - names)

    def test_household_schema_has_every_spec_field(self) -> None:
        spec_fields = "household_id member_ids region housing_tenure housing_value mortgage_balance financial_assets other_debt shared_consumption_budget num_dependents".split()
        self.assertTrue(set(spec_fields) <= {c.name for c in HOUSEHOLD_COLUMNS})

    def test_demo_files_on_disk_match_the_generator(self) -> None:
        for table, df in demo_tables().items():
            path = ROOT / "data" / "demo" / DEMO_FILES[table]
            self.assertTrue(path.is_file(), path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("synthetic_demo_only: true", text)
            self.assertIn("FICTIONAL", text)
            on_disk = pd.read_csv(path, comment="#")
            self.assertEqual(len(on_disk), len(df))

    def test_templates_are_header_only_and_not_marked_synthetic(self) -> None:
        for table in TABLE_SCHEMAS:
            text = template_text(table)
            self.assertIn("status: template", text)
            self.assertIn("synthetic_demo_only: false", text)
            self.assertEqual(len([ln for ln in text.splitlines() if not ln.startswith("#")]), 1)
            on_disk = ROOT / "data" / "templates" / f"{TABLE_SCHEMAS[table].default_dataset_id}.csv"
            self.assertEqual(on_disk.read_text(encoding="utf-8"), text)

    def test_registry_acquired_entries_point_to_existing_files(self) -> None:
        reg = load_registry(ROOT / "data" / "sources.yaml")
        self.assertEqual(len(reg), 15)
        acquired = [s for s in reg if s.status == "acquired"]
        self.assertEqual(len(acquired), 7)
        for s in acquired:
            self.assertTrue((ROOT / str(s.file_path)).is_file(), s.id)
            self.assertIn(s.reference_year, (2023, 2025))  # HFCS wave 2023 is the only non-2025 vintage
        # restricted microdata must never be marked acquired
        self.assertTrue(all(s.status == "not_acquired" for s in reg if s.restricted_microdata))

    def test_demo_population_and_households_are_consistent(self) -> None:
        t = demo_tables()
        persons_from_hh = (t["households"]["count"] * (t["households"]["n_adults"] + t["households"]["n_children"])).sum()
        self.assertAlmostEqual(persons_from_hh / t["population"]["count"].sum(), 1.0, delta=0.002)


if __name__ == "__main__":
    unittest.main()
