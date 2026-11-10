"""Real-data build: runs offline against the committed raw cache (data/real/cache)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from belgium_abm.datasources.belgium import ipf
from belgium_abm.datasources.eurostat import to_frame
from belgium_abm.io.loaders import parse_annotated_csv
from tests.helpers import ROOT

REAL = ROOT / "data" / "real" / "belgium_2025"


class JsonStatTests(unittest.TestCase):
    def test_to_frame_flattens_with_missing_values(self) -> None:
        js = {"id": ["a", "b"], "size": [2, 2],
              "dimension": {"a": {"category": {"index": {"x": 0, "y": 1}}}, "b": {"category": {"index": ["p", "q"]}}},
              "value": {"0": 1.0, "3": 4.0}}
        df = to_frame(js)
        self.assertEqual(df[["a", "b", "value"]].values.tolist(), [["x", "p", 1.0], ["y", "q", 4.0]])


class IpfTests(unittest.TestCase):
    def test_matches_margins(self) -> None:
        m = ipf(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.4, 0.6]), np.array([0.7, 0.3]))
        np.testing.assert_allclose(m.sum(axis=1), [0.4, 0.6], atol=1e-10)
        np.testing.assert_allclose(m.sum(axis=0), [0.7, 0.3], atol=1e-10)

    def test_zero_margin_column_stays_zero(self) -> None:
        m = ipf(np.ones((2, 3)), np.array([0.5, 0.5]), np.array([0.5, 0.5, 0.0]))
        self.assertEqual(m[:, 2].sum(), 0.0)
        np.testing.assert_allclose(m.sum(), 1.0)


class RealTablesTests(unittest.TestCase):
    def table(self, name: str) -> tuple[dict[str, str], pd.DataFrame]:
        meta, df = parse_annotated_csv((REAL / name).read_text(), name)
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except ValueError:
                pass
        return meta, df

    def test_all_tables_flagged_real_2025(self) -> None:
        for f in REAL.glob("*.csv"):
            meta, _ = self.table(f.name)
            self.assertEqual(meta["synthetic_demo_only"], "false", f.name)
            self.assertEqual(meta["reference_year"], "2025", f.name)
            self.assertIn(meta["status"], {"observed", "derived"}, f.name)

    def test_population_totals_are_plausible(self) -> None:
        _, pop = self.table("population_age_sex_region.csv")
        total = pop["count"].sum()
        self.assertTrue(11.5e6 < total < 12.5e6, total)
        by_region = pop.groupby("region")["count"].sum()
        self.assertGreater(by_region["BE_FLANDERS"], by_region["BE_WALLONIA"])
        self.assertGreater(by_region["BE_WALLONIA"], by_region["BE_BRUSSELS"])

    def test_households_match_population_adults_and_children(self) -> None:
        _, pop = self.table("population_age_sex_region.csv")
        _, hh = self.table("household_templates_region.csv")
        for r, g in hh.groupby("region"):
            p = pop[pop.region == r]
            self.assertAlmostEqual((g["count"] * g.n_adults).sum() / p[p.age_min >= 18]["count"].sum(), 1.0, places=6)
            self.assertAlmostEqual((g["count"] * g.n_children).sum() / p[p.age_max <= 17]["count"].sum(), 1.0, places=6)

    def test_labour_shares_are_logical(self) -> None:
        _, lab = self.table("labour_state_by_age_education_region.csv")
        old = lab[lab.age_min == 65]
        self.assertEqual(old[old.work_state == "STUDENT"]["share"].sum(), 0.0)
        self.assertEqual(old[old.work_state == "INACTIVE"]["share"].sum(), 0.0)
        young = lab[(lab.age_min == 15) & (lab.work_state == "RETIRED")]
        self.assertEqual(young["share"].sum(), 0.0)

    def test_manifest_lists_every_source_with_sha(self) -> None:
        m = json.loads((REAL / "build_manifest.json").read_text())
        self.assertGreaterEqual(len(m["sources"]), 10)
        self.assertTrue(all(len(s["sha256"]) == 64 for s in m["sources"]))
        self.assertTrue(any("proxy" in a for a in m["assumptions"]))

    def test_offline_rebuild_is_identical(self) -> None:
        from belgium_abm.datasources.belgium import build_all

        with tempfile.TemporaryDirectory() as d:
            build_all(2025, ROOT / "data" / "real" / "cache", Path(d), offline=True)
            for f in REAL.glob("*.csv"):
                a, b = f.read_text().splitlines(), (Path(d) / f.name).read_text().splitlines()
                self.assertEqual([x for x in a if not x.startswith("# source")], [x for x in b if not x.startswith("# source")], f.name)


if __name__ == "__main__":
    unittest.main()
