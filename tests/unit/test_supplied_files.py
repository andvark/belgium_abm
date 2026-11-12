"""Parsers for user-supplied workbooks (HFCS, Statbel) on tiny synthetic workbooks, and the committed cross-checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from belgium_abm.datasources.hfcs import parse_hfcs
from belgium_abm.datasources.statbel_files import parse_lfs_region_education_age
from belgium_abm.io.loaders import parse_annotated_csv
from tests.helpers import ROOT

REAL = ROOT / "data" / "real" / "belgium_2025"


def _hfcs_sheet(rows: list[list]) -> pd.DataFrame:
    head = [["Table", None, None, None, None, None], [None] * 6, [None] * 6, [None, None, None, None, "euro area", "BE"]]
    return pd.DataFrame(head + rows + [["Source: test", None, None, None, None, None]])


class HfcsParserTests(unittest.TestCase):
    def test_parses_belgian_column_and_units(self) -> None:
        sheets = {
            "A1 Main aggregates - medians": _hfcs_sheet([["All", "DN3001 - Net wealth", None, None, 1.0, 254.2], [None, None, None, None, "(1)", "(2)"],
                                                         ["All", "DL1000i - Has debt", None, None, 40.0, 48.8], ["Home owner", "Share", None, None, 60.0, 66.7],
                                                         ["Home owner", "DN3001 - Net wealth", None, None, 1.0, 375.9], ["Non owner", "DN3001 - Net wealth", None, None, 1.0, 17.7]]),
            "J3 Net wealth per household - d": _hfcs_sheet([["Mean", None, None, None, 1.0, 411.5], ["p50", None, None, None, 1.0, 254.2]]),
            "J4 Net wealth inequality indica": _hfcs_sheet([["Share indicators (%)", "Top 10% share", None, None, 1.0, 42.4], ["Share indicators (%)", "Top 5% share", None, None, 1.0, 29.4],
                                                           ["Inequality measures", "Gini coefficient", None, None, 0.5, 0.604]]),
            "A3 Net wealth medians - breakdo": _hfcs_sheet([["Age of RP", "16-34", None, None, 1.0, 97.2]]),
            "B1 Percentage of households own": _hfcs_sheet([["DA1140i - Has self-employment business wealth", None, None, None, 1.0, 12.2]]),
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "h.xlsx"
            with pd.ExcelWriter(p) as w:
                for k, v in sheets.items():
                    v.to_excel(w, sheet_name=k, header=False, index=False)
            df, meta = parse_hfcs(p)
        get = lambda i, g="TOTAL": float(df[(df.indicator == i) & (df.group == g)].value.iloc[0])  # noqa: E731
        self.assertEqual(get("net_wealth_median"), 254200.0)
        self.assertEqual(get("net_wealth_mean"), 411500.0)
        self.assertAlmostEqual(get("net_wealth_share_top10"), 0.424)
        self.assertAlmostEqual(get("net_wealth_gini"), 0.604)
        self.assertAlmostEqual(get("share_with_debt"), 0.488)
        self.assertEqual(get("net_wealth_median_by_age_of_reference_person", "16-34"), 97200.0)
        self.assertEqual(len(meta["source_sha256"]), 64)

    def test_committed_targets_match_published_belgian_values(self) -> None:
        _, df = parse_annotated_csv((REAL / "hfcs_wealth_targets.csv").read_text(), "hfcs")
        df["value"] = df["value"].astype(float)
        g = lambda i, grp="TOTAL": float(df[(df.indicator == i) & (df.group == grp)].value.iloc[0])  # noqa: E731
        self.assertEqual(g("net_wealth_median"), 254200.0)
        self.assertAlmostEqual(g("net_wealth_gini"), 0.604)
        self.assertAlmostEqual(g("homeowner_share"), 0.667)


class StatbelFilesTests(unittest.TestCase):
    def test_lfs_parser_keeps_regions_and_maps_education(self) -> None:
        d = pd.DataFrame([["Belgium", "Brussels-Capital region", "Total", "Low", "Total", "From 25 to 49 years", 2025, 0.47, 0.23, 0.6],
                          ["Belgium", "Region", "Total", "Low", "Total", "From 25 to 49 years", 2025, 0.5, 0.1, 0.6],
                          ["Belgium", "Flanders region", "Total", "Middle", "Total", "From 15 to 24 years", 2025, 0.4, np.nan, 0.5]],
                         columns=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "s.xlsx"
            d.to_excel(p, sheet_name="Data as Table", index=False, header=False)
            # the real file has a header row that is dropped by the parser's column rename; emulate it
            hdr = pd.concat([pd.DataFrame([list(d.columns)], columns=d.columns), d], ignore_index=True)
            hdr.to_excel(p, sheet_name="Data as Table", index=False, header=False)
            out = parse_lfs_region_education_age(p)
        self.assertEqual(set(out.region), {"BE_BRUSSELS", "BE_FLANDERS"})
        self.assertEqual(set(out.education), {"LOW", "MEDIUM"})

    def test_committed_crosschecks_are_within_tolerance(self) -> None:
        lab = pd.read_csv(REAL / "crosscheck_labour_vs_statbel.csv")
        self.assertGreater(len(lab), 20)
        self.assertLess(lab.employment_diff.abs().mean(), 0.03)
        self.assertLess(lab.unemployment_diff.abs().mean(), 0.03)
        edu = pd.read_csv(REAL / "crosscheck_education_vs_statbel.csv")
        self.assertLess(edu["diff"].abs().max(), 0.01)

    def test_regional_self_employment_factors_match_supplied_eurostat_extract(self) -> None:
        # values from the user-supplied lfst_r_lfe2estat extract (thousand persons, 2025, ages 15-64)
        nat = 695.1 / 5040.9
        factors = {"BE_BRUSSELS": (69.9 / 505.4) / nat, "BE_FLANDERS": (453.8 / 3075.4) / nat, "BE_WALLONIA": (171.4 / 1460.2) / nat}
        import json

        m = json.loads((REAL / "build_manifest.json").read_text())["diagnostics"]["regional_self_employment_factor"]
        for r, f in factors.items():
            self.assertAlmostEqual(m[r], f, delta=0.01)


if __name__ == "__main__":
    unittest.main()
