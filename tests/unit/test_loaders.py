import tempfile
import unittest
from pathlib import Path

from belgium_abm.demo_data import DEMO_FILES, write_demo_data
from belgium_abm.io.loaders import InputDataError, MissingInputsError, load_targets, parse_annotated_csv
from belgium_abm.io.templates import write_templates
from tests.helpers import ROOT, edit_first_data_row, modified_demo_config


def _cfg_for(directory: Path, **overrides):
    inputs = {t: str(directory / f) for t, f in DEMO_FILES.items()}
    return modified_demo_config(inputs=inputs, **overrides)


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        write_demo_data(self.dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_demo_tables_load(self) -> None:
        t = load_targets(_cfg_for(self.dir))
        self.assertTrue(t.any_synthetic)
        self.assertEqual(t.max_age, 99)
        self.assertGreater(t.total_persons, 500_000)
        self.assertEqual(set(t.meta), set(DEMO_FILES))

    def test_missing_files_raise_and_are_listed(self) -> None:
        (self.dir / DEMO_FILES["education"]).unlink()
        with self.assertRaises(MissingInputsError) as ctx:
            load_targets(_cfg_for(self.dir))
        self.assertIn("education", str(ctx.exception))

    def test_synthetic_data_refused_unless_allowed(self) -> None:
        with self.assertRaises(InputDataError):
            load_targets(_cfg_for(self.dir, allow_synthetic_demo_data=False))

    def test_templates_are_refused(self) -> None:
        tdir = self.dir / "tpl"
        paths = {p.stem: p for p in write_templates(tdir)}
        inputs = {t: str(paths[Path(f).stem]) for t, f in DEMO_FILES.items()}
        with self.assertRaises(InputDataError) as ctx:
            load_targets(modified_demo_config(inputs=inputs))
        self.assertIn("template", str(ctx.exception))

    def test_shares_must_sum_to_one(self) -> None:
        edit_first_data_row(self.dir / DEMO_FILES["education"], 4, "0.99000000")
        with self.assertRaises(InputDataError) as ctx:
            load_targets(_cfg_for(self.dir))
        self.assertIn("sum to 1", str(ctx.exception))

    def test_population_household_inconsistency_is_detected(self) -> None:
        edit_first_data_row(self.dir / DEMO_FILES["households"], 4, "99999")
        with self.assertRaises(InputDataError) as ctx:
            load_targets(_cfg_for(self.dir))
        self.assertIn("inconsistent", str(ctx.exception))

    def test_bad_domain_and_unknown_region(self) -> None:
        edit_first_data_row(self.dir / DEMO_FILES["education"], 3, "PHD")
        with self.assertRaises(InputDataError):
            load_targets(_cfg_for(self.dir))
        write_demo_data(self.dir)
        edit_first_data_row(self.dir / DEMO_FILES["population"], 0, "ATLANTIS")
        with self.assertRaises(InputDataError):
            load_targets(_cfg_for(self.dir))

    def test_non_numeric_and_negative_values(self) -> None:
        edit_first_data_row(self.dir / DEMO_FILES["population"], 4, "many")
        with self.assertRaises(InputDataError):
            load_targets(_cfg_for(self.dir))
        write_demo_data(self.dir)
        edit_first_data_row(self.dir / DEMO_FILES["population"], 4, "-5")
        with self.assertRaises(InputDataError):
            load_targets(_cfg_for(self.dir))

    def test_missing_metadata_header(self) -> None:
        p = self.dir / DEMO_FILES["tenure"]
        p.write_text("\n".join(ln for ln in p.read_text().splitlines() if "dataset_id" not in ln) + "\n")
        with self.assertRaises(InputDataError):
            load_targets(_cfg_for(self.dir))

    def test_real_data_need_matching_reference_year(self) -> None:
        for f in DEMO_FILES.values():
            p = self.dir / f
            p.write_text(p.read_text().replace("synthetic_demo_only: true", "synthetic_demo_only: false").replace("reference_year: none", "reference_year: 1999"))
        with self.assertRaises(InputDataError) as ctx:
            load_targets(_cfg_for(self.dir, allow_synthetic_demo_data=False))
        self.assertIn("baseline_year", str(ctx.exception))

    def test_mixing_synthetic_and_real_is_refused(self) -> None:
        p = self.dir / DEMO_FILES["tenure"]
        p.write_text(p.read_text().replace("synthetic_demo_only: true", "synthetic_demo_only: false").replace("reference_year: none", "reference_year: 2000"))
        with self.assertRaises(InputDataError):
            load_targets(_cfg_for(self.dir))

    def test_parse_annotated_csv(self) -> None:
        meta, df = parse_annotated_csv("# a: 1\n# b: x: y\nc,d\n1,2\n")
        self.assertEqual(meta, {"a": "1", "b": "x: y"})
        self.assertEqual(list(df.columns), ["c", "d"])
        with self.assertRaises(InputDataError):
            parse_annotated_csv("# only: header\n")

    def test_baseline_config_cannot_run_without_data(self) -> None:
        from belgium_abm.config import load_config

        cfg = load_config(ROOT / "configs" / "belgium_baseline.yaml")
        cfg = cfg.model_copy(update={"inputs": cfg.inputs.model_copy(update={"population": "data/raw/absent.csv"})})
        with self.assertRaises(MissingInputsError):
            load_targets(cfg)

    def test_baseline_real_tables_load_and_are_marked_real(self) -> None:
        from belgium_abm.config import load_config

        t = load_targets(load_config(ROOT / "configs" / "belgium_baseline.yaml"))
        self.assertFalse(t.any_synthetic)
        self.assertEqual({m.reference_year for m in t.meta.values()}, {"2025"})
        self.assertGreater(t.total_persons, 11_000_000)
        self.assertLess(t.total_persons, 12_500_000)


if __name__ == "__main__":
    unittest.main()
