import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from belgium_abm.cli import main
from tests.helpers import DEMO_CONFIG, ROOT


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def test_init_writes_all_outputs_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d1, d2 = Path(d) / "a", Path(d) / "b"
            for target in (d1, d2):
                code, out, _ = run_cli("init", "--config", str(DEMO_CONFIG), "--n-persons", "2000", "--seed", "5", "--output-dir", str(target), "--no-parquet")
                self.assertEqual(code, 0, out)
            for name in ("persons.csv", "households.csv", "calibration_comparison.csv", "calibration_report.md", "calibration_report.json", "assumptions_report.md", "required_datasets.md", "population_pyramid.png", "manifest.json"):
                self.assertTrue((d1 / name).is_file(), name)
            m1, m2 = json.loads((d1 / "manifest.json").read_text()), json.loads((d2 / "manifest.json").read_text())
            self.assertEqual(m1["outputs_sha256"], m2["outputs_sha256"])
            self.assertTrue(m1["synthetic_demo_inputs"])
            self.assertEqual(set(m1["outputs_sha256"]), {"persons.csv", "households.csv", "calibration_comparison.csv"})
            self.assertTrue(all(not v["path"].startswith("/") for v in m1["inputs"]))  # no absolute paths in the manifest
            self.assertIn("SYNTHETIC", (d1 / "calibration_report.md").read_text())
            code, out, _ = run_cli("validate", "--config", str(DEMO_CONFIG), "--run-dir", str(d1))
            self.assertEqual(code, 0, out)

    def test_different_seeds_change_the_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            hashes = []
            for seed in ("1", "2"):
                target = Path(d) / seed
                run_cli("init", "--config", str(DEMO_CONFIG), "--n-persons", "1500", "--seed", seed, "--output-dir", str(target), "--no-parquet", "--no-plot")
                hashes.append(json.loads((target / "manifest.json").read_text())["outputs_sha256"]["persons.csv"])
            self.assertNotEqual(*hashes)

    def test_baseline_config_exits_2_with_a_clear_message(self) -> None:
        code, _, err = run_cli("init", "--config", str(ROOT / "configs" / "belgium_baseline.yaml"))
        self.assertEqual(code, 2)
        self.assertIn("no real data is ever substituted", err)
        self.assertIn("data/raw", err)

    def test_bad_config_path_exits_2(self) -> None:
        self.assertEqual(run_cli("init", "--config", "does/not/exist.yaml")[0], 2)

    def test_sources_lists_every_dataset_as_not_acquired(self) -> None:
        code, out, _ = run_cli("sources", "--config", str(DEMO_CONFIG))
        self.assertEqual(code, 0)
        self.assertIn("12 of 12 registered datasets are NOT acquired", out)
        self.assertIn("statbel_population", out)

    def test_assumptions_command_matches_committed_doc(self) -> None:
        doc = ROOT / "docs" / "model_assumptions.md"
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "a.md"
            self.assertEqual(run_cli("assumptions", "--config", str(DEMO_CONFIG), "--output", str(target))[0], 0)
            if doc.is_file():
                self.assertEqual(target.read_text(), doc.read_text(), "docs/model_assumptions.md is stale: re-run `abm assumptions`")
            self.assertIn("No Belgian tax rate", target.read_text())

    def test_make_templates_and_demo_data(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_cli("make-templates", "--output-dir", str(Path(d) / "t"))[0], 0)
            self.assertEqual(run_cli("make-demo-data", "--output-dir", str(Path(d) / "d"))[0], 0)
            self.assertEqual(len(list((Path(d) / "t").glob("*.csv"))), 6)
            self.assertEqual(len(list((Path(d) / "d").glob("*.csv"))), 6)

    def test_validate_flags_a_corrupted_run(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            run_cli("init", "--config", str(DEMO_CONFIG), "--n-persons", "1500", "--output-dir", d, "--no-parquet", "--no-plot")
            p = Path(d) / "persons.csv"
            p.write_text(p.read_text().replace(",True,", ",False,", 1))
            code, out, _ = run_cli("validate", "--config", str(DEMO_CONFIG), "--run-dir", d)
            self.assertEqual(code, 1)
            self.assertIn("FAIL", out)

    def test_version_flag(self) -> None:
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(io.StringIO()):
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
