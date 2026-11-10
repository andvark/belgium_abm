import unittest

from pydantic import ValidationError

from belgium_abm.config import load_config
from belgium_abm.provenance import REQUIRED_PARAMETERS, Param, ParameterStatus, validate_parameters
from tests.helpers import DEMO_CONFIG, ROOT, modified_demo_config


class ProvenanceTests(unittest.TestCase):
    def test_demo_registry_is_complete(self) -> None:
        cfg = load_config(DEMO_CONFIG)
        self.assertEqual(validate_parameters(cfg.parameters), [])
        for req in REQUIRED_PARAMETERS:
            self.assertIn(req.name, cfg.parameters)

    def test_missing_parameter_is_reported(self) -> None:
        cfg = load_config(DEMO_CONFIG)
        params = dict(cfg.parameters)
        del params["adult_age"]
        self.assertTrue(any("adult_age" in p for p in validate_parameters(params)))

    def test_observed_parameter_requires_year(self) -> None:
        cfg = load_config(DEMO_CONFIG)
        params = dict(cfg.parameters)
        params["adult_age"] = Param(value=18, status=ParameterStatus.OBSERVED, source="x", unit="years")
        self.assertTrue(any("reference year" in p for p in validate_parameters(params)))

    def test_param_requires_source_and_unit(self) -> None:
        with self.assertRaises(ValidationError):
            Param(value=1, status="assumed", source="", unit="years")

    def test_all_demo_parameters_are_assumed(self) -> None:
        cfg = load_config(DEMO_CONFIG)
        self.assertTrue(all(p.status is ParameterStatus.ASSUMED for p in cfg.parameters.values()))


class ConfigTests(unittest.TestCase):
    def test_root_is_project_root(self) -> None:
        self.assertEqual(load_config(DEMO_CONFIG).root, ROOT)

    def test_overrides_and_hash(self) -> None:
        a = modified_demo_config(1000)
        b = modified_demo_config(1200)
        self.assertEqual(a.generator.n_persons, 1000)
        self.assertNotEqual(a.config_hash(), b.config_hash())
        self.assertEqual(a.config_hash(), modified_demo_config(1000).config_hash())
        self.assertEqual(a.root, ROOT)

    def test_unknown_key_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            load_config(DEMO_CONFIG).with_overrides(generator={"nonsense": 1})

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config(ROOT / "configs" / "nope.yaml")

    def test_baseline_config_loads_but_is_a_placeholder(self) -> None:
        cfg = load_config(ROOT / "configs" / "belgium_baseline.yaml")
        self.assertEqual(cfg.baseline_year, 0)
        self.assertFalse(cfg.allow_synthetic_demo_data)

    def test_typed_accessors(self) -> None:
        cfg = load_config(DEMO_CONFIG)
        self.assertEqual((cfg.adult_age, cfg.working_age_min), (18, 15))
        self.assertEqual(cfg.equivalence, (1.0, 0.5, 0.3, 14))


if __name__ == "__main__":
    unittest.main()
