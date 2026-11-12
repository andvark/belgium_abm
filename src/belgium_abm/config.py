"""Run configuration (pydantic) and YAML loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from belgium_abm.provenance import Param, validate_parameters


class InputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    population: str
    households: str
    education: str
    labour_state: str
    work_arrangement: str
    tenure: str | None = None


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_persons: int = Field(default=10_000, gt=0)
    pool_multiplier: float = Field(default=10.0, gt=0)
    min_pool_per_template: int = Field(default=200, gt=0)
    max_candidate_attempts: int = Field(default=50, gt=0)
    ipu_max_iterations: int = Field(default=500, gt=0)
    ipu_tolerance: float = Field(default=1e-6, gt=0)
    refine_swaps_per_household: int = Field(default=40, ge=0)
    refine_counts_proposals_per_cell: int = Field(default=300, ge=0)
    population_household_consistency_tol: float = Field(default=0.01, gt=0)


class ToleranceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    marginal_rel_warn: float = 0.01
    marginal_rel_fail: float = 0.05
    cell_rel_warn: float = 0.10
    cell_rel_fail: float = 0.25
    cells_can_fail: bool = False
    min_expected_count: float = 20
    small_cell_abs_tol: float = 3
    rounding_slack_units: float = 2
    min_group_size: int = 30


class SimulationConfig(BaseModel):
    """Inputs of the dynamic model (Phases 2-4). Optional: the population generator does not need them."""

    model_config = ConfigDict(extra="forbid")
    policy_file: str
    economy_file: str
    demography_rates: str
    validation_targets: str
    wealth_targets: str | None = None
    behaviour_parameters: str | None = None
    calibrated_parameters: str | None = None
    years: int = Field(default=30, gt=0)
    min_group_size: int = Field(default=30, gt=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    baseline_year: int = Field(ge=0)
    allow_synthetic_demo_data: bool = False
    regions: list[str] = Field(min_length=1)
    sources_registry: str = "data/sources.yaml"
    output_dir: str = "outputs/run"
    inputs: InputsConfig
    generator: GeneratorConfig = GeneratorConfig()
    tolerance: ToleranceConfig = ToleranceConfig()
    parameters: dict[str, Param]
    simulation: SimulationConfig | None = None

    _root: Path = PrivateAttr(default=Path("."))

    # -- paths -------------------------------------------------------------------------
    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, relative: str | Path) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else (self._root / p)

    # -- reproducibility ---------------------------------------------------------------
    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_overrides(self, **overrides: Any) -> ModelConfig:
        data = self.model_dump(mode="json")
        for k, v in overrides.items():
            data[k] = {**data[k], **v} if isinstance(v, dict) and isinstance(data.get(k), dict) else v
        new = ModelConfig.model_validate(data)
        new._root = self._root
        return new

    # -- typed parameter accessors -----------------------------------------------------
    def _p(self, name: str) -> Any:
        return self.parameters[name].value

    @property
    def adult_age(self) -> int:
        return int(self._p("adult_age"))

    @property
    def working_age_min(self) -> int:
        return int(self._p("working_age_min"))

    @property
    def min_parent_age_at_birth(self) -> int:
        return int(self._p("min_parent_age_at_birth"))

    @property
    def max_parent_age_at_birth(self) -> int:
        return int(self._p("max_parent_age_at_birth"))

    @property
    def couple_age_difference_sd(self) -> float:
        return float(self._p("couple_age_difference_sd"))

    @property
    def equivalence(self) -> tuple[float, float, float, int]:
        return (
            float(self._p("equivalence_first_adult_weight")),
            float(self._p("equivalence_additional_person_weight")),
            float(self._p("equivalence_child_weight")),
            int(self._p("equivalence_child_age_threshold")),
        )


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for cand in (cur, *cur.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    return start.resolve()


def load_config(path: str | Path, root: str | Path | None = None) -> ModelConfig:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config {p} must be a YAML mapping")
    cfg = ModelConfig.model_validate(raw)
    problems = validate_parameters(cfg.parameters)
    if cfg.parameters.get("within_band_age_distribution") and cfg.parameters["within_band_age_distribution"].value != "uniform":
        problems.append("within_band_age_distribution: only 'uniform' is implemented")
    if problems:
        raise ValueError("invalid parameter registry:\n  - " + "\n  - ".join(problems))
    cfg._root = Path(root).resolve() if root else find_project_root(p.resolve().parent)
    return cfg
