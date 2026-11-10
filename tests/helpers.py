"""Shared test helpers (cached demo population, config tweaks, CSV editing)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from belgium_abm.config import ModelConfig, load_config
from belgium_abm.engines.population import Population, generate_population
from belgium_abm.io.loaders import TargetBundle, load_targets

ROOT = Path(__file__).resolve().parents[1]
DEMO_CONFIG = ROOT / "configs" / "demo_synthetic.yaml"


def modified_demo_config(n_persons: int = 2000, **overrides) -> ModelConfig:
    cfg = load_config(DEMO_CONFIG)
    generator = {"n_persons": n_persons, **overrides.pop("generator", {})}
    return cfg.with_overrides(generator=generator, **overrides)


@lru_cache(maxsize=8)
def demo_population(n_persons: int = 2000, seed: int = 42) -> tuple[ModelConfig, TargetBundle, Population]:
    cfg = modified_demo_config(n_persons)
    targets = load_targets(cfg)
    return cfg, targets, generate_population(cfg, targets, seed)


def edit_first_data_row(path: Path, field_index: int, new_value: str) -> None:
    """Replace one field of the first data row of an annotated CSV."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if not ln.startswith("#")) + 1  # skip header row
    parts = lines[i].split(",")
    parts[field_index] = new_value
    lines[i] = ",".join(parts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
