"""Orchestration of ``abm init``: load -> generate -> validate -> compare -> report -> persist."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from belgium_abm import __version__
from belgium_abm.analysis.calibration import build_comparisons, classify, sample_size_report, verdict
from belgium_abm.analysis.plots import population_pyramid
from belgium_abm.analysis.reporting import banner_for, build_report, render_markdown
from belgium_abm.analysis.validation import check_population, summarize_checks
from belgium_abm.assumptions import render_assumptions_markdown, render_sources_markdown
from belgium_abm.config import ModelConfig
from belgium_abm.engines.population import Population, generate_population
from belgium_abm.io.loaders import TargetBundle, load_targets
from belgium_abm.io.outputs import parquet_available, sha256_file, write_json, write_table
from belgium_abm.io.registry import load_registry


@dataclass
class RunSummary:
    verdict: str
    output_dir: Path
    n_persons: int
    n_households: int
    invariants_failed: int
    status_counts: dict[str, int]
    files: dict[str, str] = field(default_factory=dict)
    population: Population | None = None
    comparisons: pd.DataFrame | None = None
    report: dict[str, Any] | None = None


def evaluate(cfg: ModelConfig, targets: TargetBundle, pop: Population, seed: int) -> tuple[str, dict[str, Any], pd.DataFrame, list[Any]]:
    """Validate and compare a generated population; return (verdict, report, comparisons, checks)."""
    checks = check_population(pop.persons, pop.households, cfg, max_age=targets.max_age)
    comparisons = classify(build_comparisons(pop.persons, pop.households, targets, cfg), cfg.tolerance)
    failed = summarize_checks(checks)["failed"]
    v = verdict(comparisons, failed)
    sample = sample_size_report(pop.persons, pop.households, cfg, pop.diagnostics["ipu_weight_kish_n_eff"])
    report = build_report(cfg=cfg, targets=targets, seed=seed, verdict=v, checks=checks, comparisons=comparisons, generation=pop.diagnostics, sample=sample)
    return v, report, comparisons, checks


def init_population(cfg: ModelConfig, seed: int, *, output_dir: Path | None = None, write_parquet: bool = True, make_plot: bool = True) -> RunSummary:
    targets = load_targets(cfg)
    pop = generate_population(cfg, targets, seed)
    v, report, comparisons, checks = evaluate(cfg, targets, pop, seed)

    out = output_dir if output_dir is not None else cfg.resolve(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    produced += write_table(pop.persons, out, "persons", parquet=write_parquet)
    produced += write_table(pop.households, out, "households", parquet=write_parquet)
    produced += write_table(comparisons, out, "calibration_comparison", parquet=False)
    write_json(report, out / "calibration_report.json")
    (out / "calibration_report.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")

    reg_path = cfg.resolve(cfg.sources_registry)
    registry = load_registry(reg_path) if reg_path.is_file() else []
    (out / "assumptions_report.md").write_text(render_assumptions_markdown(cfg, registry), encoding="utf-8", newline="\n")
    (out / "required_datasets.md").write_text(render_sources_markdown(registry), encoding="utf-8", newline="\n")
    if make_plot:
        population_pyramid(pop.persons, targets, cfg, out / "population_pyramid.png", banner=banner_for(targets, cfg))

    manifest = {
        "verdict": v,
        "seed": seed,
        "config_name": cfg.name,
        "config_hash_sha256": cfg.config_hash(),
        "package_version": __version__,
        "synthetic_demo_inputs": targets.any_synthetic,
        "parquet_written": bool(write_parquet and parquet_available()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "inputs": [
            {"table": m.table, "path": m.path, "sha256": m.sha256, "dataset_id": m.dataset_id, "synthetic_demo_only": m.synthetic_demo_only}
            for m in targets.meta.values()
        ],
        # canonical outputs: byte-identical for identical (config, inputs, seed, environment)
        "outputs_sha256": {p.name: sha256_file(p) for p in produced if p.suffix == ".csv"},
    }
    write_json(manifest, out / "manifest.json")

    return RunSummary(
        verdict=v,
        output_dir=out,
        n_persons=len(pop.persons),
        n_households=len(pop.households),
        invariants_failed=summarize_checks(checks)["failed"],
        status_counts={k: int(c) for k, c in comparisons["status"].value_counts().items()},
        files={p.name: str(p) for p in sorted(out.iterdir()) if p.is_file()},
        population=pop,
        comparisons=comparisons,
        report=report,
    )
