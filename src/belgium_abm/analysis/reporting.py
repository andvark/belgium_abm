"""Machine-readable (JSON) and human-readable (Markdown) calibration report."""

from __future__ import annotations

import platform
from typing import Any

import numpy as np
import pandas as pd
import scipy

from belgium_abm.analysis.calibration import summarize
from belgium_abm.analysis.validation import NOT_YET_APPLICABLE, CheckResult, summarize_checks
from belgium_abm.config import ModelConfig
from belgium_abm.io.loaders import TargetBundle

SYNTHETIC_BANNER = "SYNTHETIC DEMO - FICTIONAL INPUTS - NOT BELGIAN DATA - NO EMPIRICAL MEANING"


def banner_for(targets: TargetBundle, cfg: ModelConfig) -> str:
    if targets.any_synthetic:
        return SYNTHETIC_BANNER
    return f"Reference year {cfg.baseline_year} - data vintage listed below - Phase 1 population only, no policy effects"


def build_report(
    *,
    cfg: ModelConfig,
    targets: TargetBundle,
    seed: int,
    verdict: str,
    checks: list[CheckResult],
    comparisons: pd.DataFrame,
    generation: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, Any]:
    cs = summarize_checks(checks)
    counts = {k: int(v) for k, v in comparisons["status"].value_counts().items()}
    worst = comparisons[comparisons["status"] != "pass"].sort_values("rel_diff", ascending=False).head(15)
    if worst.empty:
        worst = comparisons.sort_values("rel_diff", ascending=False).head(10)
    return {
        "banner": banner_for(targets, cfg),
        "verdict": verdict,
        "config_name": cfg.name,
        "config_hash_sha256": cfg.config_hash(),
        "seed": seed,
        "baseline_year_label": cfg.baseline_year,
        "horizon": "initial state only (no dynamics in Phase 1)",
        "scenario": "none (no policy scenario in Phase 1)",
        "weighting_unit": f"persons (each simulated person carries one uniform expansion weight); 1 simulated person = {sample['real_persons_per_simulated_person']:.2f} real persons",
        "uncertainty": "Single stochastic realisation for the given seed. Seed-to-seed variability is not summarised in this report; re-run with other seeds and compare.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "data_vintage": [
            {"table": m.table, "dataset_id": m.dataset_id, "synthetic_demo_only": m.synthetic_demo_only, "status": m.status, "reference_year": m.reference_year, "sha256": m.sha256}
            for m in targets.meta.values()
        ],
        "invariants": {"summary": cs, "checks": [{"name": c.name, "spec": c.spec, "passed": c.passed, "detail": c.detail} for c in checks]},
        "comparison_status_counts": counts,
        "comparison_summary": summarize(comparisons).to_dict(orient="records"),
        "worst_rows": worst.to_dict(orient="records"),
        "generation": generation,
        "sample_size": sample,
        "tolerances": cfg.tolerance.model_dump(),
        "not_yet_applicable": [{"spec": s, "item": t} for s, t in NOT_YET_APPLICABLE],
        "limitations": [
            "Phase 1 builds demographic and labour-state variables only; no income, wealth, tax or benefit is modelled.",
            "Agreement with the targets says nothing about Belgium unless the targets are real, validated data.",
            "Members of a household are independent in education / labour state beyond the explicit age, sex and role rules.",
        ],
    }


def render_markdown(r: dict[str, Any]) -> str:
    L: list[str] = []
    L.append(f"# Calibration report - {r['config_name']}")
    L.append("")
    if "SYNTHETIC" in r["banner"]:
        L += [f"> **{r['banner']}**", ""]
    else:
        L += [f"> {r['banner']}", ""]
    L.append(f"**Verdict: {r['verdict']}**  (PASS = no tolerance exceeded; WARN = some rows outside the warn band; FAIL = an invariant failed or a margin exceeded the fail band)")
    L += ["", "## Run", ""]
    L.append(f"- seed: `{r['seed']}`; config hash: `{r['config_hash_sha256'][:16]}...`; baseline year label: `{r['baseline_year_label']}`")
    L.append(f"- horizon: {r['horizon']}; scenario: {r['scenario']}")
    L.append(f"- weighting unit: {r['weighting_unit']}")
    L.append(f"- uncertainty: {r['uncertainty']}")
    e = r["environment"]
    L.append(f"- python {e['python']}, numpy {e['numpy']}, pandas {e['pandas']}, scipy {e['scipy']}")
    L += ["", "## Data vintage", "", "| table | dataset id | synthetic | status | reference year | sha256 (first 12) |", "|---|---|---|---|---|---|"]
    for d in r["data_vintage"]:
        L.append(f"| {d['table']} | `{d['dataset_id']}` | {'YES' if d['synthetic_demo_only'] else 'no'} | {d['status']} | {d['reference_year']} | `{d['sha256'][:12]}` |")
    s = r["invariants"]["summary"]
    L += ["", f"## Structural invariants ({s['passed']}/{s['total']} passed, {s['failed']} failed)", "", "| check | spec | result | detail |", "|---|---|---|---|"]
    for c in r["invariants"]["checks"]:
        L.append(f"| {c['name']} | {c['spec']} | {'pass' if c['passed'] else '**FAIL**'} | {c['detail']} |")
    L += ["", "## Calibration comparison", "", f"Row status counts: {r['comparison_status_counts']}", "", "| table | level | pass | warn | fail |", "|---|---|---|---|---|"]
    for row in r["comparison_summary"]:
        L.append(f"| {row['table']} | {row['level']} | {row.get('pass', 0)} | {row.get('warn', 0)} | {row.get('fail', 0)} |")
    L += ["", "Largest deviations (full detail in `calibration_comparison.csv`):", "", "| table | level | key | target | simulated | rel. diff | status |", "|---|---|---|---|---|---|---|"]
    for w in r["worst_rows"]:
        L.append(f"| {w['table']} | {w['level']} | `{w['key']}` | {w['target']:.1f} | {w['simulated']:.0f} | {w['rel_diff']:.2%} | {w['status']} |")
    t = r["tolerances"]
    L += ["", "Tolerances: " + ", ".join(f"`{k}`={v}" for k, v in t.items()) + ". These are configuration choices, not empirical findings."]
    g = r["generation"]
    L += ["", "## Generation diagnostics", ""]
    L.append(f"- candidate households in pool: {g['candidate_pool_size']:,}; selected: {g['n_households']:,} households, {g['n_persons']:,} persons")
    L.append(f"- IPU: {g['ipu_iterations']} iterations, converged={g['ipu_converged']}, max relative constraint error {g['ipu_max_rel_error']:.2e}; Kish effective size of weights {g['ipu_weight_kish_n_eff']:,.0f}")
    L.append(f"- integerisation objective: {g['integerise_objective_after_rounding']:.2f} after rounding -> {g['integerise_objective_after_swaps']:.2f} after {g['swaps_accepted']} accepted swaps")
    if g["ipu_unsupported_constraints"]:
        L.append(f"- UNSUPPORTED constraints (no candidate household can satisfy them): {g['ipu_unsupported_constraints']}")
    if g["templates_without_ipu_support"]:
        L.append(f"- templates that fell back to uniform weights: {g['templates_without_ipu_support']}")
    sm = r["sample_size"]
    L += ["", "## Sample size", "", f"- {sm['n_persons']:,} persons / {sm['n_households']:,} households", f"- {sm['tail_warning']}"]
    if sm["groups_below_min_group_size"]:
        L.append(f"- region x sex groups below {sm['min_group_size']} persons: {sm['groups_below_min_group_size']}")
    L += ["", "## Spec section 11 checks not yet applicable", ""]
    L += [f"- {n['spec']}: {n['item']}" for n in r["not_yet_applicable"]]
    L += ["", "## Limitations", ""] + [f"- {x}" for x in r["limitations"]]
    return "\n".join(L) + "\n"
