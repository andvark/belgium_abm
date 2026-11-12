"""Command-line interface (``abm``). Built on stdlib argparse; logic lives in plain functions.

Exit codes: 0 = success (verdict PASS or WARN); 1 = run completed but verdict FAIL / validation
failed; 2 = unusable configuration or input data (e.g. real data not acquired).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from belgium_abm import __version__
from belgium_abm.agents.household import HOUSEHOLD_COLUMNS
from belgium_abm.agents.person import PERSON_COLUMNS
from belgium_abm.analysis.validation import check_population, summarize_checks
from belgium_abm.assumptions import render_assumptions_markdown, render_sources_markdown
from belgium_abm.config import ModelConfig, find_project_root, load_config
from belgium_abm.demo_data import write_demo_data
from belgium_abm.io.loaders import InputDataError, MissingInputsError
from belgium_abm.io.outputs import read_table
from belgium_abm.io.registry import DatasetSource, load_registry
from belgium_abm.io.templates import write_templates
from belgium_abm.run import init_population

DEMO_CONFIG = "configs/demo_synthetic.yaml"


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_cfg(path: str, root: str | None) -> ModelConfig:
    return load_config(path, root=root)


def _registry(cfg: ModelConfig | None, explicit: str | None) -> list[DatasetSource]:
    if explicit:
        return load_registry(explicit)
    if cfg is not None:
        p = cfg.resolve(cfg.sources_registry)
        return load_registry(p) if p.is_file() else []
    default = find_project_root(Path.cwd()) / "data" / "sources.yaml"
    return load_registry(default) if default.is_file() else []


def cmd_init(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config, args.root)
    if args.n_persons is not None:
        cfg = cfg.with_overrides(generator={"n_persons": args.n_persons})
    out = Path(args.output_dir) if args.output_dir else None
    summary = init_population(cfg, args.seed, output_dir=out, write_parquet=not args.no_parquet, make_plot=not args.no_plot)
    print(f"[abm init] config={cfg.name} seed={args.seed}")
    print(f"[abm init] generated {summary.n_persons:,} persons in {summary.n_households:,} households")
    print(f"[abm init] invariants failed: {summary.invariants_failed}; comparison rows: {summary.status_counts}")
    print(f"[abm init] verdict: {summary.verdict}")
    print(f"[abm init] outputs in {summary.output_dir}: {', '.join(sorted(summary.files))}")
    if summary.report is not None and "SYNTHETIC" in summary.report["banner"]:
        print(f"[abm init] NOTE: {summary.report['banner']}")
    return 1 if summary.verdict == "FAIL" else 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config, args.root)
    run_dir = Path(args.run_dir)
    persons = read_table(run_dir / "persons.csv", PERSON_COLUMNS)
    households = read_table(run_dir / "households.csv", HOUSEHOLD_COLUMNS)
    checks = check_population(persons, households, cfg)
    for c in checks:
        print(f"{'ok  ' if c.passed else 'FAIL'} {c.name}" + ("" if c.passed else f"  -> {c.detail}"))
    s = summarize_checks(checks)
    print(f"[abm validate] {s['passed']}/{s['total']} checks passed")
    return 0 if s["failed"] == 0 else 1


def cmd_sources(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config, args.root) if args.config else None
    sources = _registry(cfg, args.registry)
    if not sources:
        _err("no dataset registry found (expected data/sources.yaml)")
        return 2
    print(render_sources_markdown(sources))
    n = sum(s.status == "not_acquired" for s in sources)
    print(f"{n} of {len(sources)} registered datasets are NOT acquired.")
    return 0


def cmd_assumptions(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config, args.root) if args.config else None
    text = render_assumptions_markdown(cfg, _registry(cfg, args.registry))
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8", newline="\n")
        print(f"[abm assumptions] wrote {args.output}")
    else:
        print(text)
    return 0


def cmd_make_demo_data(args: argparse.Namespace) -> int:
    paths = write_demo_data(Path(args.output_dir))
    print(f"[abm make-demo-data] wrote {len(paths)} FICTIONAL demo tables to {args.output_dir}")
    return 0


def cmd_make_templates(args: argparse.Namespace) -> int:
    paths = write_templates(Path(args.output_dir))
    print(f"[abm make-templates] wrote {len(paths)} empty real-data templates to {args.output_dir}")
    return 0


def cmd_fetch_data(args: argparse.Namespace) -> int:
    from belgium_abm.datasources.belgium import build_all
    from belgium_abm.datasources.eurostat import FetchError

    root = find_project_root(Path.cwd())
    out = Path(args.output_dir) if args.output_dir else root / "data" / "real" / f"belgium_{args.year}"
    cache = Path(args.cache_dir) if args.cache_dir else root / "data" / "real" / "cache"
    try:
        m = build_all(args.year, cache, out, offline=args.offline)
    except (FetchError, RuntimeError) as exc:
        _err(f"[abm fetch-data] failed: {exc}")
        return 2
    print(f"[abm fetch-data] wrote 6 real-data tables for {args.year} to {out}")
    print(f"[abm fetch-data] {len(m['sources'])} source responses (sha256 in build_manifest.json); {len(m['assumptions'])} documented assumptions")
    for r, d in m["diagnostics"]["households"].items():
        print(f"[abm fetch-data] {r}: fitted households / published private households = {d['ratio_fitted_to_published']:.3f}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from belgium_abm.run_sim import simulate

    cfg = _load_cfg(args.config, args.root)
    out_dir = Path(args.output_dir) if args.output_dir else cfg.resolve(cfg.output_dir) / "simulation"
    outs = simulate(cfg, seed=args.seed, years=args.years, replicates=args.replicates, n_persons=args.n_persons, scenario_path=args.scenario,
                    out_dir=out_dir, policy_file=args.policy, check_invariants=args.check_invariants)
    for k, v in outs.items():
        print(f"[abm simulate] {k}: {v}")
    print("[abm simulate] NOTE: results use a stylised economy and, unless a verified policy file is supplied, an ILLUSTRATIVE policy (not Belgian law).")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from belgium_abm.run_sim import compare

    out = compare(Path(args.baseline), Path(args.scenario), Path(args.output_dir) if args.output_dir else None)
    print(f"[abm compare] wrote comparison tables and comparison_report.md to {out}")
    return 0


def cmd_arrangements(args: argparse.Namespace) -> int:
    from belgium_abm.analysis.arrangements import run_arrangement_comparison

    cfg = _load_cfg(args.config, args.root)
    out = Path(args.output_dir) if args.output_dir else cfg.resolve(cfg.output_dir) / "arrangements"
    summary = run_arrangement_comparison(cfg, out, seed=args.seed, years=args.years, replicates=args.replicates, education=args.education, region=args.region,
                                         children=args.children, partner=args.partner, start_age=args.start_age, policy_file=args.policy, scenario_path=args.scenario)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"[abm arrangements] wrote tables, plot and report to {out}")
    print("[abm arrangements] NOTE: illustrative policy and stylised business parameters; this reveals trade-offs, not Belgian facts.")
    return 0


def cmd_import_hfcs(args: argparse.Namespace) -> int:
    from belgium_abm.datasources.hfcs import parse_hfcs
    from belgium_abm.datasources.belgium import write_table

    df, meta = parse_hfcs(Path(args.xlsx))
    out = Path(args.output) if args.output else find_project_root(Path.cwd()) / "data" / "real" / "belgium_2025" / "hfcs_wealth_targets.csv"
    header = {"dataset_id": "real_hfcs_wealth_targets", "synthetic_demo_only": "false", "status": "observed", "reference_year": "2023", "unit": "mixed",
              "source": f"{meta['publication']}; file {meta['source_file']} sha256 {meta['source_sha256']}", "description": "Published HFCS aggregates for Belgium, used only to VALIDATE simulated wealth (definitions differ: HFCS net wealth includes vehicles, valuables, pension plans and private business)"}
    write_table(out, header, df)
    print(f"[abm import-hfcs] wrote {len(df)} Belgian indicators to {out}")
    return 0


def cmd_crosscheck(args: argparse.Namespace) -> int:
    from belgium_abm.analysis.crosschecks import education_vs_statbel, labour_vs_statbel

    real = Path(args.real_dir) if args.real_dir else find_project_root(Path.cwd()) / "data" / "real" / "belgium_2025"
    outd = Path(args.output_dir) if args.output_dir else real
    if args.lfs_xlsx:
        d = labour_vs_statbel(real, Path(args.lfs_xlsx))
        d.to_csv(outd / "crosscheck_labour_vs_statbel.csv", index=False, lineterminator="\n")
        print(f"[abm crosscheck] labour: {len(d)} region x education x age cells; mean |employment diff| = {d.employment_diff.abs().mean():.3f}, "
              f"mean |unemployment diff| = {d.unemployment_diff.abs().dropna().mean():.3f}; worst: {d.loc[d.employment_diff.abs().idxmax(), ['region', 'education', 'age_band']].tolist()}")
    if args.education_xls:
        e = education_vs_statbel(real, Path(args.education_xls))
        e.to_csv(outd / "crosscheck_education_vs_statbel.csv", index=False, lineterminator="\n")
        print("[abm crosscheck] education 15+: " + "; ".join(f"{r.education} model {r.model_share:.3f} vs Statbel {r.statbel_share:.3f}" for r in e.itertuples()))
    return 0


def cmd_calibrate_wealth(args: argparse.Namespace) -> int:
    from belgium_abm.analysis.calibrate_wealth import calibrate_wealth
    from belgium_abm.datasources.belgium import write_table

    cfg = _load_cfg(args.config, args.root)
    params, out = calibrate_wealth(cfg, seed=args.seed, n_persons=args.n_persons, maxiter=args.maxiter)
    path = cfg.resolve(cfg.simulation.calibrated_parameters or "data/real/belgium_2025/wealth_calibration.csv")
    header = {"dataset_id": "wealth_calibration", "synthetic_demo_only": "false", "status": "derived", "reference_year": "2023", "unit": "mixed",
              "source": "calibration of stylised wealth parameters to ECB HFCS 2023 medians (abm calibrate-wealth)", "description": "Fitted: overall and by-age medians. Held out: Gini, top shares, percentiles, owner/non-owner medians"}
    write_table(path, header, params)
    d = out["diagnostics"]
    print(f"[abm calibrate-wealth] loss {d['loss_before']:.4f} -> {d['loss_after']:.4f} in {d['evaluations']} evaluations; wrote {path}")
    print(params[["name", "value"]].to_string(index=False))
    print(out["table"][["indicator", "group", "before_calibration_simulated", "simulated", "published", "status"]].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    from belgium_abm.analysis.sensitivity import run_sensitivity

    cfg = _load_cfg(args.config, args.root)
    out = Path(args.output_dir) if args.output_dir else cfg.resolve(cfg.output_dir) / "sensitivity"
    sw = run_sensitivity(cfg, args.scenario, out, seed=args.seed, n_persons=args.n_persons, years=args.years, replicates=args.replicates)
    for m in ("wealth_top10_share", "gini_disposable", "effect_net_fiscal_balance_dynamic", "effect_bv_owner_eq_income"):
        top = sw[sw.metric == m].sort_values("rank_within_metric").head(3)
        print(f"[abm sensitivity] {m}: " + "; ".join(f"{r.parameter} ({r.swing:+.4g})" for r in top.itertuples()))
    print(f"[abm sensitivity] wrote tables, tornado plot and report to {out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    ns = argparse.Namespace(
        config=args.config, root=args.root, seed=args.seed, n_persons=10_000,
        output_dir=args.output_dir, no_parquet=args.no_parquet, no_plot=False,
    )
    return cmd_init(ns)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="abm", description="Belgian lifecycle inequality ABM (Phase 0/1: synthetic population).")
    p.add_argument("--version", action="version", version=f"abm {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser, config_required: bool = True) -> None:
        sp.add_argument("--config", required=config_required, help="run configuration (YAML)")
        sp.add_argument("--root", default=None, help="project root for relative paths (default: nearest pyproject.toml)")

    sp = sub.add_parser("init", help="generate, validate and report a synthetic population")
    common(sp)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--n-persons", type=int, default=None, help="override generator.n_persons")
    sp.add_argument("--output-dir", default=None)
    sp.add_argument("--no-parquet", action="store_true")
    sp.add_argument("--no-plot", action="store_true")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("validate", help="re-check the invariants of a saved run")
    common(sp)
    sp.add_argument("--run-dir", required=True)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("sources", help="list the real datasets still required")
    common(sp, config_required=False)
    sp.add_argument("--registry", default=None)
    sp.set_defaults(func=cmd_sources)

    sp = sub.add_parser("assumptions", help="print/write the assumptions and open-dependencies report")
    common(sp, config_required=False)
    sp.add_argument("--registry", default=None)
    sp.add_argument("--output", default=None)
    sp.set_defaults(func=cmd_assumptions)

    sp = sub.add_parser("make-demo-data", help="(re)write the FICTIONAL demo input tables")
    sp.add_argument("--output-dir", default="data/demo")
    sp.set_defaults(func=cmd_make_demo_data)

    sp = sub.add_parser("make-templates", help="(re)write the empty real-data CSV templates")
    sp.add_argument("--output-dir", default="data/templates")
    sp.set_defaults(func=cmd_make_templates)

    sp = sub.add_parser("fetch-data", help="download open Belgian data (Eurostat API, Statbel BeStat) and build the input tables")
    sp.add_argument("--year", type=int, default=2025)
    sp.add_argument("--output-dir", default=None)
    sp.add_argument("--cache-dir", default=None)
    sp.add_argument("--offline", action="store_true", help="use only cached raw responses")
    sp.set_defaults(func=cmd_fetch_data)

    sp = sub.add_parser("simulate", help="run the dynamic lifecycle model (optionally with a policy scenario)")
    common(sp)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--years", type=int, default=None)
    sp.add_argument("--replicates", type=int, default=1)
    sp.add_argument("--n-persons", type=int, default=None)
    sp.add_argument("--scenario", default=None, help="scenario YAML (policy overrides/deltas)")
    sp.add_argument("--policy", default=None, help="policy YAML overriding simulation.policy_file")
    sp.add_argument("--output-dir", default=None)
    sp.add_argument("--check-invariants", action="store_true", help="validate the state every year (slower)")
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("compare", help="compare a scenario run with its baseline (paired, common random numbers)")
    sp.add_argument("--baseline", required=True, help="baseline run directory")
    sp.add_argument("--scenario", required=True, help="scenario run directory")
    sp.add_argument("--output-dir", default=None)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("arrangements", help="employee vs sole proprietor vs BV owner-manager over a working life (spec section 12)")
    common(sp)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--years", type=int, default=40)
    sp.add_argument("--replicates", type=int, default=500)
    sp.add_argument("--education", default="MEDIUM", choices=["LOW", "MEDIUM", "HIGH"])
    sp.add_argument("--region", default="BE_FLANDERS")
    sp.add_argument("--children", type=int, default=0)
    sp.add_argument("--partner", default="none", choices=["none", "inactive", "employee"])
    sp.add_argument("--start-age", type=int, default=30)
    sp.add_argument("--policy", default=None)
    sp.add_argument("--scenario", default=None)
    sp.add_argument("--output-dir", default=None)
    sp.set_defaults(func=cmd_arrangements)

    sp = sub.add_parser("import-hfcs", help="extract Belgian wealth-validation targets from the ECB HFCS statistical-tables workbook")
    sp.add_argument("--xlsx", required=True)
    sp.add_argument("--output", default=None)
    sp.set_defaults(func=cmd_import_hfcs)

    sp = sub.add_parser("crosscheck", help="compare derived tables with independent Statbel workbooks (user-supplied)")
    sp.add_argument("--lfs-xlsx", default=None, help="Statbel LFS rates by region x education x age (xlsx)")
    sp.add_argument("--education-xls", default=None, help="Statbel education attainment series (xls)")
    sp.add_argument("--real-dir", default=None)
    sp.add_argument("--output-dir", default=None)
    sp.set_defaults(func=cmd_crosscheck)

    sp = sub.add_parser("calibrate-wealth", help="fit stylised wealth parameters to HFCS medians (age groups, overall, non-owners); the rest is held out for validation")
    common(sp)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--n-persons", type=int, default=12000)
    sp.add_argument("--maxiter", type=int, default=70)
    sp.set_defaults(func=cmd_calibrate_wealth)

    sp = sub.add_parser("sensitivity", help="one-at-a-time sensitivity of outcomes and of a scenario effect to unestimated parameters")
    common(sp)
    sp.add_argument("--scenario", default="scenarios/corporate_tax_plus_5pp.yaml")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--years", type=int, default=15)
    sp.add_argument("--n-persons", type=int, default=8000)
    sp.add_argument("--replicates", type=int, default=2)
    sp.add_argument("--output-dir", default=None)
    sp.set_defaults(func=cmd_sensitivity)

    sp = sub.add_parser("demo", help="10,000-person demo run on the FICTIONAL inputs")
    sp.add_argument("--config", default=DEMO_CONFIG)
    sp.add_argument("--root", default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--output-dir", default=None)
    sp.add_argument("--no-parquet", action="store_true")
    sp.set_defaults(func=cmd_demo)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except MissingInputsError as exc:
        _err(f"[abm] cannot run: {exc}")
        return 2
    except (InputDataError, FileNotFoundError, ValueError) as exc:
        _err(f"[abm] error: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
