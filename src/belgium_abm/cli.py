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
