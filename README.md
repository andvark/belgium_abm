# Belgian lifecycle inequality ABM — Phase 0/1 (synthetic population)

Foundation for an agent-based model of income and wealth inequality in Belgium. This stage
contains **only** the project scaffold, data schemas, accounting skeleton and a reproducible,
calibrated **synthetic population generator** (persons and households).

> **No real Belgian data are loaded.** All bundled inputs under `data/demo/` are **FICTIONAL**
> (`synthetic_demo_only: true`) and every generated report carries a SYNTHETIC banner. No
> statutory rates, demographic statistics or transition probabilities are included or invented;
> the policy configuration is an empty skeleton. **Nothing here estimates real policy impacts.**
> See `docs/model_assumptions.md` for assumptions and `abm sources` for the real datasets still
> needed. The full specification is in `docs/specification.md`.

## Install and run

Requires Python ≥ 3.12 with numpy, pandas, scipy, pydantic, pyyaml, matplotlib
(`requirements-lock.txt` is a snapshot of the versions the project was developed with, not a
resolver lock).

```bash
pip install -e .             # provides the `abm` command
# or, without installing:
./bin/abm --help             # wrapper that sets PYTHONPATH=src
```

```bash
abm demo                                  # 10,000-person run on FICTIONAL inputs -> outputs/demo/
abm init --config configs/demo_synthetic.yaml --seed 42 --n-persons 20000
abm validate --config configs/demo_synthetic.yaml --run-dir outputs/demo
abm sources                               # real datasets still required (all not_acquired)
abm assumptions --config configs/demo_synthetic.yaml --output docs/model_assumptions.md
abm make-demo-data                        # rewrite data/demo/*.csv (fictional)
abm make-templates                        # rewrite empty real-data CSV templates
```

Exit codes: 0 = PASS/WARN, 1 = verdict FAIL or validation failure, 2 = unusable config or
missing/refused input data. `abm init --config configs/belgium_baseline.yaml` exits 2 until real
tables exist in `data/raw/` (that is intended).

### Outputs (`outputs/demo/`, git-ignored)

`persons.csv`, `households.csv` (+ `.parquet` if `pyarrow` is installed),
`calibration_comparison.csv`, `calibration_report.md/.json`, `assumptions_report.md`,
`required_datasets.md`, `population_pyramid.png`, `manifest.json` (config hash, seed, input
and output sha256).

### Tests

```bash
python3 -m unittest discover -s tests -t .     # also collectable by pytest
```

Unit, integration (CLI end to end) and seeded property tests, including bit-for-bit
determinism, validator corruption tests and invariants across seeds and sizes.

## Layout

```
configs/    demo_synthetic.yaml (runnable, fictional); belgium_baseline.yaml (needs real data)
data/       sources.yaml registry; demo/ (fictional); templates/ (empty); raw/, processed/
docs/       specification.md, model_assumptions.md, calibration.md
src/belgium_abm/  agents/ engines/ policies/ io/ analysis/ + config, provenance, rng, cli
tests/      unit/ integration/ properties/
```

## Getting real data in

1. Acquire the datasets listed by `abm sources` (public Statbel / Eurostat tables first; restricted
   microdata such as EU-SILC research files, HFCS or BELMOD need access applications).
2. Convert them into the layouts in `data/templates/`, keep the `# key: value` header (set
   `synthetic_demo_only: false`, `status`, `reference_year`, source) and save under `data/raw/`.
3. Update `data/sources.yaml` (`status: acquired`, `file_path`, `reference_year`), set
   `baseline_year`, regions and parameter provenance in `configs/belgium_baseline.yaml`, and run
   `abm init --config configs/belgium_baseline.yaml`.

## Deviations from the specification's tool list

The development sandbox had no PyPI access, so: the CLI uses stdlib `argparse` (not
Typer/Click), tests use `unittest` (pytest-compatible; no `hypothesis` — property tests are
seeded loops), Parquet is optional, and `ruff` / `mypy --strict` are configured in
`pyproject.toml` but **have not been run**. The Parquet code path is untested for the same
reason.

## Not implemented (by design)

Income, wealth, taxes, benefits, behaviour and the annual loop (`engines/simulation.py`
raises `NotImplementedError`); only the integer-cent `FiscalLedger`, the empty `PolicyConfig`
and the annual-schedule convention exist, with tests. Real-data calibration awaits the
datasets in `abm sources`.
