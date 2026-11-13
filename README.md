# Belgian lifecycle inequality ABM

An agent-based model of how income and wealth inequality in Belgium emerge over life cycles, and how
taxes, transfers, social insurance and legal work arrangements change them. Persons grouped into
households live through demography, education, work (employee / sole proprietor / BV owner-manager),
taxes and benefits, consumption, saving and asset accumulation; inequality is measured at several
points of that process and policy scenarios are compared against a baseline with common random numbers.
The full brief is in [`docs/specification.md`](docs/specification.md).

> **Read this first: what is real and what is not**
>
> * **Real (2025 Belgian aggregates):** population by region x sex x age, education by age, household
>   composition (derived), labour-market status (derived), life tables and fertility, and published
>   EU-SILC income indicators and ECB HFCS 2023 wealth aggregates used for validation. See [`docs/data_real.md`](docs/data_real.md).
> * **Public-aggregate parameters:** labour-market flows, business closure rates, part-time, earnings structure and
>   house prices come from public Eurostat tables (`data/real/belgium_2025/behaviour_observed.csv`); wealth parameters are
>   calibrated to ECB HFCS medians. **Assumptions:** the rest (earnings persistence, consumption, business payout,
>   entry and education sensitivities, ...), each tagged `assumed` in `configs/economy_stylised.yaml`.
> * **Illustrative, NOT Belgian law:** the tax-benefit file `configs/policy_illustrative.yaml`. No statutory
>   rate was invented as law; every number is a labelled placeholder (`verified: false`) and **every output
>   carries a banner**. Supply a verified policy file before quoting any Belgian fiscal result.
> * **Nothing here is a validated forecast or an estimate of real policy effects.** At baseline the model
>   overstates poverty and the Gini (see the validation tables in any run report); the poverty rate is dominated
>   by the placeholder minimum-income level of the illustrative policy (see Known limitations).

## Install

Python >= 3.12 with numpy, pandas, scipy, pydantic, pyyaml, matplotlib (`pyarrow` optional).

```bash
pip install -e .          # provides the `abm` command
# or without installing:  ./bin/abm --help
```

## Quick start

```bash
abm fetch-data --year 2025                                   # (re)build the real-data tables; --offline uses the committed cache
abm init --config configs/belgium_baseline.yaml --seed 42    # 100,000-person synthetic Belgium on the real 2025 targets
abm simulate --config configs/belgium_baseline.yaml --years 30 --seed 42 --replicates 10 \
    --scenario scenarios/corporate_tax_plus_5pp.yaml --output-dir outputs/run1
abm compare --baseline outputs/run1/baseline --scenario outputs/run1/scenario_corporate_tax_plus_5pp
abm arrangements --config configs/belgium_baseline.yaml --children 2 --partner employee   # spec section 12
abm import-hfcs --xlsx <HFCS workbook>                      # Belgian wealth targets (validation only)
abm crosscheck --lfs-xlsx <file> --education-xls <file>      # test the derived tables against Statbel workbooks
abm calibrate-wealth --config configs/belgium_baseline.yaml   # fit stylised wealth parameters to HFCS medians
abm sensitivity --config configs/belgium_baseline.yaml        # one-at-a-time sensitivity of outcomes and of a scenario effect
abm sources                                                  # datasets acquired / still required
abm demo                                                     # 10,000-person run on FICTIONAL inputs (tooling check)
```

`abm simulate --n-persons 20000` speeds up replicates (30 years x 20k persons takes about 20 s).
Exit codes: 0 = success, 1 = a validation verdict FAIL, 2 = unusable configuration, refused input or
incompatible comparison.

## What the commands produce

| command | key outputs |
|---|---|
| `init` | `persons.csv`, `households.csv`, calibration comparison (832 targets), pyramid, assumptions report, manifest with input hashes |
| `simulate` | per run directory: `metrics_by_year.csv` (Gini market with pensions as transfers / as income, disposable, after VAT, P90/P10, S80/S20, poverty, wealth Gini and top shares, fiscal totals ...), `group_metrics.csv` (age, region, education, arrangement, children, tenure, current and baseline wealth quintile; small-sample warnings), `fiscal_by_year.csv`, `ledger_summary.csv` (integer cents), `events.csv`, `income_mobility.csv`, `uncertainty_summary.csv`, `validation_vs_published.csv`, plots, `report.md`, `run_manifest.json` |
| `compare` | `comparison_vs_baseline.csv` (paired differences with 5-95% bands: **static fixed-behaviour**, **dynamic total**, and their non-causal difference), `group_differences.csv`, `fiscal_differences.csv`, `winners_losers_by_baseline_wealth_quintile.csv`, `comparison_report.md` |
| `arrangements` | employee vs sole proprietor vs BV owner-manager for the same economic activity: lifecycle tables, plot and report |

Example outputs from a full 10-replicate run are in [`docs/example_results/`](docs/example_results/).

## Layout

```
configs/    belgium_baseline.yaml (real 2025 targets + simulation inputs), demo_synthetic.yaml (fictional),
            policy_illustrative.yaml (NOT law), economy_stylised.yaml (assumptions)
scenarios/  policy scenarios (overrides / deltas)
data/       sources.yaml registry; real/ (cache + 2025 tables + build manifest); demo/ (fictional); templates/
docs/       specification.md, data_real.md, model_description.md, model_assumptions.md, calibration.md
src/belgium_abm/
  datasources/   Eurostat + Statbel BeStat clients, real-table builder
  engines/       population (IPU), state, flows (fiscal engine), dynamics, simulation, crn, economy
  policies/      rules (pure tax/benefit functions, overrides), ledger (integer cents), config
  analysis/      inequality, measures, benchmarks, arrangements, calibration, validation, plots
  io/ agents/    schemas, loaders, registry; person / household / firm tables
tests/       unit / integration / properties (180 tests)
```

## Method in brief (details in `docs/model_description.md`)

1. **Population** - households first (candidate pools per region x household template), IPU re-weighting
   to person-cell and household margins, integerisation, then attribute allocation from conditional
   shares; fully reproducible per seed.
2. **Level calibration** - one scale factor makes the simulated median equivalised disposable income equal
   the published EU-SILC median. Everything else (Gini, poverty, S80/S20, quintile shares) is validation.
3. **Annual loop** - age +1, deaths and births (observed rates), partnering / separation / leaving home /
   inheritance, education, labour flows and firm entry/exit, income and fiscal flows (legal payer and
   incidence kept apart; BV retained profit is not household income), consumption and saving (no negative
   cash without an explicit credit event), asset dynamics, measurement.
4. **Common random numbers** - each random draw is keyed by (seed, stream, year, entity id), so baseline and
   scenario runs see identical shocks for the same people.
5. **Scenarios** - a scenario changes policy (or economy knobs); `compare` reports the mechanical effect
   under fixed behaviour separately from the dynamic re-run and never labels the gap causal.

## Tests

```bash
python3 -m pytest -q          # ~45 s; unittest-compatible too: python3 -m unittest discover -s tests -t .
```

Covers the mandatory invariants of the specification (households/persons, exclusive states, ledger
reconciliation to the cent, wages/dividends paid = received, BV equity counted once, household cash
identity, no negative stocks, bit-identical reruns, baseline = identical override, fixed-behaviour
isolation, Gini toy cases, tax brackets/caps/thresholds) plus the real-data build (offline rebuild).

## Known limitations

* **Poverty and the income Gini are overstated, mainly because of the placeholder policy.** The share of persons in very-low-work-intensity households matches the published figure (12% vs 11%), so household worklessness is not the cause. Poverty is highly sensitive to the illustrative social-assistance floor: raising it from 12,500 to 19,000 EUR per equivalent adult (the 60%-of-median line) cuts the poverty rate from 20% to 5%. A verified minimum-income level is needed before poverty results mean anything.
* Labour status, education and tenure are still drawn independently across members of a household; partner correlation is absent (not a driver of the poverty gap, see above).
* **Residual cohort drift.** Even with hazards derived to preserve the observed age profile, long runs drift: over 30 years unemployment rises from 6.3% to about 8% (peak 9%), employment at ages 45-54 falls about 12 points (and rises at 55-64), and the disposable-income Gini rises from about 0.26 to 0.285. The `cross_section_drift.csv` diagnostic in every run reports it. Treat 30-year levels as illustrative; paired scenario-minus-baseline differences are less affected.
* Wealth, firms and behaviour are stylised; wealth is not validated against HFCS (restricted).
* The BV owner-manager share is a coarse proxy; regional household composition is the national mix refitted.
* No migration; the population ages and shrinks. Constant prices.
* `ruff` was run (E501 line-length and SIM905 ignored; 20 minor style notes remain, none functional); `mypy --strict` is configured but has not been run.
* The pinned `requirements-lock.txt` is an environment snapshot, not a hash-pinned lock.
