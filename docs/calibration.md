# Calibration method (Phase 1)

> The method below is the same for fictional and real inputs. `configs/demo_synthetic.yaml` uses
> FICTIONAL tables (`data/demo/`); `configs/belgium_baseline.yaml` uses the real 2025 tables built by
> `abm fetch-data` (`data/real/belgium_2025/`, see `docs/data_real.md`; the labour, household,
> work-arrangement and tenure tables are *derived*, not observed joint distributions).
> On the real tables a 20,000-person run passes all 832 comparison rows with no invariant failure.

## 1. Inputs

Six annotated CSV tables (header lines `# key: value` carry `dataset_id`, `synthetic_demo_only`,
`status`, `reference_year`, `unit`, `source`, `description`). Layouts:

| table | columns |
|---|---|
| population | `region, sex_or_demographic_stratum, age_min, age_max, count` |
| households | `region, household_type, n_adults, n_children, count` |
| education | `region, age_min, age_max, education_level, share` |
| labour_state | `region, age_min, age_max, education_level, work_state, share` |
| work_arrangement | `region, age_min, age_max, primary_work_arrangement, share` |
| tenure (optional) | `region, household_type, housing_tenure, share` |

Loader rules (`io/loaders.py`): layout and domains are checked; duplicates, negative values and
shares that do not sum to 1 within their group are refused; age bands must be contiguous and
identical across groups and must cover `working_age_min` to the oldest age; the labour table
must contain every region × age band × education combination; households and population must
agree on the number of adults and children (default tolerance 1%). Synthetic tables are
refused unless `allow_synthetic_demo_data: true`; empty templates are always refused; real
tables need a `reference_year` equal to `baseline_year`; synthetic and real tables cannot be
mixed. Empty real-data templates live in `data/templates/` (`abm make-templates`).
`configs/belgium_baseline.yaml` points at `data/real/belgium_2025/*`; if a file is missing the run exits
with code 2 and lists the missing files (real data is never substituted).

## 2. Generation pipeline (`engines/population.py`)

1. **Candidate pool.** For every household template (region × type × adults × children) draw a
   pool of candidate households whose member sexes and ages follow the region's age/sex
   structure under the couple and parent–child age rules (parameters in the config, each with a
   provenance status). Candidates that violate the parent-age bounds are redrawn.
2. **IPU** (iterative proportional updating, sparse) re-weights the pool so weighted
   person-cell (region × sex × age band) and household-template margins match the targets.
   Convergence, unsupported constraints and the Kish effective size are reported.
3. **Integerisation.** Systematic rounding of the weights to household counts per template,
   then within-template swap hill-climbing on a chi-square-like objective (template counts fixed).
4. **Attribute allocation** (education, labour state, work arrangement, tenure) from the
   published conditional shares: systematic rounding per cell followed by a count-space
   hill-climb that moves units between categories/cells (cell sizes fixed) so both cells and
   their margins track expectation.
5. Household counts are scaled so the person total is close to `generator.n_persons`.
   All randomness comes from named substreams of one seed (`rng.py`), so output CSVs are
   byte-identical for the same config, inputs, seed and environment (sha256 in `manifest.json`).

Members of a household are independent apart from the age/sex/role rules above; this and other
simplifications are listed in the assumptions report.

## 3. Comparison and tolerances

`calibration_comparison.csv` holds one row per target at three levels (cell, marginal, total),
with target, simulated, absolute and relative difference, and a status. Targets for the
attribute tables are *conditional*: simulated group size × published share.

| setting | demo default | meaning |
|---|---|---|
| `marginal_rel_warn / fail` | 1% / 5% | marginal and total relative error bands |
| `cell_rel_warn / fail` | 10% / 25% | cell bands |
| `cells_can_fail` | false | cells can at worst WARN |
| `min_expected_count` | 20 | smaller rows use the absolute rule |
| `small_cell_abs_tol` | 3 | absolute tolerance for small rows |
| `rounding_slack_units` | 2 | differences ≤ this many units count as integer-rounding noise |

Verdict: **FAIL** if any marginal/total row fails or any structural invariant fails; **WARN**
if only warnings; otherwise **PASS**. The tolerances are choices for the demo, not empirical
findings; set them deliberately when real data arrive.

## 4. Structural invariants

Run by `abm init` and `abm validate` (`analysis/validation.py`): schema and domains, unique ids,
referential integrity (members ↔ households, parents, partners), exactly one head, children =
members below `adult_age`, parent age gaps within the configured bounds, symmetric partners,
household composition and type, age range, education/labour/arrangement consistency, uniform
weights, and that no income or wealth value was invented. The exact count is printed by the
command. Spec §11 checks that need later phases are listed as *not yet applicable* in the report.

## 5. What this does not do

The population is validated only against the aggregates supplied. Agreement with *derived* targets
(household mix, labour state by education) reflects the reconstruction method, not an independent
observation. Income, wealth, taxes and behaviour belong to the dynamic model (`docs/model_description.md`).
