# Belgian Lifecycle Inequality ABM — Implementation Specification

**Version:** 0.1 (implementation brief)  
**Target coding agents:** Codex or Claude Code  
**Primary language:** Python 3.12+  
**Purpose:** Explain how Belgian income/wealth inequality emerges over life cycles, and how taxation, transfers, social insurance, and legal work arrangements change it.

## 1. Product goal

Simulate a representative synthetic Belgian population consisting of persons grouped into households, interacting with employer firms, personally owned BV/SRL companies, and a government fiscal system. At each annual time step agents change demographic and labour states, receive market income, pay taxes/contributions, receive cash transfers and publicly financed in-kind benefits (reported separately), consume, save, accumulate/deplete assets and pension entitlements. Compute inequality at different points of this process and compare policy scenarios against a shared baseline.

**Research questions:**
1. How do market-income, disposable-income, and net-wealth inequalities evolve by age, region, household type, and labour arrangement?
2. How much do taxes and transfers reduce or increase each measure, by group and over the life course?
3. How do employee, sole proprietor, and BV owner-manager paths differ in earnings risk, fiscal flows, retained assets, and lifetime outcomes?
4. How does initial family wealth influence education, entrepreneurship, housing access, and later wealth?
5. How do a policy's winners and losers change over time, and what is the confidence/sensitivity of the result?

**Explicitly out of scope for MVP:** electoral politics, global political economy, endogenous macro prices/wages, detailed healthcare utilization, comprehensive firm input-output markets, a general-equilibrium claim, optimizing national policy, normative single-number 'fairness'. Do not call the prototype a validated forecast.

## 2. Fundamental design principles

- **Four kinds of entities:** Person, Household, Firm, Government (the last can be a service/ledger rather than a strategic agent).
- **Orthogonal states:** age/education, activity/labour arrangement, household relationships, asset ownership are separate; 'family with children' and 'rentier' are not exclusive employment states.
- **Assets are owned by identifiable entities:** private savings/property by households or persons, business assets by firms. A BV owner's equity value is included when calculating beneficial household net worth; never also count the same company's assets directly as household assets.
- **Keep legal payer and economic beneficiary/incidence separate.** Employer social contributions appear as a firm cashflow, not automatically an employee cash payment. Do not infer economic incidence without behavioral assumptions.
- **Distinguish fiscal cashflow and service value.** Public schooling/healthcare in-kind allocations do not increase spendable cash.
- **No imaginary empirical parameters:** each input is `observed`, `derived`, `assumed`, or `scenario`, with source, year, unit, definition, and uncertainty.
- **Accounting identity and common random numbers:** all financial transfers balance and policy comparisons use aligned seeds for counterfactual comparability.

## 3. Data models

### 3.1 Person
Required: `person_id`, `household_id`, `age`, `sex_or_demographic_stratum` (for demographic calibration, optional in policies), `region`, `education_level`, `student_status`, `work_state`, `primary_work_arrangement`, `working_hours_fraction`, `occupation_or_sector`, `skill_score`, `health_work_capacity`, `gross_employee_wage`, `self_employed_turnover`, `self_employed_costs`, `director_pay`, `dividends_received`, `other_capital_income`, `unemployment_eligibility`, `disability_eligibility`, `accrued_statutory_pension_rights`, `pension_income`, `alive`, `weight`.

`work_state` enum: `STUDENT`, `EMPLOYED`, `UNEMPLOYED`, `INACTIVE`, `RETIRED`, with optional dual flags for student-workers, working pensioners, and secondary self-employment. Do not classify a person simultaneously as EMPLOYED and UNEMPLOYED by accident. Distinguish ILO unemployment from registered unemployment and eligibility for benefits.

`primary_work_arrangement` enum: `NONE`, `EMPLOYEE`, `SOLE_PROPRIETOR`, `BV_OWNER_MANAGER`. This is **not** a claim that all independent workers use a BV. Side activities may be modeled as additional income streams.

### 3.2 Household
Required: `household_id`, `member_ids`, `region`, `housing_tenure` (`RENTER`, `MORTGAGED_OWNER`, `OUTRIGHT_OWNER`, `OTHER`), `housing_value`, `mortgage_balance`, `financial_assets`, `other_debt`, `shared_consumption_budget`, `num_dependents`, `relationship_graph` or equivalent. Keep child ages to determine allowances and needs. Household and person financial ownership must not overlap.

### 3.3 Firm
Required: `firm_id`, `legal_form` (`BV_SRL`, `SOLE_PROPRIETOR_ACTIVITY`, `OTHER_EMPLOYER`), `owner_person_ids`, `ownership_shares`, `sector`, `employee_ids`, `turnover`, `intermediate_costs`, `gross_payroll`, `employer_contributions`, `capital_stock`, `debt`, `retained_earnings`, `corporate_tax`, `cash`, `active`.

Sole proprietors are treated as person-level economic activity (optionally a bookkeeping firm record) and **not** taxed as a distinct legal corporate taxpayer. BV director remuneration is a cost to the BV subject to the applicable rules; corporate profit, retained earnings, and distributions remain distinct. Different legal-form and director rules must be parameterized by tax year.

### 3.4 Government and ledgers
Government `PolicyConfig` includes personal income tax schedule, employee/employer/self-employed social contributions, corporate income tax, dividend withholding tax, VAT by consumption group (simplified MVP), child benefits (regionalized), unemployment benefits, pensions, disability/other benefits, and targeted rebates. Distinguish federal, social insurance, region/community if data permit.

`FiscalLedger` records transfers at entity level with `period`, `payer_id`, `recipient_id`, `category`, `amount_eur`, `policy_year`. Never credit the same benefit twice. A separate `InKindLedger` tracks attributed schooling/healthcare cost/value, with no effect on household liquid cash.

## 4. Model schedule — one year

0. Load policy and economic assumptions and create a deterministic RNG substream per module.
1. **Demography:** birthday, mortality, births, partnerships, separation, household membership, dependents, migration if enabled.
2. **Education and transitions:** entry/completion, education upgrade, labour-market entry; avoid drawing new status independently of the previous one.
3. **Work and firms:** unemployment/job finding, wage progression, entrepreneurship entry/exit, hiring, business revenue/costs, firm failure. Allow switching employee ↔ sole proprietor ↔ BV manager where valid.
4. **Market incomes:** earned wages, sole-proprietor net earnings, director remuneration, dividends *actually distributed*, rents, interest, investment returns. Retained BV profit **must not** be counted as household disposable income; it may contribute to owner equity wealth.
5. **Fiscal cashflows:** calculate person/household/firm taxes and contributions; allocate benefit eligibility; disburse benefits and pensions; reconcile ledgers.
6. **Consumption and saving:** model minimum subsistence, household economies of scale, propensities varying by age/wealth/income, debt service, optional housing purchase. Prevent negative cash without explicit credit/default rules.
7. **Asset dynamics:** cash, housing, mortgages, debt, business equity, pension accrual. Deduct payouts from firm equity/cash before crediting shareholders. Apply valuations consistently.
8. **Measure:** pre-tax market-income, post-tax cash disposable-income, consumption, household equivalized income, net worth, pension entitlement (separate), poverty, Gini, P90/P10, top 1/10% wealth shares, transition/mobility, and fiscal incidence.
9. **Persist:** annual agent snapshots (optional), normalized events/ledgers, calibration diagnostics, scenario metadata.

Specify a clear convention for whether births/transitions occur at the beginning or end of year; tests must lock the chosen ordering.

## 5. Inequality and distribution metrics

- Primary income distribution unit: **household equivalized disposable income**, mapped to individuals (modified OECD equivalence scale: 1 for first adult, 0.5 for additional persons aged ≥14, 0.3 for children <14); also report per-household and individual results where meaningful.
- Pre-government: market income **before direct personal taxes and cash transfers**, documenting treatment of pensions. Publish both variants that (a) count public pensions as transfers and (b) count pensions as deferred income, because rankings change materially.
- Post-government: cash disposable income = market cash incomes + cash benefits − direct personal taxes − relevant person/household social contributions. VAT is indirect and should appear in a separate *post-consumption-tax* metric. Do not subtract employer contribution twice.
- Wealth: household net worth = financial assets + property equity + beneficial BV equity + other assets − liabilities. Report pension promises separately, not as readily marketable net worth.
- Metrics: weighted Gini, quantiles/deciles, P90/P10 (if denominator positive), top 10%/1% wealth ownership shares, poverty rate vs 60% of median equivalized disposable income, absolute poverty/survival flag, transition matrices, intergenerational rank correlation (if parent-child links available), net fiscal balance by group and birth cohort.
- Group cuts: age bands, region, education, work arrangement, household children count, renter/owner, baseline-wealth quintile and current-wealth quintile. Warn on small effective sample sizes and overlapping groups.
- Decompose policy changes into (i) direct tax-benefit effect under fixed behavior and (ii) indirect behavioral effect by re-running the ABM; do not claim causal decomposition without explicit assumptions.
- Report uncertainty across random seeds, parameters, and calibration alternatives; show interval bands and sensitivity rankings.

## 6. Transition and decision functions

All probabilities are configurable, calibrated where possible, and conditional on selected covariates:
- education: `P(complete | age, prior_education, region, household_resources)`
- labour flows: `P(next_work_state | current_state, age, education, region, sector, shocks)`
- work arrangement: `P(enter_sole_prop / enter_BV | current_state, liquid_wealth, education, business_opportunity, dependents, risk_preferences)`
- employment earnings: `wage_growth(age, education, sector, tenure, shock)`
- business: `business_revenue`, `firm_survival`, `hire`, `payout_ratio`, `investment`, conditioned on size/sector and scenario
- households: `P(partnering, separation, childbearing | age, existing_children, region, ...)`
- savings: consumption floor + marginal propensity to consume, with constraints on borrowing, debt service, and liquidity.

Use interpretable hazard models/conditional transition tables in MVP, **not** black-box reinforcement learning. Document when transition coefficients are assumptions rather than estimated from longitudinal data. Keep entry costs and credit constraints explicit to reveal opportunity inequality.

## 7. Initialization: build joint synthetic population

1. Choose a **single baseline year** with the best overlap of demographic, labour, income and wealth tables; do not mix releases silently. Where datasets differ in year, document and uprate with explicit factors.
2. Construct households first from household-type and size tables. Allocate ages/relationships consistently, then assign individual education, labour and work arrangement using conditional distributions.
3. Calibrate to joint margins available (age × sex × region, household type × region, labour status × age × education × region) using iterative proportional fitting/raking or constrained synthetic reconstruction.
4. Seed income and wealth from EU-SILC/HFCS distributions with conditional rank correlation/coupling; do **not** assume wealth independent of age, tenure or income.
5. Create firms with plausible size and legal-form margins, attach employees and owners, reconcile payroll with person-level income. Initial stocks and flows must balance.
6. Generate 100k persons by default, optionally weighted; check effective sample sizes and stochastic variability for 1% tails. Because 100k persons yields only about 1k people in the top 1%, tail calibration/sensitivity is mandatory.
7. Emit a machine-readable calibration report comparing every target vs simulated estimate; fail hard for identity violations, warn or fail for designated tolerance bands.

**Privacy:** never infer individual records from aggregate sources. Eurostat EU-SILC *public-use files* are synthetic and are mainly for tooling/schema prototyping; scientific research microdata require application/authorization. BELMOD administrative microdata require controlled access.

## 8. Calibration dataset registry and provenance

Store dataset manifest as e.g. `data/sources.yaml`:

```yaml
- id: statbel_population
  producer: Statbel
  url: https://statbel.fgov.be/en/open-data
  reference_year: null  # set when acquired
  geography: Belgium
  unit: persons
  dimensions: [age, sex, municipality, nationality]
  access: open
  file_path: null
  license_or_terms: verify
  variables_used: []
  method_notes: ''
```

**Recommended official source inventory**

| Model input | Preferred source | Access / limitation |
|---|---|---|
| Population age, sex, residence | Statbel Open Data — population by age/sex/place | Public aggregate counts and downloadable files |
| Household composition, children | Statbel household type tables | Public aggregate counts; joint person-household detail not necessarily available |
| Employment, unemployment, inactivity, education, labour transitions | Statbel Labour Force Survey (LFS) | Aggregates/tables public; research microdata require controlled access; check definitions |
| Employee vs self-employed, occupations | Statbel LFS occupations / labour status | Published aggregates; BV distinction usually requires business/legal sources |
| Incomes, cash benefits, poverty, household budget variables | EU-SILC (Eurostat/Statbel) | Published aggregates; synthetic public use files are not actual Belgian research records; scientific microdata restricted |
| Private wealth, homeownership, mortgages, businesses | HFCS (National Bank Belgium / ECB) | Published results; microdata access subject to research conditions |
| Tax-benefit laws and baseline estimates | BELMOD documentation / EUROMOD Belgium country reports / official Belgian tax and social-security agencies | Documentation public; BELMOD administrative input controlled |
| Company counts, births, deaths, size, legal form | Statbel enterprise demography | Public tables; check sector coverage and enterprise vs legal-unit definitions |
| National macro totals, IO sector constraints | Federal Planning Bureau 2022 input-output tables / NBB national accounts | Public aggregate tables |
| Fertility, deaths, population projections | Statbel vital statistics / Federal Planning Bureau demographic projections | Public aggregates / projection assumptions |

Source URLs:
- https://statbel.fgov.be/en/open-data
- https://statbel.fgov.be/en/themes/population/structure-population/households
- https://statbel.fgov.be/en/themes/work-training/labour-market/employment-and-unemployment
- https://statbel.fgov.be/en/themes/work-training/labour-market/occupations-belgium
- https://ec.europa.eu/eurostat/web/microdata/collections-research/european-union-statistics-on-income-and-living-conditions
- https://ec.europa.eu/eurostat/en/web/microdata/public-microdata/statistics-on-income-and-living-conditions
- https://www.nbb.be/en/publications-and-research/publications/other-publications/household-finance-and-consumption-survey-0
- https://socialsecurity.belgium.be/en/shaping-social-policy/belmod/what-belmod
- https://statbel.fgov.be/en/themes/enterprises/demography-enterprises/demography-enterprises-market-sector
- https://www.plan.be/en/data/input-output-tables-2022

**Known data gaps to document prominently:** cross-dataset joint distributions, person-to-BV beneficial ownership, retained profit attribution, actual lifetime pension trajectories, correlated familial wealth, firm/household matching, legal tax-policy complexity, and longitudinal behavior parameters. Use explicit model assumptions, not invented observations.

## 9. MVP simulation phases and milestones

### Phase 0 — scaffold and provenance (deliverable: reproducible project)
- Python package with `pyproject.toml`, pinned lockfile, typing, tests, CLI, documentation, dataset registry and config schema.
- No hardcoded Belgian tax percentages, demographic frequencies, or simulated parameters hidden in code; test fixtures may use clearly labeled fake numbers.

### Phase 1 — synthetic population and descriptive baselines
- Generate persons and households from supplied CSV aggregates; calibrated cross-tabs and population pyramid.
- CSV/Parquet outputs: simulated agents, households, comparison of population targets and simulation.
- CLI: `abm init --config configs/belgium_baseline.yaml --seed 42`.

### Phase 2 — static tax-benefit and distribution engine
- Pure functions calculating household/person/firm fiscal flows from versioned policy config; validate against official example cases and, where accessible, BELMOD/EUROMOD benchmarks.
- Produce before/after income distribution and fiscal transfer incidence.

### Phase 3 — dynamic lifecycle model
- Implement annual demographic/work transitions, firm income and wealth evolution; 30-year trajectories; cohorts and transition matrices.

### Phase 4 — scenarios, uncertainty and reports
- Policy JSON/YAML overrides, common random numbers, 10+ independent replicates, group comparisons, CSV outputs and charts.
- CLI `abm simulate --config ... --scenario scenarios/corporate_tax_plus_5pp.yaml --years 30 --seed 42` and `abm compare --baseline ... --scenario ...`.

### Phase 5 (later) — improvements
- Monthly cashflow realism; sector and wage feedback; endogenous rents/housing; immigration; enhanced pension accrual; calibration of elasticities; formal distributional macro/ABM coupling.

## 10. Suggested Python layout

```text
belgian-inequality-abm/
  README.md
  pyproject.toml
  configs/belgium_baseline.yaml
  scenarios/
  data/raw/.gitkeep           # acquired data; never commit restricted microdata
  data/processed/.gitkeep
  data/sources.yaml
  src/belgium_abm/
    agents/{person,household,firm}.py
    policies/{taxes,contributions,benefits,pensions}.py
    engines/{population,demography,labour,firms,consumption,wealth,simulation}.py
    analysis/{inequality,fiscal_incidence,mobility,validation}.py
    io/{schemas,loaders,outputs}.py
    cli.py
  tests/{unit,integration,properties}/
  notebooks/                       # optional, not production logic
  docs/model_assumptions.md
  docs/calibration.md
```

Use `numpy`, `pandas` (or polars), `scipy`, `pydantic`, `pyarrow`, `pytest`, and `typer`. `Mesa` optional: only adopt if its scheduler/features materially help; a vectorized population simulation plus explicit interaction objects may scale better for 100k persons. Deterministic, typed code is more important than framework choice.

## 11. Mandatory invariant, validation, and scenario tests

1. Every active person belongs to exactly one active household; no ghost members; valid parent/child ages.
2. Mutually exclusive primary activity states; retirement and pension eligibility conditions match policy.
3. Government receipts/payments reconcile exactly with tax/benefit ledgers to cent precision in deterministic fiscal tests (use decimal or integer cents for ledgers).
4. Firms' wages paid = persons' wages received; firms' dividends paid = owners' dividends received; social contributions are not doubled.
5. Total household wealth includes owned BV equity **once**, never BV cash/assets plus equity as separate household wealth.
6. Identity test: household opening cash + inflows − taxes − consumption − debt payments − investments = closing cash (allow explicit loans/asset transfers).
7. No unexplained negative stock values; if insolvency/default model active, expose events.
8. Population and household marginal calibration errors below configurable thresholds; summarize effective weighted samples.
9. Identical seed/config/input produces bit-identical canonical outputs on a pinned environment.
10. Baseline scenario and exact same policy override produce identical results.
11. Policy change using fixed behavior isolates the mechanical fiscal effects; endogenous rerun explicitly flagged.
12. Weighted Gini returns 0 for identical positive income; known toy cases match analytical result; low/negative market incomes handled with documented convention.
13. Unit and property tests for tax brackets, caps, thresholds, regional child benefits, and edge ages.
14. Every chart/report carries reference year, time horizon, base/scenario differences, weighting unit, data vintage, and uncertainty.

## 12. First working scenario

**Comparison:** employee earning salaried income versus a self-employed sole proprietor and a BV owner-manager with a comparable **economic activity/productivity**, not just equal gross salary. Give the BV adequate turnover to cover business operating expenses, accounting costs, employer-side charges where applicable, director pay, corporation tax and retained earnings. Track owner access to liquidity and personal versus corporate wealth; count actual dividends, not undistributed profits, as household cash income. Compare annually and over 30 years under different earnings paths, household structures, risk profiles and pension eligibility. The comparison should reveal tradeoffs, not presume any arrangement is advantaged.

## 13. Instructions to coding agent

Implement **Phase 0 and Phase 1 first**; do not pretend to have loaded real Belgian data until the files are genuinely acquired. Write CSV input templates and mock fixtures with `synthetic_demo_only: true`. Run tests before adding any calibrated estimates. Output an assumptions report identifying each currently uncalibrated transition, fiscal rule and data dependency. Ask for help obtaining restricted datasets instead of fabricating them. Do not use an LLM to invent Belgian statutory rates. Keep finance and demographics modular so they can later be replaced by verified modules.

**Definition of done for first PR:** user can install the package; run a documented demo command; generate a 10,000-person synthetic population; inspect demographic and household calibration diagnostics from configurable example inputs; run and pass unit tests; read a list of real datasets still required. No claims of empirical policy effect are made until real data and validated fiscal rules are integrated.
