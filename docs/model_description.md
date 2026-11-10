# Model description

Everything is vectorised pandas/numpy on person and household tables; state lives in
`engines/state.py::SimState`. All parameters are provenance-tagged (`observed`, `derived`, `assumed`,
`scenario`); see `docs/model_assumptions.md` for the full list.

## Entities

* **Person** - age, sex, region, education, work state (`STUDENT, EMPLOYED, UNEMPLOYED, INACTIVE, RETIRED`),
  primary arrangement (`EMPLOYEE, SOLE_PROPRIETOR, BV_OWNER_MANAGER`, only for the employed), persistent skill,
  earnings and business shocks, hours, wage memory, accrued pension rights, benefit eligibility.
* **Household** - region, tenure, housing value, mortgage, financial assets, other debt. Household type and
  counts are recomputed from members each year (single source of truth).
* **Firm** - one BV per BV owner-manager; equity = retained earnings (counted once in the owners' net worth).
  Sole proprietors are person-level activities. Employers of employees are an aggregate payer.
* **Government** - policy rules + integer-cent `FiscalLedger`; `InKindLedger` exists but no in-kind values are attributed.

## Annual schedule (`engines/schedule.py`)

1. **Demography** - age +1 for everyone (before any transition); deaths (life-table qx x sex); estates of
   households left without adults pass to adult children, else to a random household of the region; orphans are
   adopted; births (fertility per woman by age); separation; age-assortative partnering within region; leaving home.
2. **Education** - school entry at 15; at 18 upper-secondary completion and tertiary choice (probabilities derived
   from the observed 25-34 shares, shifted by household wealth rank); graduation at 22; school leavers enter the labour
   market with the observed 25-34 status mix.
3. **Work and firms** - AR(1) earnings and business shocks; retirement at the policy age (BV liquidated as a taxed
   dividend); job loss; job finding and labour-force re-entry (derived to keep the cross-section stationary); exits
   from self-employment; entry into self-employment (derived age-profile hazard x wealth/education odds, requires start-up money).
4. **Incomes and fiscal flows** (`engines/flows.py`, pure given state and policy)
   * employees: gross wage; employee contributions; employer contributions as an employer cash flow;
   * sole proprietors: value added - accounting cost = profit, self-employed contributions (with a minimum);
   * BV owner-managers: surplus after accounting cost -> director pay (taxed as earned income, self-employed
     contributions) -> corporate tax -> dividends (final withholding tax) and retained earnings (not household income);
   * unemployment benefit (replacement rate, bounds, duration limit), statutory pension (accrued rights, bounds),
     regional child benefit, personal income tax (allowances, progressive brackets), capital income tax,
     social-assistance top-up (equivalised floor, asset test), VAT on consumption.
5. **Consumption, saving, assets** - subsistence floor + marginal propensity, retiree drawdown, debt service,
   liquidity cap (shortfalls become explicit `other_debt` credit events), mortgage amortisation, house price growth,
   home purchase by renters (down payment), firm retained earnings and insolvency (limited liability), pension accrual.
6. **Measure** - see below.

## Measures (`analysis/measures.py`)

Primary unit: **household equivalised disposable income** (modified OECD scale: 1 / 0.5 for others aged >= 14 /
0.3 for children) mapped to every person, uniform person weights. Reported: Gini of market income with public
pensions as transfers and as deferred income, of disposable income and of income after VAT; P90/P10 (NaN when P10 <= 0);
S80/S20; poverty rate (60% of median); household net-worth Gini and top-10% / top-1% shares (negative worth excluded from
shares); pension entitlements separately; fiscal flows by category; income-quintile mobility; intergenerational
wealth-rank correlation (children at baseline vs own household rank 25+ years later); group cuts by age, region,
education, arrangement, children, tenure, current and baseline wealth quintile with small-sample warnings.

## Scenarios, replicates, uncertainty

`abm simulate --scenario X --replicates N` runs, for each seed, the baseline and the scenario from the same initial
state with common random numbers. While the baseline runs, the scenario policy is also evaluated on the unchanged
baseline state each year (**static / fixed-behaviour** effect). `compare` reports paired differences (mean and 5-95%
across replicates): `dynamic_total`, `static_fixed_behaviour` and their difference labelled
`behavioural_residual_NOT_CAUSAL`. Replicates also regenerate the synthetic population.

## Invariants (spec section 11), where they are tested

| item | test |
|---|---|
| 1 persons <-> households, valid parent ages | `engines.simulation.check_state`, every year in `tests/integration/test_simulation.py` |
| 2 exclusive states; arrangement only if employed; retirement | `check_state` |
| 3 ledger reconciles to the cent and ties to flows | `test_ledger_reconciles_and_ties_to_flows` |
| 4 wages/dividends paid = received; contributions not doubled | `test_firm_flows_match_owner_income`, `test_employer_contributions_are_not_household_income` |
| 5 BV equity once | `test_bv_equity_counted_once_in_net_worth` |
| 6 household cash identity | `max_cash_identity_residual` < 1e-6 |
| 7 no negative stocks | `check_state`, `test_household_cash_identity_and_no_negative_stocks` |
| 9 bit-identical reruns | `test_deterministic_for_identical_input` |
| 10 baseline = identical override | `test_baseline_equals_identical_policy_override` |
| 11 fixed-behaviour isolation | `test_static_shadow_isolates_mechanical_effect` |
| 12 Gini toy cases | `tests/unit/test_inequality.py` |
| 13 brackets / caps / thresholds | `tests/unit/test_policy_rules.py` |
