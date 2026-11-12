"""Assumptions / open-dependencies report (spec section 13). Edit THIS module, not the generated doc."""

from __future__ import annotations

from dataclasses import dataclass

from belgium_abm.config import ModelConfig
from belgium_abm.io.registry import DatasetSource


@dataclass(frozen=True)
class Item:
    title: str
    status: str  # assumed | not_implemented | not_calibrated | missing_data
    text: str


STRUCTURAL_ASSUMPTIONS: tuple[Item, ...] = (
    Item("Households are built first, then persons", "assumed",
         "Each simulated household is drawn from a candidate pool generated per (region, household template). IPU re-weights the pool to the person-cell (region x sex x age band) and household-template targets, so the result matches published margins only to the extent the pool supports them; unsupported constraints are reported."),
    Item("Candidate member ages follow the region's age-sex structure", "assumed",
         "Heads and other adults are drawn from the adult part of the regional age-sex pmf; a partner's age is the head's plus a rounded normal difference (SD = `couple_age_difference_sd`); a child's age must satisfy the configured minimum/maximum parent-age-at-birth bounds against every parent. These rules only propose candidates -- they are NOT estimates of Belgian partnership or fertility behaviour."),
    Item("Single-year ages inside a published band are uniform", "assumed",
         "`within_band_age_distribution = uniform` (the only option implemented). Published bands (e.g. 5-year) hide the true within-band shape."),
    Item("Household types and relationships", "assumed",
         "Five household types with declared adult/child counts. Couples are modelled as head + partner (sexes drawn independently of each other, so same-sex couples occur by construction); in OTHER_MULTI households the relationship between adults is not represented; children are linked to the head (and partner, if any)."),
    Item("Education, labour state and arrangement are allocated independently of household context", "assumed",
         "Allocation uses the published conditional shares within region x age band (x education): systematic rounding per cell, then a deterministic count-space hill-climb that moves units between categories/cells (cell sizes fixed) so both the cells and their aggregated margins stay within about one unit of expectation. Sex is not a dimension of these tables; intra-household correlation (e.g. partners' education) is absent. Only the published margins are targeted; unpublished joint distributions are not."),
    Item("Work arrangement applies to EMPLOYED persons only", "assumed",
         "EMPLOYED includes employees, sole proprietors and BV owner-managers. The BV share cannot be observed from labour-force tables alone (see `statbel_enterprise_demography`)."),
    Item("Persons below working_age_min get no education / labour assignment", "assumed",
         "`education_level = NOT_ASSESSED`, `work_state` and `student_status` are null, `primary_work_arrangement = NONE`."),
    Item("Weights are uniform", "assumed",
         "Every simulated person represents the same number of real persons (target total / simulated total); households likewise."),
    Item("Integerisation", "assumed",
         "Weights are integerised by systematic rounding followed by within-template swap hill-climbing on a chi-square-like objective. Household counts per template are preserved; person-cell counts are near, not exactly equal to, the targets."),
    Item("Timing convention", "assumed",
         "Ages increment at the start of each model year before any transition (locked by tests); the annual schedule is demography -> education -> work and firms -> incomes and fiscal flows -> consumption, saving and assets -> measure. Income measures describe the year's flows; wealth measures are opening (start-of-year, after transitions) balances."),
    Item("Calibration tolerances", "assumed",
         "Warn/fail thresholds and small-cell rules in the configuration are choices, not empirical findings."),
)

DYNAMIC_MODEL: tuple[Item, ...] = (
    Item("Mortality and fertility", "observed",
         "Annual death probabilities (Belgian life table qx by sex and age) and age-specific fertility per woman come from Eurostat (`demo_mlifetable`, `demo_frate`), reference year = baseline year - 1. Fertility does not depend on partnership, education or income; no migration; the population therefore ages and slowly shrinks (a modelling choice, not a projection). Sex of newborns uses `male_birth_share` (assumed 0.5)."),
    Item("Education pathway", "derived",
         "Everyone enters school at 15 with LOW attainment; at 18 a share `1 - LOW_25_34` completes upper secondary and a share `HIGH_25_34 / (1 - LOW_25_34)` of those continues to tertiary (graduating HIGH at 22), both derived per region from the observed 25-34 education shares so cohorts reproduce the cross-section. Household wealth rank shifts the odds with an assumed logit slope (`education_wealth_sensitivity`). Dropout and repetition are absent."),
    Item("Labour-market flows", "derived",
         "Job loss (`job_separation_rate`) and the total employment-to-inactivity flow come from the public EU-LFS quarterly flow tables (`lfsi_long_q`, 8 quarters, Markov-annualised; ages 15-74, all statuses). The model's non-retirement labour-force exit is that total minus the modelled retirement flow. Job-finding and re-entry probabilities remain derived per region x age band x education so the 2025 cross-section is stationary; they are then VALIDATED against the observed unemployment-to-employment and inactivity-to-employment flows (reported in every run: the model is lower than observed, partly because the observed inactive stock includes students and retirees). Age-specific hazards and individual histories need restricted longitudinal data."),
    Item("Self-employment dynamics", "derived",
         "Exit rates are the observed enterprise death rates by legal form (`bd_l_form`: sole proprietorships ~10%, limited-liability ~2.5%); a closure is not always an owner leaving self-employment, and the BV rate already includes winding-up at retirement, which the model also generates (the combined channel is validated and runs high). Entry hazards are derived from the age profile of the observed shares so a cohort keeps the cross-sectional profile and are validated against observed enterprise birth rates. Entry odds rise with household wealth rank (`entry_wealth_sensitivity`, assumed) and high education and require start-up money from household assets (explicit credit constraint). The sole-proprietor / BV split rests on a rough proxy from VAT-enterprise legal-form counts."),
    Item("Household formation and dissolution", "assumed",
         "Partnering (age-assortative matching within region), separation (assets split, children stay with the head), leaving home, orphan adoption and inheritance (estates of households with no surviving adult go to adult children, else a random household of the region) use assumed rates; no divorce law, no inheritance tax, no remarriage differentials."),
    Item("Earnings and productivity", "derived",
         "Log earnings = education base + age polynomial + persistent skill + AR(1) residual. The age polynomial is fitted exactly through the three published age groups of mean hourly earnings (Structure of Earnings Survey 2022, `earn_ses22_14`; a cross-section, not a lifecycle path), the education ratios are observed medians (`earn_ses_pub2i`: medium/low 1.09, high/low 1.71), and `skill_sigma` is solved so the SD of log earnings equals the lognormal equivalent of the published low-wage share (`earn_ses_pub1s`). The AR(1) persistence and innovation (`wage_shock_*`) remain assumptions (no public longitudinal earnings data). One scale factor makes the median equivalised disposable income equal the published EU-SILC median. Self-employed owners receive the employee-equivalent labour cost as value added (`*_productivity_multiplier` = 1), with their own AR(1) business shock. Part-time shares and hours are observed (`lfsa_eppga`, `lfsa_ewhuis`)."),
    Item("BV mechanics", "assumed",
         "Surplus after accounting cost is split into director pay (`bv_director_pay_share`, taxed as earned income with self-employed contributions), corporate tax, dividends (`bv_payout_ratio`, final withholding tax) and retained earnings (firm equity, counted once in owners' net worth). Negative equity triggers an insolvency event with limited liability; retirement or voluntary exit liquidates the firm as a taxed dividend."),
    Item("Wealth, housing and consumption", "derived",
         "Initial financial assets, housing and mortgages follow stylised age-income profiles whose four parameters (`fin_assets_multiple_max`, `fin_assets_age_tau`, `house_to_income`, `renter_financial_asset_factor`) are calibrated to the ECB HFCS 2023 medians (overall, non-owners, by age of reference person); Gini, top shares, percentiles and the homeowner median are held out and reported (`validation_wealth_vs_hfcs.csv`): overall and older-age medians fit within about 10-30%, wealth of households under 35 and the lowest percentiles are not matched. Retired households are seeded from a share of career-average earnings. Consumption is a subsistence floor plus a marginal propensity (assumed) with a drawdown for retirees; the implied saving rate is validated against the national-accounts household saving rate. Borrowing is constrained, unpaid obligations become explicit `other_debt` credit events, and renters can buy a home with a down payment. House price growth is the 20-year average real growth (`tipsho10`)."),
    Item("Tax-benefit system", "scenario",
         "All fiscal rules come from a versioned policy YAML. The shipped `configs/policy_illustrative.yaml` is an ILLUSTRATIVE placeholder (every value `scenario`, `verified: false`, banner on every output): it is **not** Belgian law. Statutory rates, thresholds, regional child benefits and pension rules must be supplied from verified sources before any Belgian result is quoted."),
    Item("Distributional validation", "observed",
         "At baseline the simulated Gini, at-risk-of-poverty rate, S80/S20 and income-quintile shares are compared with published EU-SILC indicators (`validation_vs_published.csv`). Only the median is calibrated. The simulation overstates the lower tail (see report). The share of persons in very-low-work-intensity households matches the published figure, so this is not due to household worklessness; the poverty rate is dominated by the level of the placeholder minimum-income floor in the illustrative policy."),
)

NOT_IMPLEMENTED: tuple[Item, ...] = (
    Item("Verified Belgian tax-benefit rules", "missing_data", "Phase 2 engine and scenario machinery exist, but the policy file is illustrative. Needs verified statutory values (and BELMOD/EUROMOD benchmarks for validation)."),
    Item("In-kind benefits (schooling, healthcare)", "not_implemented", "`InKindLedger` exists as a tested building block; no in-kind values are attributed in the simulation."),
    Item("Migration, regional mobility", "not_implemented", "Population is closed apart from births and deaths."),
    Item("Behavioural response of labour supply, fertility, savings to policy", "not_implemented", "Only the explicit scenario knobs act; the fixed-behaviour (static) and re-run (dynamic) results differ only through the channels modelled (accrual, liquidation, firm insolvency, assets)."),
    Item("Tax incidence", "not_implemented", "Legal payer and economic beneficiary are kept separate; employer contributions are a firm cash flow and no pass-through to wages is assumed."),
    Item("Wealth and tail calibration (HFCS, top-1%)", "missing_data", "Needs restricted HFCS microdata or published HFCS tables; with ~1% of persons in the top percentile, tail results need many replicates and are sensitive."),
    Item("Sensitivity analysis over parameters", "not_implemented", "Seed-to-seed uncertainty bands are produced by `--replicates`; one-at-a-time parameter sensitivity is available through scenario `economy_overrides` but no automated ranking exists."),
    Item("Monthly cash flow, sector/wage feedback, endogenous rents, macro coupling", "not_implemented", "Spec phase 5."),
)

DATA_GAPS: tuple[str, ...] = (
    "Cross-dataset joint distributions (e.g. education x labour state x household type).",
    "Person-to-BV beneficial ownership and retained-profit attribution.",
    "Actual lifetime pension trajectories (accrual here is a stylised career-average rule).",
    "Correlated familial (intergenerational) wealth.",
    "Firm-household matching.",
    "Legal tax-policy complexity by tax year and region.",
    "Longitudinal behavioural parameters (transition hazards).",
)


def render_sources_markdown(registry: list[DatasetSource]) -> str:
    L = ["# Real datasets still required", "", "Status of the dataset registry (`data/sources.yaml`). Nothing listed as `not_acquired` has been loaded.", "",
         "| id | producer | access | status | needed for phase | model input | url |", "|---|---|---|---|---|---|---|"]
    for s in registry:
        L.append(f"| `{s.id}` | {s.producer} | {s.access}{' (microdata restricted)' if s.restricted_microdata else ''} | **{s.status}** | {s.needed_for_phase} | {s.model_input} | {s.url or 'not given in the specification'} |")
    L += ["", "Restricted data (EU-SILC research microdata, HFCS microdata, BELMOD administrative data) must be requested from the data owners; this project never fabricates or infers them from aggregates."]
    return "\n".join(L) + "\n"


def render_assumptions_markdown(cfg: ModelConfig | None, registry: list[DatasetSource]) -> str:
    L: list[str] = [
        "# Model assumptions and open dependencies",
        "",
        "> **Status: all phases implemented; calibration is partial.** Real 2025 Belgian aggregates drive the",
        "> population (see `docs/data_real.md`); the lifecycle economy, behaviour and the tax-benefit file are",
        "> assumptions or an illustrative placeholder. No Belgian tax rate, demographic frequency, transition",
        "> probability or income / wealth value is hard-coded or fabricated in the code: every number is read",
        "> from a provenance-tagged data or configuration file. **No claim about real Belgian policy effects is made.**",
        "",
        "Generated by `abm assumptions` from `belgium_abm/assumptions.py`; edit that module, not this file.",
        "",
    ]
    if cfg is not None:
        L += ["## 1. Configured parameters (provenance)", "", f"Configuration: `{cfg.name}`", "", "| parameter | value | status | unit | source | demo only |", "|---|---|---|---|---|---|"]
        for name in sorted(cfg.parameters):
            p = cfg.parameters[name]
            L.append(f"| `{name}` | {p.value} | {p.status.value} | {p.unit} | {p.source} | {'yes' if p.synthetic_demo_only else 'no'} |")
        L.append("")
    L += ["## 2. Structural modelling assumptions", ""]
    for it in STRUCTURAL_ASSUMPTIONS:
        L += [f"- **{it.title}** ({it.status}). {it.text}"]
    L += ["", "## 3. Dynamic model (Phases 2-4): what is observed, derived, assumed", ""]
    for it in DYNAMIC_MODEL:
        L += [f"- **{it.title}** ({it.status}). {it.text}"]
    L += ["", "## 4. Not implemented / still missing", ""]
    for it in NOT_IMPLEMENTED:
        L += [f"- **{it.title}** ({it.status}). {it.text}"]
    L += ["", "## 5. Known data gaps (spec section 8)", ""] + [f"- {g}" for g in DATA_GAPS]
    L += ["", "## 6. Dataset registry", "", render_sources_markdown(registry).split("\n", 2)[2]]
    return "\n".join(L).rstrip("\n") + "\n"
