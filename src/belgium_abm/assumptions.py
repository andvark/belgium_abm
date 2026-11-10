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
         "Ages increment at the start of each model year before any transition (locked by tests); no dynamics are implemented yet."),
    Item("Calibration tolerances", "assumed",
         "Warn/fail thresholds and small-cell rules in the configuration are choices, not empirical findings."),
)

NOT_IMPLEMENTED: tuple[Item, ...] = (
    Item("Income, wealth, pensions, firms", "not_implemented", "Schemas exist; all values are left null. Seeding from EU-SILC/HFCS needs data that are not acquired."),
    Item("Tax and benefit rules", "not_implemented", "`PolicyConfig` is an empty skeleton. No statutory rate is stored anywhere; they must come from verified sources (never from an LLM)."),
    Item("Demographic transitions (mortality, fertility, partnership, separation, migration)", "not_calibrated", "No transition probabilities exist in the code."),
    Item("Education and labour-market transitions", "not_calibrated", "No transition probabilities exist in the code."),
    Item("Entrepreneurship entry/exit, firm dynamics, hiring", "not_calibrated", "Not implemented."),
    Item("Consumption, saving, housing purchase, credit constraints", "not_calibrated", "Not implemented."),
    Item("Fiscal ledgers in a simulation loop", "not_implemented", "`FiscalLedger` / `InKindLedger` are unit-tested building blocks only."),
    Item("Seed-to-seed uncertainty and sensitivity analysis", "not_implemented", "Reports describe a single realisation."),
)

DATA_GAPS: tuple[str, ...] = (
    "Cross-dataset joint distributions (e.g. education x labour state x household type).",
    "Person-to-BV beneficial ownership and retained-profit attribution.",
    "Actual lifetime pension trajectories.",
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
        "> **Status: Phase 0/1 only.** This repository builds and validates an initial synthetic",
        "> population. No Belgian tax rate, demographic frequency, transition probability or income /",
        "> wealth value is hard-coded or fabricated anywhere in the code. No claim about real Belgian",
        "> policy effects is made. Real input datasets are listed below as **not acquired**.",
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
    L += ["", "## 3. Not implemented / not calibrated", ""]
    for it in NOT_IMPLEMENTED:
        L += [f"- **{it.title}** ({it.status}). {it.text}"]
    L += ["", "## 4. Known data gaps (spec section 8)", ""] + [f"- {g}" for g in DATA_GAPS]
    L += ["", "## 5. Dataset registry", "", render_sources_markdown(registry).split("\n", 2)[2]]
    return "\n".join(L).rstrip("\n") + "\n"
