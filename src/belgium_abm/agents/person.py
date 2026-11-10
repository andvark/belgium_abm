"""Person table schema (spec section 3.1) plus helper columns for household structure.

Phase 1 initialises demographic and labour-state variables only. Every income, pension and
eligibility column is declared (so the schema is final) but left NULL: no value is invented.
"""

from __future__ import annotations

from belgium_abm.agents.columns import ColumnSpec
from belgium_abm.agents.enums import EducationLevel, Role, Sex, WorkArrangement, WorkState, values

_F = "float64"

PERSON_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("person_id", "int64", description="unique person id"),
    ColumnSpec("household_id", "int64", description="household the person belongs to"),
    ColumnSpec("age", "int64", description="age in completed years at the baseline"),
    ColumnSpec("sex_or_demographic_stratum", "string", domain=values(Sex), description="demographic stratum used for calibration"),
    ColumnSpec("region", "string", description="region label from the configuration"),
    ColumnSpec("education_level", "string", domain=values(EducationLevel)),
    ColumnSpec("student_status", "boolean", nullable=True, description="NULL below working_age_min"),
    ColumnSpec("work_state", "string", nullable=True, domain=values(WorkState), description="NULL below working_age_min"),
    ColumnSpec("primary_work_arrangement", "string", domain=values(WorkArrangement), description="NONE unless EMPLOYED"),
    ColumnSpec("working_hours_fraction", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("occupation_or_sector", "string", nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("skill_score", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("health_work_capacity", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("gross_employee_wage", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("self_employed_turnover", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("self_employed_costs", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("director_pay", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("dividends_received", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("other_capital_income", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("unemployment_eligibility", "boolean", nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("disability_eligibility", "boolean", nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("accrued_statutory_pension_rights", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("pension_income", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("alive", "boolean"),
    ColumnSpec("weight", _F, description="number of real persons one simulated person represents"),
    # optional dual-activity flags (spec 3.1) -- not modelled in Phase 1
    ColumnSpec("dual_student_worker", "boolean", nullable=True),
    ColumnSpec("dual_working_pensioner", "boolean", nullable=True),
    ColumnSpec("dual_secondary_self_employed", "boolean", nullable=True),
    # household-structure helpers (the 'relationship graph' in person-level form)
    ColumnSpec("role", "string", domain=values(Role)),
    ColumnSpec("partner_id", "Int64", nullable=True),
    ColumnSpec("parent1_id", "Int64", nullable=True),
    ColumnSpec("parent2_id", "Int64", nullable=True),
)

# Columns deliberately left NULL in Phase 1 (checked by the validation suite).
UNINITIALISED_PERSON_COLUMNS: tuple[str, ...] = tuple(
    c.name
    for c in PERSON_COLUMNS
    if c.nullable and c.description.startswith("not initialised")
) + ("dual_student_worker", "dual_working_pensioner", "dual_secondary_self_employed")
