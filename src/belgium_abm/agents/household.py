"""Household table schema (spec section 3.2)."""

from __future__ import annotations

from belgium_abm.agents.columns import ColumnSpec
from belgium_abm.agents.enums import HouseholdType, HousingTenure, values

_F = "float64"

HOUSEHOLD_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("household_id", "int64"),
    ColumnSpec("member_ids", "string", description="';'-joined person ids"),
    ColumnSpec("region", "string"),
    ColumnSpec("household_type", "string", domain=values(HouseholdType)),
    ColumnSpec("n_adults", "int64"),
    ColumnSpec("n_children", "int64"),
    ColumnSpec("head_person_id", "int64"),
    ColumnSpec("housing_tenure", "string", nullable=True, domain=values(HousingTenure), description="NULL when no tenure table is supplied"),
    ColumnSpec("housing_value", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("mortgage_balance", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("financial_assets", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("other_debt", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("shared_consumption_budget", _F, nullable=True, description="not initialised in Phase 1"),
    ColumnSpec("num_dependents", "int64", description="children below adult_age (child ages stay on the person table)"),
    ColumnSpec("weight", _F, description="number of real households one simulated household represents"),
)

UNINITIALISED_HOUSEHOLD_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in HOUSEHOLD_COLUMNS if c.description.startswith("not initialised")
)
