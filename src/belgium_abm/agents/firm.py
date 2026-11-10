"""Firm table schema (spec section 3.3). Schema only -- no firms are created in Phase 1."""

from __future__ import annotations

from belgium_abm.agents.columns import ColumnSpec
from belgium_abm.agents.enums import LegalForm, values

_F = "float64"

FIRM_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("firm_id", "int64"),
    ColumnSpec("legal_form", "string", domain=values(LegalForm)),
    ColumnSpec("owner_person_ids", "string", nullable=True),
    ColumnSpec("ownership_shares", "string", nullable=True),
    ColumnSpec("sector", "string", nullable=True),
    ColumnSpec("employee_ids", "string", nullable=True),
    ColumnSpec("turnover", _F, nullable=True),
    ColumnSpec("intermediate_costs", _F, nullable=True),
    ColumnSpec("gross_payroll", _F, nullable=True),
    ColumnSpec("employer_contributions", _F, nullable=True),
    ColumnSpec("capital_stock", _F, nullable=True),
    ColumnSpec("debt", _F, nullable=True),
    ColumnSpec("retained_earnings", _F, nullable=True),
    ColumnSpec("corporate_tax", _F, nullable=True),
    ColumnSpec("cash", _F, nullable=True),
    ColumnSpec("active", "boolean"),
)
