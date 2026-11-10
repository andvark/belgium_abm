"""Annual schedule and timing convention (spec section 4). Locked by tests.

Convention: ages increment at the START of each model year (step 1, before any state
transition), so every transition in a year is evaluated at the person's new age. Births occur
in the demography step and the newborn has age 0 for the rest of that year.
"""

from __future__ import annotations

BIRTHDAY_CONVENTION = "start_of_year"

ANNUAL_SCHEDULE: tuple[str, ...] = (
    "load_policy_and_rng",
    "demography",  # birthday first, then mortality, births, partnerships, household changes
    "education_transitions",
    "work_and_firms",
    "market_incomes",
    "fiscal_cashflows",
    "consumption_and_saving",
    "asset_dynamics",
    "measure",
    "persist",
)

# Which of the steps above exist in this code base (Phase 1 implements none of them).
IMPLEMENTED_STEPS: tuple[str, ...] = ()
