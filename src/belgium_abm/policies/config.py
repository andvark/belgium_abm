"""PolicyConfig skeleton (spec section 3.4).

No statutory value is stored here. Every component is empty until a verified, versioned policy
file is supplied (Phase 2). ``require_complete`` makes any attempt to run a fiscal calculation
without data fail loudly instead of silently using a made-up rate.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class MissingPolicyDataError(RuntimeError):
    pass


POLICY_COMPONENTS: tuple[str, ...] = (
    "personal_income_tax_schedule",
    "employee_social_contributions",
    "employer_social_contributions",
    "self_employed_social_contributions",
    "corporate_income_tax",
    "dividend_withholding_tax",
    "vat_by_consumption_group",
    "child_benefits_regionalised",
    "unemployment_benefits",
    "pensions",
    "disability_and_other_benefits",
    "targeted_rebates",
)


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_year: int | None = None
    components: dict[str, Any] = {}

    def missing_components(self) -> list[str]:
        return [c for c in POLICY_COMPONENTS if not self.components.get(c)]

    def require_complete(self) -> None:
        missing = self.missing_components()
        if self.policy_year is None:
            missing = ["policy_year", *missing]
        if missing:
            raise MissingPolicyDataError(
                "policy configuration is incomplete; no statutory values are built in. Missing: " + ", ".join(missing)
            )
