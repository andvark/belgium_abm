"""Versioned policy file -> typed rules, plus the pure vectorised tax/benefit functions (Phase 2).

No statutory value is built into the code: every number comes from a policy YAML file whose
blocks carry a ``status`` and ``source``. ``verified: false`` policies propagate a banner to all
reports. Overrides (``a.b.c: value`` dotted keys) implement policy scenarios.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from belgium_abm.policies.config import POLICY_COMPONENTS, MissingPolicyDataError


@dataclass(frozen=True)
class PolicyRules:
    name: str
    policy_year: int
    verified: bool
    banner: str
    components: dict[str, dict[str, Any]]
    overrides: tuple[str, ...] = ()

    # ---- construction -------------------------------------------------------------
    @staticmethod
    def from_dict(raw: dict[str, Any]) -> PolicyRules:
        comps = raw.get("components") or {}
        missing = [c for c in POLICY_COMPONENTS if c not in comps]
        if missing:
            raise MissingPolicyDataError("policy file lacks components: " + ", ".join(missing))
        for name, block in comps.items():
            if "status" not in block or "source" not in block:
                raise MissingPolicyDataError(f"policy component '{name}' needs 'status' and 'source' (no unsourced rates)")
        return PolicyRules(str(raw["name"]), int(raw["policy_year"]), bool(raw.get("verified", False)), str(raw.get("banner", "")), comps)

    @staticmethod
    def load(path: str | Path) -> PolicyRules:
        return PolicyRules.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def with_overrides(self, overrides: dict[str, Any], label: str | None = None) -> PolicyRules:
        comps = copy.deepcopy(self.components)
        applied = []
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node: Any = comps
            for p in parts[:-1]:
                if not isinstance(node, dict) or p not in node:
                    raise KeyError(f"policy override '{dotted}': no such path")
                node = node[p]
            if not isinstance(node, dict) or parts[-1] not in node:
                raise KeyError(f"policy override '{dotted}': no such parameter")
            node[parts[-1]] = value
            applied.append(f"{dotted}={value}")
        banner = self.banner if not applied else self.banner + f" | SCENARIO OVERRIDES: {', '.join(applied)}"
        return PolicyRules(label or self.name, self.policy_year, self.verified, banner, comps, tuple(self.overrides) + tuple(applied))

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.components, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def g(self, component: str, key: str) -> Any:
        return self.components[component][key]


# ----------------------------------------------------------------- pure functions ----------
def progressive_tax(taxable: np.ndarray, brackets: list[list[Any]]) -> np.ndarray:
    """Tax on ``taxable`` (>=0 clipped) with marginal ``[upper, rate]`` brackets (upper None = open)."""
    x = np.clip(np.asarray(taxable, dtype=float), 0.0, None)
    tax = np.zeros_like(x)
    lower = 0.0
    for upper, rate in brackets:
        hi = np.inf if upper is None else float(upper)
        tax += np.clip(np.minimum(x, hi) - lower, 0.0, None) * float(rate)
        lower = hi
        if not np.isfinite(hi):
            break
    return tax


def personal_income_tax(base: np.ndarray, n_adult_allowances: np.ndarray, n_child_allowances: np.ndarray, rules: PolicyRules) -> np.ndarray:
    c = rules.components["personal_income_tax_schedule"]
    taxable = base - n_adult_allowances * c["basic_allowance_eur"] - n_child_allowances * c["child_allowance_eur"]
    return progressive_tax(taxable, c["brackets"])


def self_employed_contribution(profit_or_pay: np.ndarray, rules: PolicyRules, active: np.ndarray) -> np.ndarray:
    c = rules.components["self_employed_social_contributions"]
    raw = np.clip(profit_or_pay, 0.0, None) * c["rate"]
    return np.where(active, np.maximum(raw, c["min_annual_eur"]), 0.0)


def unemployment_benefit(prev_wage: np.ndarray, eligible: np.ndarray, rules: PolicyRules) -> np.ndarray:
    c = rules.components["unemployment_benefits"]
    amt = np.clip(prev_wage * c["replacement_rate"], c["min_annual_eur"], c["max_annual_eur"])
    return np.where(eligible, amt, 0.0)


def statutory_pension(accrued_annual: np.ndarray, rules: PolicyRules) -> np.ndarray:
    c = rules.components["pensions"]
    return np.clip(accrued_annual, c["min_annual_eur"], c["max_annual_eur"])
