"""Scenario files: policy overrides/deltas and economy knobs, applied on top of a baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from belgium_abm.engines.economy import EconomyConfig
from belgium_abm.policies.rules import PolicyRules


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str = ""
    policy_overrides: dict[str, Any] = field(default_factory=dict)
    policy_deltas: dict[str, float] = field(default_factory=dict)
    economy_overrides: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> Scenario:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        unknown = set(raw) - {"name", "description", "policy_overrides", "policy_deltas", "economy_overrides"}
        if unknown:
            raise ValueError(f"{path}: unknown scenario keys {sorted(unknown)}")
        return Scenario(raw["name"], raw.get("description", ""), raw.get("policy_overrides") or {}, raw.get("policy_deltas") or {}, raw.get("economy_overrides") or {})

    def apply_policy(self, base: PolicyRules) -> PolicyRules:
        ov = dict(self.policy_overrides)
        for dotted, delta in self.policy_deltas.items():
            node: Any = base.components
            for part in dotted.split("."):
                node = node[part]
            ov[dotted] = node + delta
        return base.with_overrides(ov, label=self.name) if ov else base.with_overrides({}, label=self.name)

    def apply_economy(self, base: EconomyConfig) -> EconomyConfig:
        return base.with_values(**self.economy_overrides) if self.economy_overrides else base

    @property
    def changes_behaviour_inputs(self) -> bool:
        """Retirement age changes behaviour (participation, accrual); pure tax-rate changes act mechanically first."""
        return bool(self.economy_overrides) or any(k.startswith("pensions.retirement_age") for k in self.policy_overrides)
