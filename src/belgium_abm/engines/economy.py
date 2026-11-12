"""Stylised economy parameters (all provenance-tagged ``assumed``) loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import yaml
from pydantic import BaseModel, ConfigDict

from belgium_abm.provenance import Param, ParameterStatus


class EconomyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    banner: str
    parameters: dict[str, Param]

    @staticmethod
    def load(path: str | Path) -> EconomyConfig:
        return EconomyConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def v(self, name: str) -> Any:
        return self.parameters[name].value

    def with_derived(self, source: str, **values: Any) -> EconomyConfig:
        """Replace values with ones DERIVED at run time (e.g. solved to match a published target)."""
        params = dict(self.parameters)
        for k, val in values.items():
            params[k] = params[k].model_copy(update={"value": val, "status": ParameterStatus.DERIVED, "source": source})
        return EconomyConfig(name=self.name, banner=self.banner, parameters=params)

    def apply_observed(self, table: pd.DataFrame) -> EconomyConfig:
        """Override assumptions with parameters from the public-data table (``behaviour_observed.csv``).

        Unknown names are added (they may be run-time inputs such as calibration targets). The education wage
        ratios set ``wage_base_medium/high`` relative to ``wage_base_low``.
        """
        params = dict(self.parameters)
        for r in table.itertuples(index=False):
            params[r.name] = Param(value=float(r.value), status=ParameterStatus(r.status), source=str(r.source), unit=str(r.unit), definition=str(r.definition), year=int(r.year))
        low = float(params["wage_base_low"].value)
        for lvl, key in (("medium", "wage_base_medium"), ("high", "wage_base_high")):
            ratio = params.get(f"education_wage_ratio_{lvl}")
            if ratio is not None:
                params[key] = Param(value=low * float(ratio.value), status=ParameterStatus.DERIVED, source=f"wage_base_low x {ratio.source}", unit=params[key].unit,
                                    definition=params[key].definition + " (level set by the observed education wage ratio)", year=ratio.year)
        return EconomyConfig(name=self.name, banner=self.banner, parameters=params)

    def with_values(self, **values: Any) -> EconomyConfig:
        params = dict(self.parameters)
        for k, val in values.items():
            if k not in params:
                raise KeyError(f"unknown economy parameter '{k}'")
            params[k] = params[k].model_copy(update={"value": val, "status": ParameterStatus.SCENARIO})
        return EconomyConfig(name=self.name, banner=self.banner, parameters=params)
