"""Stylised economy parameters (all provenance-tagged ``assumed``) loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

    def with_values(self, **values: Any) -> EconomyConfig:
        params = dict(self.parameters)
        for k, val in values.items():
            if k not in params:
                raise KeyError(f"unknown economy parameter '{k}'")
            params[k] = params[k].model_copy(update={"value": val, "status": ParameterStatus.SCENARIO})
        return EconomyConfig(name=self.name, banner=self.banner, parameters=params)
