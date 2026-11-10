"""Dataset registry (``data/sources.yaml``, spec section 8)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class DatasetSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    producer: str
    url: str | None = None
    reference_year: int | None = None
    geography: str = "Belgium"
    unit: str = ""
    dimensions: list[str] = []
    access: Literal["open", "restricted", "mixed"] = "open"
    file_path: str | None = None
    license_or_terms: str = "verify"
    variables_used: list[str] = []
    method_notes: str = ""
    model_input: str = ""
    needed_for_phase: int | None = None
    restricted_microdata: bool = False
    status: Literal["not_acquired", "acquired"] = "not_acquired"


def load_registry(path: str | Path) -> list[DatasetSource]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: registry must be a YAML list")
    sources = [DatasetSource.model_validate(item) for item in raw]
    ids = [s.id for s in sources]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate dataset ids")
    for s in sources:
        if s.status == "acquired" and (s.file_path is None or s.reference_year is None):
            raise ValueError(f"{path}: dataset '{s.id}' is marked acquired without file_path/reference_year")
    return sources
