"""Provenance-tagged parameters. Nothing empirical may enter the model without a status/source."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParameterStatus(StrEnum):
    OBSERVED = "observed"  # taken directly from a published/acquired source
    DERIVED = "derived"  # computed from observed inputs by a documented rule
    ASSUMED = "assumed"  # a modelling choice, NOT an estimate
    SCENARIO = "scenario"  # a deliberate what-if value


class Param(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float | int | str | bool
    status: ParameterStatus
    source: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    definition: str = ""
    year: int | None = None
    uncertainty: str | None = None
    synthetic_demo_only: bool = False


@dataclass(frozen=True)
class RequiredParam:
    name: str
    unit: str
    definition: str


REQUIRED_PARAMETERS: tuple[RequiredParam, ...] = (
    RequiredParam("adult_age", "years", "Age at or above which a person is an adult in household construction."),
    RequiredParam("working_age_min", "years", "Lowest age for which education / labour state / arrangement are assigned."),
    RequiredParam("min_parent_age_at_birth", "years", "Minimum parent-minus-child age accepted when linking children to parents."),
    RequiredParam("max_parent_age_at_birth", "years", "Maximum parent-minus-child age accepted when linking children to parents."),
    RequiredParam("couple_age_difference_sd", "years", "SD of partner-minus-head age difference used only to propose candidate couples."),
    RequiredParam("within_band_age_distribution", "none", "Single-year age distribution inside a published age band (only 'uniform' is implemented)."),
    RequiredParam("equivalence_first_adult_weight", "weight", "Equivalence-scale weight of the first adult."),
    RequiredParam("equivalence_additional_person_weight", "weight", "Weight of each additional person aged at/above the child-age threshold."),
    RequiredParam("equivalence_child_weight", "weight", "Weight of each child below the child-age threshold."),
    RequiredParam("equivalence_child_age_threshold", "years", "Age below which a household member counts as a child in the equivalence scale."),
)


def validate_parameters(params: dict[str, Param]) -> list[str]:
    """Return a list of human-readable problems (empty when the registry is complete)."""
    problems: list[str] = []
    for req in REQUIRED_PARAMETERS:
        if req.name not in params:
            problems.append(f"missing required parameter '{req.name}' ({req.definition})")
    for name, p in params.items():
        if p.status in (ParameterStatus.OBSERVED, ParameterStatus.DERIVED) and p.year is None and not p.synthetic_demo_only:
            problems.append(f"parameter '{name}' is {p.status.value} but has no reference year")
    return problems


def as_plain(params: dict[str, Param]) -> dict[str, Any]:
    return {k: v.model_dump(mode="json") for k, v in params.items()}
