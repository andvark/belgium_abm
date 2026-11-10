"""Layouts of the aggregate input tables that the population generator is calibrated to."""

from __future__ import annotations

from dataclasses import dataclass

from belgium_abm.agents.enums import (
    EducationLevel,
    HouseholdType,
    HousingTenure,
    Sex,
    WorkArrangement,
    WorkState,
    values,
)


@dataclass(frozen=True)
class TableSchema:
    table: str
    columns: tuple[tuple[str, str], ...]  # (name, kind) kind in {str, int, float}
    key: tuple[str, ...]  # columns that identify a row uniquely
    value_column: str  # 'count' or 'share'
    group_by: tuple[str, ...] | None  # shares must sum to 1 within these groups
    domains: dict[str, tuple[str, ...]]
    optional: bool
    default_dataset_id: str
    description: str


_EDU_LEVELS = (EducationLevel.LOW.value, EducationLevel.MEDIUM.value, EducationLevel.HIGH.value)
_ARRANGEMENTS = tuple(v for v in values(WorkArrangement) if v != "NONE")

TABLE_SCHEMAS: dict[str, TableSchema] = {
    "population": TableSchema(
        "population",
        (("region", "str"), ("sex_or_demographic_stratum", "str"), ("age_min", "int"), ("age_max", "int"), ("count", "float")),
        ("region", "sex_or_demographic_stratum", "age_min"),
        "count",
        None,
        {"sex_or_demographic_stratum": values(Sex)},
        False,
        "population_age_sex_region",
        "Persons by region x sex x age band",
    ),
    "households": TableSchema(
        "households",
        (("region", "str"), ("household_type", "str"), ("n_adults", "int"), ("n_children", "int"), ("count", "float")),
        ("region", "household_type", "n_adults", "n_children"),
        "count",
        None,
        {"household_type": values(HouseholdType)},
        False,
        "household_templates_region",
        "Households by region x household type x number of adults x number of children",
    ),
    "education": TableSchema(
        "education",
        (("region", "str"), ("age_min", "int"), ("age_max", "int"), ("education_level", "str"), ("share", "float")),
        ("region", "age_min", "education_level"),
        "share",
        ("region", "age_min"),
        {"education_level": _EDU_LEVELS},
        False,
        "education_by_age_region",
        "Share of persons in each education level, by region x age band",
    ),
    "labour_state": TableSchema(
        "labour_state",
        (("region", "str"), ("age_min", "int"), ("age_max", "int"), ("education_level", "str"), ("work_state", "str"), ("share", "float")),
        ("region", "age_min", "education_level", "work_state"),
        "share",
        ("region", "age_min", "education_level"),
        {"education_level": _EDU_LEVELS, "work_state": values(WorkState)},
        False,
        "labour_state_by_age_education_region",
        "Share of persons in each work state, by region x age band x education level",
    ),
    "work_arrangement": TableSchema(
        "work_arrangement",
        (("region", "str"), ("age_min", "int"), ("age_max", "int"), ("primary_work_arrangement", "str"), ("share", "float")),
        ("region", "age_min", "primary_work_arrangement"),
        "share",
        ("region", "age_min"),
        {"primary_work_arrangement": _ARRANGEMENTS},
        False,
        "work_arrangement_by_age_region",
        "Share of EMPLOYED persons in each primary work arrangement, by region x age band",
    ),
    "tenure": TableSchema(
        "tenure",
        (("region", "str"), ("household_type", "str"), ("housing_tenure", "str"), ("share", "float")),
        ("region", "household_type", "housing_tenure"),
        "share",
        ("region", "household_type"),
        {"household_type": values(HouseholdType), "housing_tenure": values(HousingTenure)},
        True,
        "tenure_by_household_type_region",
        "Share of households in each housing tenure, by region x household type (optional)",
    ),
}

METADATA_KEYS = ("dataset_id", "synthetic_demo_only", "status", "reference_year", "unit", "source", "description")
