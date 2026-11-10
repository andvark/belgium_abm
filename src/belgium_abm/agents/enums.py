"""Controlled vocabularies (spec section 3)."""

from __future__ import annotations

from enum import StrEnum


class Sex(StrEnum):
    F = "F"
    M = "M"


class EducationLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    NOT_ASSESSED = "NOT_ASSESSED"  # below working_age_min: no education level is assigned


class WorkState(StrEnum):
    STUDENT = "STUDENT"
    EMPLOYED = "EMPLOYED"
    UNEMPLOYED = "UNEMPLOYED"
    INACTIVE = "INACTIVE"
    RETIRED = "RETIRED"


class WorkArrangement(StrEnum):
    NONE = "NONE"
    EMPLOYEE = "EMPLOYEE"
    SOLE_PROPRIETOR = "SOLE_PROPRIETOR"
    BV_OWNER_MANAGER = "BV_OWNER_MANAGER"


class HousingTenure(StrEnum):
    RENTER = "RENTER"
    MORTGAGED_OWNER = "MORTGAGED_OWNER"
    OUTRIGHT_OWNER = "OUTRIGHT_OWNER"
    OTHER = "OTHER"


class HouseholdType(StrEnum):
    SINGLE_ADULT = "SINGLE_ADULT"
    COUPLE = "COUPLE"
    COUPLE_WITH_CHILDREN = "COUPLE_WITH_CHILDREN"
    SINGLE_PARENT = "SINGLE_PARENT"
    OTHER_MULTI = "OTHER_MULTI"


class Role(StrEnum):
    HEAD = "HEAD"
    PARTNER = "PARTNER"
    CHILD = "CHILD"
    OTHER_ADULT = "OTHER_ADULT"


class LegalForm(StrEnum):
    BV_SRL = "BV_SRL"
    SOLE_PROPRIETOR_ACTIVITY = "SOLE_PROPRIETOR_ACTIVITY"
    OTHER_EMPLOYER = "OTHER_EMPLOYER"


def values(enum_cls: type[StrEnum]) -> tuple[str, ...]:
    return tuple(m.value for m in enum_cls)
