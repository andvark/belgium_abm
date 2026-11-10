"""Structural invariants of a generated population (spec section 11.1 and friends)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from belgium_abm.agents.columns import ColumnSpec
from belgium_abm.agents.household import HOUSEHOLD_COLUMNS, UNINITIALISED_HOUSEHOLD_COLUMNS
from belgium_abm.agents.person import PERSON_COLUMNS, UNINITIALISED_PERSON_COLUMNS
from belgium_abm.config import ModelConfig


@dataclass(frozen=True)
class CheckResult:
    name: str
    spec: str
    passed: bool
    detail: str = "ok"


# Spec section 11 items that cannot be tested yet because the objects do not exist in Phase 1.
NOT_YET_APPLICABLE: tuple[tuple[str, str], ...] = (
    ("11.2", "retirement / pension eligibility matches policy (no pension rules yet)"),
    ("11.3", "government receipts/payments reconcile with ledgers (ledger unit-tested only; no fiscal engine)"),
    ("11.4", "firm wages = person wages; dividends paid = received (no firms or incomes yet)"),
    ("11.5", "BV equity counted once in household wealth (no wealth yet)"),
    ("11.6", "household cash identity (no cash flows yet)"),
    ("11.7", "no unexplained negative stock values (no stocks yet)"),
    ("11.10", "baseline vs identical policy override (no policy scenarios yet)"),
    ("11.11", "fixed-behaviour policy isolation (no policy scenarios yet)"),
    ("11.13", "tax bracket / cap / threshold property tests (no tax rules yet)"),
    ("11.14", "charts carry uncertainty bands (single realisation only)"),
)


def _schema_problems(df: pd.DataFrame, cols: tuple[ColumnSpec, ...], table: str) -> list[str]:
    probs = []
    if list(df.columns) != [c.name for c in cols]:
        probs.append(f"{table}: columns differ from the schema")
        return probs
    for c in cols:
        if str(df[c.name].dtype) != c.dtype and not (c.dtype == "string" and str(df[c.name].dtype) in ("string", "str")):
            probs.append(f"{table}.{c.name}: dtype {df[c.name].dtype} != {c.dtype}")
        if not c.nullable and df[c.name].isna().any():
            probs.append(f"{table}.{c.name}: nulls in a required column")
        if c.domain is not None:
            bad = set(df[c.name].dropna().astype(str)) - set(c.domain)
            if bad:
                probs.append(f"{table}.{c.name}: values outside domain: {sorted(bad)}")
    return probs


def check_population(persons: pd.DataFrame, households: pd.DataFrame, cfg: ModelConfig, max_age: int | None = None) -> list[CheckResult]:
    res: list[CheckResult] = []

    def add(name: str, spec: str, ok: bool, detail: str = "") -> None:
        res.append(CheckResult(name, spec, bool(ok), "ok" if ok else (detail or "violated")))

    adult_age, working_min = cfg.adult_age, cfg.working_age_min
    p, h = persons, households

    probs = _schema_problems(p, PERSON_COLUMNS, "persons") + _schema_problems(h, HOUSEHOLD_COLUMNS, "households")
    add("table schemas and domains match the specification", "3.1/3.2", not probs, "; ".join(probs[:5]))
    if probs:  # later checks assume a sane schema
        return res

    add("unique person and household ids", "11.1", p["person_id"].is_unique and h["household_id"].is_unique)
    hh_ids = set(h["household_id"])
    add("every person belongs to an existing household", "11.1", p["household_id"].isin(hh_ids).all())
    if not (p["person_id"].is_unique and h["household_id"].is_unique and p["household_id"].isin(hh_ids).all()):
        return res  # the remaining checks join on these keys and are meaningless (or ambiguous) without them
    sizes = p.groupby("household_id").size()
    add("no empty households", "11.1", set(sizes.index) == hh_ids)

    counts = h["member_ids"].str.count(";").to_numpy() + 1
    flat = pd.DataFrame({"hid": np.repeat(h["household_id"].to_numpy(), counts), "pid": [int(x) for m in h["member_ids"] for x in m.split(";")]})
    merged = flat.merge(p[["person_id", "household_id"]], left_on="pid", right_on="person_id", how="outer", indicator=True)
    members_ok = bool((merged["_merge"] == "both").all() and (merged["hid"] == merged["household_id"]).all())
    detail = ""
    add("household member_ids match person.household_id exactly (no ghost members)", "11.1", members_ok, detail)

    reg = p.merge(h[["household_id", "region"]], on="household_id", suffixes=("", "_h"))
    add("members share their household's region", "11.1", (reg["region"] == reg["region_h"]).all())

    heads = p[p["role"] == "HEAD"].groupby("household_id").size()
    add("exactly one HEAD per household", "11.1", len(heads) == len(h) and (heads == 1).all())
    head_ok = h.merge(p[["person_id", "role", "household_id"]], left_on="head_person_id", right_on="person_id", suffixes=("", "_p"))
    add("household.head_person_id points at its HEAD", "11.1", len(head_ok) == len(h) and (head_ok["role"] == "HEAD").all() and (head_ok["household_id"] == head_ok["household_id_p"]).all())

    add("children are exactly the members younger than adult_age", "11.1", ((p["role"] == "CHILD") == (p["age"] < adult_age)).all())

    known = set(p["person_id"])
    refs = pd.concat([p["parent1_id"], p["parent2_id"], p["partner_id"]]).dropna().astype(int)
    add("parent and partner ids refer to existing persons", "11.1", refs.isin(known).all())
    if not refs.isin(known).all():
        return res
    idx = p.set_index("person_id")
    for k, label in ((1, "first"), (2, "second")):
        pc = p[p[f"parent{k}_id"].notna()]
        par = idx.loc[pc[f"parent{k}_id"].astype(int), ["household_id", "age"]]
        gap = pc["age"].to_numpy() * -1 + par["age"].to_numpy()
        ok = (par["household_id"].to_numpy() == pc["household_id"].to_numpy()).all() and ((gap >= cfg.min_parent_age_at_birth) & (gap <= cfg.max_parent_age_at_birth)).all()
        add(f"{label} parent: same household, age gap within configured bounds", "11.1", ok)
    add("every child has a first parent; only children have parents", "11.1", ((p["role"] == "CHILD") == p["parent1_id"].notna()).all() and (p["parent2_id"].isna() | p["parent1_id"].notna()).all())

    pp = p[p["partner_id"].notna()]
    back = idx.loc[pp["partner_id"].astype(int), "partner_id"].to_numpy()
    same_hh = idx.loc[pp["partner_id"].astype(int), "household_id"].to_numpy() == pp["household_id"].to_numpy()
    add("partners are symmetric, adults and in the same household", "11.1", (back == pp["person_id"].to_numpy()).all() and same_hh.all() and (pp["age"] >= adult_age).all())

    comp = p.groupby("household_id").agg(n_adults=("age", lambda a: int((a >= adult_age).sum())), n_children=("age", lambda a: int((a < adult_age).sum())))
    hc = h.set_index("household_id").join(comp, rsuffix="_p")
    add("household n_adults / n_children / num_dependents match members", "3.2", (hc["n_adults"] == hc["n_adults_p"]).all() and (hc["n_children"] == hc["n_children_p"]).all() and (hc["num_dependents"] == hc["n_children_p"]).all())

    def type_ok(r: pd.Series) -> bool:
        a, c, t = int(r["n_adults"]), int(r["n_children"]), r["household_type"]
        return {"SINGLE_ADULT": a == 1 and c == 0, "COUPLE": a == 2 and c == 0, "COUPLE_WITH_CHILDREN": a == 2 and c >= 1, "SINGLE_PARENT": a == 1 and c >= 1, "OTHER_MULTI": a >= 3}[t]

    add("household type is consistent with its composition", "3.2", h.apply(type_ok, axis=1).all())

    top = max_age if max_age is not None else int(p["age"].max())
    add("ages are within [0, max_age]", "11.1", ((p["age"] >= 0) & (p["age"] <= top)).all())

    young = p["age"] < working_min
    add("education is NOT_ASSESSED exactly below working_age_min", "3.1", ((p["education_level"] == "NOT_ASSESSED") == young).all())
    add("work_state and student_status are null exactly below working_age_min", "11.2", (p["work_state"].isna() == young).all() and (p["student_status"].isna() == young).all())
    add("student_status agrees with work_state", "3.1", ((p["work_state"] == "STUDENT") == p["student_status"].fillna(False).astype(bool)).all())
    add("primary_work_arrangement is not NONE exactly for EMPLOYED persons", "3.1", ((p["primary_work_arrangement"] != "NONE") == (p["work_state"] == "EMPLOYED").fillna(False)).all())

    add("all persons alive; weights positive and uniform", "3.1", p["alive"].all() and (p["weight"] > 0).all() and np.isclose(p["weight"].min(), p["weight"].max()))
    unset = [c for c in UNINITIALISED_PERSON_COLUMNS if p[c].notna().any()] + [c for c in UNINITIALISED_HOUSEHOLD_COLUMNS if h[c].notna().any()]
    add("no income, wealth or eligibility values were invented (Phase 1 leaves them null)", "13", not unset, f"non-null: {unset}")
    return res


def summarize_checks(checks: list[CheckResult]) -> dict[str, int]:
    failed = sum(not c.passed for c in checks)
    return {"total": len(checks), "passed": len(checks) - failed, "failed": failed}
