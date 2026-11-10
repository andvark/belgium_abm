"""Compare targets with the generated population (cell, marginal and total level)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from belgium_abm.config import ModelConfig, ToleranceConfig
from belgium_abm.engines.sampling import band_index
from belgium_abm.io.loaders import TargetBundle

COLUMNS = ["table", "level", "dimensions", "key", "target", "simulated", "abs_diff", "rel_diff", "status"]


def _emit(rows: list[dict[str, Any]], table: str, cells: pd.DataFrame, dims: Sequence[str], marginals: Sequence[Sequence[str]], total: bool) -> None:
    """Append cell rows, then marginal rows (sums of cells over the other dimensions)."""

    def add(level: str, frame: pd.DataFrame, d: Sequence[str]) -> None:
        for key, r in frame.iterrows():
            k = key if isinstance(key, tuple) else (key,)
            rows.append({"table": table, "level": level, "dimensions": ",".join(d), "key": "|".join(str(x) for x in k), "target": float(r["target"]), "simulated": float(r["simulated"])})

    add("cell", cells.set_index(list(dims))[["target", "simulated"]], dims)
    for m in marginals:
        add("marginal", cells.groupby(list(m))[["target", "simulated"]].sum(), m)
    if total:
        rows.append({"table": table, "level": "total", "dimensions": "", "key": "ALL", "target": float(cells["target"].sum()), "simulated": float(cells["simulated"].sum())})


def _share_cells(sim: pd.DataFrame, group: list[str], cat: str, shares: pd.DataFrame) -> pd.DataFrame:
    """Expected counts = simulated group size x published share, versus simulated counts."""
    sizes = sim.groupby(group).size().rename("group_size").reset_index()
    counts = sim.groupby(group + [cat]).size().rename("simulated").reset_index()
    df = shares[group + [cat, "share"]].merge(sizes, on=group, how="left").merge(counts, on=group + [cat], how="left")
    df["group_size"] = df["group_size"].fillna(0)
    df["simulated"] = df["simulated"].fillna(0)
    df["target"] = df["group_size"] * df["share"]
    return df[group + [cat, "target", "simulated"]]


def build_comparisons(persons: pd.DataFrame, households: pd.DataFrame, targets: TargetBundle, cfg: ModelConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    f = cfg.generator.n_persons / targets.total_persons
    pop = targets.population
    starts = np.array(sorted(pop["age_min"].unique()), dtype=np.int64)

    sim = persons.assign(age_min=starts[band_index(persons["age"].to_numpy(), starts)])
    cells = pop.rename(columns={"count": "target"}).assign(target=lambda d: d["target"] * f)
    got = sim.groupby(["region", "sex_or_demographic_stratum", "age_min"]).size().rename("simulated").reset_index()
    cells = cells[["region", "sex_or_demographic_stratum", "age_min", "target"]].merge(got, how="left", on=["region", "sex_or_demographic_stratum", "age_min"]).fillna({"simulated": 0})
    _emit(rows, "population", cells, ["region", "sex_or_demographic_stratum", "age_min"], [["region", "age_min"], ["region", "sex_or_demographic_stratum"], ["region"]], True)

    hh = targets.households
    keys = ["region", "household_type", "n_adults", "n_children"]
    cells = hh.rename(columns={"count": "target"}).assign(target=lambda d: d["target"] * f)[keys + ["target"]]
    got = households.groupby(keys).size().rename("simulated").reset_index()
    cells = cells.merge(got, how="left", on=keys).fillna({"simulated": 0})
    _emit(rows, "households", cells, keys, [["region"], ["household_type"]], True)

    wa = cfg.working_age_min
    work = persons[persons["age"] >= wa]
    edu_t, lab_t, arr_t = targets.education, targets.labour_state, targets.work_arrangement
    es = np.array(sorted(edu_t["age_min"].unique()), dtype=np.int64)
    ls = np.array(sorted(lab_t["age_min"].unique()), dtype=np.int64)
    as_ = np.array(sorted(arr_t["age_min"].unique()), dtype=np.int64)

    s = work.assign(age_min=es[np.maximum(band_index(work["age"].to_numpy(), es), 0)])
    cells = _share_cells(s, ["region", "age_min"], "education_level", edu_t)
    _emit(rows, "education", cells, ["region", "age_min", "education_level"], [["region", "education_level"], ["education_level"]], False)

    s = work.assign(age_min=ls[np.maximum(band_index(work["age"].to_numpy(), ls), 0)])
    cells = _share_cells(s, ["region", "age_min", "education_level"], "work_state", lab_t)
    _emit(rows, "labour_state", cells, ["region", "age_min", "education_level", "work_state"], [["region", "age_min", "work_state"], ["region", "work_state"], ["work_state"]], False)

    emp = persons[persons["work_state"] == "EMPLOYED"]
    s = emp.assign(age_min=as_[np.maximum(band_index(emp["age"].to_numpy(), as_), 0)])
    cells = _share_cells(s, ["region", "age_min"], "primary_work_arrangement", arr_t)
    _emit(rows, "work_arrangement", cells, ["region", "age_min", "primary_work_arrangement"], [["region", "primary_work_arrangement"], ["primary_work_arrangement"]], False)

    if targets.tenure is not None:
        cells = _share_cells(households, ["region", "household_type"], "housing_tenure", targets.tenure)
        _emit(rows, "tenure", cells, ["region", "household_type", "housing_tenure"], [["region", "housing_tenure"], ["housing_tenure"]], False)

    df = pd.DataFrame(rows)
    df["abs_diff"] = (df["simulated"] - df["target"]).abs()
    df["rel_diff"] = df["abs_diff"] / np.maximum(df["target"], 1.0)
    df["status"] = ""
    return df[COLUMNS]


def classify(comparisons: pd.DataFrame, tol: ToleranceConfig) -> pd.DataFrame:
    out = comparisons.copy()
    status = []
    for level, target, ad, rel in zip(out["level"], out["target"], out["abs_diff"], out["rel_diff"], strict=True):
        is_cell = level == "cell"
        warn_t = tol.cell_rel_warn if is_cell else tol.marginal_rel_warn
        fail_t = tol.cell_rel_fail if is_cell else tol.marginal_rel_fail
        if ad <= tol.rounding_slack_units:
            status.append("pass")
        elif target < tol.min_expected_count:
            status.append("pass" if ad <= tol.small_cell_abs_tol else "warn")
        elif rel <= warn_t:
            status.append("pass")
        elif rel <= fail_t:
            status.append("warn")
        else:
            status.append("fail" if (not is_cell or tol.cells_can_fail) else "warn")
    out["status"] = status
    return out


def summarize(comparisons: pd.DataFrame) -> pd.DataFrame:
    return comparisons.groupby(["table", "level", "status"]).size().unstack("status", fill_value=0).reset_index()


def verdict(comparisons: pd.DataFrame, invariants_failed: int) -> str:
    if invariants_failed > 0 or (comparisons["status"] == "fail").any():
        return "FAIL"
    if (comparisons["status"] == "warn").any():
        return "WARN"
    return "PASS"


def sample_size_report(persons: pd.DataFrame, households: pd.DataFrame, cfg: ModelConfig, ipu_kish_n_eff: float) -> dict[str, Any]:
    n = len(persons)
    cell = persons.groupby(["region", "sex_or_demographic_stratum"]).size()
    small = [f"{r}|{s}" for (r, s), c in cell.items() if c < cfg.tolerance.min_group_size]
    return {
        "n_persons": n,
        "n_households": len(households),
        "real_persons_per_simulated_person": float(persons["weight"].iloc[0]),
        "ipu_weight_kish_effective_size": ipu_kish_n_eff,
        "expected_persons_in_top_1_percent": 0.01 * n,
        "tail_warning": "Tail (top 1%) estimates would rest on about "
        f"{0.01 * n:,.0f} simulated persons; tail calibration and sensitivity analysis are mandatory before any wealth-tail claim.",
        "groups_below_min_group_size": small,
        "min_group_size": cfg.tolerance.min_group_size,
    }
