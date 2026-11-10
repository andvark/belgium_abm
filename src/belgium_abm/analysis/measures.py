"""Yearly distribution, fiscal-incidence, group and mobility measures (spec section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from belgium_abm.analysis.inequality import (
    p90_p10,
    poverty_rate,
    top_share,
    weighted_gini,
    weighted_quantile,
)
from belgium_abm.engines.flows import REVENUE, SPENDING, Flows, net_fiscal_balance
from belgium_abm.engines.state import SimState, net_worth

AGE_GROUPS = [(0, 17, "00-17"), (18, 29, "18-29"), (30, 44, "30-44"), (45, 59, "45-59"), (60, 74, "60-74"), (75, 200, "75+")]


def age_group(age: np.ndarray) -> np.ndarray:
    out = np.full(age.shape, "", dtype=object)
    for lo, hi, lab in AGE_GROUPS:
        out[(age >= lo) & (age <= hi)] = lab
    return out


def s80_s20(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cw = np.cumsum(w) / w.sum()
    contrib = v * w
    bottom = contrib[cw <= 0.2].sum()
    top = contrib[cw > 0.8].sum()
    return float(top / bottom) if bottom > 0 else float("nan")


def person_view(st: SimState, fl: Flows) -> pd.DataFrame:
    """One row per person with household equivalised incomes mapped to individuals."""
    p, H = st.persons, fl.h
    hid = p["household_id"]
    v = pd.DataFrame({"household_id": hid.to_numpy(), "age": p["age"].to_numpy(), "region": p["region"].to_numpy(),
                      "education": p["education_level"].to_numpy(), "work_state": p["work_state"].to_numpy(),
                      "arrangement": p["arrangement"].to_numpy(), "sex": p["sex_or_demographic_stratum"].to_numpy(),
                      "wq0": p["wq0"].to_numpy() if "wq0" in p else np.nan}, index=p.index)
    eq = H["eq_size"].reindex(hid).to_numpy()
    for col in ("market_a", "market_b", "disp"):
        v["eq_" + col] = H[col].reindex(hid).to_numpy() / eq
    v["eq_post_vat"] = (H["disp"] - H["vat"]).reindex(hid).to_numpy() / eq
    nw = net_worth(st)
    v["hh_net_worth"] = nw.reindex(hid).to_numpy()
    v["hh_n_children"] = st.households["n_children"].reindex(hid).to_numpy()
    v["tenure"] = st.households["housing_tenure"].reindex(hid).to_numpy()
    nfp = (H["direct_taxes"] + H["vat"] - H["transfers"]) if "direct_taxes" in H else (H["pit"] + H["ssc_personal"] + H["cap_tax"] + H["div_tax"] + H["vat"] - (H["child_ben"] + H["assist"] + H["pension"] + H["unemp"]))
    v["hh_net_fiscal_per_capita"] = (nfp.reindex(hid).to_numpy()) / st.households["n_adults"].add(st.households["n_children"]).reindex(hid).to_numpy()
    wr = nw.rank(pct=True)
    v["wealth_q"] = np.ceil(wr.reindex(hid).to_numpy() * 5).clip(1, 5)
    return v


def yearly_metrics(st: SimState, fl: Flows, label: str = "dynamic") -> dict[str, float | int | str]:
    v = person_view(st, fl)
    w = np.full(len(v), st.person_weight)
    eqd = v["eq_disp"].to_numpy()
    hh = st.households
    nw = net_worth(st).to_numpy()
    hw = np.full(len(hh), st.household_weight)
    pw_total_pension = float((st.persons["pension_rights"] * (st.persons["work_state"] == "RETIRED")).sum() * st.person_weight)
    work = st.persons["work_state"]
    active = work.isin(["EMPLOYED", "UNEMPLOYED"]) & (st.persons["age"] <= 74)
    arr = st.persons["arrangement"]
    emp = work == "EMPLOYED"
    out: dict[str, float | int | str] = {
        "scenario": label, "year": st.year, "n_persons_sim": len(v), "population": float(w.sum()), "n_households_sim": len(hh),
        "gini_market_pensions_as_transfers": weighted_gini(v["eq_market_a"].to_numpy(), w),
        "gini_market_pensions_as_income": weighted_gini(v["eq_market_b"].to_numpy(), w),
        "gini_disposable": weighted_gini(eqd, w),
        "gini_post_consumption_tax": weighted_gini(v["eq_post_vat"].to_numpy(), w),
        "p90_p10_disposable": p90_p10(eqd, w), "s80_s20_disposable": s80_s20(eqd, w),
        "poverty_rate_60pct_median": poverty_rate(eqd, w), "median_eq_disposable_income": float(weighted_quantile(eqd, 0.5, w)),
        "mean_eq_disposable_income": float(np.average(eqd, weights=w)),
        "median_eq_market_income": float(weighted_quantile(v["eq_market_a"].to_numpy(), 0.5, w)),
        "wealth_gini": weighted_gini(nw, hw), "wealth_top10_share": top_share(np.clip(nw, 0, None), 0.10, hw),
        "wealth_top1_share": top_share(np.clip(nw, 0, None), 0.01, hw), "wealth_median": float(weighted_quantile(nw, 0.5, hw)),
        "wealth_mean": float(np.average(nw, weights=hw)),
        "pension_entitlement_annual_retired_total": pw_total_pension,
        "unemployment_rate": float(((work == "UNEMPLOYED") & active).sum() / max(active.sum(), 1)),
        "employment_share_sole_prop": float(((arr == "SOLE_PROPRIETOR") & emp).sum() / max(emp.sum(), 1)),
        "employment_share_bv": float(((arr == "BV_OWNER_MANAGER") & emp).sum() / max(emp.sum(), 1)),
        "n_bv_firms": len(st.firms),
    }
    out["redistribution_gini_reduction"] = out["gini_market_pensions_as_transfers"] - out["gini_disposable"]
    for k, val in fl.gov.items():
        out["gov_" + k] = val
    out["gov_net_fiscal_balance"] = net_fiscal_balance(fl.gov)
    return out


def group_metrics(st: SimState, fl: Flows, label: str, min_group_size: int = 30) -> pd.DataFrame:
    v = person_view(st, fl)
    line = 0.6 * float(weighted_quantile(v["eq_disp"].to_numpy(), 0.5))
    adults_emp = v["work_state"].eq("EMPLOYED")
    v["children_bucket"] = np.minimum(v["hh_n_children"].fillna(0).to_numpy(), 3).astype(int).astype(str).astype(object)
    v["age_group"] = age_group(v["age"].to_numpy())
    v["arrangement_emp"] = np.where(adults_emp, v["arrangement"], None)
    v["wq0_s"] = v["wq0"].astype("Int64").astype(str)
    specs = {"age_group": "age_group", "region": "region", "education": "education", "work_arrangement": "arrangement_emp",
             "children_in_household": "children_bucket", "tenure": "tenure", "current_wealth_quintile": "wealth_q", "baseline_wealth_quintile": "wq0_s"}
    rows = []
    for name, col in specs.items():
        for key, g in v.dropna(subset=[col]).groupby(col):
            if str(key) in ("<NA>", "nan", "NOT_ASSESSED"):
                continue
            n = len(g)
            rows.append({"scenario": label, "year": st.year, "group_var": name, "group": str(key), "n_sim": n, "small_sample_warning": n < min_group_size,
                         "mean_eq_market_income": g["eq_market_a"].mean(), "mean_eq_disposable_income": g["eq_disp"].mean(),
                         "median_eq_disposable_income": g["eq_disp"].median(), "poverty_rate": float((g["eq_disp"] < line).mean()),
                         "gini_disposable_within": weighted_gini(g["eq_disp"].to_numpy()) if n > 1 else np.nan,
                         "mean_hh_net_worth": g["hh_net_worth"].mean(), "mean_net_fiscal_position_per_capita": g["hh_net_fiscal_per_capita"].mean()})
    return pd.DataFrame(rows)


def fiscal_row(st: SimState, fl: Flows, label: str) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"scenario": label, "year": st.year}
    row.update({k: fl.gov[k] for k in (*REVENUE, *SPENDING)})
    row["net_fiscal_balance"] = net_fiscal_balance(fl.gov)
    return row


def quintile(values: pd.Series) -> pd.Series:
    return np.ceil(values.rank(pct=True, method="first") * 5).clip(1, 5).astype(int)


def mobility_matrix(initial_q: pd.Series, final_values: pd.Series) -> pd.DataFrame:
    """Row-normalised transition matrix between baseline and final equivalised-income quintiles (survivors)."""
    both = initial_q.index.intersection(final_values.index)
    fq = quintile(final_values.loc[both])
    tab = pd.crosstab(initial_q.loc[both].astype(int), fq, normalize="index")
    return tab.reindex(index=range(1, 6), columns=range(1, 6), fill_value=0.0)
