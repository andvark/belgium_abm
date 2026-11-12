"""Simulation state, initialisation of incomes / firms / wealth, household bookkeeping.

Wealth, firm and income initial conditions are STYLISED (parameters in ``economy_stylised.yaml``,
all ``assumed``): the project has no access to the restricted EU-SILC / HFCS microdata. The only
empirical anchoring is an optional one-point level calibration to the published median
equivalised disposable income.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri

from belgium_abm.engines import crn
from belgium_abm.engines.economy import EconomyConfig
from belgium_abm.policies.rules import PolicyRules, statutory_pension

EDU_BASE = {"LOW": "wage_base_low", "MEDIUM": "wage_base_medium", "HIGH": "wage_base_high"}


@dataclass
class SimState:
    year: int
    seed: int
    persons: pd.DataFrame  # index person_id (living persons only)
    households: pd.DataFrame  # index household_id
    firms: pd.DataFrame  # index firm_id (= owner person_id); BV owner-managers only
    econ: EconomyConfig
    adult_age: int
    equivalence: tuple[float, float, float, int]
    level_scale: float
    person_weight: float
    household_weight: float
    labour_table: pd.DataFrame  # region, age_min, age_max, education_level, work_state, share
    arrangement_table: pd.DataFrame
    education_table: pd.DataFrame
    mortality: pd.DataFrame  # sex, age, rate
    fertility: pd.DataFrame
    events: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, float] = field(default_factory=dict)
    next_person_id: int = 0
    next_household_id: int = 0
    audit: Any = None

    def log(self, kind: str, n: float) -> None:
        self.counters[kind] = self.counters.get(kind, 0.0) + float(n)

    def copy(self) -> SimState:
        s = SimState(**{k: getattr(self, k) for k in self.__dataclass_fields__ if k not in ("persons", "households", "firms", "events", "counters")},
                     persons=self.persons.copy(), households=self.households.copy(), firms=self.firms.copy(),
                     events=list(self.events), counters=dict(self.counters))
        return s


# ------------------------------------------------------------------ productivity ----------
def labour_productivity(persons: pd.DataFrame, econ: EconomyConfig, year: int, level_scale: float) -> np.ndarray:
    """Full-time employee-equivalent gross annual earnings of each person (EUR, constant prices)."""
    base = persons["education_level"].map({k: float(econ.v(v)) for k, v in EDU_BASE.items()}).fillna(0.0).to_numpy(float)
    x = (persons["age"].to_numpy(float) - 40.0) / 10.0
    logw = np.log(np.where(base > 0, base, 1.0)) + econ.v("wage_age_slope") * x + econ.v("wage_age_curvature") * x**2
    logw += econ.v("skill_sigma") * persons["skill_z"].to_numpy(float) + persons["resid"].to_numpy(float)
    out = np.exp(logw) * level_scale * (1.0 + econ.v("real_wage_growth")) ** year
    return np.where(base > 0, out, 0.0)


def calibrate_skill_sigma(p: pd.DataFrame, econ: EconomyConfig, level_scale: float) -> EconomyConfig:
    """Solve ``skill_sigma`` so the total SD of log hourly earnings among employees equals the lognormal-equivalent SD implied by the
    published low-wage share (share of employees below 2/3 of the median: sigma = |ln(2/3)| / |Phi^-1(share)|). The simulated low-wage
    share itself is reported as a validation."""
    from belgium_abm.datasources.behaviour import lognormal_sd_from_low_wage_share

    if "low_wage_share_target" not in econ.parameters:
        return econ
    target_share = float(econ.v("low_wage_share_target"))
    target_sd = lognormal_sd_from_low_wage_share(target_share)
    emp = ((p["work_state"] == "EMPLOYED") & (p["arrangement"] == "EMPLOYEE")).to_numpy()
    if emp.sum() < 50:
        return econ
    base = np.log(np.clip(labour_productivity(p, econ.with_values(skill_sigma=0.0), 0, level_scale), 1e-9, None))[emp]
    z = p["skill_z"].to_numpy(float)[emp]
    lo, hi = 0.0, 1.5
    for _ in range(50):
        mid = (lo + hi) / 2
        if float(np.std(base + mid * z)) < target_sd:
            lo = mid
        else:
            hi = mid
    return econ.with_derived(f"solved so the SD of log hourly earnings equals {target_sd:.3f}, the lognormal equivalent of the published low-wage share {target_share:.3f} (Eurostat earn_ses_pub1s, 2022)", skill_sigma=(lo + hi) / 2)


def equivalence_sizes(persons: pd.DataFrame, households: pd.DataFrame, eq: tuple[float, float, float, int]) -> pd.Series:
    first, extra, child, thr = eq
    ge = (persons["age"] >= thr).groupby(persons["household_id"]).sum()
    lt = (persons["age"] < thr).groupby(persons["household_id"]).sum()
    ge = ge.reindex(households.index, fill_value=0)
    lt = lt.reindex(households.index, fill_value=0)
    size = np.where(ge >= 1, first + extra * (ge - 1), 0.0) + child * lt
    return pd.Series(size, index=households.index)


def refresh_households(persons: pd.DataFrame, households: pd.DataFrame, adult_age: int) -> pd.DataFrame:
    """Recompute counts, type and head from the person table (single source of truth)."""
    hh = households.loc[households.index.isin(persons["household_id"])].copy()
    adult = persons["age"] >= adult_age
    na = adult.groupby(persons["household_id"]).sum().reindex(hh.index, fill_value=0)
    nc = (~adult).groupby(persons["household_id"]).sum().reindex(hh.index, fill_value=0)
    hh["n_adults"], hh["n_children"], hh["num_dependents"] = na.astype(int), nc.astype(int), nc.astype(int)
    typ = np.where(na >= 3, "OTHER_MULTI", np.where(na == 2, np.where(nc > 0, "COUPLE_WITH_CHILDREN", "COUPLE"),
                                                      np.where(nc > 0, "SINGLE_PARENT", "SINGLE_ADULT")))
    hh["household_type"] = typ
    return hh


def assign_heads(persons: pd.DataFrame) -> pd.DataFrame:
    """Head = HEAD role if present else the oldest adult (ties: lowest id). Roles are normalised."""
    p = persons
    order = p.assign(_pid=p.index).sort_values(["household_id", "age", "_pid"], ascending=[True, False, True])
    head_id = order.groupby("household_id").head(1).set_index("household_id")["_pid"]
    p["head_flag"] = p.index.to_numpy() == p["household_id"].map(head_id).to_numpy()
    return p


# ------------------------------------------------------------------- initialisation -------
def _hour_fraction(p: pd.DataFrame, econ: EconomyConfig, seed: int, stream: str, year: int) -> np.ndarray:
    u = crn.uniform(seed, stream, year, p.index.to_numpy())
    share = np.where(p["sex_or_demographic_stratum"].to_numpy() == "F", econ.v("parttime_share_f"), econ.v("parttime_share_m"))
    return np.where(u < share, econ.v("parttime_hours"), 1.0)


def build_state(persons_in: pd.DataFrame, households_in: pd.DataFrame, labour: pd.DataFrame, arrangement: pd.DataFrame, education: pd.DataFrame,
                demography: pd.DataFrame, econ: EconomyConfig, policy: PolicyRules, cfg_adult_age: int,
                equivalence: tuple[float, float, float, int], seed: int, level_scale: float = 1.0) -> SimState:
    from belgium_abm.engines.flows import compute_flows  # local import: flows depends on state

    p = persons_in.copy()
    p.index = p["person_id"].to_numpy()
    p = p.rename(columns={"work_state": "work_state", "primary_work_arrangement": "arrangement"})
    p["sex_or_demographic_stratum"] = p["sex_or_demographic_stratum"].astype(str)
    for c in ("education_level", "work_state", "arrangement", "region", "role"):
        p[c] = p[c].astype(object).where(p[c].notna(), None)
    p["work_state"] = p["work_state"].where(p["work_state"].notna(), "NONE")
    p["age"] = p["age"].astype(int)
    ids = p.index.to_numpy()
    p["skill_z"] = crn.normal(seed, "init_skill", 0, ids)
    p["resid"] = crn.normal(seed, "init_resid", 0, ids) * econ.v("wage_shock_sigma") / np.sqrt(1 - econ.v("wage_shock_rho") ** 2)
    p["biz"] = crn.normal(seed, "init_biz", 0, ids) * econ.v("business_shock_sigma") / np.sqrt(1 - econ.v("business_shock_rho") ** 2)
    p["hours"] = _hour_fraction(p, econ, seed, "init_hours", 0)
    econ = calibrate_skill_sigma(p, econ, level_scale)
    p["last_wage"] = 0.0
    p["pension_rights"] = 0.0
    p["unemp_elig"] = False
    p["unemp_years"] = 0
    p["birth_year"] = -p["age"]
    # students aged 18-21 with MEDIUM/HIGH attainment are treated as tertiary students (graduating at 22)
    p["tertiary"] = ((p["work_state"] == "STUDENT") & p["education_level"].isin(["MEDIUM", "HIGH"]) & p["age"].between(18, 21)).to_numpy()
    elig = crn.uniform(seed, "init_unemp_elig", 0, ids) < econ.v("unemployment_eligibility_share")
    unemp = p["work_state"].to_numpy() == "UNEMPLOYED"
    p["unemp_elig"] = unemp & elig
    p["unemp_years"] = np.where(unemp, (crn.uniform(seed, "init_unemp_years", 0, ids) * 2).astype(int), 0)

    # initial last_wage and pension rights (career-average convention)
    prod = labour_productivity(p, econ, 0, level_scale)
    emp = (p["work_state"] == "EMPLOYED").to_numpy()
    p["last_wage"] = np.where(emp & (p["arrangement"] == "EMPLOYEE").to_numpy(), prod * p["hours"].to_numpy(), 0.0)
    ref = p.copy()
    ref["age"] = np.minimum(p["age"], 55)
    ref_wage = labour_productivity(ref, econ, 0, level_scale)
    pc = policy.components["pensions"]
    ret_age = pc["retirement_age"]
    years_worked = np.clip(np.minimum(p["age"].to_numpy(), ret_age) - econ.v("start_work_age"), 0, 44) * 0.9
    se_factor = np.where(p["arrangement"].isin(["SOLE_PROPRIETOR", "BV_OWNER_MANAGER"]).to_numpy(), pc.get("self_employed_accrual_factor", 1.0), 1.0)
    has_career = p["work_state"].isin(["EMPLOYED", "UNEMPLOYED", "RETIRED"]).to_numpy() | (p["age"].to_numpy() >= 30)
    p["pension_rights"] = np.where(has_career, pc["accrual_rate_per_year"] * years_worked * ref_wage * se_factor, 0.0)
    # career-based permanent income, used only to seed the wealth of retired households (their current income is a pension)
    p["perm_income"] = np.where(has_career, ref_wage * se_factor, 0.0)

    hh = households_in.copy()
    hh.index = hh["household_id"].to_numpy()
    if "housing_tenure" not in hh or hh["housing_tenure"].isna().all():
        hh["housing_tenure"] = "OUTRIGHT_OWNER"
    hh["housing_tenure"] = hh["housing_tenure"].astype(object).where(hh["housing_tenure"].notna(), "OTHER")
    for c in ("housing_value", "mortgage_balance", "financial_assets", "other_debt"):
        hh[c] = 0.0

    # firms: one BV per BV owner-manager
    bv = p.index[(p["arrangement"] == "BV_OWNER_MANAGER") & (p["work_state"] == "EMPLOYED")]
    firms = pd.DataFrame({"firm_id": bv, "owner_person_id": bv, "equity": 0.0, "active": True}, index=bv)

    st = SimState(year=0, seed=seed, persons=p, households=hh, firms=firms, econ=econ, adult_age=cfg_adult_age, equivalence=equivalence,
                  level_scale=level_scale, person_weight=float(p["weight"].iloc[0]), household_weight=float(hh["weight"].iloc[0]),
                  labour_table=labour, arrangement_table=arrangement, education_table=education, mortality=demography[demography.kind == "mortality_qx"],
                  fertility=demography[demography.kind == "fertility_per_woman"],
                  next_person_id=int(p.index.max()) + 1, next_household_id=int(hh.index.max()) + 1)
    st.persons = assign_heads(st.persons)
    _init_wealth(st, policy, compute_flows)
    return st


def _init_wealth(st: SimState, policy: PolicyRules, compute_flows) -> None:
    econ, p, hh = st.econ, st.persons, st.households
    fl = compute_flows(st, policy, consume=False)
    inc = fl.h["market_a"].reindex(hh.index).fillna(0.0).to_numpy()
    head = p[p["head_flag"]].set_index("household_id")
    hid = hh.index.to_numpy()
    head_age = head["age"].reindex(hh.index).fillna(40).to_numpy(float)
    head_z = head["skill_z"].reindex(hh.index).fillna(0.0).to_numpy(float)
    c = econ.v("fin_assets_skill_coupling")
    e = np.sqrt(c) * head_z + np.sqrt(1 - c) * crn.normal(st.seed, "init_assets", 0, hid)
    sig = econ.v("fin_assets_sigma")
    mult = econ.v("fin_assets_multiple_max") * (1 - np.exp(-np.clip(head_age - 20, 0, None) / econ.v("fin_assets_age_tau")))
    retired = (p["work_state"] == "RETIRED").to_numpy()
    perm = pd.Series(np.where(retired, p["perm_income"].to_numpy(float), 0.0), index=p.index).groupby(p["household_id"]).sum().reindex(hh.index, fill_value=0.0).to_numpy()
    # retired households accumulated wealth over a career: use the larger of current income and a share of past earnings
    income_ref = np.maximum(np.maximum(inc, 0.0), econ.v("retiree_permanent_income_share") * perm) + 0.1 * st.level_scale * econ.v("wage_base_medium")
    fin = mult * income_ref * np.exp(sig * e - sig**2 / 2)
    tenure = hh["housing_tenure"].to_numpy()
    owner = np.isin(tenure, ["MORTGAGED_OWNER", "OUTRIGHT_OWNER"])
    hs = econ.v("housing_value_sigma")
    housing = np.where(owner, econ.v("house_to_income") * income_ref * np.exp(hs * crn.normal(st.seed, "init_house", 0, hid) - hs**2 / 2), 0.0)
    ltv = np.clip(0.70 - 0.015 * (head_age - 30), 0.05, 0.70)
    renter_like = np.isin(tenure, ["RENTER", "OTHER"])
    fin = np.where(renter_like, fin * float(econ.v("renter_financial_asset_factor")), fin)
    hh["financial_assets"] = fin
    hh["housing_value"] = housing
    hh["mortgage_balance"] = np.where(tenure == "MORTGAGED_OWNER", ltv * housing, 0.0)
    hh["other_debt"] = 0.0
    # BV equity: multiple of one year of retained profit
    if len(st.firms):
        retained = fl.firms["retained_delta"].reindex(st.firms.index).fillna(0.0).to_numpy()
        st.firms["equity"] = np.clip(retained, 0.0, None) * econ.v("bv_initial_retained_years")


def net_worth(st: SimState) -> pd.Series:
    """Household net worth: financial + housing - mortgage - other debt + beneficial BV equity (counted once)."""
    hh = st.households
    eq = pd.Series(0.0, index=hh.index)
    if len(st.firms):
        owner_hh = st.persons["household_id"].reindex(st.firms["owner_person_id"].to_numpy()).to_numpy()
        eq = pd.Series(st.firms["equity"].to_numpy(), index=owner_hh).groupby(level=0).sum().reindex(hh.index, fill_value=0.0)
    return hh["financial_assets"] + hh["housing_value"] - hh["mortgage_balance"] - hh["other_debt"] + eq


__all__ = ["SimState", "assign_heads", "build_state", "equivalence_sizes", "labour_productivity", "ndtri", "net_worth", "refresh_households", "statutory_pension"]
