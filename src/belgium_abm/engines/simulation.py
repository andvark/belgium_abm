"""Dynamic simulation driver (Phases 3-4): calibration of the income level, the annual loop,
shadow-policy (fixed-behaviour) evaluation, ledger posting and invariant checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from belgium_abm.analysis.inequality import weighted_quantile
from belgium_abm.analysis.behaviour_validation import cross_section_drift
from belgium_abm.analysis.measures import (
    fiscal_row,
    group_metrics,
    mobility_matrix,
    quintile,
    yearly_metrics,
)
from belgium_abm.engines.dynamics import step_demography, step_education, step_finances, step_work
from belgium_abm.engines.economy import EconomyConfig
from belgium_abm.engines.flows import compute_flows
from belgium_abm.engines.state import SimState, build_state, net_worth
from belgium_abm.policies.ledger import FiscalLedger
from belgium_abm.policies.rules import PolicyRules

LEDGER_PARTIES = {
    "personal_income_tax": ("HOUSEHOLDS", "GOVERNMENT"), "employee_ssc": ("HOUSEHOLDS", "GOVERNMENT"),
    "employer_ssc": ("EMPLOYERS", "GOVERNMENT"), "self_employed_ssc": ("HOUSEHOLDS", "GOVERNMENT"),
    "corporate_tax": ("FIRMS", "GOVERNMENT"), "dividend_withholding_tax": ("FIRMS", "GOVERNMENT"),
    "capital_income_tax": ("HOUSEHOLDS", "GOVERNMENT"), "vat": ("HOUSEHOLDS", "GOVERNMENT"),
    "unemployment_benefits": ("GOVERNMENT", "HOUSEHOLDS"), "pensions": ("GOVERNMENT", "HOUSEHOLDS"),
    "child_benefits": ("GOVERNMENT", "HOUSEHOLDS"), "social_assistance": ("GOVERNMENT", "HOUSEHOLDS"),
}


@dataclass
class SimResult:
    metrics: pd.DataFrame
    groups: pd.DataFrame
    fiscal: pd.DataFrame
    events: pd.DataFrame
    mobility: pd.DataFrame
    ledger: FiscalLedger
    final_state: SimState
    banner: str
    max_cash_identity_residual: float = 0.0
    invariant_failures: list[str] = field(default_factory=list)
    extras: dict[str, float] = field(default_factory=dict)
    snapshots: dict[int, pd.DataFrame] = field(default_factory=dict)
    drift: pd.DataFrame | None = None


def _median_eq_disp(st: SimState, policy: PolicyRules) -> float:
    fl = compute_flows(st, policy, consume=False)
    eq = fl.h["eq_disp"].reindex(st.persons["household_id"]).to_numpy()
    return float(weighted_quantile(eq, 0.5))


def initial_state(persons: pd.DataFrame, households: pd.DataFrame, targets, demography: pd.DataFrame, econ: EconomyConfig,
                  policy: PolicyRules, adult_age: int, equivalence, seed: int, target_median: float | None = None, iterations: int = 5) -> tuple[SimState, dict[str, float]]:
    """Build the t=0 state; optionally scale earnings/wealth so the median equivalised disposable income hits a published target."""
    scale = 1.0
    info: dict[str, float] = {"level_scale": 1.0}
    if "labour_force_exit_total_observed" in econ.parameters:
        # observed employment -> inactivity flow includes retirement; the model's non-retirement exit is the remainder
        ret_age = policy.g("pensions", "retirement_age")
        emp = persons[(persons["work_state"] == "EMPLOYED") & (persons["age"] <= 74)]
        retire_flow = float((emp["age"] == ret_age - 1).sum() / max(len(emp), 1))
        l_exit = max(float(econ.v("labour_force_exit_total_observed")) - retire_flow, 0.0)
        econ = econ.with_derived("observed EMP->INAC annual flow (Eurostat lfsi_long_q) minus the modelled retirement flow", labour_force_exit_rate=l_exit)
        info["labour_force_exit_rate_derived"] = l_exit
        info["retirement_flow_share"] = retire_flow
    for _ in range(iterations if (target_median and econ.v("calibrate_level_to_target")) else 1):
        st = build_state(persons, households, targets.labour_state, targets.work_arrangement, targets.education, demography, econ, policy, adult_age, equivalence, seed, scale)
        if not (target_median and econ.v("calibrate_level_to_target")):
            break
        med = _median_eq_disp(st, policy)
        info["median_before_final_step"] = med
        if abs(med / target_median - 1) < 2e-3:
            break
        scale *= target_median / med
    info["level_scale"] = scale
    info["skill_sigma"] = float(st.econ.v("skill_sigma"))
    info["target_median"] = float(target_median) if target_median else float("nan")
    info["achieved_median"] = _median_eq_disp(st, policy)
    # baseline markers for group cuts and mobility
    fl0 = compute_flows(st, policy, consume=False)
    p = st.persons
    nw = net_worth(st)
    p["wq0"] = np.ceil(nw.rank(pct=True).reindex(p["household_id"]).to_numpy() * 5).clip(1, 5)
    p["wr0"] = nw.rank(pct=True).reindex(p["household_id"]).to_numpy()
    p["child_at_t0"] = (p["age"] < adult_age).to_numpy()
    eq0 = fl0.h["eq_disp"].reindex(p["household_id"]).to_numpy()
    p["iq0"] = quintile(pd.Series(eq0, index=p.index)).to_numpy()
    st.persons = p
    return st, info


def check_state(st: SimState) -> list[str]:
    """Structural invariants of the dynamic state (spec section 11 items 1, 2, 5, 7)."""
    bad: list[str] = []
    p, hh, f = st.persons, st.households, st.firms
    if p.index.has_duplicates:
        bad.append("duplicate person ids")
    if hh.index.has_duplicates:
        bad.append("duplicate household ids")
    if not p["household_id"].isin(hh.index).all():
        bad.append("person in a non-existent household")
    if not hh.index.isin(p["household_id"]).all():
        bad.append("household without members")
    adults = (p["age"] >= st.adult_age).groupby(p["household_id"]).sum().reindex(hh.index, fill_value=0)
    if (adults < 1).any():
        bad.append(f"{int((adults < 1).sum())} households without an adult")
    heads = p["head_flag"].groupby(p["household_id"]).sum().reindex(hh.index, fill_value=0)
    if (heads != 1).any():
        bad.append("households without exactly one head")
    ws = p["work_state"]
    if (p["arrangement"].ne("NONE") & ws.ne("EMPLOYED")).any():
        bad.append("work arrangement set for a non-employed person")
    if (ws.eq("EMPLOYED") & p["arrangement"].eq("NONE")).any():
        bad.append("employed person without an arrangement")
    if (ws.eq("RETIRED") & (p["age"] < 50)).any():
        bad.append("retired person below 50")
    if (p["age"] < 15).mul(ws.isin(["EMPLOYED", "UNEMPLOYED", "STUDENT", "RETIRED"])).any():
        bad.append("child with a labour state")
    bv_people = set(p.index[(p["arrangement"] == "BV_OWNER_MANAGER") & (ws == "EMPLOYED")])
    if set(f.index) != bv_people:
        bad.append("firm table does not match BV owner-managers")
    for c in ("financial_assets", "mortgage_balance", "other_debt", "housing_value"):
        if (hh[c] < -1e-6).any():
            bad.append(f"negative {c}")
    if len(f) and (f["equity"] < -1e-6).any():
        bad.append("negative firm equity")
    kids = p[p["parent1_id"].notna()]
    par = kids["parent1_id"].astype("Int64")
    known = par.isin(p.index.to_numpy())
    if known.any():
        gap = p.loc[par[known].astype("int64").to_numpy(), "age"].to_numpy() - kids.loc[known[known].index, "age"].to_numpy()
        if (gap < 12).any():
            bad.append("parent-child age gap below 12")
    return bad


def run_simulation(st0: SimState, policy: PolicyRules, years: int, *, shadow: dict[str, PolicyRules] | None = None, label: str = "dynamic",
                   check_invariants: bool = False, min_group_size: int = 30, group_years: tuple[int, ...] | None = None) -> SimResult:
    st = st0.copy()
    ledger = FiscalLedger()
    metrics: list[dict] = []
    groups: list[pd.DataFrame] = []
    fiscal: list[dict] = []
    ev_rows: list[dict] = []
    failures: list[str] = []
    snaps: dict[int, pd.DataFrame] = {}
    max_res = 0.0
    shadow = shadow or {}
    gyears = set(group_years) if group_years is not None else {0, years // 2, years}

    liq = {"gross": 0.0}

    def add_liquidation(fl, pol: PolicyRules):
        fl.gov["dividend_withholding_tax"] += pol.g("dividend_withholding_tax", "rate") * liq["gross"]
        return fl

    def record(fl) -> None:
        metrics.append(yearly_metrics(st, fl, label))
        fiscal.append(fiscal_row(st, fl, label))
        if st.year in gyears:
            groups.append(group_metrics(st, fl, label, min_group_size))
            snaps[st.year] = pd.DataFrame({"eq_disp": fl.h["eq_disp"].reindex(st.persons["household_id"]).to_numpy(),
                                          "wq0": st.persons["wq0"].to_numpy(), "age": st.persons["age"].to_numpy()}, index=st.persons.index)
        for name, pol in shadow.items():
            fls = add_liquidation(compute_flows(st, pol), pol)
            tag = f"static:{name}"
            metrics.append(yearly_metrics(st, fls, tag))
            fiscal.append(fiscal_row(st, fls, tag))

    fl0 = compute_flows(st, policy)
    record(fl0)
    for _ in range(years):
        before = dict(st.counters)
        step_demography(st)
        step_education(st)
        step_work(st, policy)
        liq["gross"] = st.counters.pop("gov:liquidation_dividends_gross", 0.0)
        fl = add_liquidation(compute_flows(st, policy), policy)
        record(fl)
        fl = step_finances(st, policy, fl)
        for cat, amt in fl.gov.items():
            payer, recipient = LEDGER_PARTIES[cat]
            ledger.post(period=st.year, payer_id=payer, recipient_id=recipient, category=cat, amount_eur=round(amt, 2), policy_year=policy.policy_year)
        a = st.audit
        res = a["closing"] - (a["opening"] + a["disp"] - a["debt_service"] - a["consumption"] + a["credit_event"] - a["down_payment"])
        max_res = max(max_res, float(res.abs().max()))
        for k, vv in st.counters.items():
            d = vv - before.get(k, 0.0)
            if d:
                ev_rows.append({"scenario": label, "year": st.year, "event": k, "value": d})
        if check_invariants:
            failures += [f"year {st.year}: {m}" for m in check_state(st)]
    p = st.persons
    last = compute_flows(st, policy, consume=False)
    eq_final = pd.Series(last.h["eq_disp"].reindex(p["household_id"]).to_numpy(), index=p.index)
    mob = mobility_matrix(p.loc[p["iq0"].notna(), "iq0"], eq_final)
    extras: dict[str, float] = {}
    kids = p[p["child_at_t0"].fillna(False).astype(bool) & (p["age"] >= 25)]
    if len(kids) > 30:
        own = net_worth(st).rank(pct=True).reindex(kids["household_id"]).to_numpy()
        extras["intergenerational_wealth_rank_correlation"] = float(pd.Series(kids["wr0"].to_numpy()).corr(pd.Series(own), method="spearman"))
        extras["intergenerational_wealth_rank_correlation_n"] = float(len(kids))
    return SimResult(pd.DataFrame(metrics), pd.concat(groups, ignore_index=True) if groups else pd.DataFrame(), pd.DataFrame(fiscal),
                     pd.DataFrame(ev_rows), mob, ledger, st, policy.banner, max_res, failures, extras, snaps, cross_section_drift(st0.persons, st.persons))
