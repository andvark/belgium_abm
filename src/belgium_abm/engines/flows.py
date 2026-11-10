"""Annual income, fiscal and household cash-flow computation (Phase 2 engine).

Everything is vectorised and driven by a :class:`PolicyRules` object, so changing the policy
(without changing behaviour or state) isolates the mechanical fiscal effect (spec section 11.11).

Conventions
  * Legal payer vs incidence: employer social contributions are a cash flow of employers to the
    government and never part of the employee's income; they are not subtracted twice.
  * BV retained profit is not household income; only director pay and distributed dividends are.
  * In-kind services are not modelled here (they never touch cash).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from belgium_abm.engines.state import SimState, equivalence_sizes, labour_productivity
from belgium_abm.policies.rules import PolicyRules, personal_income_tax, self_employed_contribution, statutory_pension, unemployment_benefit


@dataclass
class Flows:
    p: pd.DataFrame  # per person
    h: pd.DataFrame  # per household
    firms: pd.DataFrame  # per BV firm
    gov: dict[str, float]  # EUR, population totals (weights applied)
    policy: PolicyRules


def _business_va(st: SimState, prod: np.ndarray, er: float, mult: float) -> np.ndarray:
    e = st.econ
    var = e.v("business_shock_sigma") ** 2 / (1 - e.v("business_shock_rho") ** 2)
    return prod * (1.0 + er) * mult * np.exp(st.persons["biz"].to_numpy(float) - 0.5 * var)


def compute_flows(st: SimState, policy: PolicyRules, consume: bool = True) -> Flows:
    e, p, hh = st.econ, st.persons, st.households
    g = policy.g
    er, ee = g("employer_social_contributions", "rate"), g("employee_social_contributions", "rate")
    prod = labour_productivity(p, e, st.year, st.level_scale)
    employed = (p["work_state"] == "EMPLOYED").to_numpy()
    arr = p["arrangement"].to_numpy()
    is_emp = employed & (arr == "EMPLOYEE")
    is_sp = employed & (arr == "SOLE_PROPRIETOR")
    is_bv = employed & (arr == "BV_OWNER_MANAGER")

    wage = np.where(is_emp, prod * p["hours"].to_numpy(float), 0.0)
    emp_ssc, employer_ssc = wage * ee, wage * er

    va_sp = np.where(is_sp, _business_va(st, prod, er, e.v("sole_prop_productivity_multiplier")), 0.0)
    profit_sp = np.where(is_sp, va_sp - e.v("sole_prop_accounting_cost"), 0.0)
    se_ssc = self_employed_contribution(profit_sp, policy, is_sp)

    va_bv = np.where(is_bv, _business_va(st, prod, er, e.v("bv_productivity_multiplier")), 0.0)
    surplus = np.where(is_bv, va_bv - e.v("bv_accounting_cost"), 0.0)
    dir_pay = e.v("bv_director_pay_share") * np.clip(surplus, 0.0, None)
    dir_ssc = self_employed_contribution(dir_pay, policy, is_bv)
    pbt = np.where(is_bv, surplus - dir_pay, 0.0)
    corp_tax = g("corporate_income_tax", "rate") * np.clip(pbt, 0.0, None)
    post = pbt - corp_tax
    dividends = e.v("bv_payout_ratio") * np.clip(post, 0.0, None)
    div_tax = g("dividend_withholding_tax", "rate") * dividends
    retained_delta = np.where(is_bv, post - dividends, 0.0)

    max_years = policy.components["unemployment_benefits"]["max_years_by_age"][0][1]
    unemployed = (p["work_state"] == "UNEMPLOYED").to_numpy()
    elig = unemployed & p["unemp_elig"].to_numpy(bool) & (p["unemp_years"].to_numpy() < max_years)
    unemp = unemployment_benefit(p["last_wage"].to_numpy(float), elig, policy)
    retired = (p["work_state"] == "RETIRED").to_numpy()
    pension = np.where(retired, statutory_pension(p["pension_rights"].to_numpy(float), policy), 0.0)

    net_wage = wage - emp_ssc
    net_se = profit_sp - se_ssc
    net_dir = dir_pay - dir_ssc
    taxable = net_wage + net_se + net_dir + pension + unemp

    adult = (p["age"] >= st.adult_age).to_numpy()
    hid = p["household_id"].to_numpy()
    n_child_hh = pd.Series((~adult).astype(int), index=p.index).groupby(p["household_id"]).sum()
    head = p["head_flag"].to_numpy(bool)
    child_allow = np.where(head, n_child_hh.reindex(hid).to_numpy(), 0)
    pit = personal_income_tax(taxable, adult.astype(float), child_allow.astype(float), policy)

    # household level --------------------------------------------------------------------
    fin = hh["financial_assets"].to_numpy(float)
    cap_gross = np.clip(fin, 0.0, None) * e.v("fin_return")
    cap_tax = g("capital_income_tax", "rate") * cap_gross
    n_child_h = n_child_hh.reindex(hh.index, fill_value=0).to_numpy()
    cb_rate = hh["region"].map(g("child_benefits_regionalised", "per_child_eur")).to_numpy(float)
    child_ben = cb_rate * n_child_h

    market_a_p = wage + profit_sp + dir_pay + dividends
    market_b_p = market_a_p + pension
    cash_p = net_wage + net_se + net_dir + dividends - div_tax + pension + unemp - pit
    P = pd.DataFrame({"wage": wage, "emp_ssc": emp_ssc, "employer_ssc": employer_ssc, "profit_sp": profit_sp, "se_ssc": se_ssc,
                      "dir_pay": dir_pay, "dir_ssc": dir_ssc, "dividends": dividends, "div_tax": div_tax, "unemp": unemp,
                      "pension": pension, "pit": pit, "market_a": market_a_p, "market_b": market_b_p, "cash": cash_p,
                      "prod": prod, "va": va_sp + va_bv}, index=p.index)
    byh = P.groupby(p["household_id"]).sum().reindex(hh.index, fill_value=0.0)

    market_a = byh["market_a"].to_numpy() + cap_gross
    market_b = byh["market_b"].to_numpy() + cap_gross
    pre_assist = byh["cash"].to_numpy() + child_ben + cap_gross - cap_tax
    eq = equivalence_sizes(p, hh, st.equivalence).to_numpy(float)
    floor_inc = g("disability_and_other_benefits", "social_assistance_floor_per_equivalent_eur") * eq
    eligible_assist = fin < g("disability_and_other_benefits", "asset_limit_eur")
    assist = np.where(eligible_assist, np.clip(floor_inc - pre_assist, 0.0, None), 0.0)
    disp = pre_assist + assist

    owner_debt = hh["mortgage_balance"].to_numpy(float)
    interest = owner_debt * e.v("mortgage_rate") + hh["other_debt"].to_numpy(float) * e.v("mortgage_rate")
    repay = owner_debt * e.v("mortgage_repay_rate")
    debt_service = interest + repay

    H = pd.DataFrame({"region": hh["region"].to_numpy(), "eq_size": eq, "market_a": market_a, "market_b": market_b,
                      "cap_gross": cap_gross, "cap_tax": cap_tax, "child_ben": child_ben, "assist": assist,
                      "pit": byh["pit"].to_numpy(), "disp": disp, "debt_service": debt_service, "interest": interest, "repay": repay,
                      "ssc_personal": (byh["emp_ssc"] + byh["se_ssc"] + byh["dir_ssc"]).to_numpy(),
                      "pension": byh["pension"].to_numpy(), "unemp": byh["unemp"].to_numpy(), "div_tax": byh["div_tax"].to_numpy()}, index=hh.index)
    H["eq_disp"] = np.where(eq > 0, disp / np.where(eq > 0, eq, 1.0), np.nan)
    if consume:
        retired_hh = (p["work_state"] == "RETIRED").groupby(p["household_id"]).all().reindex(hh.index, fill_value=False).to_numpy()
        avail = disp - debt_service
        floor_c = e.v("consumption_floor_per_equivalent") * eq
        target = floor_c + e.v("marginal_propensity_to_consume") * np.clip(avail - floor_c, 0.0, None)
        target = target + np.where(retired_hh, e.v("retiree_wealth_drawdown") * np.clip(fin, 0.0, None), 0.0)
        liquidity = np.clip(avail + np.clip(fin, 0.0, None), 0.0, None)
        cons = np.minimum(target, liquidity)
        shortfall = np.clip(-(avail + np.clip(fin, 0.0, None)), 0.0, None)  # unpaid obligations -> explicit credit event
        H["consumption"] = cons
        H["credit_event"] = shortfall
        H["vat"] = g("vat_by_consumption_group", "effective_rate") / (1 + g("vat_by_consumption_group", "effective_rate")) * cons
        H["saving"] = avail - cons

    F = pd.DataFrame(index=st.firms.index)
    if len(st.firms):
        ix = p.index.get_indexer(st.firms["owner_person_id"].to_numpy())
        okay = ix >= 0
        ixs = np.where(okay, ix, 0)
        va = va_bv[ixs] * okay
        cr = e.v("intermediate_cost_ratio")
        F["turnover"] = (va + e.v("bv_accounting_cost")) / (1 - cr)
        F["intermediate_costs"] = F["turnover"] * cr
        F["accounting_cost"] = e.v("bv_accounting_cost") * okay
        F["director_pay"] = dir_pay[ixs] * okay
        F["corporate_tax"] = corp_tax[ixs] * okay
        F["dividends"] = dividends[ixs] * okay
        F["retained_delta"] = retained_delta[ixs] * okay

    w = st.person_weight
    gov = {
        "personal_income_tax": float(pit.sum() * w), "employee_ssc": float(emp_ssc.sum() * w), "employer_ssc": float(employer_ssc.sum() * w),
        "self_employed_ssc": float((se_ssc + dir_ssc).sum() * w), "corporate_tax": float(corp_tax.sum() * w),
        "dividend_withholding_tax": float(div_tax.sum() * w), "capital_income_tax": float(cap_tax.sum() * w),
        "vat": float(H["vat"].sum() * w) if consume else 0.0,
        "unemployment_benefits": float(unemp.sum() * w), "pensions": float(pension.sum() * w),
        "child_benefits": float(child_ben.sum() * w), "social_assistance": float(assist.sum() * w),
    }
    return Flows(P, H, F, gov, policy)


REVENUE = ("personal_income_tax", "employee_ssc", "employer_ssc", "self_employed_ssc", "corporate_tax", "dividend_withholding_tax", "capital_income_tax", "vat")
SPENDING = ("unemployment_benefits", "pensions", "child_benefits", "social_assistance")


def net_fiscal_balance(gov: dict[str, float]) -> float:
    return sum(gov[k] for k in REVENUE) - sum(gov[k] for k in SPENDING)
