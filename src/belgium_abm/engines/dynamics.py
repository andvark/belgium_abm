"""Annual transitions (Phase 3): demography, education, labour/firms, finances.

Timing convention (locked by tests): at the start of model year k every living person is one
year older (step 1) before any other transition; newborns have age 0 for the rest of the year.
All random draws use common random numbers keyed by (seed, stream, year, entity id).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from belgium_abm.engines import crn
from belgium_abm.engines.flows import Flows, compute_flows
from belgium_abm.engines.state import SimState, assign_heads, net_worth, refresh_households
from belgium_abm.policies.rules import PolicyRules

AGE_BANDS = [(15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 99)]


def _band_of(age: np.ndarray) -> np.ndarray:
    lo = np.array([b[0] for b in AGE_BANDS])
    return lo[np.clip(np.searchsorted(lo, age, side="right") - 1, 0, len(lo) - 1)]


def wealth_rank(st: SimState) -> pd.Series:
    """Household net-worth percentile rank in (0,1)."""
    nw = net_worth(st)
    return nw.rank(pct=True, method="average")


# ------------------------------------------------------------------------- demography ------
def _qx_lookup(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for sex, g in df.groupby("sex"):
        arr = np.zeros(121)
        s = g.set_index("age")["rate"].sort_index()
        arr[: s.index.max() + 1] = s.reindex(range(s.index.max() + 1)).ffill().to_numpy()
        arr[s.index.max() + 1 :] = s.iloc[-1]
        out[sex] = arr
    return out


def hh_add(st: SimState, col: str, hids: np.ndarray, amounts: np.ndarray, floor_zero: bool = False) -> None:
    """Duplicate-safe in-place add of ``amounts`` to ``col`` of the households ``hids``."""
    arr = st.households[col].to_numpy(float).copy()
    np.add.at(arr, st.households.index.get_indexer(np.asarray(hids)), np.asarray(amounts, dtype=float))
    st.households[col] = np.clip(arr, 0.0, None) if floor_zero else arr


def _new_household_ids(st: SimState, n: int) -> np.ndarray:
    ids = np.arange(st.next_household_id, st.next_household_id + n)
    st.next_household_id += n
    return ids


def _add_households(st: SimState, ids: np.ndarray, region: np.ndarray, tenure: str = "RENTER") -> None:
    new = pd.DataFrame({"household_id": ids, "region": region, "housing_tenure": tenure, "housing_value": 0.0, "mortgage_balance": 0.0,
                        "financial_assets": 0.0, "other_debt": 0.0, "weight": st.household_weight, "household_type": "SINGLE_ADULT",
                        "n_adults": 1, "n_children": 0, "num_dependents": 0}, index=ids)
    for c in st.households.columns:
        if c not in new:
            new[c] = np.nan
    st.households = pd.concat([st.households, new[st.households.columns]])


def step_demography(st: SimState) -> None:
    e, yr = st.econ, st.year + 1
    st.year = yr
    p = st.persons
    p["age"] = p["age"] + 1
    ids = p.index.to_numpy()

    # mortality --------------------------------------------------------------------------
    q = _qx_lookup(st.mortality)
    age_i = np.clip(p["age"].to_numpy(), 0, 120)
    qx = np.where(p["sex_or_demographic_stratum"].to_numpy() == "F", q["F"][age_i], q["M"][age_i]) * e.v("mortality_multiplier")
    dead = crn.uniform(st.seed, "mortality", yr, ids) < qx
    st.log("deaths", dead.sum())
    dead_ids = ids[dead]
    if dead.any():
        dead_hh = p.loc[dead, "household_id"]
        # firm equity of dead owners joins the estate of their household
        if len(st.firms):
            fd = st.firms[st.firms["owner_person_id"].isin(dead_ids)]
            hh_add(st, "financial_assets", p.loc[fd["owner_person_id"].to_numpy(), "household_id"].to_numpy(), np.clip(fd["equity"].to_numpy(), 0, None))
            st.firms = st.firms.drop(fd.index)
        p = p.loc[~dead].copy()
        p.loc[p["partner_id"].isin(dead_ids), "partner_id"] = pd.NA
        st.persons = p
        affected = np.unique(dead_hh.to_numpy())
        _resolve_estates(st, affected, dead_ids)
        p = st.persons

    # births -------------------------------------------------------------------------------
    fert = st.fertility.set_index("age")["rate"]
    women = p[(p["sex_or_demographic_stratum"] == "F") & p["age"].between(15, 49)]
    rate = women["age"].map(fert).fillna(0.0).to_numpy() * e.v("fertility_multiplier")
    mothers = women[crn.uniform(st.seed, "birth", yr, women.index.to_numpy()) < rate].sort_index()
    if len(mothers):
        nid = np.arange(st.next_person_id, st.next_person_id + len(mothers))
        st.next_person_id += len(mothers)
        male = crn.uniform(st.seed, "birth_sex", yr, mothers.index.to_numpy()) < e.v("male_birth_share")
        k = float(e.v("skill_inheritance"))
        z = k * mothers["skill_z"].to_numpy() + np.sqrt(1 - k**2) * crn.normal(st.seed, "birth_skill", yr, mothers.index.to_numpy())
        babies = pd.DataFrame(index=nid)
        babies["person_id"] = nid
        babies["household_id"] = mothers["household_id"].to_numpy()
        babies["age"] = 0
        babies["sex_or_demographic_stratum"] = np.where(male, "M", "F")
        babies["region"] = mothers["region"].to_numpy()
        babies["education_level"] = "NOT_ASSESSED"
        babies["work_state"] = "NONE"
        babies["arrangement"] = "NONE"
        babies["role"] = "CHILD"
        babies["partner_id"] = pd.array([pd.NA] * len(nid), dtype="Int64")
        babies["parent1_id"] = pd.array(mothers.index.to_numpy(), dtype="Int64")
        babies["parent2_id"] = mothers["partner_id"].astype("Int64").array
        babies["weight"] = st.person_weight
        babies["alive"] = True
        babies["skill_z"], babies["resid"], babies["biz"], babies["hours"] = z, 0.0, 0.0, 1.0
        babies["last_wage"], babies["pension_rights"], babies["unemp_elig"], babies["unemp_years"] = 0.0, 0.0, False, 0
        babies["birth_year"] = yr
        babies["head_flag"] = False
        for c in p.columns:
            if c not in babies:
                babies[c] = np.nan
        st.persons = pd.concat([p, babies[p.columns]])
        st.log("births", len(nid))
        p = st.persons

    _partnering_and_separation(st)
    _leave_home(st)
    st.persons = assign_heads(st.persons)
    st.households = refresh_households(st.persons, st.households, st.adult_age)
    _normalise_roles(st)


def _normalise_roles(st: SimState) -> None:
    p = st.persons
    head_of = pd.Series(p.index[p["head_flag"].to_numpy(bool)].to_numpy(), index=p.loc[p["head_flag"].to_numpy(bool), "household_id"].to_numpy())
    head_id = p["household_id"].map(head_of)
    adult = (p["age"] >= st.adult_age).to_numpy()
    is_partner = (p["partner_id"].astype("float64") == head_id.astype("float64")).to_numpy() & ~p["head_flag"].to_numpy(bool)
    p["role"] = np.where(~adult, "CHILD", np.where(p["head_flag"], "HEAD", np.where(is_partner, "PARTNER", "OTHER_ADULT")))


def _resolve_estates(st: SimState, affected: np.ndarray, dead_ids: np.ndarray) -> None:
    p, hh = st.persons, st.households
    adults = (p["age"] >= st.adult_age).groupby(p["household_id"]).sum()
    for hid in affected:
        n_adult = int(adults.get(hid, 0))
        if n_adult > 0 or hid not in hh.index:
            continue
        net = float(hh.at[hid, "financial_assets"] + hh.at[hid, "housing_value"] - hh.at[hid, "mortgage_balance"] - hh.at[hid, "other_debt"])
        region = hh.at[hid, "region"]
        orphans = p.index[p["household_id"] == hid]
        target_hhs: list[int] = []
        if len(orphans):  # adopted by a random household of the region that still has an adult
            cand = hh.index[(hh["region"] == region) & hh.index.isin(adults.index[adults > 0])]
            cand = cand[cand != hid]
            if len(cand) == 0:
                continue
            u = crn.uniform(st.seed, "adopt", st.year, np.array([int(orphans[0])]))[0]
            adopt = int(cand[int(u * len(cand))])
            p.loc[orphans, "household_id"] = adopt
            target_hhs = [adopt]
            st.log("orphans_adopted", len(orphans))
        else:
            heirs = p[(p["parent1_id"].isin(dead_ids) | p["parent2_id"].isin(dead_ids)) & (p["age"] >= st.adult_age)]
            heirs = heirs[heirs["household_id"] != hid]
            if len(heirs):
                target_hhs = list(heirs["household_id"].unique())
            else:
                cand = hh.index[(hh["region"] == region) & hh.index.isin(adults.index[adults > 0])]
                if len(cand) == 0:
                    continue
                u = crn.uniform(st.seed, "distant_heir", st.year, np.array([int(hid)]))[0]
                target_hhs = [int(cand[int(u * len(cand))])]
        if net > 0:
            hh_add(st, "financial_assets", np.array(target_hhs), np.full(len(target_hhs), net / len(target_hhs)))
            hh = st.households
            st.log("estate_transferred_eur", net)
        hh.loc[hid, ["financial_assets", "housing_value", "mortgage_balance", "other_debt"]] = 0.0
    st.persons, st.households = p, hh.loc[hh.index.isin(p["household_id"])]


def _partnering_and_separation(st: SimState) -> None:
    e, yr = st.econ, st.year
    p, hh = st.persons, st.households
    # separation --------------------------------------------------------------------------
    heads = p[p["head_flag"] & p["partner_id"].notna()]
    sep = heads[crn.uniform(st.seed, "separation", yr, heads.index.to_numpy()) < e.v("separation_rate")]
    if len(sep):
        leavers = sep["partner_id"].astype("int64").to_numpy()
        leavers = leavers[np.isin(leavers, p.index)]
        sep = sep[sep["partner_id"].isin(leavers)]
        new_ids = _new_household_ids(st, len(leavers))
        old_hh = p.loc[leavers, "household_id"].to_numpy()
        half = 0.5 * st.households.loc[old_hh, "financial_assets"].to_numpy()
        _add_households(st, new_ids, p.loc[leavers, "region"].to_numpy())
        st.households.loc[old_hh, "financial_assets"] -= half
        st.households.loc[new_ids, "financial_assets"] = half
        p.loc[leavers, "household_id"] = new_ids
        p.loc[leavers, "partner_id"] = pd.NA
        p.loc[sep.index, "partner_id"] = pd.NA
        st.log("separations", len(leavers))
        p = assign_heads(p)
        st.households = refresh_households(p, st.households, st.adult_age)
    # partnering --------------------------------------------------------------------------
    hh = st.households
    single_heads = p[p["head_flag"] & p["partner_id"].isna() & p["age"].between(20, 60) & (hh["n_adults"].reindex(p["household_id"]).to_numpy() == 1)]
    pick = single_heads[crn.uniform(st.seed, "partnering", yr, single_heads.index.to_numpy()) < e.v("partnering_rate")]
    for _region, g in pick.groupby("region"):
        men = g[g["sex_or_demographic_stratum"] == "M"].sort_values("age")
        women = g[g["sex_or_demographic_stratum"] == "F"].sort_values("age")
        n = min(len(men), len(women))
        if n == 0:
            continue
        m_idx = np.linspace(0, len(men) - 1, n).round().astype(int)
        w_idx = np.linspace(0, len(women) - 1, n).round().astype(int)
        m, w = men.iloc[m_idx], women.iloc[w_idx]
        ok = np.abs(m["age"].to_numpy() - w["age"].to_numpy()) <= 10
        m, w = m[ok], w[ok]
        if len(m) == 0:
            continue
        mh, wh = m["household_id"].to_numpy(), w["household_id"].to_numpy()
        H = st.households
        # merge woman's household into man's: pool financial assets; keep the larger property, cash out the other
        keep_w = H.loc[wh, "housing_value"].to_numpy() > H.loc[mh, "housing_value"].to_numpy()
        eq_m = H.loc[mh, "housing_value"].to_numpy() - H.loc[mh, "mortgage_balance"].to_numpy()
        eq_w = H.loc[wh, "housing_value"].to_numpy() - H.loc[wh, "mortgage_balance"].to_numpy()
        cash_out = np.where(keep_w, eq_m, eq_w)
        for col in ("housing_value", "mortgage_balance"):
            H.loc[mh, col] = np.where(keep_w, H.loc[wh, col].to_numpy(), H.loc[mh, col].to_numpy())
        H.loc[mh, "housing_tenure"] = np.where(keep_w, H.loc[wh, "housing_tenure"].to_numpy(), H.loc[mh, "housing_tenure"].to_numpy())
        H.loc[mh, "financial_assets"] = H.loc[mh, "financial_assets"].to_numpy() + H.loc[wh, "financial_assets"].to_numpy() + np.clip(cash_out, 0, None)
        H.loc[mh, "other_debt"] = H.loc[mh, "other_debt"].to_numpy() + H.loc[wh, "other_debt"].to_numpy()
        mapping = dict(zip(wh, mh, strict=True))
        mask = p["household_id"].isin(wh)
        p.loc[mask, "household_id"] = p.loc[mask, "household_id"].map(mapping)
        p.loc[m.index, "partner_id"] = w.index.to_numpy()
        p.loc[w.index, "partner_id"] = m.index.to_numpy()
        st.households = H.drop(index=wh)
        st.log("partnerships", len(m))
    st.persons = p
    st.households = refresh_households(p, st.households, st.adult_age)


def _leave_home(st: SimState) -> None:
    p = st.persons
    adults_in_hh = (p["age"] >= st.adult_age).groupby(p["household_id"]).sum()
    cand = p[(~p["head_flag"]) & p["age"].between(20, 29) & p["partner_id"].isna() & (p["parent1_id"].notna() | p["parent2_id"].notna())
             & (adults_in_hh.reindex(p["household_id"]).to_numpy() >= 2)]
    go = cand[crn.uniform(st.seed, "leave_home", st.year, cand.index.to_numpy()) < st.econ.v("leave_home_prob")]
    if len(go) == 0:
        return
    new_ids = _new_household_ids(st, len(go))
    _add_households(st, new_ids, go["region"].to_numpy())
    p.loc[go.index, "household_id"] = new_ids
    st.persons = p
    st.log("left_home", len(go))


# --------------------------------------------------------------------------- education ------
def step_education(st: SimState) -> None:
    """Education: everyone enters school at 15 with LOW attainment. At 18 they complete upper secondary
    (LOW -> MEDIUM, probability implied by the observed 25-34 shares) and some continue to tertiary
    (probability implied by the observed HIGH share); tertiary students graduate (HIGH) at 22 and
    enter the labour market. Parental/household wealth shifts both odds (logit, centred at the median rank)."""
    e, yr, p = st.econ, st.year, st.persons
    ids = p.index.to_numpy()
    new15 = (p["age"] == 15) & (p["education_level"] == "NOT_ASSESSED")
    p.loc[new15, ["education_level", "work_state"]] = ["LOW", "STUDENT"]
    p.loc[new15, "tertiary"] = False
    st.log("entered_education", new15.sum())
    rank = wealth_rank(st).reindex(p["household_id"]).to_numpy()
    sens = e.v("education_wealth_sensitivity")
    stud = (p["work_state"] == "STUDENT").to_numpy()
    base_m, base_h = _upgrade_bases(st, p["region"].to_numpy())
    pm = expit(logit(np.clip(base_m, 1e-6, 1 - 1e-6)) + sens * (rank - 0.5))
    ph = expit(logit(np.clip(base_h, 1e-6, 1 - 1e-6)) + sens * (rank - 0.5))
    at18 = stud & (p["age"] == 18).to_numpy() & (p["education_level"] == "LOW").to_numpy()
    up_m = at18 & (crn.uniform(st.seed, "edu_secondary", yr, ids) < pm)
    p.loc[up_m, "education_level"] = "MEDIUM"
    go_tert = up_m & (crn.uniform(st.seed, "edu_tertiary", yr, ids) < ph)
    p.loc[go_tert, "tertiary"] = True
    grad = stud & p["tertiary"].fillna(False).to_numpy(bool) & (p["age"] >= 22).to_numpy()
    p.loc[grad, "education_level"] = "HIGH"
    p.loc[grad, "tertiary"] = False
    st.log("education_upgrades", up_m.sum() + grad.sum())
    # leave school: no tertiary plans and at least 18, or any student reaching 22 (initial stock) / 25
    tert = p["tertiary"].fillna(False).to_numpy(bool)
    leave = stud & (((~tert) & (p["age"] >= 18).to_numpy()) | (p["age"] >= 22).to_numpy())
    if leave.any():
        sub = p[leave]
        pe = _entry_probs(st, sub)
        v = crn.uniform(st.seed, "school_exit", yr, sub.index.to_numpy())
        state = np.where(v < pe[:, 0], "EMPLOYED", np.where(v < pe[:, 0] + pe[:, 1], "UNEMPLOYED", "INACTIVE"))
        p.loc[sub.index, "work_state"] = state
        st.log("school_leavers_employed", int((state == "EMPLOYED").sum()))
        p.loc[sub.index, "tertiary"] = False
        emp = sub.index[state == "EMPLOYED"]
        p.loc[emp, "arrangement"] = "EMPLOYEE"
        p.loc[emp, "hours"] = _hours(st, p.loc[emp], "school_hours")
        st.log("school_leavers", len(sub))
    st.persons = p


def _upgrade_bases(st: SimState, regions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upgrade probabilities implied by the observed 25-34 education shares (everyone starts LOW):
    P(LOW->MEDIUM at 18) = 1 - share_LOW; P(MEDIUM->HIGH at 21) = share_HIGH / (1 - share_LOW)."""
    t = st.education_table[st.education_table["age_min"] == 25].pivot_table(index="region", columns="education_level", values="share", aggfunc="sum")
    a = (1.0 - t["LOW"]).clip(1e-6, 1.0)
    b = (t["HIGH"] / a).clip(0.0, 1.0)
    return a.reindex(regions).fillna(a.mean()).to_numpy(), b.reindex(regions).fillna(b.mean()).to_numpy()


def _hours(st: SimState, sub: pd.DataFrame, stream: str) -> np.ndarray:
    e = st.econ
    share = np.where(sub["sex_or_demographic_stratum"].to_numpy() == "F", e.v("parttime_share_f"), e.v("parttime_share_m"))
    return np.where(crn.uniform(st.seed, stream, st.year, sub.index.to_numpy()) < share, e.v("parttime_hours"), 1.0)


def _entry_probs(st: SimState, sub: pd.DataFrame) -> np.ndarray:
    """P(EMPLOYED), P(UNEMPLOYED) for school leavers from the 25-34 cross-section (non-students)."""
    lab = st.labour_table[st.labour_table["age_min"] == 25].pivot_table(index=["region", "education_level"], columns="work_state", values="share", aggfunc="sum").fillna(0)
    key = list(zip(sub["region"], sub["education_level"], strict=True))
    rows = lab.reindex(key).fillna(0.0)
    non_student = (1.0 - rows.get("STUDENT", 0.0)).to_numpy()
    non_student = np.where(non_student > 0, non_student, 1.0)
    emp = rows.get("EMPLOYED", pd.Series(0.0, index=rows.index)).to_numpy() / non_student
    une = rows.get("UNEMPLOYED", pd.Series(0.0, index=rows.index)).to_numpy() / non_student
    return np.column_stack([emp, une])


# ------------------------------------------------------------------------ labour & firms ----
def _job_finding(st: SimState, sub: pd.DataFrame) -> np.ndarray:
    """Stationary finding probability f = s*E/U per region x age band x education, capped."""
    e = st.econ
    lab = st.labour_table.pivot_table(index=["region", "age_min", "education_level"], columns="work_state", values="share", aggfunc="sum").fillna(0)
    key = list(zip(sub["region"], _band_of(sub["age"].to_numpy()), sub["education_level"], strict=True))
    rows = lab.reindex(key).fillna(0.0)
    E, U = rows["EMPLOYED"].to_numpy(), rows["UNEMPLOYED"].to_numpy()
    f = np.where(U > 1e-9, e.v("job_separation_rate") * E / np.maximum(U, 1e-9), e.v("unemployment_reentry_cap"))
    return np.clip(f, 0.0, e.v("unemployment_reentry_cap"))


def _liquidate_firm(st: SimState, owner_ids: np.ndarray, policy: PolicyRules) -> None:
    """Wind up BV firms: retained equity is paid out as a taxed dividend to the owner's household."""
    f = st.firms.loc[st.firms["owner_person_id"].isin(owner_ids)]
    if f.empty:
        return
    wht = policy.g("dividend_withholding_tax", "rate")
    hid = st.persons.loc[f["owner_person_id"].to_numpy(), "household_id"].to_numpy()
    net = np.clip(f["equity"].to_numpy(), 0.0, None) * (1 - wht)
    hh_add(st, "financial_assets", hid, net)
    st.counters["gov:liquidation_dividends_gross"] = st.counters.get("gov:liquidation_dividends_gross", 0.0) + float(np.clip(f["equity"], 0, None).sum() * st.person_weight)
    st.log("bv_liquidations", len(f))
    st.firms = st.firms.drop(f.index)


def step_work(st: SimState, policy: PolicyRules) -> None:
    e, yr, p = st.econ, st.year, st.persons
    ids = p.index.to_numpy()
    rho, sg = e.v("wage_shock_rho"), e.v("wage_shock_sigma")
    p["resid"] = rho * p["resid"] + sg * crn.normal(st.seed, "resid", yr, ids)
    p["biz"] = e.v("business_shock_rho") * p["biz"] + e.v("business_shock_sigma") * crn.normal(st.seed, "biz", yr, ids)
    ret_age = policy.g("pensions", "retirement_age")

    # retirement -------------------------------------------------------------------------
    ret = (p["age"] >= ret_age) & p["work_state"].isin(["EMPLOYED", "UNEMPLOYED", "INACTIVE"])
    if ret.any():
        st.log("retirements_employed", int((ret & (p["work_state"] == "EMPLOYED")).sum()))
        bv_ret = p.index[ret & (p["arrangement"] == "BV_OWNER_MANAGER")].to_numpy()
        _liquidate_firm(st, bv_ret, policy)
        p.loc[ret, "work_state"] = "RETIRED"
        p.loc[ret, "arrangement"] = "NONE"
        p.loc[ret, "unemp_elig"] = False
        st.log("retirements", ret.sum())

    # job loss and job finding -----------------------------------------------------------
    emp_e = (p["work_state"] == "EMPLOYED") & (p["arrangement"] == "EMPLOYEE")
    lose = emp_e & (crn.uniform(st.seed, "job_loss", yr, ids) < e.v("job_separation_rate"))
    p.loc[lose, "work_state"] = "UNEMPLOYED"
    p.loc[lose, "arrangement"] = "NONE"
    p.loc[lose, "unemp_elig"] = crn.uniform(st.seed, "elig", yr, ids[lose.to_numpy()]) < e.v("unemployment_eligibility_share")
    p.loc[lose, "unemp_years"] = 0
    st.log("job_losses", lose.sum())

    unemp = (p["work_state"] == "UNEMPLOYED") & (~lose) & (p["age"] < ret_age)
    if unemp.any():
        sub = p[unemp]
        find = crn.uniform(st.seed, "job_find", yr, sub.index.to_numpy()) < _job_finding(st, sub)
        found = sub.index[find]
        p.loc[found, ["work_state", "arrangement"]] = ["EMPLOYED", "EMPLOYEE"]
        p.loc[found, "hours"] = _hours(st, p.loc[found], "find_hours")
        p.loc[found, "unemp_elig"] = False
        p.loc[sub.index[~find], "unemp_years"] += 1
        st.log("job_findings", find.sum())

    # labour-force exit and re-entry (re-entry rate derived so cross-sectional shares are stationary)
    lf_out = (p["work_state"] == "EMPLOYED") & (p["arrangement"] == "EMPLOYEE") & (p["age"] < ret_age) & (~lose) & (crn.uniform(st.seed, "lf_exit", yr, ids) < e.v("labour_force_exit_rate"))
    inact = (p["work_state"] == "INACTIVE") & (p["age"] < ret_age) & (p["age"] >= 18)
    if inact.any():
        sub = p[inact]
        lab = st.labour_table.pivot_table(index=["region", "age_min", "education_level"], columns="work_state", values="share", aggfunc="sum").fillna(0)
        rows = lab.reindex(list(zip(sub["region"], _band_of(sub["age"].to_numpy()), sub["education_level"], strict=True))).fillna(0.0)
        g = np.clip(e.v("labour_force_exit_rate") * rows["EMPLOYED"].to_numpy() / np.maximum(rows["INACTIVE"].to_numpy(), 1e-9), 0.0, 0.5)
        u3 = crn.uniform(st.seed, "lf_entry", yr, sub.index.to_numpy())
        enter = u3 < g
        to_emp = enter & (crn.uniform(st.seed, "lf_entry_kind", yr, sub.index.to_numpy()) < e.v("inactive_entry_to_employment_share"))
        to_une = enter & ~to_emp
        ne = sub.index[to_emp]
        p.loc[ne, ["work_state", "arrangement"]] = ["EMPLOYED", "EMPLOYEE"]
        p.loc[ne, "hours"] = _hours(st, p.loc[ne], "lf_hours")
        p.loc[sub.index[to_une], "work_state"] = "UNEMPLOYED"
        p.loc[sub.index[to_une], ["unemp_elig", "unemp_years"]] = [False, 0]
        st.log("labour_force_entries", enter.sum())
        st.log("lf_entries_employed", int(to_emp.sum()))
        st.log("lf_entries_unemployed", int(to_une.sum()))
    p.loc[lf_out, ["work_state", "arrangement"]] = ["INACTIVE", "NONE"]
    st.log("labour_force_exits", lf_out.sum())

    # exits from self-employment ---------------------------------------------------------
    for arrn, rate_name in (("SOLE_PROPRIETOR", "sole_prop_exit_rate"), ("BV_OWNER_MANAGER", "bv_exit_rate")):
        m = (p["work_state"] == "EMPLOYED") & (p["arrangement"] == arrn)
        out = m & (crn.uniform(st.seed, "exit_" + arrn, yr, ids) < e.v(rate_name))
        if out.any():
            if arrn == "BV_OWNER_MANAGER":
                _liquidate_firm(st, p.index[out].to_numpy(), policy)
            fail = out & (crn.uniform(st.seed, "exit_fail_" + arrn, yr, ids) < 0.5)
            p.loc[out, ["work_state", "arrangement"]] = ["EMPLOYED", "EMPLOYEE"]
            p.loc[out, "hours"] = 1.0
            p.loc[fail, ["work_state", "arrangement"]] = ["UNEMPLOYED", "NONE"]
            p.loc[fail, "unemp_elig"] = False
            p.loc[fail, "unemp_years"] = 0
            st.log("exits_" + arrn, out.sum())

    # entry into self-employment ---------------------------------------------------------
    emp_e = (p["work_state"] == "EMPLOYED") & (p["arrangement"] == "EMPLOYEE") & (p["age"] < ret_age) & ~lose
    cand = p[emp_e]
    if len(cand):
        tab = st.arrangement_table
        mids = np.array([(a + b) / 2 for a, b in AGE_BANDS[:-1]] + [70.0])
        base = {}
        for kind, rate_name in (("SOLE_PROPRIETOR", "sole_prop_exit_rate"), ("BV_OWNER_MANAGER", "bv_exit_rate")):
            hz = np.zeros(len(cand))
            for region, idx in cand.groupby("region").indices.items():
                sub = tab[(tab["region"] == region) & (tab["primary_work_arrangement"] == kind)].sort_values("age_min")
                sh = sub["share"].to_numpy()
                a = cand["age"].to_numpy()[idx].astype(float)
                s0, s1 = np.interp(a, mids, sh), np.interp(a + 1, mids, sh)
                x = e.v(rate_name)
                # hazard that keeps the cross-sectional age profile of shares as a cohort ages
                hz[idx] = np.clip((s1 - s0 * (1 - x)) / np.maximum(1 - s0, 1e-9), 0.0, None)
            base[kind] = hz
        base_sp, base_bv = base["SOLE_PROPRIETOR"], base["BV_OWNER_MANAGER"]
        rank = wealth_rank(st).reindex(cand["household_id"]).to_numpy()
        mult = np.exp(e.v("entry_wealth_sensitivity") * (rank - 0.5)) * np.where(cand["education_level"].to_numpy() == "HIGH", e.v("entry_education_high_multiplier"), 1.0)
        p_sp = np.minimum(base_sp * mult, e.v("max_entry_prob"))
        p_bv = np.minimum(base_bv * mult, e.v("max_entry_prob"))
        u = crn.uniform(st.seed, "entry", yr, cand.index.to_numpy())
        fin = st.households.loc[cand["household_id"], "financial_assets"].to_numpy()
        go_sp = (u < p_sp) & (fin >= e.v("business_entry_cost_sole_prop"))
        go_bv = (~go_sp) & (u >= p_sp) & (u < p_sp + p_bv) & (fin >= e.v("bv_capital_contribution"))
        if go_sp.any():
            sp_ids = cand.index[go_sp]
            p.loc[sp_ids, "arrangement"] = "SOLE_PROPRIETOR"
            p.loc[sp_ids, "hours"] = 1.0
            hh_add(st, "financial_assets", cand.loc[sp_ids, "household_id"].to_numpy(), -np.full(len(sp_ids), float(e.v("business_entry_cost_sole_prop"))), floor_zero=True)
            st.log("entries_SOLE_PROPRIETOR", len(sp_ids))
        if go_bv.any():
            bv_ids = cand.index[go_bv]
            p.loc[bv_ids, "arrangement"] = "BV_OWNER_MANAGER"
            p.loc[bv_ids, "hours"] = 1.0
            hh_add(st, "financial_assets", cand.loc[bv_ids, "household_id"].to_numpy(), -np.full(len(bv_ids), float(e.v("bv_capital_contribution"))), floor_zero=True)
            new = pd.DataFrame({"firm_id": bv_ids, "owner_person_id": bv_ids, "equity": float(e.v("bv_capital_contribution")), "active": True}, index=bv_ids)
            st.firms = pd.concat([st.firms, new])
            st.log("entries_BV_OWNER_MANAGER", len(bv_ids))
    st.persons = p


# --------------------------------------------------------------------------- finances ------
def step_finances(st: SimState, policy: PolicyRules, fl: Flows | None = None) -> Flows:
    """Fiscal cash flows -> consumption & saving -> asset dynamics. Returns the year's flows."""
    e, p, hh = st.econ, st.persons, st.households
    fl = fl or compute_flows(st, policy)
    H, Pf, Ff = fl.h, fl.p, fl.firms
    opening = hh["financial_assets"].to_numpy(float).copy()

    credit = H["credit_event"].to_numpy()
    avail = (H["disp"] - H["debt_service"]).to_numpy()
    cons = H["consumption"].to_numpy()

    # home purchase by renters (explicit down payment from financial assets)
    head = p[p["head_flag"]]
    hh_head = head.set_index("household_id")
    age_h = hh_head["age"].reindex(hh.index).to_numpy(float)
    renter = (hh["housing_tenure"] == "RENTER").to_numpy()
    income_ref = np.clip(H["market_a"].to_numpy(), 0, None) + 0.1 * st.level_scale * e.v("wage_base_medium")
    price = e.v("house_to_income") * income_ref
    down = e.v("home_purchase_down_payment") * price
    fin_after = np.clip(opening + avail - cons, 0.0, None)
    wants = renter & (age_h >= 25) & (age_h <= 45) & (crn.uniform(st.seed, "home_purchase", st.year, hh.index.to_numpy()) < e.v("home_purchase_prob"))
    buy = wants & (fin_after >= down)
    down_paid = np.where(buy, down, 0.0)

    fin_close = fin_after - down_paid
    fin_close = np.where(credit > 0, 0.0, fin_close)
    hh["financial_assets"] = fin_close
    hh["other_debt"] = hh["other_debt"].to_numpy() + credit
    hh["mortgage_balance"] = np.clip(hh["mortgage_balance"].to_numpy() - H["repay"].to_numpy(), 0.0, None) + np.where(buy, price - down, 0.0)
    hh["housing_value"] = hh["housing_value"].to_numpy() * (1 + e.v("house_price_growth")) + np.where(buy, price, 0.0)
    hh.loc[buy, "housing_tenure"] = "MORTGAGED_OWNER"
    paid_off = (hh["housing_tenure"] == "MORTGAGED_OWNER") & (hh["mortgage_balance"] < 1.0)
    hh.loc[paid_off, "housing_tenure"] = "OUTRIGHT_OWNER"
    hh.loc[paid_off, "mortgage_balance"] = 0.0
    st.log("home_purchases", buy.sum())
    st.audit = pd.DataFrame({"opening": opening, "disp": H["disp"].to_numpy(), "debt_service": H["debt_service"].to_numpy(), "consumption": cons,
                             "credit_event": credit, "down_payment": down_paid, "closing": fin_close}, index=hh.index)

    # firms: retained profit, insolvency (limited liability) ------------------------------
    if len(st.firms):
        st.firms["equity"] = st.firms["equity"].to_numpy() + Ff["retained_delta"].reindex(st.firms.index).fillna(0.0).to_numpy()
        broke = st.firms.index[st.firms["equity"] < 0]
        if len(broke):
            p.loc[broke, ["work_state", "arrangement"]] = ["UNEMPLOYED", "NONE"]
            p.loc[broke, "unemp_elig"] = False
            p.loc[broke, "unemp_years"] = 0
            st.firms = st.firms.drop(broke)
            st.log("bv_insolvencies", len(broke))

    # pension accrual and wage memory ------------------------------------------------------
    pc = policy.components["pensions"]
    working = (p["work_state"] == "EMPLOYED") & (p["age"] < pc["retirement_age"])
    cover = Pf["wage"] + pc["self_employed_accrual_factor"] * (np.clip(Pf["profit_sp"], 0, None) + Pf["dir_pay"])
    p["pension_rights"] = p["pension_rights"] + np.where(working, pc["accrual_rate_per_year"] * cover, 0.0)
    p["last_wage"] = np.where((Pf["wage"] > 0), Pf["wage"], p["last_wage"])
    p["unemp_years"] = p["unemp_years"]
    st.persons = p
    return fl
