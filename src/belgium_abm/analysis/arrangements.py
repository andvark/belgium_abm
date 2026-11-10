"""First working scenario (spec section 12): employee vs sole proprietor vs BV owner-manager.

Identical households (same education, skill, region, family, starting assets) differ ONLY in how
the same economic activity is organised. Shocks are paired across arrangements (common random
numbers): the earnings residual is shared by all three, the business shock by the two business
forms. Everything else (taxes, contributions, pensions, consumption, saving, housing purchase,
liquidation of the BV at retirement) runs through the same engines as the population model.
The comparison is designed to reveal trade-offs; it does not presume any form is advantaged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from belgium_abm.analysis.plots import plot_timeseries  # noqa: F401  (re-export convenience)
from belgium_abm.config import ModelConfig
from belgium_abm.engines import crn
from belgium_abm.engines.dynamics import _liquidate_firm, step_finances
from belgium_abm.engines.economy import EconomyConfig
from belgium_abm.engines.population import generate_population
from belgium_abm.engines.simulation import initial_state
from belgium_abm.engines.state import SimState, assign_heads, labour_productivity, net_worth, refresh_households
from belgium_abm.io.loaders import load_targets
from belgium_abm.io.outputs import write_json
from belgium_abm.policies.rules import PolicyRules, statutory_pension
from belgium_abm.run_sim import load_inputs
from belgium_abm.scenarios import Scenario

ARRANGEMENTS = ("EMPLOYEE", "SOLE_PROPRIETOR", "BV_OWNER_MANAGER")


def _mini_state(econ: EconomyConfig, policy: PolicyRules, cfg: ModelConfig, level_scale: float, seed: int, replicates: int, education: str, region: str,
                children: int, partner: str, start_age: int, assets_years: float) -> SimState:
    rows, hrows = [], []
    pid = 0
    hid = 0
    var_r = econ.v("wage_shock_sigma") ** 2 / (1 - econ.v("wage_shock_rho") ** 2)
    var_b = econ.v("business_shock_sigma") ** 2 / (1 - econ.v("business_shock_rho") ** 2)
    base_income = level_scale * float(econ.v({"LOW": "wage_base_low", "MEDIUM": "wage_base_medium", "HIGH": "wage_base_high"}[education]))
    for ai, arr in enumerate(ARRANGEMENTS):
        for m in range(replicates):
            hid += 1
            r0 = crn.normal(seed, "arr_init_resid", 0, np.array([m]))[0] * np.sqrt(var_r)
            b0 = crn.normal(seed, "arr_init_biz", 0, np.array([m]))[0] * np.sqrt(var_b)
            members = [(start_age, "M", arr, "EMPLOYED", "HEAD")]
            if partner != "none":
                members.append((start_age, "F", "EMPLOYEE" if partner == "employee" else "NONE", "EMPLOYED" if partner == "employee" else "INACTIVE", "PARTNER"))
            members += [(max(2 + 3 * k, 0), "F" if k % 2 else "M", "NONE", "NONE", "CHILD") for k in range(children)]
            ids = []
            for age, sex, a, ws, role in members:
                pid += 1
                ids.append(pid)
                rows.append({"person_id": pid, "household_id": hid, "age": age, "sex_or_demographic_stratum": sex, "region": region,
                             "education_level": education if age >= 15 else "NOT_ASSESSED", "work_state": ws, "arrangement": a, "role": role,
                             "partner_id": pd.NA, "parent1_id": pd.NA, "parent2_id": pd.NA, "weight": 1.0, "skill_z": 0.0,
                             "resid": r0, "biz": b0, "hours": 1.0, "last_wage": 0.0, "pension_rights": 0.0, "unemp_elig": False, "unemp_years": 0,
                             "birth_year": -age, "tertiary": False, "head_flag": role == "HEAD", "rep": m, "arr_index": ai})
            if partner != "none":
                rows[-len(members) + 0]["partner_id"] = ids[1]
                rows[-len(members) + 1]["partner_id"] = ids[0]
            hrows.append({"household_id": hid, "region": region, "housing_tenure": "RENTER", "housing_value": 0.0, "mortgage_balance": 0.0,
                          "financial_assets": assets_years * base_income, "other_debt": 0.0, "weight": 1.0, "household_type": "SINGLE_ADULT",
                          "n_adults": 1, "n_children": 0, "num_dependents": 0})
    p = pd.DataFrame(rows)
    p.index = p["person_id"].to_numpy()
    p["partner_id"] = p["partner_id"].astype("Int64")
    p["parent1_id"] = p["parent1_id"].astype("Int64")
    p["parent2_id"] = p["parent2_id"].astype("Int64")
    h = pd.DataFrame(hrows)
    h.index = h["household_id"].to_numpy()
    e = pd.DataFrame
    st = SimState(year=0, seed=seed, persons=p, households=h, firms=pd.DataFrame({"firm_id": [], "owner_person_id": [], "equity": [], "active": []}, dtype=float),
                  econ=econ, adult_age=cfg.adult_age, equivalence=cfg.equivalence, level_scale=level_scale, person_weight=1.0, household_weight=1.0,
                  labour_table=e(), arrangement_table=e(), education_table=e(), mortality=e(), fertility=e(),
                  next_person_id=pid + 1, next_household_id=hid + 1)
    # BV firms and start-up costs, paid out of the same starting assets
    bv = p.index[(p["arrangement"] == "BV_OWNER_MANAGER") & p["head_flag"]]
    st.firms = pd.DataFrame({"firm_id": bv, "owner_person_id": bv, "equity": float(econ.v("bv_capital_contribution")), "active": True}, index=bv)
    hh_bv = p.loc[bv, "household_id"].to_numpy()
    st.households.loc[hh_bv, "financial_assets"] -= econ.v("bv_capital_contribution")
    sp = p.index[(p["arrangement"] == "SOLE_PROPRIETOR") & p["head_flag"]]
    st.households.loc[p.loc[sp, "household_id"].to_numpy(), "financial_assets"] -= econ.v("business_entry_cost_sole_prop")
    st.households["financial_assets"] = st.households["financial_assets"].clip(lower=0.0)
    st.persons = assign_heads(st.persons)
    st.households = refresh_households(st.persons, st.households, st.adult_age)
    # initial wage memory for employees
    prod = labour_productivity(st.persons, econ, 0, level_scale)
    st.persons["last_wage"] = np.where(st.persons["arrangement"] == "EMPLOYEE", prod, 0.0)
    return st


def run_arrangement_comparison(cfg: ModelConfig, out: Path, *, seed: int, years: int, replicates: int, education: str, region: str, children: int, partner: str,
                               start_age: int, policy_file: str | None = None, scenario_path: str | None = None, assets_years: float = 0.5) -> pd.DataFrame:
    inp = load_inputs(cfg, policy_file)
    policy = inp.policy
    if scenario_path:
        policy = Scenario.load(scenario_path).apply_policy(policy)
    econ = inp.econ
    # level calibration identical to the population model (small population just to obtain the scale)
    small = cfg.with_overrides(generator={"n_persons": 8000})
    targets = load_targets(small)
    pop = generate_population(small, targets, seed)
    _, info = initial_state(pop.persons, pop.households, targets, inp.demography, econ, policy, cfg.adult_age, cfg.equivalence, seed, inp.target_median)
    scale = info["level_scale"]

    st = _mini_state(econ, policy, cfg, scale, seed, replicates, education, region, children, partner, start_age, assets_years)
    ret_age = policy.g("pensions", "retirement_age")
    rho_r, sg_r = econ.v("wage_shock_rho"), econ.v("wage_shock_sigma")
    rho_b, sg_b = econ.v("business_shock_rho"), econ.v("business_shock_sigma")
    recs = []
    p0 = st.persons
    head_ix = p0.index[p0["head_flag"]]
    rep_of = p0["rep"].to_numpy()
    for yr in range(1, years + 1):
        st.year = yr
        p = st.persons
        p["age"] = p["age"] + 1
        shock_r = crn.normal(seed, "arr_resid", yr, np.arange(replicates))
        shock_b = crn.normal(seed, "arr_biz", yr, np.arange(replicates))
        p["resid"] = rho_r * p["resid"] + sg_r * shock_r[rep_of]
        p["biz"] = rho_b * p["biz"] + sg_b * shock_b[rep_of]
        ret = (p["age"] >= ret_age) & p["work_state"].isin(["EMPLOYED"])
        liq_hh = pd.Series(0.0, index=st.households.index)
        if ret.any():
            bv_ret = p.index[ret & (p["arrangement"] == "BV_OWNER_MANAGER")].to_numpy()
            if len(bv_ret):  # withholding tax on the liquidation dividend counts as a fiscal contribution of the owner's household
                eq = np.clip(st.firms.loc[bv_ret, "equity"].to_numpy(), 0.0, None)
                liq_hh.loc[p.loc[bv_ret, "household_id"].to_numpy()] = policy.g("dividend_withholding_tax", "rate") * eq
            _liquidate_firm(st, bv_ret, policy)
            p.loc[ret, ["work_state", "arrangement"]] = ["RETIRED", "NONE"]
        st.counters.pop("gov:liquidation_dividends_gross", None)
        fl = step_finances(st, policy)
        P, H, F = fl.p, fl.h, fl.firms
        by_h = lambda col: P[col].groupby(st.persons["household_id"]).sum().reindex(st.households.index, fill_value=0.0)  # noqa: E731
        firm_tax = pd.Series(0.0, index=st.households.index)
        if len(F):
            firm_tax = pd.Series(F["corporate_tax"].to_numpy(), index=st.persons.loc[F.index, "household_id"].to_numpy()).groupby(level=0).sum().reindex(st.households.index, fill_value=0.0)
        taxes = H["pit"] + H["ssc_personal"] + H["cap_tax"] + H["div_tax"] + H["vat"] + by_h("employer_ssc") + firm_tax + liq_hh
        net_fiscal = taxes - (H["child_ben"] + H["assist"] + H["pension"] + H["unemp"])
        nw = net_worth(st)
        heads = st.persons[st.persons["head_flag"]]
        hh_of_head = heads["household_id"].to_numpy()
        df = pd.DataFrame({"arrangement": heads["arr_index"].map(dict(enumerate(ARRANGEMENTS))).to_numpy(), "replicate": heads["rep"].to_numpy(), "year": yr,
                           "age": heads["age"].to_numpy(), "disposable_income": H["disp"].reindex(hh_of_head).to_numpy(),
                           "eq_disposable_income": H["eq_disp"].reindex(hh_of_head).to_numpy(), "market_income": H["market_a"].reindex(hh_of_head).to_numpy(),
                           "consumption": H["consumption"].reindex(hh_of_head).to_numpy(), "net_fiscal_contribution": net_fiscal.reindex(hh_of_head).to_numpy(),
                           "financial_assets": st.households["financial_assets"].reindex(hh_of_head).to_numpy(),
                           "net_worth": nw.reindex(hh_of_head).to_numpy(), "pension_rights_annual": heads["pension_rights"].to_numpy(),
                           "retired": (heads["work_state"] == "RETIRED").to_numpy()})
        recs.append(df)
    long = pd.concat(recs, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "arrangements_by_replicate_year.csv", index=False, lineterminator="\n")

    q = lambda s, a: s.quantile(a)  # noqa: E731
    by = long.groupby(["arrangement", "year"])
    ts = by.agg(age=("age", "first"), mean_disposable=("disposable_income", "mean"), p10_disposable=("disposable_income", lambda s: q(s, .1)), median_disposable=("disposable_income", "median"),
                p90_disposable=("disposable_income", lambda s: q(s, .9)), mean_net_worth=("net_worth", "mean"), p10_net_worth=("net_worth", lambda s: q(s, .1)),
                p90_net_worth=("net_worth", lambda s: q(s, .9)), mean_net_fiscal=("net_fiscal_contribution", "mean"), mean_pension_rights=("pension_rights_annual", "mean")).reset_index()
    ts.to_csv(out / "arrangements_by_year.csv", index=False, lineterminator="\n")

    piv = long.pivot_table(index="replicate", columns="arrangement", values="disposable_income", aggfunc="sum")
    last = long[long.year == years].set_index(["replicate", "arrangement"])
    ret_year = long[long.retired].groupby(["replicate", "arrangement"])["year"].min()
    rows = []
    for arr in ARRANGEMENTS:
        g = long[long.arrangement == arr]
        cum = piv[arr]
        vol = g.sort_values(["replicate", "year"]).groupby("replicate")["disposable_income"].apply(lambda s: np.std(np.diff(np.log(np.clip(s.to_numpy(), 1.0, None)))))
        fin_min = g.groupby("replicate")["financial_assets"].min()
        working = g[~g.retired]
        rows.append({"arrangement": arr, "lifetime_disposable_income_mean": cum.mean(), "lifetime_disposable_income_p10": cum.quantile(.1), "lifetime_disposable_income_p90": cum.quantile(.9),
                     "share_below_employee_lifetime_income": float((cum < piv["EMPLOYEE"]).mean()) if arr != "EMPLOYEE" else np.nan,
                     "mean_working_year_disposable_income": working["disposable_income"].mean(), "income_volatility_sd_log_change": vol.mean(),
                     "net_worth_at_end_mean": last.xs(arr, level="arrangement")["net_worth"].mean(), "net_worth_at_end_p10": last.xs(arr, level="arrangement")["net_worth"].quantile(.1),
                     "annual_pension_entitlement_at_end_mean": statutory_pension(last.xs(arr, level="arrangement")["pension_rights_annual"].to_numpy(), policy).mean(),
                     "lifetime_net_fiscal_contribution_mean": g.groupby("replicate")["net_fiscal_contribution"].sum().mean(),
                     "min_financial_assets_p10": fin_min.quantile(.1)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "arrangements_summary.csv", index=False, lineterminator="\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (col, lab, lo, hi) in zip(axes, [("mean_disposable", "Household disposable income (EUR/yr)", "p10_disposable", "p90_disposable"),
                                             ("mean_net_worth", "Household net worth incl. BV equity (EUR)", "p10_net_worth", "p90_net_worth"),
                                             ("mean_net_fiscal", "Net fiscal contribution (EUR/yr)", None, None)], strict=True):
        for arr in ARRANGEMENTS:
            g = ts[ts.arrangement == arr]
            ax.plot(g["age"], g[col], label=arr)
            if lo:
                ax.fill_between(g["age"], g[lo], g[hi], alpha=0.12)
        ax.set_xlabel("age of household head")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Same economic activity, three legal arrangements ({replicates} paired replicates; mean and 10-90% band)", fontsize=11)
    fig.text(0.5, 0.005, policy.banner[:240], ha="center", fontsize=6, color="firebrick")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out / "arrangements_lifecycle.png", dpi=130)
    plt.close(fig)

    meta = {"education": education, "region": region, "children": children, "partner": partner, "start_age": start_age, "years": years, "replicates": replicates, "seed": seed,
            "policy": policy.name, "policy_fingerprint": policy.fingerprint(), "policy_verified": policy.verified, "banner": policy.banner, "level_scale": scale,
            "sole_prop_productivity_multiplier": econ.v("sole_prop_productivity_multiplier"), "bv_productivity_multiplier": econ.v("bv_productivity_multiplier"),
            "assets_years_of_income": assets_years}
    write_json(meta, out / "arrangements_manifest.json")
    L = ["# Employee vs sole proprietor vs BV owner-manager", "", f"> **{policy.banner}**", "",
         "> Stylised business parameters (assumptions) and an illustrative policy: this exposes trade-offs under stated assumptions; it is not a statement about Belgium.", "",
         f"Household: head aged {start_age} ({education} education, {region}), partner: {partner}, children: {children}; {years} years; {replicates} paired replicates (common random numbers).",
         "Same skill, same earnings residual path and the same starting assets in every arrangement; the BV pays its capital in out of those assets, retains profit, pays director pay and dividends, and is liquidated (dividend taxed) at retirement.", "",
         "| arrangement | lifetime disposable income (mean) | p10 | p90 | P(below employee) | volatility | net worth at end (mean) | p10 | pension at end | lifetime net fiscal contribution |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        L.append(f"| {r['arrangement']} | {r['lifetime_disposable_income_mean']:,.0f} | {r['lifetime_disposable_income_p10']:,.0f} | {r['lifetime_disposable_income_p90']:,.0f} | "
                 f"{'' if pd.isna(r['share_below_employee_lifetime_income']) else format(r['share_below_employee_lifetime_income'], '.0%')} | {r['income_volatility_sd_log_change']:.3f} | "
                 f"{r['net_worth_at_end_mean']:,.0f} | {r['net_worth_at_end_p10']:,.0f} | {r['annual_pension_entitlement_at_end_mean']:,.0f} | {r['lifetime_net_fiscal_contribution_mean']:,.0f} |")
    (out / "arrangements_report.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    return summary
