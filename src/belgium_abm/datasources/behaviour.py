"""Behavioural parameters derived from PUBLIC aggregate statistics (replacing earlier assumptions).

Each row: name, value, status (observed | derived), unit, source, year, definition. None of these is
estimated from individual-level longitudinal data; where an aggregate is only an approximation to the
model parameter, the definition says so. Validation of the flows against the model is done in
``analysis/behaviour_validation``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

from belgium_abm.datasources.belgium import Build, EDU


def _row(name: str, value: float, status: str, unit: str, source: str, year: int, definition: str) -> dict:
    return {"name": name, "value": float(value), "status": status, "unit": unit, "source": source, "year": year, "definition": definition}


def labour_transitions(b: Build) -> tuple[np.ndarray, list[str]]:
    """Annual transition matrix EMP/UNE/INAC (ages 15-74) from the last 8 quarterly LFS flow tables (Markov annualisation)."""
    d = b.es_any("lfsi_long_q", geo="BE", s_adj="NSA", sex="T", unit="THS_PER", lastTimePeriod="8")
    qs = sorted(d.time.unique())
    states = ["EMP", "UNE", "INAC"]
    mats = []
    for q in qs:
        x = d[d.time == q].set_index("wstatus").value
        m = np.array([[x[f"{a}_{c}"] for c in states] for a in states])
        mats.append(m / m.sum(axis=1, keepdims=True))
    quarterly = np.mean(mats, axis=0)
    b.diagnostics["labour_flow_quarters"] = f"{qs[0]}..{qs[-1]}"
    return np.linalg.matrix_power(quarterly, 4), states


def build_behaviour(b: Build) -> pd.DataFrame:
    rows: list[dict] = []
    y = b.year
    # --- labour-market flows --------------------------------------------------------------
    A, S = labour_transitions(b)
    ix = {s: i for i, s in enumerate(S)}
    src = f"Eurostat lfsi_long_q (EU-LFS quarterly flows, {b.diagnostics['labour_flow_quarters']}, NSA, quarterly matrix averaged and raised to the 4th power)"
    rows.append(_row("job_separation_rate", A[ix["EMP"], ix["UNE"]], "derived", "per year", src, y, "Annual probability an employed person is unemployed a year later (all ages 15-74, all employment statuses)"))
    rows.append(_row("labour_force_exit_total_observed", A[ix["EMP"], ix["INAC"]], "derived", "per year", src, y,
                     "Annual probability an employed person is outside the labour force a year later; includes retirement. The model's non-retirement exit rate is derived as this minus the modelled retirement flow"))
    rows.append(_row("observed_flow_unemployed_to_employed", A[ix["UNE"], ix["EMP"]], "derived", "per year", src, y, "VALIDATION ONLY: annual unemployment-to-employment probability"))
    rows.append(_row("observed_flow_inactive_to_employed", A[ix["INAC"], ix["EMP"]], "derived", "per year", src, y, "VALIDATION ONLY: annual inactivity-to-employment probability (inactive includes students and retirees aged <= 74)"))
    rows.append(_row("observed_flow_inactive_to_unemployed", A[ix["INAC"], ix["UNE"]], "derived", "per year", src, y, "VALIDATION ONLY: annual inactivity-to-unemployment probability"))
    inac_to_emp, inac_to_une = A[ix["INAC"], ix["EMP"]], A[ix["INAC"], ix["UNE"]]
    rows.append(_row("inactive_entry_to_employment_share", inac_to_emp / (inac_to_emp + inac_to_une), "derived", "share", src, y, "Share of labour-force entrants from inactivity who go to employment rather than unemployment"))

    # --- business demography --------------------------------------------------------------
    ry = y - 1
    bd = b.es_any("bd_l_form", geo="BE", nace_r2="B-S_X_O_S94", time=str(ry))
    p = bd.pivot_table(index="indic_sbs", columns="leg_form", values="value")
    rows.append(_row("sole_prop_exit_rate", p.loc["ENT_DTHR_PC", "ENT_SOLE"] / 100, "observed", "per year", f"Eurostat bd_l_form, enterprise death rate of sole proprietorships, {ry}", ry,
                     "Annual enterprise death rate used as the exit rate of sole proprietors (a closure is not always an owner leaving self-employment)"))
    rows.append(_row("bv_exit_rate", p.loc["ENT_DTHR_PC", "ENT_LL"] / 100, "observed", "per year", f"Eurostat bd_l_form, enterprise death rate of limited-liability enterprises, {ry}", ry,
                     "Annual closure rate of limited-liability enterprises; includes insolvency and winding-up at owner retirement, which the model also generates separately"))
    rows.append(_row("observed_birth_rate_sole_prop", p.loc["ENT_BRTHR_PC", "ENT_SOLE"] / 100, "observed", "per year", f"Eurostat bd_l_form, {ry}", ry, "VALIDATION ONLY: enterprise birth rate of sole proprietorships"))
    rows.append(_row("observed_birth_rate_bv", p.loc["ENT_BRTHR_PC", "ENT_LL"] / 100, "observed", "per year", f"Eurostat bd_l_form, {ry}", ry, "VALIDATION ONLY: enterprise birth rate of limited-liability enterprises"))

    # --- hours --------------------------------------------------------------------------------
    pt = b.es_any("lfsa_eppga", geo="BE", time=str(y), age="Y15-64").set_index("sex").value
    rows.append(_row("parttime_share_f", pt["F"] / 100, "observed", "share", f"Eurostat lfsa_eppga, ages 15-64, {y}", y, "Part-time share of female employment"))
    rows.append(_row("parttime_share_m", pt["M"] / 100, "observed", "share", f"Eurostat lfsa_eppga, ages 15-64, {y}", y, "Part-time share of male employment"))
    wh = b.es_any("lfsa_ewhuis", geo="BE", time=str(y), wstatus="EMP", isco08="TOTAL", age="Y15-64", sex="T").set_index("worktime").value
    rows.append(_row("parttime_hours", wh["PT"] / wh["FT"], "derived", "fraction of full time", f"Eurostat lfsa_ewhuis, usual weekly hours part-time / full-time, {y}", y, "Part-time hours as a fraction of full-time hours"))

    # --- earnings structure (Structure of Earnings Survey 2022, employees in firms with >= 10 employees) ----
    ey = 2022
    ea = b.es_any("earn_ses22_14", geo="BE", time=str(ey), sex="T", indic_se="ERN", unit="EUR", isco08="TOTAL", sizeclas="GE10").set_index("age").value
    ages = np.array([25.0, 39.5, 56.0])
    x = (ages - 40.0) / 10.0
    lv = np.log(np.array([ea["Y_LT30"], ea["Y30-49"], ea["Y_GE50"]]) / ea["Y30-49"])
    # exact quadratic through the three age groups: lv = s x + c x^2 (level normalised at the middle group)
    m = np.array([[x[0], x[0] ** 2], [x[2], x[2] ** 2]])
    s, c = np.linalg.solve(m, np.array([lv[0], lv[2]]))
    src = f"Eurostat earn_ses22_14, mean hourly gross earnings by age group (<30, 30-49, 50+), {ey}"
    rows.append(_row("wage_age_slope", s, "derived", "log points per decade", src, ey, "Linear term of the cross-sectional age-earnings profile through three age groups (cross-section, not a lifecycle path)"))
    rows.append(_row("wage_age_curvature", c, "derived", "log points per decade^2", src, ey, "Quadratic term of the same profile"))
    ed = b.es_any("earn_ses_pub2i", geo="BE", time=str(ey), unit="EUR").set_index("isced11").value
    rows.append(_row("education_wage_ratio_medium", ed["ED3_4"] / ed["ED0-2"], "observed", "ratio", f"Eurostat earn_ses_pub2i, median hourly earnings, {ey}", ey, "Median hourly earnings of upper-secondary relative to lower-secondary or less"))
    rows.append(_row("education_wage_ratio_high", ed["ED5-8"] / ed["ED0-2"], "observed", "ratio", f"Eurostat earn_ses_pub2i, median hourly earnings, {ey}", ey, "Median hourly earnings of tertiary relative to lower-secondary or less"))
    lw = float(b.es_any("earn_ses_pub1s", geo="BE", time=str(ey), sex="T").value.iloc[0]) / 100
    rows.append(_row("low_wage_share_target", lw, "observed", "share", f"Eurostat earn_ses_pub1s, employees below 2/3 of median hourly earnings, {ey}", ey,
                     "Target for the dispersion of earnings: skill_sigma is solved so the SD of log earnings equals the lognormal-equivalent implied by this share; the simulated share is then a validation"))

    # --- housing and saving ---------------------------------------------------------------------
    h = b.es_any("tipsho10", geo="BE", unit="I15_A_AVG").set_index("time").value
    h.index = h.index.astype(int)
    hh = h[(h.index >= y - 20) & (h.index <= y)]
    g = float(np.log(hh.iloc[-1] / hh.iloc[0]) / (len(hh) - 1))
    rows.append(_row("house_price_growth", g, "derived", "per year", f"Eurostat tipsho10 (deflated house price index), log growth {hh.index[0]}-{hh.index[-1]}", y, "Average real annual growth of house prices over the last 20 years"))
    sr = b.es_any("nasa_10_ki", geo="BE", time=str(y - 1)).set_index("na_item").value
    rows.append(_row("observed_household_saving_rate", sr["SRG_S14_S15"] / 100, "observed", "share of disposable income", f"Eurostat nasa_10_ki, gross household saving rate, {y - 1}", y - 1, "VALIDATION ONLY: national-accounts gross saving rate of households and NPISH"))
    b.diagnostics["behaviour_education_ratio"] = {k: float(v) for k, v in {"medium": ed["ED3_4"] / ed["ED0-2"], "high": ed["ED5-8"] / ed["ED0-2"]}.items()}
    return pd.DataFrame(rows)


def lognormal_sd_from_low_wage_share(share: float, threshold_ratio: float = 2 / 3) -> float:
    """Closed form for a lognormal earnings distribution: share = Phi(ln(r)/sigma)."""
    return float(abs(np.log(threshold_ratio)) / abs(ndtri(share)))


__all__ = ["EDU", "build_behaviour", "labour_transitions", "lognormal_sd_from_low_wage_share"]
