"""Compare simulated distribution indicators with published figures (validation, NOT fitting)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from belgium_abm.analysis.inequality import weighted_gini, weighted_quantile


def quintile_shares(values: np.ndarray) -> np.ndarray:
    v = np.sort(values)
    c = np.cumsum(v)
    n = len(v)
    cuts = [int(round(n * q)) for q in (0.2, 0.4, 0.6, 0.8, 1.0)]
    shares, prev = [], 0.0
    for k in cuts:
        shares.append((c[k - 1] - prev) / c[-1])
        prev = c[k - 1]
    return np.array(shares)


def validation_table(metrics_row: dict, eq_disp: np.ndarray, targets: pd.DataFrame, calibrated: set[str] | None = None) -> pd.DataFrame:
    """One row per published indicator: simulated vs published, relative gap, status."""
    calibrated = calibrated or {"median_equivalised_income"}
    t = targets.copy()
    t["value"] = t["value"].astype(float)
    pick = lambda ind, grp="TOTAL": float(t[(t.indicator == ind) & (t.group == grp)].value.iloc[0]) if ((t.indicator == ind) & (t.group == grp)).any() else np.nan  # noqa: E731
    qs = quintile_shares(eq_disp)
    rows = [
        ("median_equivalised_income", "TOTAL", float(weighted_quantile(eq_disp, 0.5)), pick("median_equivalised_income")),
        ("mean_equivalised_income", "TOTAL", float(np.mean(eq_disp)), pick("mean_equivalised_income")),
        ("gini_disposable_income", "TOTAL", weighted_gini(eq_disp), pick("gini_disposable_income")),
        ("at_risk_of_poverty_rate_60pct_median", "TOTAL", float(metrics_row["poverty_rate_60pct_median"]), pick("at_risk_of_poverty_rate_60pct_median")),
        ("s80_s20", "TOTAL", float(metrics_row["s80_s20_disposable"]), pick("s80_s20")),
    ]
    for i, q in enumerate(("QU1", "QU2", "QU3", "QU4", "QU5")):
        rows.append(("income_share_quintile", q, float(qs[i]), pick("income_share_quintile", q)))
    df = pd.DataFrame(rows, columns=["indicator", "group", "simulated", "published"])
    df["abs_gap"] = df["simulated"] - df["published"]
    df["rel_gap"] = df["abs_gap"] / df["published"]
    df["role"] = np.where(df["indicator"].isin(calibrated), "calibration target (one-point level)", "validation (not fitted)")
    df["status"] = np.where(df["role"].str.startswith("calibration"), "calibrated", np.where(df["rel_gap"].abs() <= 0.15, "close (<=15%)", np.where(df["rel_gap"].abs() <= 0.35, "gap (15-35%)", "LARGE GAP (>35%)")))
    return df


def wealth_validation_table(st, targets: pd.DataFrame) -> pd.DataFrame:
    """Simulated household net worth (baseline) vs HFCS wave-2023 aggregates. NOT like-for-like:
    HFCS net wealth also includes vehicles, valuables, voluntary pension plans and private business."""
    from belgium_abm.analysis.inequality import top_share
    from belgium_abm.engines.state import net_worth

    hh = st.households
    nw = net_worth(st)
    w = np.ones(len(nw))
    t = targets.copy()
    t["value"] = t["value"].astype(float)

    def pub(ind: str, grp: str = "TOTAL") -> float:
        m = (t.indicator == ind) & (t.group == grp)
        return float(t[m].value.iloc[0]) if m.any() else np.nan

    head_age = st.persons[st.persons["head_flag"]].set_index("household_id")["age"].reindex(hh.index)
    owner = hh["housing_tenure"].isin(["MORTGAGED_OWNER", "OUTRIGHT_OWNER"]).to_numpy()
    rows = [
        ("net_wealth_median", "TOTAL", float(weighted_quantile(nw.to_numpy(), 0.5, w)), pub("net_wealth_median")),
        ("net_wealth_mean", "TOTAL", float(nw.mean()), pub("net_wealth_mean")),
        ("net_wealth_gini", "TOTAL", weighted_gini(nw.to_numpy(), w), pub("net_wealth_gini")),
        ("net_wealth_share_top10", "TOTAL", top_share(np.clip(nw.to_numpy(), 0, None), 0.10, w), pub("net_wealth_share_top10")),
        ("net_wealth_share_top5", "TOTAL", top_share(np.clip(nw.to_numpy(), 0, None), 0.05, w), pub("net_wealth_share_top5")),
        ("homeowner_share", "TOTAL", float(owner.mean()), pub("homeowner_share")),
        ("share_with_debt", "TOTAL", float(((hh["mortgage_balance"] + hh["other_debt"]) > 0).mean()), pub("share_with_debt")),
        ("net_wealth_median", "HOMEOWNER", float(nw[owner].median()), pub("net_wealth_median", "HOMEOWNER")),
        ("net_wealth_median", "NON_OWNER", float(nw[~owner].median()), pub("net_wealth_median", "NON_OWNER")),
    ]
    for q in range(10, 100, 10):
        g = f"p{q}"
        if ((t.indicator == "net_wealth_percentile") & (t.group == g)).any():
            rows.append(("net_wealth_percentile", g, float(weighted_quantile(nw.to_numpy(), q / 100, w)), pub("net_wealth_percentile", g)))
    for lo, hi, lab in ((16, 34, "16-34"), (35, 44, "35-44"), (45, 54, "45-54"), (55, 64, "55-64"), (65, 74, "65-74"), (75, 200, "75+")):
        m = head_age.between(lo, hi).to_numpy()
        if m.any():
            rows.append(("net_wealth_median_by_age_of_reference_person", lab, float(nw[m].median()), pub("net_wealth_median_by_age_of_reference_person", lab)))
    df = pd.DataFrame(rows, columns=["indicator", "group", "simulated", "published"])
    df["rel_gap"] = df["simulated"] / df["published"] - 1
    df["status"] = np.where(df["rel_gap"].abs() <= 0.15, "close (<=15%)", np.where(df["rel_gap"].abs() <= 0.35, "gap (15-35%)", "LARGE GAP (>35%)"))
    df["note"] = "HFCS 2023 vs simulated 2025 baseline; definitions differ (see docs); not calibrated"
    return df
