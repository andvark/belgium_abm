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
