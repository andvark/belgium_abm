"""Compare the model's simulated flows and structure with the public aggregates that informed (or did not inform) it."""

from __future__ import annotations

import numpy as np
import pandas as pd


def behaviour_validation(metrics: pd.DataFrame, events: pd.DataFrame, behaviour: pd.DataFrame, info: dict, label: str = "baseline") -> pd.DataFrame:
    """Annual simulated transition / business flow rates (mean over years 1..T) against observed public rates."""
    obs = behaviour.set_index("name")["value"].astype(float)
    m = metrics[metrics.scenario == label].set_index("year").sort_index()
    ev = events[events.scenario == label].pivot_table(index="year", columns="event", values="value", aggfunc="sum").fillna(0.0)
    years = [y for y in ev.index if y >= 1 and (y - 1) in m.index]
    if not years:
        return pd.DataFrame()

    def rate(num: pd.Series, stock_col: str) -> float:
        return float(np.mean([num.get(y, 0.0) / max(m.loc[y - 1, stock_col], 1) for y in years]))

    g = lambda k: ev[k] if k in ev else pd.Series(0.0, index=ev.index)  # noqa: E731
    rows = [
        ("employment_to_unemployment", rate(g("job_losses"), "n_employed_le74"), obs.get("job_separation_rate"), "input (observed); model must reproduce it"),
        ("employment_to_inactivity (incl. retirement)", rate(g("labour_force_exits") + g("retirements_employed"), "n_employed_le74"), obs.get("labour_force_exit_total_observed"), "input (derived); retirement part is modelled"),
        ("unemployment_to_employment", rate(g("job_findings"), "n_unemployed"), obs.get("observed_flow_unemployed_to_employed"), "validation (derived to keep cross-section stationary)"),
        ("inactivity_to_employment (inactive incl. students, retirees <=74)", rate(g("lf_entries_employed") + g("school_leavers_employed"), "n_inactive_like_le74"), obs.get("observed_flow_inactive_to_employed"), "validation"),
        ("sole_prop_birth_rate", rate(g("entries_SOLE_PROPRIETOR"), "n_sole_prop"), obs.get("observed_birth_rate_sole_prop"), "validation (entry flows are derived to keep the age profile)"),
        ("sole_prop_exit_rate", rate(g("exits_SOLE_PROPRIETOR"), "n_sole_prop"), obs.get("sole_prop_exit_rate"), "input (observed)"),
        ("bv_birth_rate", rate(g("entries_BV_OWNER_MANAGER"), "n_bv_owner_managers"), obs.get("observed_birth_rate_bv"), "validation"),
        ("bv_closure_rate (voluntary + insolvency + retirement liquidation)", rate(g("exits_BV_OWNER_MANAGER") + g("bv_insolvencies") + g("bv_liquidations"), "n_bv_owner_managers"), obs.get("bv_exit_rate"), "validation of the combined channels"),
        ("household_saving_rate", float(m.loc[m.index >= 1, "household_saving_rate"].mean()), obs.get("observed_household_saving_rate"), "validation (national accounts, households incl. NPISH)"),
    ]
    df = pd.DataFrame(rows, columns=["indicator", "simulated", "observed", "role"])
    df["rel_gap"] = df["simulated"] / df["observed"] - 1
    df["status"] = np.where(df["rel_gap"].abs() <= 0.15, "close (<=15%)", np.where(df["rel_gap"].abs() <= 0.35, "gap (15-35%)", "LARGE GAP (>35%)"))
    return df
