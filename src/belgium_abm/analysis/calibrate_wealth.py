"""Calibrate the stylised wealth-seeding parameters to published HFCS medians (age groups, overall and non-owner medians).

Fitted: overall median, non-owner median, medians by age of the reference person. Held out for VALIDATION: wealth Gini, top shares, percentiles, homeowner median, share with debt.
Fitting is by simulated method of moments on the initial state only (cheap: no dynamics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from belgium_abm.analysis.benchmarks import wealth_validation_table
from belgium_abm.config import ModelConfig
from belgium_abm.engines.population import generate_population
from belgium_abm.engines.simulation import initial_state
from belgium_abm.io.loaders import load_targets
from belgium_abm.run_sim import load_inputs

PARAMS = ("fin_assets_multiple_max", "fin_assets_age_tau", "house_to_income", "renter_financial_asset_factor")


def calibrate_wealth(cfg: ModelConfig, seed: int = 42, n_persons: int = 12000, maxiter: int = 70) -> tuple[pd.DataFrame, dict]:
    inp = load_inputs(cfg)
    if inp.wealth is None:
        raise ValueError("no HFCS wealth targets configured (simulation.wealth_targets)")
    small = cfg.with_overrides(generator={"n_persons": n_persons})
    targets = load_targets(small)
    pop = generate_population(small, targets, seed)
    x0 = np.log([float(inp.econ.v(p)) for p in PARAMS])
    cache: dict[tuple, float] = {}

    def table(x: np.ndarray) -> pd.DataFrame:
        econ = inp.econ.with_values(**{p: float(np.exp(v)) for p, v in zip(PARAMS, x, strict=True)})
        st, _ = initial_state(pop.persons, pop.households, targets, inp.demography, econ, inp.policy, cfg.adult_age, cfg.equivalence, seed, inp.target_median)
        return wealth_validation_table(st, inp.wealth)

    def loss(x: np.ndarray) -> float:
        key = tuple(np.round(x, 5))
        if key not in cache:
            t = table(x)
            t = t[(t.indicator == "net_wealth_median_by_age_of_reference_person") | ((t.indicator == "net_wealth_median") & t.group.isin(["TOTAL", "NON_OWNER"]))]
            gap = np.log(np.clip(t.simulated.to_numpy(), 1.0, None) / t.published.to_numpy())
            cache[key] = float(np.mean(gap**2))
        return cache[key]

    res = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 0.01, "fatol": 1e-4, "initial_simplex": x0 + np.vstack([np.zeros(len(x0)), 0.35 * np.eye(len(x0))])})
    best = table(res.x)
    base = table(x0)
    rows = [{"name": p, "value": float(np.exp(v)), "status": "derived", "unit": inp.econ.parameters[p].unit,
             "source": "calibrated to ECB HFCS 2023 median net wealth by age of reference person, overall and non-owner (simulated method of moments on the initial state)",
             "year": 2023, "definition": inp.econ.parameters[p].definition} for p, v in zip(PARAMS, res.x, strict=True)]
    diag = {"loss_before": loss(x0), "loss_after": float(res.fun), "evaluations": int(res.nfev), "converged": bool(res.success)}
    best["before_calibration_simulated"] = base["simulated"].to_numpy()
    return pd.DataFrame(rows), {"diagnostics": diag, "table": best}
