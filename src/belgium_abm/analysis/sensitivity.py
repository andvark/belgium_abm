"""One-at-a-time sensitivity of baseline outcomes and of a policy scenario's effect to the unestimated parameters.

Each parameter is moved to a low and a high value (others at baseline); every setting reuses the same synthetic populations and the
same common random numbers. Ranges are JUDGMENTS (documented below): +-30% around public proxies, wide plausible intervals for pure
assumptions. One-at-a-time analysis shows first-order swings only, not interactions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from belgium_abm.config import ModelConfig
from belgium_abm.engines.population import generate_population
from belgium_abm.engines.simulation import initial_state, run_simulation
from belgium_abm.io.loaders import load_targets
from belgium_abm.run_sim import load_inputs
from belgium_abm.scenarios import Scenario

# name -> (low, high, why this range). 'public' = observed/derived value with +-30% uncertainty, 'assumed' = plausible interval.
RANGES: dict[str, tuple[float | None, float | None, str]] = {
    "job_separation_rate": (None, None, "public proxy +-30%"),
    "sole_prop_exit_rate": (None, None, "public proxy +-30%"),
    "bv_exit_rate": (None, None, "public proxy +-30%"),
    "parttime_share_f": (None, None, "public proxy +-30%"),
    "bv_director_pay_share": (0.4, 0.8, "assumed: owner-manager pay policy"),
    "bv_payout_ratio": (0.1, 0.6, "assumed: dividend policy"),
    "bv_productivity_multiplier": (0.8, 1.2, "assumed: comparable productivity +-20%"),
    "business_shock_sigma": (0.15, 0.35, "assumed: business income risk"),
    "wage_shock_rho": (0.80, 0.95, "assumed: earnings persistence"),
    "wage_shock_sigma": (0.05, 0.12, "assumed: earnings innovation"),
    "real_wage_growth": (0.0, 0.02, "assumed: productivity growth"),
    "marginal_propensity_to_consume": (0.70, 0.95, "assumed: consumption behaviour"),
    "fin_return": (0.0, 0.03, "assumed: real return on savings"),
    "entry_wealth_sensitivity": (0.0, 2.0, "assumed: credit constraint on entrepreneurship"),
    "education_wealth_sensitivity": (0.0, 3.0, "assumed: family wealth and education"),
    "separation_rate": (0.006, 0.024, "assumed: couple separation"),
}
OUTCOMES = ["gini_disposable", "poverty_rate_60pct_median", "wealth_gini", "wealth_top10_share", "median_eq_disposable_income", "gov_net_fiscal_balance"]


def _setting(inp, name: str, which: str, base_val: float):
    lo, hi, _ = RANGES[name]
    if which == "base":
        return inp.econ
    v = (lo if which == "low" else hi) if lo is not None else base_val * (0.7 if which == "low" else 1.3)
    return inp.econ.with_values(**{name: v}), v


def run_sensitivity(cfg: ModelConfig, scenario_path: str, out: Path, *, seed: int = 42, n_persons: int = 8000, years: int = 15, replicates: int = 2,
                    params: list[str] | None = None) -> pd.DataFrame:
    inp = load_inputs(cfg)
    cfg = cfg.with_overrides(generator={"n_persons": n_persons})
    targets = load_targets(cfg)
    scen = Scenario.load(scenario_path)
    scen_pol = scen.apply_policy(inp.policy)
    pops = [generate_population(cfg, targets, seed + r) for r in range(replicates)]
    names = params or list(RANGES)
    settings: list[tuple[str, str, float, object]] = [("(baseline)", "base", float("nan"), inp.econ)]
    for n in names:
        base_val = float(inp.econ.v(n))
        for which in ("low", "high"):
            econ, v = _setting(inp, n, which, base_val)
            settings.append((n, which, float(v), econ))
    rows = []
    for name, which, value, econ in settings:
        acc: list[dict] = []
        for rep, pop in enumerate(pops):
            st0, _ = initial_state(pop.persons, pop.households, targets, inp.demography, econ, inp.policy, cfg.adult_age, cfg.equivalence, seed + rep, inp.target_median)
            b = run_simulation(st0, inp.policy, years, shadow={scen.name: scen_pol}, label="baseline", group_years=(years,), min_group_size=cfg.simulation.min_group_size)
            st_s = st0.copy()
            s = run_simulation(st_s, scen_pol, years, label="scenario", group_years=(years,), min_group_size=cfg.simulation.min_group_size)
            mb = b.metrics[(b.metrics.scenario == "baseline") & (b.metrics.year == years)].iloc[0]
            ms = s.metrics[(s.metrics.scenario == "scenario") & (s.metrics.year == years)].iloc[0]
            mst = b.metrics[(b.metrics.scenario == f"static:{scen.name}") & (b.metrics.year == years)].iloc[0]
            gb = b.groups[(b.groups.scenario == "baseline") & (b.groups.group_var == "work_arrangement") & (b.groups.group == "BV_OWNER_MANAGER")]
            gs = s.groups[(s.groups.scenario == "scenario") & (s.groups.group_var == "work_arrangement") & (s.groups.group == "BV_OWNER_MANAGER")]
            row = {o: float(mb[o]) for o in OUTCOMES}
            row["effect_net_fiscal_balance_dynamic"] = float(ms["gov_net_fiscal_balance"] - mb["gov_net_fiscal_balance"])
            row["effect_net_fiscal_balance_static"] = float(mst["gov_net_fiscal_balance"] - mb["gov_net_fiscal_balance"])
            row["effect_wealth_top10_share"] = float(ms["wealth_top10_share"] - mb["wealth_top10_share"])
            row["effect_bv_owner_eq_income"] = float(gs["mean_eq_disposable_income"].iloc[0] - gb["mean_eq_disposable_income"].iloc[0]) if len(gb) and len(gs) else float("nan")
            acc.append(row)
        agg = pd.DataFrame(acc).mean()
        rows.append({"parameter": name, "setting": which, "value": value, **agg.to_dict(), "n_replicates": replicates})
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sensitivity_by_setting.csv", index=False, lineterminator="\n")
    base = df[df.parameter == "(baseline)"].iloc[0]
    metrics = [c for c in df.columns if c in OUTCOMES or c.startswith("effect_")]
    sw = []
    for n in names:
        lo = df[(df.parameter == n) & (df.setting == "low")].iloc[0]
        hi = df[(df.parameter == n) & (df.setting == "high")].iloc[0]
        for m in metrics:
            sw.append({"parameter": n, "metric": m, "low_value": lo["value"], "high_value": hi["value"], "at_low": lo[m], "at_baseline": base[m], "at_high": hi[m],
                       "swing": hi[m] - lo[m], "range_basis": RANGES[n][2]})
    swing = pd.DataFrame(sw)
    swing["abs_swing"] = swing["swing"].abs()
    swing["rank_within_metric"] = swing.groupby("metric")["abs_swing"].rank(ascending=False, method="first").astype(int)
    swing.sort_values(["metric", "rank_within_metric"]).to_csv(out / "sensitivity_ranking.csv", index=False, lineterminator="\n")
    _plot(swing, out / "sensitivity_tornado.png", years, replicates, inp.policy.banner)
    _report(swing, base, out / "sensitivity_report.md", years, replicates, n_persons, inp.policy.banner, scen.name)
    return swing


def _plot(swing: pd.DataFrame, path: Path, years: int, reps: int, banner: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mets = [("wealth_top10_share", "Top-10% wealth share"), ("gini_disposable", "Gini, disposable income"), ("effect_net_fiscal_balance_dynamic", "Corp-tax effect on net fiscal balance (EUR)"),
            ("effect_bv_owner_eq_income", "Corp-tax effect on BV-owner income (EUR)")]
    fig, axes = plt.subplots(1, len(mets), figsize=(5.2 * len(mets), 5.0))
    for ax, (m, lab) in zip(axes, mets, strict=True):
        g = swing[swing.metric == m].sort_values("abs_swing").tail(10)
        base = g["at_baseline"].iloc[0]
        y = np.arange(len(g))
        ax.barh(y, g["at_high"] - base, left=base, alpha=0.7, label="high")
        ax.barh(y, g["at_low"] - base, left=base, alpha=0.7, label="low")
        ax.axvline(base, color="k", lw=0.8)
        ax.set_yticks(y, g["parameter"], fontsize=8)
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"One-at-a-time sensitivity at year {years} ({reps} replicate(s)); ranges are judgments", fontsize=10)
    fig.text(0.5, 0.005, banner[:230], ha="center", fontsize=6, color="firebrick")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _report(swing: pd.DataFrame, base: pd.Series, path: Path, years: int, reps: int, n: int, banner: str, scen: str) -> None:
    L = ["# Sensitivity analysis (one-at-a-time)", "", f"> **{banner}**", "",
         f"Year {years}, {n:,} simulated persons, {reps} paired replicate(s), scenario `{scen}`. Each parameter is moved to a low and a high value with all others at baseline. "
         "Ranges are judgments (public proxies +-30%; plausible intervals for pure assumptions). First-order swings only; no interactions. Few replicates: treat small swings as noise.", ""]
    for m in ["wealth_top10_share", "gini_disposable", "poverty_rate_60pct_median", "median_eq_disposable_income", "effect_net_fiscal_balance_dynamic", "effect_bv_owner_eq_income", "effect_wealth_top10_share"]:
        g = swing[swing.metric == m].sort_values("rank_within_metric").head(6)
        L += [f"## {m} (baseline {base[m]:.4g})", "", "| rank | parameter | low | high | value at low | value at high | swing |", "|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| {r['rank_within_metric']} | {r['parameter']} | {r['low_value']:.4g} | {r['high_value']:.4g} | {r['at_low']:.4g} | {r['at_high']:.4g} | {r['swing']:+.4g} |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8", newline="\n")
