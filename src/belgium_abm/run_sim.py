"""Orchestration of ``abm simulate`` and ``abm compare`` (Phase 4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from belgium_abm import __version__
from belgium_abm.analysis.benchmarks import validation_table, wealth_validation_table
from belgium_abm.analysis.behaviour_validation import behaviour_validation
from belgium_abm.analysis.plots import plot_timeseries
from belgium_abm.config import ModelConfig
from belgium_abm.engines.economy import EconomyConfig
from belgium_abm.engines.population import generate_population
from belgium_abm.engines.simulation import SimResult, initial_state, run_simulation
from belgium_abm.io.loaders import load_targets, parse_annotated_csv
from belgium_abm.io.outputs import write_json
from belgium_abm.policies.rules import PolicyRules
from belgium_abm.scenarios import Scenario

KEY_METRICS = {
    "gini_market_pensions_as_transfers": "Gini, market income (pensions as transfers)",
    "gini_market_pensions_as_income": "Gini, market income (pensions as deferred income)",
    "gini_disposable": "Gini, disposable income",
    "gini_post_consumption_tax": "Gini, after VAT",
    "poverty_rate_60pct_median": "At-risk-of-poverty rate",
    "s80_s20_disposable": "S80/S20",
    "p90_p10_disposable": "P90/P10",
    "wealth_gini": "Gini, net worth",
    "wealth_top10_share": "Top-10% wealth share",
    "wealth_top1_share": "Top-1% wealth share",
    "median_eq_disposable_income": "Median equivalised disposable income (EUR)",
    "unemployment_rate": "Unemployment rate",
    "gov_net_fiscal_balance": "Net fiscal balance (EUR)",
}


def _read_numeric(path: Path) -> pd.DataFrame:
    meta, df = parse_annotated_csv(path.read_text(encoding="utf-8"), path.name)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except ValueError:
            pass
    df.attrs["meta"] = meta
    return df


@dataclass
class Inputs:
    cfg: ModelConfig
    econ: EconomyConfig
    policy: PolicyRules
    demography: pd.DataFrame
    validation: pd.DataFrame
    target_median: float
    wealth: pd.DataFrame | None = None
    behaviour: pd.DataFrame | None = None


def load_inputs(cfg: ModelConfig, policy_file: str | None = None) -> Inputs:
    if cfg.simulation is None:
        raise ValueError(f"configuration '{cfg.name}' has no `simulation:` section (policy, economy, demography, validation targets)")
    sim = cfg.simulation
    dem = _read_numeric(cfg.resolve(sim.demography_rates))
    val = _read_numeric(cfg.resolve(sim.validation_targets))
    tm = float(val[(val.indicator == "median_equivalised_income") & (val.group == "TOTAL")].value.iloc[0])
    wealth = _read_numeric(cfg.resolve(sim.wealth_targets)) if sim.wealth_targets and cfg.resolve(sim.wealth_targets).is_file() else None
    econ = EconomyConfig.load(cfg.resolve(sim.economy_file))
    beh = None
    if sim.behaviour_parameters and cfg.resolve(sim.behaviour_parameters).is_file():
        beh = _read_numeric(cfg.resolve(sim.behaviour_parameters))
        econ = econ.apply_observed(beh)
    if sim.calibrated_parameters and cfg.resolve(sim.calibrated_parameters).is_file():
        econ = econ.apply_observed(_read_numeric(cfg.resolve(sim.calibrated_parameters)))
    return Inputs(cfg, econ, PolicyRules.load(cfg.resolve(policy_file or sim.policy_file)), dem, val, tm, wealth, beh)


def _combine(results: list[tuple[int, SimResult]], attr: str) -> pd.DataFrame:
    frames = []
    for rep, r in results:
        df = getattr(r, attr)
        if len(df):
            frames.append(df.assign(replicate=rep))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_run(out: Path, label: str, results: list[tuple[int, SimResult]], seeds: list[int], inp: Inputs, years: int, info: list[dict], policy: PolicyRules,
               extra_validation: list[pd.DataFrame], static_name: str | None, wealth_validation: list[pd.DataFrame] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    metrics = _combine(results, "metrics")
    groups = _combine(results, "groups")
    fiscal = _combine(results, "fiscal")
    events = _combine(results, "events")
    for name, df in (("metrics_by_year", metrics), ("group_metrics", groups), ("fiscal_by_year", fiscal), ("events", events)):
        df.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    mob = pd.concat([r.mobility.assign(replicate=rep).reset_index() for rep, r in results], ignore_index=True)
    mob.to_csv(out / "income_mobility.csv", index=False, lineterminator="\n")
    dyn = metrics[metrics["scenario"] == label]
    cols = [c for c in KEY_METRICS if c in dyn]
    summ = []
    for y in sorted({0, years // 3, 2 * years // 3, years}):
        g = dyn[dyn["year"] == y]
        for c in cols:
            v = g[c].dropna()
            summ.append({"year": y, "metric": c, "n_replicates": len(v), "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                         "p05": v.quantile(0.05) if len(v) > 1 else np.nan, "p95": v.quantile(0.95) if len(v) > 1 else np.nan})
    pd.DataFrame(summ).to_csv(out / "uncertainty_summary.csv", index=False, lineterminator="\n")
    if extra_validation:
        vt = pd.concat(extra_validation, ignore_index=True)
        agg = vt.groupby(["indicator", "group", "role"], as_index=False).agg(simulated=("simulated", "mean"), published=("published", "first"))
        agg["rel_gap"] = agg["simulated"] / agg["published"] - 1
        agg["status"] = np.where(agg["role"].str.startswith("calibration"), "calibrated", np.where(agg["rel_gap"].abs() <= 0.15, "close (<=15%)", np.where(agg["rel_gap"].abs() <= 0.35, "gap (15-35%)", "LARGE GAP (>35%)")))
        agg.to_csv(out / "validation_vs_published.csv", index=False, lineterminator="\n")
    if wealth_validation:
        wv = pd.concat(wealth_validation, ignore_index=True)
        wa = wv.groupby(["indicator", "group"], as_index=False, sort=False).agg(simulated=("simulated", "mean"), published=("published", "first"))
        wa["rel_gap"] = wa["simulated"] / wa["published"] - 1
        wa["status"] = np.where(wa["rel_gap"].abs() <= 0.15, "close (<=15%)", np.where(wa["rel_gap"].abs() <= 0.35, "gap (15-35%)", "LARGE GAP (>35%)"))
        wa.to_csv(out / "validation_wealth_vs_hfcs.csv", index=False, lineterminator="\n")
    if inp.behaviour is not None:
        bvs = [behaviour_validation(r.metrics, r.events, inp.behaviour, i, label).assign(replicate=rep) for (rep, r), i in zip(results, info, strict=False)]
        bvs = [x for x in bvs if len(x)]
        if bvs:
            bv = pd.concat(bvs, ignore_index=True).groupby(["indicator", "role"], as_index=False, sort=False).agg(simulated=("simulated", "mean"), observed=("observed", "first"))
            bv["rel_gap"] = bv["simulated"] / bv["observed"] - 1
            bv["status"] = np.where(bv["rel_gap"].abs() <= 0.15, "close (<=15%)", np.where(bv["rel_gap"].abs() <= 0.35, "gap (15-35%)", "LARGE GAP (>35%)"))
            bv.to_csv(out / "validation_behaviour_vs_public.csv", index=False, lineterminator="\n")
    ledger_rows = [{"replicate": rep, "year": e.period, "category": e.category, "payer": e.payer_id, "recipient": e.recipient_id, "amount_cents": e.amount_cents}
                   for rep, r in results for e in r.ledger.entries]
    pd.DataFrame(ledger_rows).to_csv(out / "ledger_summary.csv", index=False, lineterminator="\n")
    plot_timeseries(metrics, {"gini_disposable": "Gini, disposable income", "poverty_rate_60pct_median": "At-risk-of-poverty rate",
                              "wealth_top10_share": "Top-10% wealth share", "gov_net_fiscal_balance": "Net fiscal balance (EUR)"},
                    out / "inequality_over_time.png", f"{label}: {years}-year projection from baseline {inp.cfg.baseline_year} ({len(seeds)} replicate(s))", policy.banner)
    manifest = {
        "label": label, "package_version": __version__, "config": inp.cfg.name, "config_hash_sha256": inp.cfg.config_hash(), "baseline_year": inp.cfg.baseline_year,
        "horizon_years": years, "seeds": seeds, "policy": {"name": policy.name, "fingerprint": policy.fingerprint(), "verified": policy.verified, "banner": policy.banner},
        "economy": {"name": inp.econ.name, "banner": inp.econ.banner, "all_parameters_status": sorted({p.status.value for p in inp.econ.parameters.values()})},
        "weighting_unit": "uniform person weights; households scaled by member share; incomes equivalised (modified OECD)", "static_shadow_scenario": static_name,
        "level_calibration": info, "max_cash_identity_residual_eur": max(r.max_cash_identity_residual for _, r in results),
        "invariant_failures": [m for _, r in results for m in r.invariant_failures][:20],
        "intergenerational": {str(rep): r.extras for rep, r in results},
    }
    write_json(manifest, out / "run_manifest.json")
    (out / "report.md").write_text(render_report(label, manifest, summ, out), encoding="utf-8", newline="\n")
    return manifest


def render_report(label: str, m: dict, summary: list[dict], out: Path) -> str:
    L = [f"# Simulation report: `{label}`", "", f"> **{m['policy']['banner']}**", "",
         f"> **{m['economy']['banner']}**. Population dynamics use *observed* Belgian life tables and fertility; labour, education and household flows are derived from 2025 cross-sections;",
         "> wealth, behaviour, business and all tax/benefit values are assumptions. These results are **mechanical demonstrations, not forecasts and not estimates of Belgian policy effects**.", "",
         f"- Baseline year: **{m['baseline_year']}**, horizon: **{m['horizon_years']} years** (constant prices, no migration)",
         f"- Replicates / seeds: {len(m['seeds'])} / {m['seeds']}",
         f"- Weighting and unit: {m['weighting_unit']}",
         f"- Policy file `{m['policy']['name']}` fingerprint `{m['policy']['fingerprint']}`, verified against law: **{m['policy']['verified']}**",
         f"- Max household cash-identity residual: {m['max_cash_identity_residual_eur']:.2e} EUR; invariant failures: {len(m['invariant_failures'])}", "",
         "## Key indicators (mean across replicates; 5-95% band when more than one)", "", "| year | indicator | mean | p05 | p95 |", "|---|---|---|---|---|"]
    for r in summary:
        L.append(f"| {r['year']} | {KEY_METRICS.get(r['metric'], r['metric'])} | {r['mean']:.4g} | {'' if np.isnan(r['p05']) else format(r['p05'], '.4g')} | {'' if np.isnan(r['p95']) else format(r['p95'], '.4g')} |")
    vv = out / "validation_vs_published.csv"
    if vv.is_file():
        v = pd.read_csv(vv)
        L += ["", "## Baseline validation against published EU-SILC indicators", "",
              "Only the median equivalised income is a (one-point) calibration target; everything else is a genuine out-of-sample check.", "",
              "| indicator | group | simulated | published | rel. gap | status |", "|---|---|---|---|---|---|"]
        for _, r in v.iterrows():
            L.append(f"| {r['indicator']} | {r['group']} | {r['simulated']:.4g} | {r['published']:.4g} | {r['rel_gap']:+.1%} | {r['status']} |")
    bvf = out / "validation_behaviour_vs_public.csv"
    if bvf.is_file():
        v = pd.read_csv(bvf)
        L += ["", "## Simulated flows vs public aggregates (Eurostat LFS flows, business demography, national accounts)", "",
              "| indicator | simulated | observed | rel. gap | status | role |", "|---|---|---|---|---|---|"]
        for _, r in v.iterrows():
            L.append(f"| {r['indicator']} | {r['simulated']:.4g} | {r['observed']:.4g} | {r['rel_gap']:+.1%} | {r['status']} | {r['role']} |")
    wv = out / "validation_wealth_vs_hfcs.csv"
    if wv.is_file():
        v = pd.read_csv(wv)
        L += ["", "## Baseline wealth vs ECB HFCS 2023 (validation only, NOT calibrated)", "",
              "Definitions differ: HFCS net wealth also includes vehicles, valuables, voluntary pension plans and private business, so simulated wealth is expected to be lower.", "",
              "| indicator | group | simulated | published | rel. gap | status |", "|---|---|---|---|---|---|"]
        for _, r in v.iterrows():
            L.append(f"| {r['indicator']} | {r['group']} | {r['simulated']:.4g} | {r['published']:.4g} | {r['rel_gap']:+.1%} | {r['status']} |")
    L += ["", "Files: `metrics_by_year.csv`, `group_metrics.csv`, `fiscal_by_year.csv`, `events.csv`, `income_mobility.csv`, `uncertainty_summary.csv`, `ledger_summary.csv`, `inequality_over_time.png`, `run_manifest.json`.", ""]
    return "\n".join(L)


def simulate(cfg: ModelConfig, *, seed: int, years: int | None, replicates: int, n_persons: int | None, scenario_path: str | None, out_dir: Path,
             policy_file: str | None = None, check_invariants: bool = False) -> dict[str, Path]:
    inp = load_inputs(cfg, policy_file)
    if n_persons:
        cfg = cfg.with_overrides(generator={"n_persons": n_persons})
        inp.cfg = cfg
    years = years or cfg.simulation.years
    targets = load_targets(cfg)
    scen = Scenario.load(scenario_path) if scenario_path else None
    base_pol = inp.policy
    scen_pol = scen.apply_policy(base_pol) if scen else None
    scen_econ = scen.apply_economy(inp.econ) if scen else None
    seeds = [seed + r for r in range(replicates)]
    base_res: list[tuple[int, SimResult]] = []
    scen_res: list[tuple[int, SimResult]] = []
    infos: list[dict] = []
    vals: list[pd.DataFrame] = []
    wvals: list[pd.DataFrame] = []
    wl_rows: list[pd.DataFrame] = []
    for rep, sd in enumerate(seeds):
        pop = generate_population(cfg, targets, sd)
        st0, info = initial_state(pop.persons, pop.households, targets, inp.demography, inp.econ, base_pol, cfg.adult_age, cfg.equivalence, sd, inp.target_median)
        infos.append({"replicate": rep, "seed": sd, **info})
        shadow = {scen.name: scen_pol} if scen else None
        b = run_simulation(st0, base_pol, years, shadow=shadow, label="baseline", check_invariants=check_invariants, min_group_size=cfg.simulation.min_group_size)
        base_res.append((rep, b))
        m0 = b.metrics[(b.metrics.scenario == "baseline") & (b.metrics.year == 0)].iloc[0].to_dict()
        vals.append(validation_table(m0, b.snapshots[0]["eq_disp"].to_numpy(), inp.validation))
        if inp.wealth is not None:
            wvals.append(wealth_validation_table(st0, inp.wealth))
        if scen:
            st_s = st0.copy()
            st_s.econ = scen.apply_economy(st0.econ)  # scenario knobs on top of the calibrated economy
            s = run_simulation(st_s, scen_pol, years, label=scen.name, check_invariants=check_invariants, min_group_size=cfg.simulation.min_group_size)
            scen_res.append((rep, s))
            for y, snap in b.snapshots.items():
                if y in s.snapshots:
                    j = snap[["eq_disp", "wq0"]].join(s.snapshots[y][["eq_disp"]], lsuffix="_base", rsuffix="_scen", how="inner")
                    rel = j["eq_disp_scen"] / j["eq_disp_base"] - 1
                    j["outcome"] = np.where(rel > 0.01, "winner (>+1%)", np.where(rel < -0.01, "loser (<-1%)", "unchanged (+-1%)"))
                    t = pd.crosstab(j["wq0"], j["outcome"], normalize="index").reset_index().melt(id_vars="wq0", var_name="outcome", value_name="share")
                    wl_rows.append(t.assign(year=y, replicate=rep, mean_rel_change=j.groupby("wq0")["eq_disp_scen"].sum().div(j.groupby("wq0")["eq_disp_base"].sum()).sub(1).reindex(t["wq0"]).to_numpy()))
    outs: dict[str, Path] = {}
    root = out_dir
    bdir = root / "baseline"
    _write_run(bdir, "baseline", base_res, seeds, inp, years, infos, base_pol, vals, scen.name if scen else None, wvals)
    outs["baseline"] = bdir
    if scen:
        sdir = root / f"scenario_{scen.name}"
        _write_run(sdir, scen.name, scen_res, seeds, inp, years, infos, scen_pol, [], None)
        pd.concat(wl_rows, ignore_index=True).to_csv(sdir / "winners_losers_by_baseline_wealth_quintile.csv", index=False, lineterminator="\n")
        write_json({"name": scen.name, "description": scen.description, "policy_overrides": scen.policy_overrides, "policy_deltas": scen.policy_deltas,
                    "economy_overrides": scen.economy_overrides}, sdir / "scenario.json")
        outs["scenario"] = sdir
    return outs


def compare(baseline_dir: Path, scenario_dir: Path, out_dir: Path | None = None) -> Path:
    bm = pd.read_csv(baseline_dir / "metrics_by_year.csv")
    sm = pd.read_csv(scenario_dir / "metrics_by_year.csv")
    scen_name = json.loads((scenario_dir / "run_manifest.json").read_text())["label"]
    sman = json.loads((scenario_dir / "run_manifest.json").read_text())
    bman = json.loads((baseline_dir / "run_manifest.json").read_text())
    if bman["seeds"] != sman["seeds"]:
        raise ValueError("baseline and scenario were run with different seeds; common random numbers are required for a paired comparison")
    base = bm[bm.scenario == "baseline"].set_index(["replicate", "year"])
    dyn = sm[sm.scenario == scen_name].set_index(["replicate", "year"])
    stat = bm[bm.scenario == f"static:{scen_name}"].set_index(["replicate", "year"])
    cols = [c for c in KEY_METRICS if c in base]
    rows = []
    for c in cols:
        for kind, other in (("dynamic_total", dyn), ("static_fixed_behaviour", stat)):
            if other.empty:
                continue
            d = (other[c] - base[c]).dropna()
            for y, g in d.groupby(level="year"):
                rows.append({"metric": c, "effect": kind, "year": y, "n_replicates": len(g), "mean_diff": g.mean(), "p05": g.quantile(0.05) if len(g) > 1 else np.nan,
                             "p95": g.quantile(0.95) if len(g) > 1 else np.nan, "baseline_mean": base[c].xs(y, level="year").mean()})
        if not stat.empty:
            d = ((dyn[c] - base[c]) - (stat[c] - base[c])).dropna()
            for y, g in d.groupby(level="year"):
                rows.append({"metric": c, "effect": "behavioural_residual_NOT_CAUSAL", "year": y, "n_replicates": len(g), "mean_diff": g.mean(),
                             "p05": g.quantile(0.05) if len(g) > 1 else np.nan, "p95": g.quantile(0.95) if len(g) > 1 else np.nan, "baseline_mean": base[c].xs(y, level="year").mean()})
    comp = pd.DataFrame(rows)
    out = out_dir or scenario_dir
    out.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out / "comparison_vs_baseline.csv", index=False, lineterminator="\n")
    bg, sg = pd.read_csv(baseline_dir / "group_metrics.csv"), pd.read_csv(scenario_dir / "group_metrics.csv")
    bg, sg = bg[bg.scenario == "baseline"], sg[sg.scenario == scen_name]
    keys = ["replicate", "year", "group_var", "group"]
    mg = bg.merge(sg, on=keys, suffixes=("_base", "_scen"))
    for c in ("mean_eq_disposable_income", "mean_net_fiscal_position_per_capita", "poverty_rate"):
        mg[f"diff_{c}"] = mg[f"{c}_scen"] - mg[f"{c}_base"]
    mg["small_sample_warning"] = mg["small_sample_warning_base"] | mg["small_sample_warning_scen"]
    gcols = keys + ["n_sim_base", "small_sample_warning"] + [c for c in mg if c.startswith("diff_")] + ["mean_eq_disposable_income_base", "mean_eq_disposable_income_scen"]
    gg = mg[gcols].groupby(["year", "group_var", "group"], as_index=False).mean(numeric_only=True)
    gg.to_csv(out / "group_differences.csv", index=False, lineterminator="\n")
    fb = pd.read_csv(baseline_dir / "fiscal_by_year.csv")
    fs = pd.read_csv(scenario_dir / "fiscal_by_year.csv")
    fb, fs = fb[fb.scenario == "baseline"], fs[fs.scenario == scen_name]
    fm = fb.merge(fs, on=["replicate", "year"], suffixes=("_base", "_scen"))
    fcols = [c for c in fb.columns if c not in ("scenario", "year", "replicate")]
    fd = pd.DataFrame({"replicate": fm["replicate"], "year": fm["year"], **{f"diff_{c}": fm[f"{c}_scen"] - fm[f"{c}_base"] for c in fcols}})
    fd.groupby("year", as_index=False).mean(numeric_only=True).drop(columns="replicate").to_csv(out / "fiscal_differences.csv", index=False, lineterminator="\n")
    final = int(comp["year"].max())
    L = [f"# Scenario comparison: `{scen_name}` vs baseline", "", f"> **{sman['policy']['banner']}**", "",
         "> Results are mechanical demonstrations on a stylised model, not estimates of real policy effects.", "",
         f"- Baseline year {bman['baseline_year']}, horizon {bman['horizon_years']} years, {len(bman['seeds'])} paired replicate(s) with common random numbers (seeds {bman['seeds']}).",
         "- **static_fixed_behaviour**: the scenario policy applied to the unchanged baseline trajectory (direct mechanical effect).",
         "- **dynamic_total**: the ABM re-run under the scenario. **behavioural_residual_NOT_CAUSAL** = dynamic - static; it is only a descriptive gap, not a causal decomposition.", "",
         f"## Differences (scenario - baseline) at year {final}", "", "| metric | effect | mean diff | p05 | p95 | baseline |", "|---|---|---|---|---|---|"]
    for _, r in comp[comp.year == final].iterrows():
        L.append(f"| {KEY_METRICS.get(r['metric'], r['metric'])} | {r['effect']} | {r['mean_diff']:.4g} | {'' if pd.isna(r['p05']) else format(r['p05'], '.4g')} | {'' if pd.isna(r['p95']) else format(r['p95'], '.4g')} | {r['baseline_mean']:.4g} |")
    L += ["", "Group-level winners and losers: `group_differences.csv`, `winners_losers_by_baseline_wealth_quintile.csv`; fiscal flows: `fiscal_differences.csv`.", ""]
    (out / "comparison_report.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    return out
