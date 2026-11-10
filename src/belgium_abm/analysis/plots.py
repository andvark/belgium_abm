"""Population pyramid: scaled target vs generated, per region and in total."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from belgium_abm.config import ModelConfig
from belgium_abm.engines.sampling import band_index
from belgium_abm.io.loaders import TargetBundle


def population_pyramid(persons: pd.DataFrame, targets: TargetBundle, cfg: ModelConfig, path: Path, *, banner: str) -> None:
    pop = targets.population
    starts = np.array(sorted(pop["age_min"].unique()), dtype=np.int64)
    ends = {int(a): int(b) for a, b in pop[["age_min", "age_max"]].drop_duplicates().itertuples(index=False)}
    labels = [f"{s}-{ends[int(s)]}" for s in starts]
    scale = cfg.generator.n_persons / targets.total_persons
    sim = persons.assign(band=band_index(persons["age"].to_numpy(), starts))

    panels = [*cfg.regions, "ALL REGIONS"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 6.2), sharey=True)
    y = np.arange(len(starts))
    for ax, name in zip(np.atleast_1d(axes), panels, strict=True):
        t = pop if name == "ALL REGIONS" else pop[pop["region"] == name]
        s = sim if name == "ALL REGIONS" else sim[sim["region"] == name]
        for sex, sign, color in (("F", -1.0, "#c0504d"), ("M", 1.0, "#4f81bd")):
            tgt = t[t["sex_or_demographic_stratum"] == sex].groupby("age_min")["count"].sum().reindex(starts).fillna(0).to_numpy() * scale
            got = s[s["sex_or_demographic_stratum"] == sex].groupby("band").size().reindex(range(len(starts))).fillna(0).to_numpy()
            ax.barh(y, sign * got, height=0.8, color=color, alpha=0.55, label=f"generated {sex}")
            ax.vlines(sign * tgt, y - 0.4, y + 0.4, color="black", linewidth=1.8, label="target (scaled)" if sex == "F" else None)
        ax.axvline(0, color="grey", linewidth=0.6)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("persons (female < 0 < male)")
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{abs(int(v))}" for v in ticks])
    np.atleast_1d(axes)[0].set_yticks(y)
    np.atleast_1d(axes)[0].set_yticklabels(labels)
    np.atleast_1d(axes)[0].set_ylabel("age band")
    handles, lbls = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        f"Population pyramid - {cfg.name} - n={len(persons):,} simulated persons, 1 simulated person = "
        f"{persons['weight'].iloc[0]:,.1f} real persons\n{banner}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_timeseries(metrics, columns: dict[str, str], path, title: str, banner: str, scenario_col: str = "scenario") -> None:
    """Small-multiples line chart of metrics by year; shaded 5-95% band across replicates when present."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    for ax, (col, label) in zip(axes[0], columns.items(), strict=True):
        for scen, g in metrics.groupby(scenario_col):
            if str(scen).startswith("static:"):
                continue
            by = g.groupby("year")[col]
            mean = by.mean()
            ax.plot(mean.index, mean.values, label=str(scen))
            if g["replicate"].nunique() > 1:
                ax.fill_between(mean.index, by.quantile(0.05).values, by.quantile(0.95).values, alpha=0.2)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("model year")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.005, banner[:230], ha="center", fontsize=6, color="firebrick")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)
