"""Population pyramid: scaled target vs generated, per region and in total."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from belgium_abm.config import ModelConfig  # noqa: E402
from belgium_abm.engines.sampling import band_index  # noqa: E402
from belgium_abm.io.loaders import TargetBundle  # noqa: E402


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
