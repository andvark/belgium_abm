"""Cross-checks of the derived real-data tables against independent Statbel publications."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from belgium_abm.datasources.statbel_files import parse_education_series, parse_lfs_region_education_age

CMAP = {"From 15 to 24 years": [(15, 1.0)], "From 25 to 49 years": [(25, 1.0), (35, 1.0), (45, 0.5)], "From 50 to 64 years": [(45, 0.5), (55, 1.0)]}
BAND_END = {15: 24, 25: 34, 35: 44, 45: 54, 55: 64}


def _read(path: Path) -> pd.DataFrame:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(lines)))


def labour_vs_statbel(real_dir: Path, lfs_xlsx: Path, year: int = 2025) -> pd.DataFrame:
    """Model (derived) employment / unemployment rates by region x education x Statbel age band vs Statbel's published LFS rates.

    Statbel's bands (15-24 / 25-49 / 50-64) are rebuilt from the model's bands, splitting 45-54 in half (approximation).
    """
    pop, edu, lab = (_read(real_dir / f) for f in ("population_age_sex_region.csv", "education_by_age_region.csv", "labour_state_by_age_education_region.csv"))
    obs = parse_lfs_region_education_age(lfs_xlsx)
    obs = obs[(obs.year == year) & obs.age_band.isin(CMAP)]
    pw = {(r, a): pop[(pop.region == r) & (pop.age_min >= a) & (pop.age_max <= BAND_END[a])]["count"].sum() for r in pop.region.unique() for a in BAND_END}
    rows = []
    for _, o in obs.iterrows():
        num_e = num_u = den = 0.0
        for a, w in CMAP[o.age_band]:
            es = edu[(edu.region == o.region) & (edu.age_min == a) & (edu.education_level == o.education)].share.iloc[0]
            sh = lab[(lab.region == o.region) & (lab.age_min == a) & (lab.education_level == o.education)].set_index("work_state").share
            wt = pw[(o.region, a)] * es * w
            num_e += wt * sh["EMPLOYED"]
            num_u += wt * sh["UNEMPLOYED"]
            den += wt
        rows.append((o.region, o.education, o.age_band, o.employment_rate, num_e / den, o.unemployment_rate, num_u / max(num_e + num_u, 1e-12)))
    df = pd.DataFrame(rows, columns=["region", "education", "age_band", "employment_statbel", "employment_model", "unemployment_statbel", "unemployment_model"])
    df["employment_diff"] = df["employment_model"] - df["employment_statbel"]
    df["unemployment_diff"] = df["unemployment_model"] - df["unemployment_statbel"]
    return df.dropna(subset=["employment_statbel"]).reset_index(drop=True)


def education_vs_statbel(real_dir: Path, edu_xls: Path, year: int = 2025) -> pd.DataFrame:
    """National education shares (15+, 3 groups) from the model's table vs the Statbel series."""
    pop, edu = _read(real_dir / "population_age_sex_region.csv"), _read(real_dir / "education_by_age_region.csv")
    ser = parse_education_series(edu_xls)
    ser = ser[ser.year == year].set_index("category").share
    obs = {"LOW": ser["PRIMARY"] + ser["LOWER_SECONDARY"], "MEDIUM": ser["UPPER_SECONDARY"], "HIGH": ser["SHORT_TERTIARY"] + ser["LONG_TERTIARY"]}
    tot, mod = 0.0, {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0}
    for r in edu.region.unique():
        for a in edu.age_min.unique():
            hi = 99 if a == 65 else BAND_END[a]
            w = pop[(pop.region == r) & (pop.age_min >= a) & (pop.age_max <= hi)]["count"].sum()
            tot += w
            for e in mod:
                mod[e] += w * edu[(edu.region == r) & (edu.age_min == a) & (edu.education_level == e)].share.iloc[0]
    out = pd.DataFrame({"education": list(mod), "model_share": [mod[k] / tot for k in mod], "statbel_share": [float(obs[k]) for k in mod]})
    out["diff"] = out["model_share"] - out["statbel_share"]
    return out
