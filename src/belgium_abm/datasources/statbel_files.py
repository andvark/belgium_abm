"""Parsers for Statbel workbooks downloaded from the Statbel / BeStat table viewers (user-supplied files).

These are used as INDEPENDENT cross-checks of the derived input tables; they are never fed back
into the generator, so agreement is a genuine test of the derivation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

REGION = {"Brussels-Capital region": "BE_BRUSSELS", "Flanders region": "BE_FLANDERS", "Flemish region": "BE_FLANDERS", "Walloon region": "BE_WALLONIA"}
EDU = {"Low": "LOW", "Middle": "MEDIUM", "High": "HIGH"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_lfs_region_education_age(xlsx: Path) -> pd.DataFrame:
    """Statbel LFS 'Data as Table' sheet: employment / unemployment / activity rate by region x education x age band x year."""
    d = pd.read_excel(xlsx, sheet_name="Data as Table")
    d.columns = ["country", "region", "r_total", "education", "e_total", "age_band", "year", "employment_rate", "unemployment_rate", "activity_rate"]
    d = d[d["region"].isin(REGION) & d["education"].isin(EDU)].copy()
    d["region"] = d["region"].map(REGION)
    d["education"] = d["education"].map(EDU)
    d["year"] = d["year"].astype(int)
    return d[["region", "education", "age_band", "year", "employment_rate", "unemployment_rate", "activity_rate"]].reset_index(drop=True)


def parse_education_series(xls: Path) -> pd.DataFrame:
    """Statbel education attainment of the population aged 15+ (national, shares, 1987-2025). Returns year x 5 categories."""
    raw = pd.read_excel(xls, sheet_name="en %", header=None)
    rows = []
    year_row = None
    for _, r in raw.iterrows():
        if r[0] == "Diplôme obtenu":
            year_row = [str(v).split(" ")[0].split(".")[0] for v in r[1:]]
        elif year_row and isinstance(r[0], str) and r[0] != "Total":
            for y, v in zip(year_row, r[1:], strict=False):
                if y.isdigit() and pd.notna(v):
                    rows.append((int(y), r[0], float(v)))
    df = pd.DataFrame(rows, columns=["year", "category", "share"])
    key = {"Primaire ou sans diplôme": "PRIMARY", "Secondaire inférieur": "LOWER_SECONDARY", "Secondaire supérieur": "UPPER_SECONDARY"}
    df["category"] = df["category"].map(lambda c: key.get(c, "SHORT_TERTIARY" if "court" in c else "LONG_TERTIARY" if "long" in c else c))
    return df
