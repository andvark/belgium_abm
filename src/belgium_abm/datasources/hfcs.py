"""Parse Belgian figures from the ECB HFCS wave-2023 statistical-tables workbook.

The workbook is a public aggregate publication (not microdata). Values are EUR thousands or per
cent as published; they are converted to EUR / ratios here. Standard errors are not kept.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

SHEETS = {
    "A1": "A1 Main aggregates - medians", "A3": "A3 Net wealth medians - breakdo", "B1": "B1 Percentage of households own",
    "J3": "J3 Net wealth per household - d", "J4": "J4 Net wealth inequality indica",
}


def _be_rows(xl: pd.ExcelFile, sheet: str, country: str = "BE") -> list[tuple[str | None, str | None, float]]:
    d = xl.parse(sheet, header=None)
    hdr_idx = d.index[d.eq(country).any(axis=1)]
    if len(hdr_idx) == 0:
        raise ValueError(f"{sheet}: no column for {country}")
    hdr = hdr_idx[0]
    col = d.columns[d.loc[hdr].eq(country)][0]
    out: list[tuple[str | None, str | None, float]] = []
    lab1 = lab2 = None
    for i in range(hdr + 1, len(d)):
        r = d.loc[i]
        if isinstance(r[0], str) and r[0].startswith("Source:"):
            break
        if isinstance(r[0], str):
            lab1 = r[0]
        if isinstance(r[1], str):
            lab2 = r[1]
        v = r[col]
        if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and (v.startswith("(") or v in ("M", "N"))):
            continue
        try:
            out.append((lab1, lab2, float(v)))
        except (TypeError, ValueError):
            continue
    return out


def parse_hfcs(xlsx: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    xl = pd.ExcelFile(xlsx)
    rows: list[tuple[str, str, float, str]] = []
    k = 1000.0
    a1 = {(a, b): v for a, b, v in _be_rows(xl, SHEETS["A1"])}
    rows.append(("net_wealth_median", "TOTAL", a1[("All", "DN3001 - Net wealth")] * k, "EUR"))
    rows.append(("share_with_debt", "TOTAL", a1[("All", "DL1000i - Has debt")] / 100, "ratio"))
    rows.append(("homeowner_share", "TOTAL", a1[("Home owner", "Share")] / 100, "ratio"))
    rows.append(("net_wealth_median", "HOMEOWNER", a1[("Home owner", "DN3001 - Net wealth")] * k, "EUR"))
    rows.append(("net_wealth_median", "NON_OWNER", a1[("Non owner", "DN3001 - Net wealth")] * k, "EUR"))
    for a, b, v in _be_rows(xl, SHEETS["J3"]):
        if a == "Mean":
            rows.append(("net_wealth_mean", "TOTAL", v * k, "EUR"))
        elif a and a.startswith("p"):
            rows.append(("net_wealth_percentile", a, v * k, "EUR"))
    for a, b, v in _be_rows(xl, SHEETS["J4"]):
        if b == "Top 10% share":
            rows.append(("net_wealth_share_top10", "TOTAL", v / 100, "ratio"))
        elif b == "Top 5% share":
            rows.append(("net_wealth_share_top5", "TOTAL", v / 100, "ratio"))
        elif b == "Gini coefficient":
            rows.append(("net_wealth_gini", "TOTAL", v, "ratio"))
    for a, b, v in _be_rows(xl, SHEETS["A3"]):
        if a == "Age of RP":
            rows.append(("net_wealth_median_by_age_of_reference_person", b or "", v * k, "EUR"))
    for a, b, v in _be_rows(xl, SHEETS["B1"]):
        if a and a.startswith("DA1140i"):
            rows.append(("owns_self_employment_business_share", "TOTAL", v / 100, "ratio"))
    df = pd.DataFrame(rows, columns=["indicator", "group", "value", "unit"])
    meta = {"source_file": xlsx.name, "source_sha256": hashlib.sha256(xlsx.read_bytes()).hexdigest(),
            "publication": "ECB Household Finance and Consumption Survey, wave 2023, statistical tables (Belgium, BE column)",
            "fieldwork": "Belgium 01/23-12/23; assets and liabilities at time of interview; income reference year 2022"}
    return df, meta
