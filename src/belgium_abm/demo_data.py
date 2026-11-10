"""FICTIONAL demo input tables.

Everything here is invented with simple formulas so the pipeline can be demonstrated end to end.
The numbers do NOT describe Belgium or any other real population, are not estimates of
anything, and every file is stamped ``synthetic_demo_only: true``. The household table is
derived from the population table so the two are mutually consistent for ``adult_age = 18``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEMO_FILES = {
    "population": "population_age_sex_region.csv",
    "households": "household_templates_region.csv",
    "education": "education_by_age_region.csv",
    "labour_state": "labour_state_by_age_education_region.csv",
    "work_arrangement": "work_arrangement_by_age_region.csv",
    "tenure": "tenure_by_household_type_region.csv",
}

REGIONS = ("REGION_A", "REGION_B", "REGION_C")
REGION_PERSONS = {"REGION_A": 270_000, "REGION_B": 160_000, "REGION_C": 112_800}
POP_BANDS = [(a, a + 4) for a in range(0, 85, 5)] + [(85, 99)]
BAND_WEIGHTS = [1.00, 1.02, 1.04, 1.06, 1.10, 1.15, 1.17, 1.16, 1.12, 1.10, 1.08, 1.04, 0.96, 0.86, 0.74, 0.60, 0.50, 0.40]
YOUNG_TILT = {"REGION_A": 0.0, "REGION_B": 0.10, "REGION_C": -0.10}
WORK_BANDS = [(15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 99)]
DEMO_ADULT_AGE = 18


def _round_shares(vals: list[float], decimals: int = 8) -> list[float]:
    arr = np.clip(np.asarray(vals, dtype=float), 0.0, None)
    arr = arr / arr.sum()
    r = np.round(arr, decimals)
    r[int(np.argmax(r))] += round(1.0 - float(r.sum()), decimals)
    return [float(x) for x in r]


def population_table() -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        mids = [(lo + hi) / 2 for lo, hi in POP_BANDS]
        w = np.array([bw * (1 + YOUNG_TILT[region] * (40 - m) / 40) for bw, m in zip(BAND_WEIGHTS, mids, strict=True)])
        w = w / w.sum() * REGION_PERSONS[region]
        for (lo, hi), m, wb in zip(POP_BANDS, mids, w, strict=True):
            f_share = min(0.66, 0.488 + 0.17 * max(0.0, m - 55) / 45)
            rows.append((region, "F", lo, hi, int(round(wb * f_share))))
            rows.append((region, "M", lo, hi, int(round(wb * (1 - f_share)))))
    return pd.DataFrame(rows, columns=["region", "sex_or_demographic_stratum", "age_min", "age_max", "count"])


# Base mix of templates with children (weights are fictional) and of those without.
_CHILD_TEMPLATES = [("COUPLE_WITH_CHILDREN", 2, 1, 0.30), ("COUPLE_WITH_CHILDREN", 2, 2, 0.25), ("COUPLE_WITH_CHILDREN", 2, 3, 0.08), ("SINGLE_PARENT", 1, 1, 0.12), ("SINGLE_PARENT", 1, 2, 0.06)]
_SINGLE_SHIFT = {"REGION_A": 0.05, "REGION_B": 0.0, "REGION_C": -0.05}


def household_table(pop: pd.DataFrame, adult_age: int = DEMO_ADULT_AGE) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        g = pop[pop["region"] == region]
        adults = children = 0.0
        for lo, hi, cnt in g[["age_min", "age_max", "count"]].itertuples(index=False):
            width = hi - lo + 1
            n_adult_years = max(0, hi - max(lo, adult_age) + 1)
            adults += cnt * n_adult_years / width
            children += cnt * (width - n_adult_years) / width
        w_sum = sum(t[3] for t in _CHILD_TEMPLATES)
        c_per = sum(t[2] * t[3] for t in _CHILD_TEMPLATES) / w_sum
        a_per = sum(t[1] * t[3] for t in _CHILD_TEMPLATES) / w_sum
        n_child_hh = children / c_per
        adults_rest = adults - n_child_hh * a_per
        if adults_rest <= 0:
            raise ValueError("demo constants give a negative number of childless adults")
        for typ, a, c, w in _CHILD_TEMPLATES:
            rows.append((region, typ, a, c, int(round(n_child_hh * w / w_sum))))
        p_single = 0.45 + _SINGLE_SHIFT[region]
        p_couple = 0.45 - _SINGLE_SHIFT[region]
        p_multi = 0.10
        unit = adults_rest / (p_single + 2 * p_couple + 3 * p_multi)
        rows.append((region, "SINGLE_ADULT", 1, 0, int(round(unit * p_single))))
        rows.append((region, "COUPLE", 2, 0, int(round(unit * p_couple))))
        rows.append((region, "OTHER_MULTI", 3, 0, int(round(unit * p_multi))))
    return pd.DataFrame(rows, columns=["region", "household_type", "n_adults", "n_children", "count"])


_EDU_LOW = [0.55, 0.20, 0.24, 0.30, 0.38, 0.52]
_EDU_HIGH = [0.05, 0.38, 0.36, 0.32, 0.27, 0.18]
_EDU_SHIFT = {"REGION_A": 0.0, "REGION_B": -0.03, "REGION_C": 0.03}
EDU_LEVELS = ("LOW", "MEDIUM", "HIGH")


def education_table() -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        for i, (lo, hi) in enumerate(WORK_BANDS):
            low = _EDU_LOW[i] + _EDU_SHIFT[region]
            high = _EDU_HIGH[i] - _EDU_SHIFT[region]
            sh = _round_shares([low, 1.0 - low - high, high])
            rows += [(region, lo, hi, lvl, s) for lvl, s in zip(EDU_LEVELS, sh, strict=True)]
    return pd.DataFrame(rows, columns=["region", "age_min", "age_max", "education_level", "share"])


_STATES = ("STUDENT", "EMPLOYED", "UNEMPLOYED", "INACTIVE", "RETIRED")
_LAB_BASE = [
    [0.50, 0.30, 0.05, 0.15, 0.00],
    [0.05, 0.75, 0.07, 0.13, 0.00],
    [0.01, 0.80, 0.06, 0.13, 0.00],
    [0.00, 0.78, 0.06, 0.15, 0.01],
    [0.00, 0.50, 0.05, 0.17, 0.28],
    [0.00, 0.03, 0.01, 0.08, 0.88],
]
_ACT = [0.3, 1.0, 1.0, 1.0, 1.0, 0.1]
_LAB_SHIFT = {"REGION_A": 0.0, "REGION_B": 0.02, "REGION_C": -0.02}
_EDU_ADJ = {"LOW": (-0.08, 0.03, 0.05), "MEDIUM": (0.0, 0.0, 0.0), "HIGH": (0.06, -0.03, -0.03)}


def labour_table() -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        for i, (lo, hi) in enumerate(WORK_BANDS):
            for lvl in EDU_LEVELS:
                st, em, un, ina, ret = _LAB_BASE[i]
                d_emp, d_un, d_in = _EDU_ADJ[lvl]
                em += d_emp * _ACT[i] + _LAB_SHIFT[region] * _ACT[i]
                un += d_un * _ACT[i]
                ina += d_in * _ACT[i] - _LAB_SHIFT[region] * _ACT[i]
                sh = _round_shares([st, em, un, ina, ret])
                rows += [(region, lo, hi, lvl, s, v) for s, v in zip(_STATES, sh, strict=True)]
    df = pd.DataFrame(rows, columns=["region", "age_min", "age_max", "education_level", "work_state", "share"])
    return df[["region", "age_min", "age_max", "education_level", "work_state", "share"]]


_ARR = ("EMPLOYEE", "SOLE_PROPRIETOR", "BV_OWNER_MANAGER")
_ARR_BASE = [[0.97, 0.025, 0.005], [0.88, 0.08, 0.04], [0.84, 0.09, 0.07], [0.82, 0.10, 0.08], [0.80, 0.10, 0.10], [0.60, 0.20, 0.20]]
_ARR_SHIFT = {"REGION_A": 0.0, "REGION_B": 0.01, "REGION_C": -0.01}


def arrangement_table() -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        for i, (lo, hi) in enumerate(WORK_BANDS):
            e, s, b = _ARR_BASE[i]
            s += _ARR_SHIFT[region]
            e -= _ARR_SHIFT[region]
            sh = _round_shares([e, s, b])
            rows += [(region, lo, hi, a, v) for a, v in zip(_ARR, sh, strict=True)]
    return pd.DataFrame(rows, columns=["region", "age_min", "age_max", "primary_work_arrangement", "share"])


_TEN = ("RENTER", "MORTGAGED_OWNER", "OUTRIGHT_OWNER", "OTHER")
_TEN_BASE = {
    "SINGLE_ADULT": [0.45, 0.15, 0.33, 0.07],
    "COUPLE": [0.20, 0.30, 0.45, 0.05],
    "COUPLE_WITH_CHILDREN": [0.17, 0.55, 0.25, 0.03],
    "SINGLE_PARENT": [0.45, 0.25, 0.25, 0.05],
    "OTHER_MULTI": [0.30, 0.25, 0.40, 0.05],
}
_TEN_SHIFT = {"REGION_A": 0.04, "REGION_B": 0.0, "REGION_C": -0.04}


def tenure_table() -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        for typ, base in _TEN_BASE.items():
            r, m, o, x = base
            sh = _round_shares([r + _TEN_SHIFT[region], m, o - _TEN_SHIFT[region], x])
            rows += [(region, typ, t, v) for t, v in zip(_TEN, sh, strict=True)]
    return pd.DataFrame(rows, columns=["region", "household_type", "housing_tenure", "share"])


_DESCRIPTIONS = {
    "population": "FICTIONAL persons by region x sex x age band",
    "households": "FICTIONAL households by region x type x adults x children",
    "education": "FICTIONAL education shares by region x age band",
    "labour_state": "FICTIONAL work-state shares by region x age band x education",
    "work_arrangement": "FICTIONAL work-arrangement shares among employed, by region x age band",
    "tenure": "FICTIONAL housing-tenure shares by region x household type",
}


def _write(path: Path, table: str, df: pd.DataFrame) -> None:
    unit = "persons" if table == "population" else ("households" if table == "households" else "share")
    header = [
        f"# dataset_id: DEMO_FICTIONAL_{table}",
        "# synthetic_demo_only: true",
        "# status: assumed",
        "# reference_year: none",
        f"# unit: {unit}",
        "# source: FICTIONAL - generated by belgium_abm.demo_data formulas; NOT Belgian data, NOT an estimate",
        f"# description: {_DESCRIPTIONS[table]}",
    ]
    body = df.to_csv(index=False, float_format="%.8f", lineterminator="\n")
    path.write_text("\n".join(header) + "\n" + body, encoding="utf-8", newline="\n")


def demo_tables() -> dict[str, pd.DataFrame]:
    pop = population_table()
    return {
        "population": pop,
        "households": household_table(pop),
        "education": education_table(),
        "labour_state": labour_table(),
        "work_arrangement": arrangement_table(),
        "tenure": tenure_table(),
    }


def write_demo_data(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for table, df in demo_tables().items():
        p = directory / DEMO_FILES[table]
        _write(p, table, df)
        out.append(p)
    return out
