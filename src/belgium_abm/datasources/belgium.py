"""Build the model's six input tables for Belgium from official open data.

Provenance policy (matches the project rule "no imaginary empirical parameters"):

* ``observed``  -- a published count/rate used as is (possibly summed over published cells).
* ``derived``   -- computed from observed margins by a documented, deterministic procedure
                   (IPF / maximum-entropy fitting). Nothing is invented; unpublished joint
                   distributions are *reconstructed*, not observed.
* ``assumed``   -- listed explicitly in ``ASSUMPTIONS`` below and in the build manifest.

All data come from Eurostat's dissemination API (which republishes Statbel/LFS figures for
Belgium) and from Statbel's BeStat API. Raw responses are cached with sha256 checksums.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from belgium_abm.datasources.bestat import fetch_view_csv
from belgium_abm.datasources.eurostat import fetch_raw, to_frame

REGIONS = ("BE_BRUSSELS", "BE_FLANDERS", "BE_WALLONIA")
NUTS2 = {
    "BE_BRUSSELS": ["BE10"],
    "BE_FLANDERS": ["BE21", "BE22", "BE23", "BE24", "BE25"],
    "BE_WALLONIA": ["BE31", "BE32", "BE33", "BE34", "BE35"],
}
LFS_GEO = {"BE_BRUSSELS": "BE10", "BE_FLANDERS": "BE2", "BE_WALLONIA": "BE3"}
BESTAT_REGION = {"BE_BRUSSELS": "Brussels-Capital region", "BE_FLANDERS": "Flanders region", "BE_WALLONIA": "Walloon region"}
BESTAT_LEGAL_FORM_VIEW = "efedbaa1-1478-4366-9d9a-5318af8f90ab"  # VAT-subject enterprises by legal form and region

POP_BANDS = [(0, 4), (5, 9), (10, 14), (15, 17), (18, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44),
             (45, 49), (50, 54), (55, 59), (60, 64), (65, 69), (70, 74), (75, 79), (80, 84), (85, 99)]
ATTR_BANDS = [(15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 99)]
LFS_REGIONAL_AGE = {(15, 24): "Y15-24", (25, 34): "Y25-34", (35, 44): "Y35-44", (45, 54): "Y45-54", (55, 64): "Y55-64", (65, 99): "Y_GE65"}
NAT_AGE_PARTS = {(15, 24): ["Y15-24"], (25, 34): ["Y25-29", "Y30-34"], (35, 44): ["Y35-39", "Y40-44"],
                 (45, 54): ["Y45-49", "Y50-54"], (55, 64): ["Y55-59", "Y60-64"]}  # 65+ = Y15-74 minus Y15-64
EDU = {"ED0-2": "LOW", "ED3_4": "MEDIUM", "ED5-8": "HIGH"}
STATES = ["STUDENT", "EMPLOYED", "UNEMPLOYED", "INACTIVE", "RETIRED"]
EDULEVELS = ["LOW", "MEDIUM", "HIGH"]

ASSUMPTIONS = [
    "Regional household-type mix equals the national mix (Eurostat publishes household composition for Belgium only); it is then re-fitted per region to the regional adult and child totals (maximum entropy).",
    "Households with 3+ children are split 80/20 into 3 and 4 children, and OTHER_MULTI households 75/25 into 3 and 4 adults (seed only; the maximum-entropy fit moves these to match regional totals).",
    "OTHER_MULTI requires >= 3 adults in this model; 2-adult non-couple households published as 'other type' cannot be represented separately.",
    "Population in institutions (about 1-2% of residents, mostly elderly) is not covered by the private-household tables; the household table is fitted to the full resident population, so the implied household count exceeds the published private-household count (reported in the manifest).",
    "STUDENT = (share of not-employed 15-24 year-olds in formal or non-formal education/training, national, Eurostat edat_lfse_18) x (national not-employed share of 15-24); applied to ages 15-24 only and capped by the regional inactive share. Non-formal training makes this an upper bound; working students are EMPLOYED; no students are assumed above 24.",
    "RETIRED is defined by convention as inactive persons aged 65+; inactive persons below 65 (including early retirees) are INACTIVE. The LFS does not publish a 'retired' status here.",
    "At ages 65+ the unemployed share of the population equals the national 65-74 share in every region (no regional unemployment rate is published for 65+).",
    "Education x labour-state joint distribution by region is reconstructed by IPF from national age x education x labour-state seeds and observed regional margins (education by age; employment and unemployment rates by age).",
    "Self-employment share of employment by age is national (no regional x age breakdown); it is applied in every region.",
    "The BV/SRL share of the self-employed is proxied by regional counts of VAT-subject enterprises: private limited liability companies / (private limited liability companies + natural persons), 2024. Enterprise counts are not persons; this is a rough proxy, not an observation.",
    "Housing-tenure shares are national population shares by household type (EU-SILC via Eurostat) applied to households and to all regions. RENTER = tenant at market rent; OTHER = tenant at reduced or free rent.",
    "Age 100+ residents are included in the 85-99 band; within-band ages are uniform (model convention).",
]


@dataclass
class Build:
    cache_dir: Path
    year: int
    offline: bool = False
    sources: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    # -- data access ---------------------------------------------------------------
    def es(self, code: str, **params) -> pd.DataFrame:
        p = {"time": str(self.year), **params}
        js, sha = fetch_raw(code, p, self.cache_dir, offline=self.offline)
        self.sources.append({"provider": "Eurostat", "dataset": code, "params": p, "updated": js.get("updated"), "label": js.get("label"), "sha256": sha})
        return to_frame(js)

    def bestat(self, view_id: str) -> pd.DataFrame:
        df, sha = fetch_view_csv(view_id, self.cache_dir, offline=self.offline)
        self.sources.append({"provider": "Statbel BeStat", "view": view_id, "sha256": sha})
        return df


# ---------------------------------------------------------------------- helpers ------------
def ipf(seed: np.ndarray, rows: np.ndarray, cols: np.ndarray, iters: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """2-D iterative proportional fitting. Zero margins force zero rows/cols."""
    m = seed.astype(float).copy()
    m[:, cols <= 0] = 0.0
    m[rows <= 0, :] = 0.0
    for _ in range(iters):
        rs = m.sum(axis=1)
        m *= np.where(rs > 0, rows / np.where(rs > 0, rs, 1), 0.0)[:, None]
        cs = m.sum(axis=0)
        m *= np.where(cs > 0, cols / np.where(cs > 0, cs, 1), 0.0)[None, :]
        if np.abs(m.sum(axis=1) - rows).max() < tol and np.abs(m.sum(axis=0) - cols).max() < tol:
            break
    return m


def _band_label(a: tuple[int, int]) -> str:
    return f"{a[0]}-{a[1]}"


# ------------------------------------------------------------------ tables ----------------
def build_population(b: Build) -> pd.DataFrame:
    geos = [g for r in REGIONS for g in NUTS2[r]]
    d = b.es("demo_r_d2jan", geo=geos, sex=["M", "F"], unit="NR")
    d = d[~d.age.isin(["TOTAL", "UNK"])].copy()

    def age_num(code: str) -> int:
        if code == "Y_LT1":
            return 0
        if code == "Y_OPEN":
            return 100
        return int(code[1:])

    d["a"] = d.age.map(age_num)
    d["region"] = d.geo.map({g: r for r, gs in NUTS2.items() for g in gs})
    rows = []
    for r in REGIONS:
        for s in ("F", "M"):
            g = d[(d.region == r) & (d.sex == s)]
            for lo, hi in POP_BANDS:
                top = 100 if hi == 99 else hi
                rows.append((r, s, lo, hi, float(g[(g.a >= lo) & (g.a <= top)].value.sum())))
    return pd.DataFrame(rows, columns=["region", "sex_or_demographic_stratum", "age_min", "age_max", "count"])


def build_households(b: Build, pop: pd.DataFrame) -> pd.DataFrame:
    h = b.es("lfst_hhnhtych", geo="BE", agechild="TOTAL")
    h = h[h.n_child != "TOTAL"].pivot_table(index="phhcomp", columns="n_child", values="value")
    total_priv = b.es("lfst_r_lfsd2hh", geo=list(LFS_GEO.values()), deg_urb="TOTAL").set_index("geo").value * 1000

    def v(comp: str, n: str) -> float:
        x = h.loc[comp, n]
        return float(0.0 if pd.isna(x) else x) * 1000

    seeds: list[tuple[str, int, int, float]] = []
    seeds.append(("SINGLE_ADULT", 1, 0, v("A1_NCH", "0")))
    seeds.append(("COUPLE", 2, 0, v("CPL_NCH", "0")))
    for typ, comp, adults in (("COUPLE_WITH_CHILDREN", "CPL_CH", 2), ("SINGLE_PARENT", "A1_CH", 1)):
        seeds += [(typ, adults, 1, v(comp, "1")), (typ, adults, 2, v(comp, "2")),
                  (typ, adults, 3, 0.8 * v(comp, "GE3")), (typ, adults, 4, 0.2 * v(comp, "GE3"))]
    other = {0: v("OTH_NCH", "0"), 1: v("OTH_CH", "1"), 2: v("OTH_CH", "2"), 3: v("OTH_CH", "GE3")}
    for k, val in other.items():
        seeds += [("OTHER_MULTI", 3, k, 0.75 * val), ("OTHER_MULTI", 4, k, 0.25 * val)]
    tpl = pd.DataFrame(seeds, columns=["household_type", "n_adults", "n_children", "seed"])
    tpl = tpl[tpl.seed > 0].reset_index(drop=True)

    rows = []
    diag = {}
    for r in REGIONS:
        g = pop[pop.region == r]
        adults = float(g[g.age_min >= 18]["count"].sum())
        children = float(g[g.age_max <= 17]["count"].sum())
        a, c = tpl.n_adults.to_numpy(float), tpl.n_children.to_numpy(float)
        w0 = tpl.seed.to_numpy(float)
        # scale seed so the household count is of the right order before solving for the tilts
        w0 = w0 * (adults / (w0 * a).sum())

        x = np.stack([a, c], axis=1)
        target = np.array([adults, children]) / adults  # in units of the adult total

        def dual(lam: np.ndarray, w0=w0, x=x, target=target) -> tuple[float, np.ndarray]:
            w = w0 / adults * np.exp(x @ lam)
            return float(w.sum() - lam @ target), x.T @ w - target

        sol = minimize(dual, x0=np.zeros(2), jac=True, method="BFGS", options={"gtol": 1e-13, "maxiter": 1000})
        if np.abs(sol.jac).max() > 1e-8:
            raise RuntimeError(f"household max-entropy fit failed for {r}: {sol.message} (grad={sol.jac})")
        w = w0 * np.exp(x @ sol.x)
        out = tpl.assign(region=r, count=w)
        rows.append(out)
        diag[r] = {
            "fitted_households": float(w.sum()),
            "published_private_households": float(total_priv[LFS_GEO[r]]),
            "ratio_fitted_to_published": float(w.sum() / total_priv[LFS_GEO[r]]),
            "adults": adults, "children": children, "tilt_adults": float(sol.x[0]), "tilt_children": float(sol.x[1]),
        }
    b.diagnostics["households"] = diag
    df = pd.concat(rows, ignore_index=True)
    return df[["region", "household_type", "n_adults", "n_children", "count"]]


def _sum_age(df: pd.DataFrame, parts: list[str], edu: str) -> float:
    s = df[(df.isced11 == edu) & (df.age.isin(parts))].value
    return float(s.sum()) if len(s) else 0.0


def _national_band(df: pd.DataFrame, band: tuple[int, int], edu: str) -> float:
    if band == (65, 99):
        return _sum_age(df, ["Y15-74"], edu) - _sum_age(df, ["Y15-64"], edu)
    return _sum_age(df, NAT_AGE_PARTS[band], edu)


def build_education(b: Build) -> pd.DataFrame:
    d = b.es("lfst_r_lfsd2pop", geo=list(LFS_GEO.values()), sex="T", unit="THS_PER")
    rows = []
    for r in REGIONS:
        for band in ATTR_BANDS:
            g = d[(d.geo == LFS_GEO[r]) & (d.age == LFS_REGIONAL_AGE[band])].set_index("isced11").value
            vals = np.array([g.get(k, 0.0) for k in EDU])
            vals = vals / vals.sum()
            for lvl, s in zip(EDU.values(), vals, strict=True):
                rows.append((r, band[0], band[1], lvl, float(s)))
    return pd.DataFrame(rows, columns=["region", "age_min", "age_max", "education_level", "share"])


def build_labour(b: Build, edu_tbl: pd.DataFrame) -> pd.DataFrame:
    pop = b.es("lfsa_pgaed", geo="BE", sex="T", unit="THS_PER")
    emp = b.es("lfsa_egaed", geo="BE", sex="T", unit="THS_PER")
    act = b.es("lfsa_agaed", geo="BE", sex="T", unit="THS_PER")
    erate = b.es("lfst_r_lfe2emprt", geo=list(LFS_GEO.values()), sex="T", unit="PC")
    urate = b.es("lfst_r_lfu3rt", geo=list(LFS_GEO.values()), sex="T", isced11="TOTAL", unit="PC")
    edu_part = b.es("edat_lfse_18", geo="BE", sex="T", wstatus="NEMP", training="FE_NFE", unit="PC")

    # national seeds [edu, state-group] per band: emp, une, inac as shares of population
    seeds: dict[tuple[int, int], np.ndarray] = {}
    nat_share: dict[tuple[int, int], dict[str, float]] = {}
    for band in ATTR_BANDS:
        m = np.zeros((3, 3))
        for i, e in enumerate(EDU):
            p, em, ac = (_national_band(t, band, e) for t in (pop, emp, act))
            if p <= 0:
                continue
            m[i] = [em / p, max(ac - em, 0.0) / p, max(p - ac, 0.0) / p]
        seeds[band] = m
        tp = sum(_national_band(pop, band, e) for e in EDU)
        te = sum(_national_band(emp, band, e) for e in EDU)
        ta = sum(_national_band(act, band, e) for e in EDU)
        nat_share[band] = {"pop": tp, "emp": te / tp, "une": max(ta - te, 0) / tp}

    # students = not-employed young people in education/training (Eurostat edat_lfse_18, formal + non-formal)
    # x the national not-employed share; applied to ages 15-24 only.
    pop_t = _sum_age(pop, ["Y15-24"], "TOTAL")
    not_emp_1524 = 1.0 - _sum_age(emp, ["Y15-24"], "TOTAL") / pop_t
    part = float(edu_part[edu_part.age == "Y15-24"].value.iloc[0]) / 100.0
    stu = {band: 0.0 for band in ATTR_BANDS}
    stu[(15, 24)] = part * not_emp_1524
    b.diagnostics["student_share"] = {_band_label(k): v for k, v in stu.items()}

    rows = []
    flags: list[str] = []
    for r in REGIONS:
        for band in ATTR_BANDS:
            ag = LFS_REGIONAL_AGE[band]
            E = float(erate[(erate.geo == LFS_GEO[r]) & (erate.age == ag)].value.iloc[0]) / 100.0
            if band == (65, 99):
                u = nat_share[band]["une"]
            else:
                U = float(urate[(urate.geo == LFS_GEO[r]) & (urate.age == ag)].value.iloc[0]) / 100.0
                u = E * U / (1.0 - U)
            retired = 0.0
            student = min(stu[band], max(1.0 - E - u, 0.0))
            if band == (65, 99):
                retired = max(1.0 - E - u, 0.0)
                inactive = 0.0
            else:
                inactive = 1.0 - E - u - student
            cols = np.array([student, E, u, inactive, retired])
            if cols.min() < -1e-9:
                flags.append(f"{r} {band}: negative margin {cols}")
                cols = np.clip(cols, 0, None) / np.clip(cols, 0, None).sum()
            pi = (edu_tbl[(edu_tbl.region == r) & (edu_tbl.age_min == band[0])].set_index("education_level").share.reindex(EDULEVELS)).to_numpy()
            s = seeds[band]  # columns: emp, une, inac
            full = np.zeros((3, 5))
            full[:, 1], full[:, 2] = s[:, 0], s[:, 1]
            full[:, 0] = s[:, 2] if student > 0 else 0.0  # student seed ~ inactive profile by education
            full[:, 3] = s[:, 2] if band != (65, 99) else 0.0
            full[:, 4] = s[:, 2] if band == (65, 99) else 0.0
            full = np.maximum(full, 1e-9) * (cols > 0)[None, :]
            fit = ipf(full * pi[:, None], pi, cols)
            cond = fit / pi[:, None]
            for i, lvl in enumerate(EDULEVELS):
                row = cond[i] / cond[i].sum()
                for st, share in zip(STATES, row, strict=True):
                    rows.append((r, band[0], band[1], lvl, st, float(share)))
    b.diagnostics["labour_flags"] = flags
    return pd.DataFrame(rows, columns=["region", "age_min", "age_max", "education_level", "work_state", "share"])


def build_arrangement(b: Build) -> pd.DataFrame:
    selfe = b.es("lfsa_esgan", geo="BE", sex="T", unit="THS_PER", wstatus="SELF", citizen="TOTAL")
    empl = b.es("lfsa_egan", geo="BE", sex="T", unit="THS_PER", citizen="TOTAL")
    lf = b.bestat(BESTAT_LEGAL_FORM_VIEW)
    lf = lf[lf["Year"].astype(str) == "2024"]
    q = {}
    for r in REGIONS:
        reg = lf[lf["Region"] == BESTAT_REGION[r]].set_index("Legal Form")["Number of ent. subject to vat"]
        natural, bv = float(reg["Natural persons"]), float(reg["Private limited liability company"])
        q[r] = bv / (bv + natural)
    b.diagnostics["bv_share_of_self_employed_proxy"] = q
    parts = {(15, 24): ["Y15-24"], (25, 34): ["Y25-29", "Y30-34"], (35, 44): ["Y35-39", "Y40-44"],
             (45, 54): ["Y45-49", "Y50-54"], (55, 64): ["Y55-64"], (65, 99): ["Y_GE65"]}
    rows = []
    sh_report = {}
    for band, ages in parts.items():
        se = float(selfe[selfe.age.isin(ages)].value.sum())
        em = float(empl[empl.age.isin(ages)].value.sum())
        sh = se / em
        sh_report[_band_label(band)] = sh
        for r in REGIONS:
            rows += [(r, band[0], band[1], "EMPLOYEE", 1.0 - sh),
                     (r, band[0], band[1], "SOLE_PROPRIETOR", sh * (1.0 - q[r])),
                     (r, band[0], band[1], "BV_OWNER_MANAGER", sh * q[r])]
    b.diagnostics["self_employed_share_of_employment"] = sh_report
    return pd.DataFrame(rows, columns=["region", "age_min", "age_max", "primary_work_arrangement", "share"])


def build_tenure(b: Build) -> pd.DataFrame:
    t = b.es("ilc_lvho02", geo="BE", rskpovth="TOTAL", unit="PC")
    t = t.pivot_table(index="hhcomp", columns="tenure", values="value")
    groups = {"SINGLE_ADULT": ["A1"], "COUPLE": ["A2"], "COUPLE_WITH_CHILDREN": ["A2_DCH1", "A2_DCH2", "A2_DCH_GE3"],
              "SINGLE_PARENT": ["A1_DCH"], "OTHER_MULTI": ["A_GE3", "A_GE3_DCH"]}
    cols = {"RENTER": "RENT_MKT", "MORTGAGED_OWNER": "OWN_L", "OUTRIGHT_OWNER": "OWN_NL", "OTHER": "RENT_FR"}
    rows = []
    for r in REGIONS:
        for typ, comps in groups.items():
            vals = np.array([t.loc[comps, c].sum() for c in cols.values()])
            vals = vals / vals.sum()
            for k, s in zip(cols, vals, strict=True):
                rows.append((r, typ, k, float(s)))
    return pd.DataFrame(rows, columns=["region", "household_type", "housing_tenure", "share"])


def _age_num(code: str) -> int:
    return 0 if code == "Y_LT1" else int(code[1:])


def build_demography(b: Build, rate_year: int) -> pd.DataFrame:
    """Observed annual death probabilities (life table qx) and fertility rates, Belgium."""
    rows = []
    mort = b.es("demo_mlifetable", geo="BE", sex=["M", "F"], indic_de="PROBDEATH", time=str(rate_year))
    for sex, age, v in mort[["sex", "age", "value"]].itertuples(index=False):
        if age.startswith("Y") and age[1:].isdigit() or age == "Y_LT1":
            rows.append(("mortality_qx", sex, _age_num(age), float(v)))
    fert = b.es("demo_frate", geo="BE", unit="NR", agedef="COMPLET", time=str(rate_year))
    for age, v in fert[["age", "value"]].itertuples(index=False):
        if age[1:].isdigit():
            rows.append(("fertility_per_woman", "F", int(age[1:]), float(v)))
    df = pd.DataFrame(rows, columns=["kind", "sex", "age", "rate"]).sort_values(["kind", "sex", "age"]).reset_index(drop=True)
    return df


def build_validation_targets(b: Build) -> pd.DataFrame:
    """Published income-distribution indicators used only to VALIDATE simulated outputs."""
    yr = str(b.year)
    rows: list[tuple[str, str, float, str, str]] = []
    d3 = b.es("ilc_di03", geo="BE", sex="T", unit="EUR", time=yr)
    for st, nm in (("MEAN_EI", "mean_equivalised_income"), ("MED_EI", "median_equivalised_income")):
        for age, v in d3[d3.statinfo == st][["age", "value"]].itertuples(index=False):
            rows.append((nm, age, float(v), "EUR/year", yr))
    g = b.es("ilc_di12", geo="BE", time=yr)
    for age, v in g[["age", "value"]].itertuples(index=False):
        rows.append(("gini_disposable_income", age, float(v) / 100.0, "ratio", yr))
    a = b.es("ilc_li02", geo="BE", sex="T", unit="PC", rskpovth="B_60", statinfo="MED_EI", time=yr)
    for age, v in a[["age", "value"]].itertuples(index=False):
        rows.append(("at_risk_of_poverty_rate_60pct_median", age, float(v) / 100.0, "ratio", yr))
    s = b.es("ilc_di11", geo="BE", sex="T", unit="RAT", time=yr)
    for age, v in s[["age", "value"]].itertuples(index=False):
        rows.append(("s80_s20", age, float(v), "ratio", yr))
    q = b.es("ilc_di01", geo="BE", unit="EUR", statinfo="SHARE", time=yr)
    q2 = b.es("ilc_di01", geo="BE", unit="EUR", statinfo="TC", time=yr)
    for code, v in q[q.quant_inc.str.startswith("QU")][["quant_inc", "value"]].itertuples(index=False):
        rows.append(("income_share_quintile", code, float(v) / 100.0, "ratio", yr))
    for code, v in q2[q2.quant_inc.isin(["D1", "D9", "QU1", "QU4"])][["quant_inc", "value"]].itertuples(index=False):
        rows.append(("income_cutoff", code, float(v), "EUR/year", yr))
    return pd.DataFrame(rows, columns=["indicator", "group", "value", "unit", "income_reference_note"]).assign(
        income_reference_note=lambda d: "EU-SILC survey year " + d["income_reference_note"] + " (income refers to the previous calendar year)"
    )


# -------------------------------------------------------------------- output ---------------
def _src(b: Build, codes: list[str]) -> str:
    return "; ".join(sorted({s.get("dataset", s.get("view", "")) for s in b.sources if s.get("dataset", s.get("view")) in codes}))


def write_table(path: Path, header: dict[str, str], df: pd.DataFrame) -> None:
    lines = [f"# {k}: {v}" for k, v in header.items()]
    body = df.to_csv(index=False, float_format="%.8f", lineterminator="\n")
    path.write_text("\n".join(lines) + "\n" + body, encoding="utf-8", newline="\n")


def build_all(year: int, cache_dir: Path, out_dir: Path, *, offline: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    b = Build(cache_dir=cache_dir, year=year, offline=offline)
    pop = build_population(b)
    n0 = len(b.sources)
    hh = build_households(b, pop)
    edu = build_education(b)
    lab = build_labour(b, edu)
    arr = build_arrangement(b)
    ten = build_tenure(b)
    demo = build_demography(b, rate_year=year - 1)
    val = build_validation_targets(b)

    def hdr(dsid: str, status: str, unit: str, desc: str, codes: list[str]) -> dict[str, str]:
        return {"dataset_id": dsid, "synthetic_demo_only": "false", "status": status, "reference_year": str(year), "unit": unit,
                "source": f"Eurostat dissemination API / Statbel BeStat ({', '.join(codes)}); fetched {pd.Timestamp.now('UTC').date()}; see build_manifest.json",
                "description": desc}

    write_table(out_dir / "population_age_sex_region.csv",
                hdr("real_population", "observed", "persons", "Resident population on 1 January by region x sex x age band (NUTS-2 summed to regions)", ["demo_r_d2jan"]), pop)
    write_table(out_dir / "household_templates_region.csv",
                hdr("real_households", "derived", "households", "Household templates fitted (max-entropy) to regional adult/child totals; seed = national composition", ["lfst_hhnhtych", "lfst_r_lfsd2hh", "demo_r_d2jan"]), hh)
    write_table(out_dir / "education_by_age_region.csv",
                hdr("real_education", "observed", "share", "Highest education (ISCED 2011 3 groups) by region x age band, LFS", ["lfst_r_lfsd2pop"]), edu)
    write_table(out_dir / "labour_state_by_age_education_region.csv",
                hdr("real_labour_state", "derived", "share", "Work-state shares by region x age band x education; IPF of national seeds to regional margins", ["lfsa_pgaed", "lfsa_egaed", "lfsa_agaed", "lfst_r_lfe2emprt", "lfst_r_lfu3rt", "edat_lfse_18"]), lab)
    write_table(out_dir / "work_arrangement_by_age_region.csv",
                hdr("real_work_arrangement", "derived", "share", "Employee / sole proprietor / BV owner-manager shares among employed; BV split is a proxy", ["lfsa_esgan", "lfsa_egan", BESTAT_LEGAL_FORM_VIEW]), arr)
    write_table(out_dir / "tenure_by_household_type_region.csv",
                hdr("real_tenure", "derived", "share", "Housing tenure shares by household type (national, EU-SILC); no regional variation", ["ilc_lvho02"]), ten)

    write_table(out_dir / "demography_rates.csv",
                hdr("real_demography_rates", "observed", "annual probability / births per woman", f"Belgian life-table death probabilities and age-specific fertility, {year - 1} (latest complete year)", ["demo_mlifetable", "demo_frate"]) | {"reference_year": str(year - 1)}, demo)
    write_table(out_dir / "validation_targets.csv",
                hdr("real_validation_targets", "observed", "mixed", "Published EU-SILC income-distribution indicators for VALIDATION only (never used to fit the distribution)", ["ilc_di03", "ilc_di12", "ilc_li02", "ilc_di11", "ilc_di01"]), val)

    manifest = {
        "year": year,
        "built_utc": pd.Timestamp.now("UTC").isoformat(),
        "sources": b.sources,
        "assumptions": ASSUMPTIONS,
        "diagnostics": b.diagnostics,
        "n_sources_before_households": n0,
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
