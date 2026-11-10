"""Load and validate the aggregate calibration tables.

Rules: no real data is ever substituted for a missing file; synthetic demo tables are refused
unless the configuration opts in; templates (empty files) are refused; every table is checked
for layout, domains, duplicates, share sums, band structure and mutual consistency.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from belgium_abm.config import ModelConfig
from belgium_abm.io.schemas import METADATA_KEYS, TABLE_SCHEMAS, TableSchema


class MissingInputsError(RuntimeError):
    """Required input files do not exist (real data not acquired)."""


class InputDataError(ValueError):
    """An input file exists but is malformed, inconsistent or not allowed."""


@dataclass(frozen=True)
class TableMeta:
    table: str
    path: str
    sha256: str
    dataset_id: str
    synthetic_demo_only: bool
    status: str
    reference_year: str
    unit: str
    source: str
    description: str


@dataclass
class TargetBundle:
    population: pd.DataFrame
    households: pd.DataFrame
    education: pd.DataFrame
    labour_state: pd.DataFrame
    work_arrangement: pd.DataFrame
    tenure: pd.DataFrame | None
    meta: dict[str, TableMeta]
    regions: list[str]
    max_age: int
    total_persons: float
    total_households: float
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def any_synthetic(self) -> bool:
        return any(m.synthetic_demo_only for m in self.meta.values())

    def table(self, name: str) -> pd.DataFrame | None:
        return getattr(self, name)


# ------------------------------------------------------------------ parsing -------------------
def parse_annotated_csv(text: str, name: str = "<csv>") -> tuple[dict[str, str], pd.DataFrame]:
    meta: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        body = lines[i][1:].strip()
        if ":" in body:
            k, v = body.split(":", 1)
            meta[k.strip()] = v.strip()
        i += 1
    body_text = "\n".join(lines[i:])
    if not body_text.strip():
        raise InputDataError(f"{name}: no data rows")
    df = pd.read_csv(io.StringIO(body_text), dtype=str, keep_default_na=False)
    return meta, df


def _convert(df: pd.DataFrame, schema: TableSchema, name: str) -> pd.DataFrame:
    expected = [c for c, _ in schema.columns]
    if list(df.columns) != expected:
        raise InputDataError(f"{name}: columns must be exactly {expected}, found {list(df.columns)}")
    out = pd.DataFrame(index=df.index)
    for col, kind in schema.columns:
        s = df[col].str.strip()
        if (s == "").any():
            raise InputDataError(f"{name}: column '{col}' has empty values")
        if kind == "str":
            out[col] = s
            continue
        try:
            num = pd.to_numeric(s, errors="raise")
        except (ValueError, TypeError) as exc:
            raise InputDataError(f"{name}: column '{col}' must be numeric ({exc})") from exc
        if not np.isfinite(num.to_numpy(dtype=float)).all():
            raise InputDataError(f"{name}: column '{col}' has non-finite values")
        if kind == "int":
            if (num != np.floor(num)).any():
                raise InputDataError(f"{name}: column '{col}' must be integer-valued")
            out[col] = num.astype("int64")
        else:
            out[col] = num.astype("float64")
    return out


def _check_bands(df: pd.DataFrame, group_cols: list[str], name: str) -> list[tuple[int, int]]:
    """Bands must be contiguous, non-overlapping, and identical across groups; return them."""
    structures: dict[tuple[object, ...], list[tuple[int, int]]] = {}
    for key, g in df.groupby(group_cols, sort=True):
        bands = sorted({(int(a), int(b)) for a, b in zip(g["age_min"], g["age_max"], strict=True)})
        for (a, b) in bands:
            if a > b or a < 0:
                raise InputDataError(f"{name}: invalid age band {a}-{b}")
        for (_, b1), (a2, _) in zip(bands, bands[1:], strict=False):
            if a2 != b1 + 1:
                raise InputDataError(f"{name}: age bands must be contiguous and non-overlapping (group {key}: ... {b1} -> {a2})")
        structures[key if isinstance(key, tuple) else (key,)] = bands
    first = next(iter(structures.values()))
    for key, bands in structures.items():
        if bands != first:
            raise InputDataError(f"{name}: age bands differ between groups (group {key})")
    return first


def _validate_table(df: pd.DataFrame, schema: TableSchema, regions: list[str], name: str) -> None:
    for col, allowed in schema.domains.items():
        bad = sorted(set(df[col]) - set(allowed))
        if bad:
            raise InputDataError(f"{name}: column '{col}' has values outside {list(allowed)}: {bad}")
    bad_regions = sorted(set(df["region"]) - set(regions))
    if bad_regions:
        raise InputDataError(f"{name}: unknown regions {bad_regions} (configured: {regions})")
    missing = sorted(set(regions) - set(df["region"]))
    if missing:
        raise InputDataError(f"{name}: no rows for configured regions {missing}")
    if df.duplicated(list(schema.key)).any():
        dup = df[df.duplicated(list(schema.key), keep=False)].head(3)
        raise InputDataError(f"{name}: duplicate keys {schema.key}, e.g. {dup.iloc[0].to_dict()}")
    v = df[schema.value_column]
    if (v < 0).any():
        raise InputDataError(f"{name}: negative {schema.value_column}")
    if schema.value_column == "share":
        if (v > 1 + 1e-9).any():
            raise InputDataError(f"{name}: shares above 1")
        sums = df.groupby(list(schema.group_by or ()))["share"].sum()
        off = sums[(sums - 1.0).abs() > 1e-6]
        if len(off):
            raise InputDataError(f"{name}: shares must sum to 1 within {schema.group_by}; e.g. {off.index[0]} sums to {off.iloc[0]:.6f}")
    if schema.value_column == "count" and float(v.sum()) <= 0:
        raise InputDataError(f"{name}: counts sum to zero")


def _adult_child_expectation(pop: pd.DataFrame, adult_age: int) -> pd.DataFrame:
    rows = []
    for region, g in pop.groupby("region"):
        adults = children = 0.0
        for lo, hi, cnt in g[["age_min", "age_max", "count"]].itertuples(index=False):
            width = hi - lo + 1
            n_adult_years = max(0, hi - max(lo, adult_age) + 1)
            adults += cnt * n_adult_years / width
            children += cnt * (width - n_adult_years) / width
        rows.append((region, adults, children))
    return pd.DataFrame(rows, columns=["region", "adults", "children"]).set_index("region")


def _check_household_type(hh: pd.DataFrame, name: str) -> None:
    for r in hh.itertuples(index=False):
        a, c, t = int(r.n_adults), int(r.n_children), r.household_type
        ok = {
            "SINGLE_ADULT": a == 1 and c == 0,
            "COUPLE": a == 2 and c == 0,
            "COUPLE_WITH_CHILDREN": a == 2 and c >= 1,
            "SINGLE_PARENT": a == 1 and c >= 1,
            "OTHER_MULTI": a >= 3,
        }[t]
        if not ok:
            raise InputDataError(f"{name}: household_type {t} is inconsistent with n_adults={a}, n_children={c}")


# ------------------------------------------------------------------ main entry ----------------
def load_targets(cfg: ModelConfig) -> TargetBundle:
    wanted: dict[str, Path] = {}
    for table, schema in TABLE_SCHEMAS.items():
        rel = getattr(cfg.inputs, table)
        if rel is None:
            if not schema.optional:
                raise MissingInputsError(f"no path configured for required table '{table}'")
            continue
        wanted[table] = cfg.resolve(rel)

    missing = [(t, p) for t, p in wanted.items() if not p.is_file()]
    if missing:
        lines = "\n".join(f"  - {t}: {p}" for t, p in missing)
        raise MissingInputsError(
            "required input files are missing (no real data is ever substituted):\n"
            f"{lines}\nSee `abm sources` for the datasets still to be acquired and `data/templates/` for the expected file layouts."
        )

    frames: dict[str, pd.DataFrame] = {}
    metas: dict[str, TableMeta] = {}
    for table, path in wanted.items():
        schema = TABLE_SCHEMAS[table]
        raw_bytes = path.read_bytes()
        meta, raw = parse_annotated_csv(raw_bytes.decode("utf-8"), str(path.name))
        absent = [k for k in ("dataset_id", "synthetic_demo_only", "status") if k not in meta]
        if absent:
            raise InputDataError(f"{path.name}: missing metadata header lines {absent} (expected keys: {list(METADATA_KEYS)})")
        flag = meta["synthetic_demo_only"].lower()
        if flag not in ("true", "false"):
            raise InputDataError(f"{path.name}: synthetic_demo_only must be true or false")
        synthetic = flag == "true"
        if meta["status"].lower() == "template":
            raise InputDataError(f"{path.name}: this file is an empty template, not data")
        if synthetic and not cfg.allow_synthetic_demo_data:
            raise InputDataError(f"{path.name}: is marked synthetic_demo_only but this configuration does not allow synthetic demo data")
        ref = meta.get("reference_year", "null")
        if not synthetic:
            if ref.lower() in ("", "null", "none", "n/a"):
                raise InputDataError(f"{path.name}: real data need a reference_year")
            if ref.isdigit() and int(ref) != cfg.baseline_year:
                raise InputDataError(f"{path.name}: reference_year {ref} differs from baseline_year {cfg.baseline_year} (one baseline year per run)")
        df = _convert(raw, schema, path.name)
        _validate_table(df, schema, cfg.regions, path.name)
        frames[table] = df
        metas[table] = TableMeta(
            table=table,
            path=str(path.relative_to(cfg.root)) if path.is_relative_to(cfg.root) else path.name,
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            dataset_id=meta["dataset_id"],
            synthetic_demo_only=synthetic,
            status=meta["status"],
            reference_year=ref,
            unit=meta.get("unit", ""),
            source=meta.get("source", ""),
            description=meta.get("description", ""),
        )

    flags = {m.synthetic_demo_only for m in metas.values()}
    if len(flags) > 1:
        raise InputDataError("synthetic demo tables and real tables must not be mixed in one run")

    pop, hh = frames["population"], frames["households"]
    pop_bands = _check_bands(pop, ["region", "sex_or_demographic_stratum"], "population")
    if pop_bands[0][0] != 0:
        raise InputDataError("population: age bands must start at age 0")
    max_age = pop_bands[-1][1]
    for table in ("education", "labour_state", "work_arrangement"):
        df = frames[table]
        group = ["region"] + (["education_level"] if table == "labour_state" else [])
        bands = _check_bands(df, group, table)
        if bands[0][0] > cfg.working_age_min or bands[-1][1] < max_age:
            raise InputDataError(f"{table}: age bands must cover working_age_min={cfg.working_age_min} to the oldest population age {max_age}")
    edu_levels = sorted(set(frames["education"]["education_level"]))
    lab = frames["labour_state"]
    expected_groups = {(r, a, e) for r in cfg.regions for a in sorted(set(lab["age_min"])) for e in edu_levels}
    found = {(r, a, e) for r, a, e in lab[["region", "age_min", "education_level"]].drop_duplicates().itertuples(index=False)}
    if expected_groups - found:
        raise InputDataError(f"labour_state: missing region x age band x education groups, e.g. {sorted(expected_groups - found)[0]}")
    if "tenure" in frames:
        t = frames["tenure"]
        types = set(hh["household_type"])
        exp = {(r, ty) for r in cfg.regions for ty in types}
        got = {(r, ty) for r, ty in t[["region", "household_type"]].drop_duplicates().itertuples(index=False)}
        if exp - got:
            raise InputDataError(f"tenure: missing region x household_type groups, e.g. {sorted(exp - got)[0]}")
    _check_household_type(hh, "households")

    exp = _adult_child_expectation(pop, cfg.adult_age)
    tol = cfg.generator.population_household_consistency_tol
    for region in cfg.regions:
        g = hh[hh["region"] == region]
        adults = float((g["count"] * g["n_adults"]).sum())
        children = float((g["count"] * g["n_children"]).sum())
        for label, got, want in (("adults", adults, exp.loc[region, "adults"]), ("children", children, exp.loc[region, "children"])):
            if abs(got - want) > tol * max(want, 1.0):
                raise InputDataError(
                    f"population vs households are inconsistent in {region}: households imply {got:,.0f} {label}, "
                    f"the population table implies {want:,.0f} (tolerance {tol:.1%}, adult_age={cfg.adult_age})"
                )

    return TargetBundle(
        population=pop,
        households=hh,
        education=frames["education"],
        labour_state=lab,
        work_arrangement=frames["work_arrangement"],
        tenure=frames.get("tenure"),
        meta=metas,
        regions=list(cfg.regions),
        max_age=int(max_age),
        total_persons=float(pop["count"].sum()),
        total_households=float(hh["count"].sum()),
    )
