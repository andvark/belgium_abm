"""Reading and writing run outputs (CSV is canonical; Parquet only if pyarrow is available)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from belgium_abm.agents.columns import ColumnSpec, coerce_frame


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_table(df: pd.DataFrame, directory: Path, stem: str, *, parquet: bool) -> list[Path]:
    """Write ``stem.csv`` (always) and ``stem.parquet`` (when requested and possible)."""
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{stem}.csv"
    df.to_csv(csv_path, index=False, lineterminator="\n")
    written = [csv_path]
    if parquet and parquet_available():
        pq_path = directory / f"{stem}.parquet"
        df.to_parquet(pq_path, index=False)
        written.append(pq_path)
    return written


def read_table(path: Path, columns: Sequence[ColumnSpec]) -> pd.DataFrame:
    """Read a CSV written by :func:`write_table` back into the declared dtypes."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    out = pd.DataFrame(index=raw.index)
    for col in columns:
        s = raw[col.name]
        if col.dtype in ("bool", "boolean"):
            out[col.name] = s.map({"True": True, "False": False}).astype("boolean")
        elif col.dtype in ("int64", "Int64"):
            out[col.name] = pd.to_numeric(s).astype("Int64")
        elif col.dtype == "float64":
            out[col.name] = pd.to_numeric(s).astype("float64")
        else:
            out[col.name] = s.astype("string")
    return coerce_frame(out, columns)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
