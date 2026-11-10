"""Column specifications shared by the person / household / firm tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str  # int64 | Int64 | float64 | boolean | string
    nullable: bool = False
    domain: tuple[str, ...] | None = None
    description: str = ""


def _empty(dtype: str, n: int) -> pd.Series:
    if dtype == "float64":
        return pd.Series([float("nan")] * n, dtype="float64")
    return pd.Series([pd.NA] * n, dtype=dtype)


def coerce_frame(df: pd.DataFrame, columns: Sequence[ColumnSpec]) -> pd.DataFrame:
    """Return ``df`` restricted to the declared columns, in order, with declared dtypes.

    Missing nullable columns are created as all-null; a missing non-nullable column is an error.
    """
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col.name in df.columns:
            s = df[col.name]
        elif col.nullable:
            s = _empty(col.dtype, len(df)).set_axis(df.index)
        else:
            raise KeyError(f"required column '{col.name}' is missing")
        out[col.name] = s.astype(col.dtype)
    return out


def column_names(columns: Sequence[ColumnSpec]) -> list[str]:
    return [c.name for c in columns]
