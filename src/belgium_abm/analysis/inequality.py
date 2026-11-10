"""Weighted inequality metrics (spec section 5).

Conventions (documented, locked by tests):
  * Gini is computed from the weighted Lorenz curve with trapezoids. Negative values are allowed
    (net wealth); the result can then exceed 1. All-zero or empty input returns NaN.
  * Quantiles use Hazen plotting positions on the weighted empirical distribution.
  * P90/P10 is NaN when P10 <= 0.
"""

from __future__ import annotations

import numpy as np


def _clean(values: np.ndarray, weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    keep = w > 0
    return v[keep], w[keep]


def weighted_gini(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    v, w = _clean(values, weights)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    total_w, total_v = w.sum(), float((v * w).sum())
    if total_v == 0:
        return float("nan")
    cum_w = np.concatenate([[0.0], np.cumsum(w)]) / total_w
    cum_v = np.concatenate([[0.0], np.cumsum(v * w)]) / total_v
    area = np.trapezoid(cum_v, cum_w)
    return float(1.0 - 2.0 * area)


def weighted_quantile(values: np.ndarray, q: float | np.ndarray, weights: np.ndarray | None = None) -> float | np.ndarray:
    v, w = _clean(values, weights)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    pos = (np.cumsum(w) - 0.5 * w) / w.sum()
    out = np.interp(q, pos, v)
    return float(out) if np.ndim(q) == 0 else out


def p90_p10(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    p10 = float(weighted_quantile(values, 0.10, weights))
    p90 = float(weighted_quantile(values, 0.90, weights))
    return p90 / p10 if p10 > 0 else float("nan")


def top_share(values: np.ndarray, top_fraction: float, weights: np.ndarray | None = None) -> float:
    """Share of the total held by the richest ``top_fraction`` of the weighted population."""
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    v, w = _clean(values, weights)
    if v.size == 0 or float((v * w).sum()) == 0:
        return float("nan")
    order = np.argsort(-v, kind="stable")
    v, w = v[order], w[order]
    cum_w = np.cumsum(w) / w.sum()
    prev = np.concatenate([[0.0], cum_w[:-1]])
    take = np.clip((top_fraction - prev) / np.maximum(cum_w - prev, 1e-300), 0.0, 1.0)  # fractional last unit
    return float((v * w * take).sum() / (v * w).sum())


def poverty_rate(values: np.ndarray, weights: np.ndarray | None = None, line_fraction: float = 0.6) -> float:
    """Weighted share below ``line_fraction`` x weighted median."""
    v, w = _clean(values, weights)
    if v.size == 0:
        return float("nan")
    line = line_fraction * float(weighted_quantile(v, 0.5, w))
    return float(w[v < line].sum() / w.sum())


def equivalised_size(ages: np.ndarray, first_adult: float, additional: float, child: float, child_age_threshold: int) -> float:
    """Modified-OECD-style equivalence size of one household from its members' ages."""
    a = np.sort(np.asarray(ages))[::-1]
    if a.size == 0:
        return 0.0
    n_child = int((a < child_age_threshold).sum())
    n_other = a.size - n_child - 1 if a[0] >= child_age_threshold else a.size - n_child
    first = first_adult if a[0] >= child_age_threshold else 0.0
    return float(first + additional * max(n_other, 0) + child * n_child)
