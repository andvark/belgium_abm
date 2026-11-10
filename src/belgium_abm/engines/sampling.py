"""Sampling / allocation primitives used by the population generator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def systematic_counts(rng: np.random.Generator, n: int, shares: np.ndarray) -> np.ndarray:
    """Allocate ``n`` units over categories with randomised systematic rounding.

    The counts sum to exactly ``n`` and each count is within 1 of ``n * share`` (unbiased in
    expectation), which removes the sampling noise a multinomial draw would add.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    s = np.asarray(shares, dtype=float)
    if s.ndim != 1 or s.size == 0 or (s < 0).any() or not np.isfinite(s).all():
        raise ValueError("shares must be a non-empty vector of non-negative finite numbers")
    total = s.sum()
    if total <= 0:
        raise ValueError("shares must have a positive sum")
    cum = np.cumsum(s / total) * n
    cum[-1] = float(n)  # remove floating-point drift so the counts sum to n exactly
    u = rng.random()
    edges = np.floor(cum + u).astype(np.int64)
    return np.diff(edges, prepend=0)


def allocate_labels(
    rng: np.random.Generator, n: int, shares: np.ndarray, labels: Sequence[str]
) -> np.ndarray:
    """Return ``n`` labels (shuffled) whose counts follow ``shares`` via systematic rounding."""
    if len(labels) != len(shares):
        raise ValueError("labels and shares must have the same length")
    counts = systematic_counts(rng, n, shares)
    out = np.repeat(np.asarray(labels, dtype=object), counts)
    return out[rng.permutation(n)]


def band_index(ages: np.ndarray, band_starts: np.ndarray) -> np.ndarray:
    """Index of the band (by ascending start) that contains each age; -1 below the first."""
    return np.searchsorted(band_starts, ages, side="right") - 1


def age_pmf_from_bands(bands: pd.DataFrame, max_age: int) -> np.ndarray:
    """Single-year age pmf from banded counts under the 'uniform within band' assumption.

    ``bands`` has columns ``age_min``, ``age_max``, ``count`` (already summed over sex).
    """
    pmf = np.zeros(max_age + 1, dtype=float)
    for lo, hi, cnt in bands[["age_min", "age_max", "count"]].itertuples(index=False):
        hi_c = min(int(hi), max_age)
        if hi_c >= lo:
            pmf[int(lo) : hi_c + 1] = float(cnt) / (int(hi) - int(lo) + 1)
    return pmf


def sample_ages(
    rng: np.random.Generator, pmf: np.ndarray, lo: int, hi: int, size: int
) -> np.ndarray:
    """Draw integer ages in ``[lo, hi]`` proportionally to ``pmf`` restricted to that range."""
    hi = min(hi, pmf.size - 1)
    if lo > hi:
        raise ValueError(f"empty age range [{lo}, {hi}]")
    p = pmf[lo : hi + 1]
    total = p.sum()
    if total <= 0:
        raise ValueError(f"population table gives no support for ages {lo}..{hi}")
    return rng.choice(np.arange(lo, hi + 1, dtype=np.int64), size=size, p=p / total)


def sample_ages_in_ranges(
    rng: np.random.Generator, pmf: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised inverse-CDF draw of one age per row from ``pmf`` restricted to ``[lo, hi]``.

    Returns ``(ages, ok)``; ``ok`` is False where the range is empty or carries no probability
    mass (the age is then meaningless).
    """
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.minimum(np.asarray(hi, dtype=np.int64), pmf.size - 1)
    cdf = np.concatenate([[0.0], np.cumsum(pmf)])
    valid = (lo <= hi) & (lo >= 0)
    lo_c = np.where(valid, lo, 0)
    hi_c = np.where(valid, hi, 0)
    mass = cdf[hi_c + 1] - cdf[lo_c]
    ok = valid & (mass > 0)
    u = rng.random(lo.shape[0])
    target = cdf[lo_c] + u * np.where(ok, mass, 1.0)
    ages = np.searchsorted(cdf, target, side="right") - 1
    ages = np.clip(ages, lo_c, hi_c)
    return ages.astype(np.int64), ok


def refine_group_counts(
    rng: np.random.Generator,
    sizes: np.ndarray,
    shares: np.ndarray,
    margin_maps: Sequence[np.ndarray],
    n_proposals: int,
) -> np.ndarray:
    """Integer category counts per group that track BOTH cells and margins.

    ``sizes`` (G,) are group sizes, ``shares`` (G, K) the published conditional shares and
    ``margin_maps`` a sequence of integer vectors (length G) assigning each group to a margin
    cell; the grand total over all groups is always included. Each group first gets
    systematically rounded counts (exact row sums). Hill-climbing then applies moves that
    preserve every group size -- shifting one unit between two categories inside a group, or
    a 2x2 exchange between two groups -- and keeps a move only if it lowers

        sum_cells (x - e)^2 / max(e, 1)  +  sum_margins (x - e)^2 / max(e, 1)

    where ``e`` is the expected count (group size x published share, aggregated for margins).
    Independent per-cell rounding lets errors accumulate over the 10-20 cells behind a
    margin; this keeps them near one unit. Deterministic given ``rng``.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    shares = np.asarray(shares, dtype=float)
    n_groups, n_cat = shares.shape
    expected = sizes[:, None] * shares
    counts = np.zeros((n_groups, n_cat), dtype=np.int64)
    for g in range(n_groups):
        if sizes[g] > 0:
            counts[g] = systematic_counts(rng, int(sizes[g]), shares[g])
    if n_proposals <= 0 or n_groups == 0 or n_cat < 2:
        return counts

    maps = [np.zeros(n_groups, dtype=np.int64), *[np.asarray(m, dtype=np.int64) for m in margin_maps]]
    exp_m: list[list[list[float]]] = []
    cur_m: list[list[list[int]]] = []
    for mp in maps:
        n_m = int(mp.max()) + 1
        e = np.zeros((n_m, n_cat))
        c = np.zeros((n_m, n_cat), dtype=np.int64)
        np.add.at(e, mp, expected)
        np.add.at(c, mp, counts)
        exp_m.append(e.tolist())
        cur_m.append(c.tolist())
    e_cell = expected.tolist()
    cell = counts.tolist()
    map_lists = [mp.tolist() for mp in maps]

    def delta(changes: list[tuple[int, int, int]]) -> float:
        total = 0.0
        for g, k, d in changes:
            e, x = e_cell[g][k], cell[g][k]
            total += ((x + d - e) ** 2 - (x - e) ** 2) / max(e, 1.0)
        acc: dict[tuple[int, int, int], int] = {}
        for g, k, d in changes:
            for j, mp in enumerate(map_lists):
                key = (j, mp[g], k)
                acc[key] = acc.get(key, 0) + d
        for (j, m, k), d in acc.items():
            if d:
                e, x = exp_m[j][m][k], cur_m[j][m][k]
                total += ((x + d - e) ** 2 - (x - e) ** 2) / max(e, 1.0)
        return total

    active = [g for g in range(n_groups) if sizes[g] > 0]
    if not active:
        return counts
    g1s = rng.integers(0, len(active), n_proposals).tolist()
    g2s = rng.integers(0, len(active), n_proposals).tolist()
    ks = rng.integers(0, n_cat, (n_proposals, 2)).tolist()
    pair = rng.random(n_proposals).tolist()
    for i in range(n_proposals):
        a, b = ks[i]
        if a == b:
            continue
        g1 = active[g1s[i]]
        if cell[g1][a] <= 0:
            continue
        if pair[i] < 0.5:
            changes = [(g1, a, -1), (g1, b, +1)]
        else:
            g2 = active[g2s[i]]
            if g2 == g1 or cell[g2][b] <= 0:
                continue
            changes = [(g1, a, -1), (g1, b, +1), (g2, a, +1), (g2, b, -1)]
        if delta(changes) < -1e-12:
            for g, k, d in changes:
                cell[g][k] += d
                for j, mp in enumerate(map_lists):
                    cur_m[j][mp[g]][k] += d
    return np.asarray(cell, dtype=np.int64)


def assign_grouped_labels(
    rng: np.random.Generator,
    out: np.ndarray,
    members: Sequence[np.ndarray],
    shares: np.ndarray,
    labels: Sequence[str],
    margin_maps: Sequence[np.ndarray],
    proposals_per_cell: int,
) -> np.ndarray:
    """Write category labels into ``out`` for every member of every group; return the counts."""
    sizes = np.array([len(m) for m in members], dtype=np.int64)
    shares = np.asarray(shares, dtype=float)
    counts = refine_group_counts(rng, sizes, shares, margin_maps, int(proposals_per_cell * shares.size))
    label_arr = np.asarray(labels, dtype=object)
    for g, idx in enumerate(members):
        if len(idx):
            out[idx] = np.repeat(label_arr, counts[g])[rng.permutation(len(idx))]
    return counts
