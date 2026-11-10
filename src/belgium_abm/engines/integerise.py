"""Turn fractional household weights into a concrete integer selection of households."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from belgium_abm.engines.sampling import systematic_counts


def systematic_integerise(rng: np.random.Generator, weights: np.ndarray, total: int) -> np.ndarray:
    """Integer multiplicities summing to ``total``, each within 1 of its proportional share."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    return systematic_counts(rng, int(total), w)


def objective(counts: np.ndarray, targets: np.ndarray) -> float:
    """Chi-square-like distance  sum (x - t)^2 / max(t, 1)."""
    x = np.asarray(counts, dtype=float)
    t = np.asarray(targets, dtype=float)
    return float((((x - t) ** 2) / np.maximum(t, 1.0)).sum())


@dataclass
class SwapResult:
    selection: np.ndarray
    objective_before: float
    objective_after: float
    accepted: int


def refine_by_swaps(
    rng: np.random.Generator,
    selection: np.ndarray,
    template: np.ndarray,
    M: sp.spmatrix,
    targets: np.ndarray,
    n_proposals: int,
    proposal_weights: np.ndarray | None = None,
) -> SwapResult:
    """Hill-climb the selected households to lower :func:`objective`.

    ``selection`` lists candidate indices (with repeats); ``template[j]`` is candidate j's
    template. A proposal replaces one selected household by another candidate of the SAME
    template, so template counts never change. A swap is kept only if the objective drops.
    """
    selection = np.asarray(selection, dtype=np.int64).copy()
    template = np.asarray(template, dtype=np.int64)
    targets = np.asarray(targets, dtype=float)
    Mc = sp.csc_matrix(M)
    n_cand = Mc.shape[1]
    cols_idx = [Mc.indices[Mc.indptr[j] : Mc.indptr[j + 1]].tolist() for j in range(n_cand)]
    cols_val = [Mc.data[Mc.indptr[j] : Mc.indptr[j + 1]].tolist() for j in range(n_cand)]

    x = np.asarray(Mc[:, selection].sum(axis=1)).ravel() if selection.size else np.zeros(Mc.shape[0])
    before = objective(x, targets)
    if n_proposals <= 0 or selection.size == 0:
        return SwapResult(selection, before, before, 0)

    xs = x.tolist()
    ts = targets.tolist()
    denom = [max(t, 1.0) for t in ts]

    pw = np.ones(n_cand) if proposal_weights is None else np.asarray(proposal_weights, dtype=float).copy()
    pools: dict[int, tuple[list[int], list[float], float]] = {}
    for t in np.unique(template[np.unique(selection)]):
        members = np.flatnonzero(template == t)
        w = pw[members]
        if w.sum() <= 0:
            w = np.ones_like(w)
        keep = w > 0
        members, w = members[keep], w[keep]
        cum = np.cumsum(w)
        pools[int(t)] = (members.tolist(), cum.tolist(), float(cum[-1]))

    pos_draw = rng.integers(0, selection.size, n_proposals).tolist()
    u_draw = rng.random(n_proposals).tolist()
    sel = selection.tolist()
    tmpl = template.tolist()
    accepted = 0
    for i in range(n_proposals):
        pos = pos_draw[i]
        a = sel[pos]
        members, cum, total = pools[tmpl[a]]
        b = members[min(bisect_right(cum, u_draw[i] * total), len(members) - 1)]
        if b == a:
            continue
        diff: dict[int, float] = {}
        for c, v in zip(cols_idx[a], cols_val[a], strict=True):
            diff[c] = diff.get(c, 0.0) - v
        for c, v in zip(cols_idx[b], cols_val[b], strict=True):
            diff[c] = diff.get(c, 0.0) + v
        delta = 0.0
        for c, d in diff.items():
            if d:
                delta += ((xs[c] + d - ts[c]) ** 2 - (xs[c] - ts[c]) ** 2) / denom[c]
        if delta < -1e-12:
            for c, d in diff.items():
                xs[c] += d
            sel[pos] = b
            accepted += 1
    after = objective(np.asarray(xs), targets)
    return SwapResult(np.asarray(sel, dtype=np.int64), before, after, accepted)
