"""Iterative proportional updating (IPU) of household weights.

``M`` is a (constraints x households) sparse matrix whose entry (c, h) is how many units of
constraint c (persons in a demographic cell, or 1 for the household's template) household h
contributes. IPU cycles over constraints, rescaling the weights of every household that
contributes to the constraint so that it is met exactly; at convergence all feasible
constraints hold simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class IPUResult:
    weights: np.ndarray
    iterations: int
    converged: bool
    max_rel_error: float
    unsupported: list[int]  # constraints with a positive target but no contributing household


def kish_effective_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    s2 = float((w**2).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def run_ipu(
    M: sp.csr_matrix,
    targets: np.ndarray,
    w0: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> IPUResult:
    M = sp.csr_matrix(M)
    n_c, n_h = M.shape
    targets = np.asarray(targets, dtype=float)
    if targets.shape != (n_c,):
        raise ValueError("targets must have one entry per constraint")
    if (targets < 0).any():
        raise ValueError("targets must be non-negative")
    w = np.ones(n_h) if w0 is None else np.asarray(w0, dtype=float).copy()

    rows = [(M.indices[M.indptr[c] : M.indptr[c + 1]], M.data[M.indptr[c] : M.indptr[c + 1]].astype(float)) for c in range(n_c)]
    unsupported = [c for c in range(n_c) if targets[c] > 0 and rows[c][0].size == 0]
    active = [c for c in range(n_c) if rows[c][0].size > 0]
    denom = np.maximum(targets, 1.0)

    max_err = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        for c in active:
            idx, vals = rows[c]
            cur = float(vals @ w[idx])
            if targets[c] == 0.0:
                w[idx] = 0.0
            elif cur > 0.0:
                w[idx] *= targets[c] / cur
        resid = np.abs(M @ w - targets) / denom
        max_err = float(resid[active].max()) if active else 0.0
        if max_err < tol:
            return IPUResult(w, it, True, max_err, unsupported)
    return IPUResult(w, it, False, max_err, unsupported)
