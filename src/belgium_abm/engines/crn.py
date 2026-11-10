"""Common random numbers: uniform draws keyed by (seed, stream, year, entity id).

A draw depends only on those four values, never on how many other entities exist or in which
order they are processed, so a baseline run and a policy-scenario run that share a seed see the
same shock for the same person in the same year.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.special import ndtri

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLD = np.uint64(0x9E3779B97F4A7C15)


def _stream_key(name: str) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little"))


def _splitmix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + _GOLD
        x = (x ^ (x >> np.uint64(30))) * _M1
        x = (x ^ (x >> np.uint64(27))) * _M2
        return x ^ (x >> np.uint64(31))


def uniform(seed: int, stream: str, year: int, ids: np.ndarray) -> np.ndarray:
    """Uniform(0,1) draws, one per id (open interval)."""
    ids = np.asarray(ids).astype(np.uint64)
    with np.errstate(over="ignore"):
        k = _splitmix(np.array([np.uint64(seed) * _GOLD + _stream_key(stream) + np.uint64(year) * _M1], dtype=np.uint64))[0]
        h = _splitmix(ids * _M2 + k)
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def normal(seed: int, stream: str, year: int, ids: np.ndarray) -> np.ndarray:
    return ndtri(uniform(seed, stream, year, ids))
