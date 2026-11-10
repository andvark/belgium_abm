"""Deterministic random-number substreams, one per named module.

Every module asks for ``factory.stream("module-name")``. The stream depends only on the master
seed and the name, so adding or reordering modules never perturbs the draws of another module
(the basis for common random numbers across scenarios later on).
"""

from __future__ import annotations

import hashlib

import numpy as np


class RngFactory:
    def __init__(self, seed: int) -> None:
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")
        self.seed = int(seed)

    def stream(self, name: str) -> np.random.Generator:
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        key = tuple(int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4))
        ss = np.random.SeedSequence(self.seed, spawn_key=key)
        return np.random.Generator(np.random.PCG64(ss))
