"""Dynamic simulation -- intentionally NOT implemented (Phase 3)."""

from __future__ import annotations

from typing import Any


def run_simulation(*_args: Any, **_kwargs: Any) -> None:
    raise NotImplementedError(
        "The annual lifecycle simulation is not implemented. Phase 0/1 only builds and validates the "
        "initial synthetic population; behavioural rules and policy scenarios come after the "
        "population model is validated against real data."
    )
