"""Integer-cent fiscal ledger and a separate in-kind ledger (spec section 3.4).

Amounts are stored as integer cents so reconciliation is exact. An idempotency key prevents
the same benefit from being credited twice. The in-kind ledger records attributed service
value only and, by construction, has no effect on liquid cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal


class DuplicatePostingError(ValueError):
    """Raised when a posting with an already-used idempotency key is attempted."""


def eur_to_cents(amount_eur: float | int | str | Decimal) -> int:
    """Convert euros to integer cents (half-up on the decimal representation)."""
    d = Decimal(str(amount_eur)) if not isinstance(amount_eur, Decimal) else amount_eur
    return int((d * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class LedgerEntry:
    period: int
    payer_id: str
    recipient_id: str
    category: str
    amount_cents: int
    policy_year: int


@dataclass
class FiscalLedger:
    entries: list[LedgerEntry] = field(default_factory=list)
    _keys: set[str] = field(default_factory=set)

    def post(
        self,
        *,
        period: int,
        payer_id: str,
        recipient_id: str,
        category: str,
        amount_eur: float | int | str | Decimal,
        policy_year: int,
        key: str | None = None,
    ) -> LedgerEntry:
        cents = eur_to_cents(amount_eur)
        if cents < 0:
            raise ValueError("ledger amounts must be non-negative; swap payer and recipient instead")
        if key is not None:
            if key in self._keys:
                raise DuplicatePostingError(f"duplicate posting for key {key!r}")
            self._keys.add(key)
        entry = LedgerEntry(period, payer_id, recipient_id, category, cents, policy_year)
        self.entries.append(entry)
        return entry

    def post_benefit(
        self, *, period: int, payer_id: str, recipient_id: str, category: str, amount_eur: float | int | str | Decimal, policy_year: int
    ) -> LedgerEntry:
        """A benefit can be credited at most once per (period, recipient, category)."""
        return self.post(
            period=period,
            payer_id=payer_id,
            recipient_id=recipient_id,
            category=category,
            amount_eur=amount_eur,
            policy_year=policy_year,
            key=f"benefit|{period}|{recipient_id}|{category}",
        )

    def net_cents_by_entity(self) -> dict[str, int]:
        net: dict[str, int] = {}
        for e in self.entries:
            net[e.payer_id] = net.get(e.payer_id, 0) - e.amount_cents
            net[e.recipient_id] = net.get(e.recipient_id, 0) + e.amount_cents
        return net

    def total_cents_by_category(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.category] = out.get(e.category, 0) + e.amount_cents
        return out

    def reconcile(self) -> bool:
        """Every transfer debits one entity and credits another: the net over all entities is 0."""
        return sum(self.net_cents_by_entity().values()) == 0


@dataclass
class InKindLedger:
    """Attributed value of publicly financed services (schooling, healthcare...)."""

    entries: list[LedgerEntry] = field(default_factory=list)

    def attribute(self, *, period: int, provider_id: str, recipient_id: str, category: str, value_eur: float | int | str | Decimal, policy_year: int) -> LedgerEntry:
        entry = LedgerEntry(period, provider_id, recipient_id, category, eur_to_cents(value_eur), policy_year)
        self.entries.append(entry)
        return entry

    def value_cents_by_recipient(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.recipient_id] = out.get(e.recipient_id, 0) + e.amount_cents
        return out

    @staticmethod
    def cash_effect_cents(_recipient_id: str) -> int:
        """In-kind attribution never changes spendable cash."""
        return 0
