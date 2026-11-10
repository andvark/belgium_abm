import unittest

from belgium_abm.engines import schedule
from belgium_abm.policies import (
    POLICY_COMPONENTS,
    DuplicatePostingError,
    FiscalLedger,
    InKindLedger,
    MissingPolicyDataError,
    PolicyConfig,
    eur_to_cents,
)


class LedgerTests(unittest.TestCase):
    def test_eur_to_cents_is_exact(self) -> None:
        self.assertEqual(eur_to_cents("0.1") + eur_to_cents("0.2"), eur_to_cents("0.3"))
        self.assertEqual(eur_to_cents(10.005), 1001)  # half-up on the decimal string
        self.assertEqual(eur_to_cents(0), 0)

    def test_postings_reconcile_to_the_cent(self) -> None:
        led = FiscalLedger()
        for i in range(1000):
            led.post(period=1, payer_id="gov", recipient_id=f"h{i % 7}", category="x", amount_eur="0.01", policy_year=2000)
        led.post(period=1, payer_id="h1", recipient_id="gov", category="tax", amount_eur="12.34", policy_year=2000)
        self.assertTrue(led.reconcile())
        self.assertEqual(sum(led.net_cents_by_entity().values()), 0)
        self.assertEqual(led.total_cents_by_category()["x"], 1000)

    def test_duplicate_benefit_is_refused(self) -> None:
        led = FiscalLedger()
        kw = dict(period=3, payer_id="gov", recipient_id="h1", category="child_benefit", amount_eur=100, policy_year=2000)
        led.post_benefit(**kw)
        with self.assertRaises(DuplicatePostingError):
            led.post_benefit(**kw)
        led.post_benefit(**{**kw, "period": 4})  # another period is fine
        self.assertEqual(len(led.entries), 2)

    def test_negative_amount_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FiscalLedger().post(period=1, payer_id="a", recipient_id="b", category="c", amount_eur=-1, policy_year=1)

    def test_in_kind_ledger_never_touches_cash(self) -> None:
        fiscal, kind = FiscalLedger(), InKindLedger()
        kind.attribute(period=1, provider_id="school", recipient_id="h1", category="schooling", value_eur=5000, policy_year=1)
        self.assertEqual(kind.value_cents_by_recipient()["h1"], 500000)
        self.assertEqual(InKindLedger.cash_effect_cents("h1"), 0)
        self.assertEqual(fiscal.net_cents_by_entity(), {})


class PolicyAndScheduleTests(unittest.TestCase):
    def test_policy_config_contains_no_statutory_values(self) -> None:
        pc = PolicyConfig()
        self.assertEqual(set(pc.missing_components()), set(POLICY_COMPONENTS))
        with self.assertRaises(MissingPolicyDataError):
            pc.require_complete()

    def test_birthday_convention_is_locked(self) -> None:
        self.assertEqual(schedule.BIRTHDAY_CONVENTION, "start_of_year")
        self.assertEqual(schedule.ANNUAL_SCHEDULE[0], "load_policy_and_rng")
        self.assertEqual(schedule.ANNUAL_SCHEDULE[1], "demography")  # ages increment before any other transition
        self.assertLess(schedule.ANNUAL_SCHEDULE.index("market_incomes"), schedule.ANNUAL_SCHEDULE.index("fiscal_cashflows"))
        self.assertEqual(schedule.IMPLEMENTED_STEPS, schedule.ANNUAL_SCHEDULE)


if __name__ == "__main__":
    unittest.main()
