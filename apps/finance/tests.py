from datetime import date, timedelta

from django.core.exceptions import ValidationError
from django.db import IntegrityError, connection, transaction
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from apps.academics.models import AcademicYear, Grade, SchoolLevel
from apps.identity.models import Person, StaffProfile, StudentProfile

from .models import (
    Adjustment,
    Charge,
    Payment,
    PriceList,
    PricingRule,
    Refund,
    Wallet,
    WalletTransaction,
)
from .selectors import (
    adjustments_for_person,
    balance_for,
    charges_for_person,
    get_enabled_pricing_rule,
    get_price_list_by_name,
    get_wallet_for_person,
    ledger_balance_for,
    ledger_for_person,
    list_pricing_rules,
    list_wallets,
    payments_for_person,
    refunds_for_charge,
    refunds_for_person,
    transaction_history,
)
from .services import (
    charge,
    check_balance,
    close_wallet,
    create_charge,
    create_wallet,
    get_or_create_wallet_for_person,
    get_or_create_wallet_for_staff,
    get_or_create_wallet_for_student,
    record_adjustment,
    record_payment,
    record_refund,
    reactivate_wallet,
    recompute_balance,
    refund,
    resolve_price,
    suspend_wallet,
)
from .validators import (
    validate_adjustment_reason_required,
    validate_charge_amounts,
    validate_no_duplicate_charge_reference,
    validate_positive_amount,
    validate_refund_reason_required,
    validate_source_module_required,
    validate_wallet_status_transition,
)


class FinanceBaseData(TestCase):
    def setUp(self):
        self.person_student = Person.objects.create(
            code="P-S1", first_name="Stu", last_name="Dent"
        )
        self.student = StudentProfile.objects.create(
            person=self.person_student, code="S-1001"
        )
        self.person_staff = Person.objects.create(
            code="P-T1", first_name="Tea", last_name="Cher"
        )
        self.staff = StaffProfile.objects.create(person=self.person_staff, code="T-2001")
        self.year = AcademicYear.objects.create(
            name="2026-2027", code="2026-27",
            start_date=date(2026, 9, 1), end_date=date(2027, 6, 30),
            is_active=True,
        )
        self.level = SchoolLevel.objects.create(name="Primary", code="PRI", order=1)
        self.grade = Grade.objects.create(
            name="Grade 1", code="G1", level=self.level, order=1
        )


# ---------------------------------------------------------------------------
# Wallet creation helpers
# ---------------------------------------------------------------------------

class WalletCreationTests(FinanceBaseData):
    def test_create_wallet(self):
        wallet = create_wallet(person=self.person_student)
        self.assertEqual(wallet.balance_iqd, 0)
        self.assertEqual(wallet.status, Wallet.Status.ACTIVE)
        self.assertEqual(wallet.currency, "IQD")
        self.assertEqual(wallet.credit_limit_iqd, 0)

    def test_create_wallet_uppercases_currency(self):
        wallet = create_wallet(person=self.person_staff, currency="usd")
        self.assertEqual(wallet.currency, "USD")

    def test_get_or_create_wallet_for_person_creates(self):
        wallet, created = get_or_create_wallet_for_person(person=self.person_student)
        self.assertTrue(created)
        self.assertEqual(wallet.person, self.person_student)

    def test_get_or_create_wallet_for_person_returns_existing(self):
        first, _ = get_or_create_wallet_for_person(person=self.person_student)
        second, created = get_or_create_wallet_for_person(person=self.person_student)
        self.assertFalse(created)
        self.assertEqual(first.pk, second.pk)

    def test_get_or_create_wallet_for_student(self):
        wallet, created = get_or_create_wallet_for_student(student_profile=self.student)
        self.assertTrue(created)
        self.assertEqual(wallet.person, self.person_student)

    def test_get_or_create_wallet_for_staff(self):
        wallet, created = get_or_create_wallet_for_staff(staff_profile=self.staff)
        self.assertTrue(created)
        self.assertEqual(wallet.person, self.person_staff)

    def test_duplicate_wallet_rejected(self):
        create_wallet(person=self.person_student)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                create_wallet(person=self.person_student)

    def test_selectors_get_wallet_for_person(self):
        wallet = create_wallet(person=self.person_student)
        self.assertEqual(get_wallet_for_person(self.person_student), wallet)

    def test_selectors_list_wallets_search(self):
        create_wallet(person=self.person_student)
        self.assertEqual(list_wallets(search="Stu").count(), 1)
        self.assertEqual(list_wallets(search="Nope").count(), 0)


# ---------------------------------------------------------------------------
# WalletTransaction creation / ledger integrity
# ---------------------------------------------------------------------------

class WalletTransactionTests(FinanceBaseData):
    def test_topup_snapshots_balance_before_after(self):
        wallet = create_wallet(person=self.person_student)
        payment, tx = record_payment(
            person=self.person_student, amount_iqd=5000, method=Payment.Method.CASH
        )
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.TOPUP)
        self.assertEqual(tx.amount_iqd, 5000)
        self.assertEqual(tx.balance_before_iqd, 0)
        self.assertEqual(tx.balance_after_iqd, 5000)
        wallet.refresh_from_db()
        self.assertEqual(wallet.balance_iqd, 5000)

    def test_debit_negative_amount(self):
        wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        charge, tx = create_charge(
            person=self.person_student,
            product_code="meal_lunch",
            price_base_iqd=3000,
            source_module="meal",
        )
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.DEBIT)
        self.assertEqual(tx.amount_iqd, -3000)
        self.assertEqual(charge.status, Charge.Status.SETTLED)
        wallet.refresh_from_db()
        self.assertEqual(wallet.balance_iqd, 7000)

    def test_transaction_person_must_match_wallet(self):
        wallet = create_wallet(person=self.person_student)
        other = Person.objects.create(code="P-X1", first_name="X", last_name="Y")
        tx = WalletTransaction(
            wallet=wallet, person=other, tx_type=WalletTransaction.TxType.TOPUP,
            amount_iqd=100, source_module="manual_topup",
        )
        with self.assertRaises(ValidationError):
            tx.full_clean()

    def test_unpaid_amount_must_be_zero(self):
        wallet = create_wallet(person=self.person_student)
        with self.assertRaises(ValidationError):
            # Bypass via low-level service call: amount nonzero + UNPAID
            from .services import create_wallet_transaction
            create_wallet_transaction(
                wallet=wallet,
                tx_type=WalletTransaction.TxType.UNPAID,
                amount_iqd=5,
                source_module="meal",
            )


# ---------------------------------------------------------------------------
# Charges
# ---------------------------------------------------------------------------

class ChargeTests(FinanceBaseData):
    def setUp(self):
        super().setUp()
        self.wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)

    def test_settled_charge(self):
        charge, tx = create_charge(
            person=self.person_student,
            product_code="meal_lunch",
            price_base_iqd=2500,
            source_module="meal",
            reference_type="LunchServiceEvent",
            reference_id=1,
            academic_year=self.year,
        )
        self.assertEqual(charge.status, Charge.Status.SETTLED)
        self.assertEqual(charge.final_charge_iqd, 2500)
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.DEBIT)
        self.assertEqual(tx.reference_type, "Charge")
        self.assertEqual(tx.reference_id, charge.pk)
        self.wallet.refresh_from_db()
        self.assertEqual(self.wallet.balance_iqd, 7500)

    def test_charge_with_discount_snapshot(self):
        charge, tx = create_charge(
            person=self.person_student,
            product_code="tuition_term1",
            price_base_iqd=10000,
            discount_iqd=2000,
            discount_reason_code="scholarship",
            source_module="tuition",
        )
        self.assertEqual(charge.final_charge_iqd, 8000)
        self.assertEqual(charge.discount_iqd, 2000)
        self.assertEqual(tx.amount_iqd, -8000)
        self.wallet.refresh_from_db()
        self.assertEqual(self.wallet.balance_iqd, 2000)

    def test_charge_discount_exceeds_price_rejected(self):
        with self.assertRaises(ValidationError):
            create_charge(
                person=self.person_student,
                product_code="x",
                price_base_iqd=1000,
                discount_iqd=2000,
                source_module="meal",
            )

    def test_charge_zero_amount_settles_with_unpaid_tx(self):
        # A 0-cost charge settles with an UNPAID (zero-impact) ledger row.
        charge, tx = create_charge(
            person=self.person_student,
            product_code="free_event",
            price_base_iqd=0,
            source_module="events",
        )
        self.assertEqual(charge.status, Charge.Status.SETTLED)
        self.assertEqual(tx.amount_iqd, 0)
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.UNPAID)
        self.wallet.refresh_from_db()
        self.assertEqual(self.wallet.balance_iqd, 10000)

    def test_charge_insufficient_funds_rejected(self):
        with self.assertRaises(ValidationError):
            create_charge(
                person=self.person_student,
                product_code="big",
                price_base_iqd=100000,
                source_module="meal",
            )

    def test_charge_insufficient_funds_unpaid_allowed(self):
        charge, tx = create_charge(
            person=self.person_student,
            product_code="big",
            price_base_iqd=100000,
            source_module="meal",
            allow_unpaid=True,
        )
        self.assertEqual(charge.status, Charge.Status.UNPAID)
        self.assertEqual(tx.amount_iqd, 0)
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.UNPAID)
        self.wallet.refresh_from_db()
        self.assertEqual(self.wallet.balance_iqd, 10000)

    def test_duplicate_charge_reference_rejected(self):
        create_charge(
            person=self.person_student,
            product_code="meal_lunch",
            price_base_iqd=1000,
            source_module="meal",
            reference_type="LunchServiceEvent",
            reference_id=42,
        )
        with self.assertRaises(ValidationError):
            create_charge(
                person=self.person_student,
                product_code="meal_lunch",
                price_base_iqd=1000,
                source_module="meal",
                reference_type="LunchServiceEvent",
                reference_id=42,
            )

    def test_charge_with_credit_limit_uses_credit(self):
        # Drain wallet, then charge within credit limit.
        record_adjustment(
            person=self.person_student, amount_iqd=-10000, reason_code="drain",
        )
        # Now balance is 0; allow a -2500 charge with a credit limit.
        self.wallet.refresh_from_db()
        self.wallet.credit_limit_iqd = 5000
        self.wallet.save(update_fields=["credit_limit_iqd"])
        charge, tx = create_charge(
            person=self.person_student,
            product_code="meal_lunch",
            price_base_iqd=2500,
            source_module="meal",
        )
        self.assertEqual(charge.status, Charge.Status.SETTLED)
        self.wallet.refresh_from_db()
        self.assertEqual(self.wallet.balance_iqd, -2500)

    def test_charge_beyond_credit_limit_rejected(self):
        # Drain the wallet to zero first so the credit limit is the binding
        # constraint.
        record_adjustment(
            person=self.person_student, amount_iqd=-10000, reason_code="drain",
        )
        self.wallet.refresh_from_db()
        self.wallet.credit_limit_iqd = 1000
        self.wallet.save(update_fields=["credit_limit_iqd"])
        with self.assertRaises(ValidationError):
            create_charge(
                person=self.person_student,
                product_code="meal_lunch",
                price_base_iqd=2000,
                source_module="meal",
            )


# ---------------------------------------------------------------------------
# Payments
# ---------------------------------------------------------------------------

class PaymentTests(FinanceBaseData):
    def test_record_payment_increases_balance(self):
        payment, tx = record_payment(
            person=self.person_student, amount_iqd=5000, method=Payment.Method.CASH,
        )
        self.assertEqual(payment.amount_iqd, 5000)
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.TOPUP)
        wallet = get_wallet_for_person(self.person_student)
        self.assertEqual(wallet.balance_iqd, 5000)

    def test_negative_payment_rejected(self):
        with self.assertRaises(ValidationError):
            record_payment(person=self.person_student, amount_iqd=-100)

    def test_zero_payment_rejected(self):
        with self.assertRaises(ValidationError):
            record_payment(person=self.person_student, amount_iqd=0)

    def test_duplicate_card_reference_rejected(self):
        record_payment(
            person=self.person_student, amount_iqd=1000,
            method=Payment.Method.CARD, reference="RC-1",
        )
        with self.assertRaises(ValidationError):
            record_payment(
                person=self.person_student, amount_iqd=1000,
                method=Payment.Method.CARD, reference="RC-1",
            )

    def test_cash_reference_not_deduped(self):
        # Cash payments may share a blank reference; they are not deduped.
        record_payment(person=self.person_student, amount_iqd=1000, method=Payment.Method.CASH)
        record_payment(person=self.person_student, amount_iqd=1000, method=Payment.Method.CASH)
        wallet = get_wallet_for_person(self.person_student)
        self.assertEqual(wallet.balance_iqd, 2000)


# ---------------------------------------------------------------------------
# Refunds
# ---------------------------------------------------------------------------

class RefundTests(FinanceBaseData):
    def setUp(self):
        super().setUp()
        self.wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        self.charge, _ = create_charge(
            person=self.person_student,
            product_code="meal_lunch",
            price_base_iqd=4000,
            source_module="meal",
        )

    def test_full_refund_reverses_charge(self):
        refund, tx = record_refund(
            original_charge=self.charge,
            amount_iqd=4000,
            reason_code="error",
            reason_notes="wrong charge",
        )
        self.assertEqual(refund.amount_iqd, 4000)
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.REFUND)
        self.assertTrue(tx.is_reversal)
        self.assertEqual(tx.reverses, self.charge.settled_transaction)
        self.charge.refresh_from_db()
        self.assertEqual(self.charge.status, Charge.Status.REVERSED)
        self.wallet.refresh_from_db()
        self.assertEqual(self.wallet.balance_iqd, 10000)

    def test_partial_refund_keeps_charge_settled(self):
        record_refund(original_charge=self.charge, amount_iqd=1000, reason_code="partial")
        self.charge.refresh_from_db()
        self.assertEqual(self.charge.status, Charge.Status.SETTLED)
        self.wallet.refresh_from_db()
        # 10000 - 4000 (charge) + 1000 (refund) = 7000
        self.assertEqual(self.wallet.balance_iqd, 7000)

    def test_refund_exceeding_charge_rejected(self):
        with self.assertRaises(ValidationError):
            record_refund(original_charge=self.charge, amount_iqd=5000, reason_code="x")

    def test_refund_exceeding_total_rejected(self):
        record_refund(original_charge=self.charge, amount_iqd=3000, reason_code="r1")
        with self.assertRaises(ValidationError):
            record_refund(original_charge=self.charge, amount_iqd=2000, reason_code="r2")

    def test_refund_without_reason_rejected(self):
        with self.assertRaises(ValidationError):
            record_refund(original_charge=self.charge, amount_iqd=1000, reason_code="")

    def test_refund_on_voided_charge_rejected(self):
        # Manually void the charge to make it non-refundable.
        self.charge.status = Charge.Status.VOIDED
        self.charge.save(update_fields=["status"])
        with self.assertRaises(ValidationError):
            record_refund(original_charge=self.charge, amount_iqd=100, reason_code="x")


# ---------------------------------------------------------------------------
# Adjustments
# ---------------------------------------------------------------------------

class AdjustmentTests(FinanceBaseData):
    def test_credit_adjustment(self):
        create_wallet(person=self.person_student)
        adj, tx = record_adjustment(
            person=self.person_student, amount_iqd=1500, reason_code="goodwill",
        )
        self.assertEqual(tx.amount_iqd, 1500)
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.ADJUSTMENT)
        wallet = get_wallet_for_person(self.person_student)
        self.assertEqual(wallet.balance_iqd, 1500)

    def test_debit_adjustment_subject_to_credit_limit(self):
        wallet = create_wallet(person=self.person_student, credit_limit_iqd=1000)
        adj, tx = record_adjustment(
            person=self.person_student, amount_iqd=-500, reason_code="correction",
        )
        self.assertEqual(tx.amount_iqd, -500)
        wallet.refresh_from_db()
        self.assertEqual(wallet.balance_iqd, -500)

    def test_debit_adjustment_beyond_credit_rejected(self):
        create_wallet(person=self.person_student, credit_limit_iqd=100)
        with self.assertRaises(ValidationError):
            record_adjustment(
                person=self.person_student, amount_iqd=-500, reason_code="correction",
            )

    def test_zero_adjustment_rejected(self):
        create_wallet(person=self.person_student)
        with self.assertRaises(ValidationError):
            record_adjustment(
                person=self.person_student, amount_iqd=0, reason_code="zero",
            )

    def test_adjustment_reason_required(self):
        create_wallet(person=self.person_student)
        with self.assertRaises(ValidationError):
            record_adjustment(person=self.person_student, amount_iqd=100, reason_code="")


# ---------------------------------------------------------------------------
# Ledger integrity / cached balance
# ---------------------------------------------------------------------------

class LedgerIntegrityTests(FinanceBaseData):
    def test_cached_balance_matches_ledger_after_ops(self):
        create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        create_charge(
            person=self.person_student, product_code="m", price_base_iqd=3000,
            source_module="meal",
        )
        record_adjustment(person=self.person_student, amount_iqd=-500, reason_code="x")
        wallet = get_wallet_for_person(self.person_student)
        self.assertEqual(wallet.balance_iqd, ledger_balance_for(person=self.person_student))

    def test_recompute_balance_no_drift(self):
        create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=1000)
        wallet = get_wallet_for_person(self.person_student)
        ledger, cached, drift = recompute_balance(wallet)
        self.assertFalse(drift)
        self.assertEqual(ledger, 1000)
        self.assertEqual(cached, 1000)

    def test_recompute_balance_fixes_drift(self):
        create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=1000)
        wallet = get_wallet_for_person(self.person_student)
        # Manually corrupt the cache to simulate drift.
        Wallet.objects.filter(pk=wallet.pk).update(balance_iqd=999)
        wallet.refresh_from_db()
        ledger, cached, drift = recompute_balance(wallet)
        self.assertTrue(drift)
        self.assertEqual(ledger, 1000)
        self.assertEqual(cached, 999)
        wallet.refresh_from_db()
        self.assertEqual(wallet.balance_iqd, 1000)

    def test_reversal_chain(self):
        create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        charge, original_tx = create_charge(
            person=self.person_student, product_code="m", price_base_iqd=2000,
            source_module="meal",
        )
        refund, rev_tx = record_refund(
            original_charge=charge, amount_iqd=2000, reason_code="err",
        )
        self.assertEqual(rev_tx.reverses_id, original_tx.pk)
        self.assertTrue(rev_tx.is_reversal)

    def _assert_no_update_on_wallet_transaction(self, captured):
        """Helper: no UPDATE statement targets finance_wallettransaction."""
        for entry in captured.captured_queries:
            sql = entry["sql"]
            normalized = " ".join(sql.split())
            if "update" in normalized.lower() and "finance_wallettransaction" in normalized.lower():
                raise AssertionError(
                    "WalletTransaction row was UPDATEd — ledger immutability "
                    f"violated by: {normalized}"
                )

    def test_payment_does_not_update_ledger_row(self):
        create_wallet(person=self.person_student)
        with CaptureQueriesContext(connection) as captured:
            record_payment(
                person=self.person_student, amount_iqd=1000,
                method=Payment.Method.CASH,
            )
        self._assert_no_update_on_wallet_transaction(captured)

    def test_refund_does_not_update_ledger_row(self):
        create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        charge, _ = create_charge(
            person=self.person_student, product_code="m", price_base_iqd=2000,
            source_module="meal",
        )
        with CaptureQueriesContext(connection) as captured:
            record_refund(original_charge=charge, amount_iqd=2000, reason_code="err")
        self._assert_no_update_on_wallet_transaction(captured)

    def test_adjustment_does_not_update_ledger_row(self):
        create_wallet(person=self.person_student)
        with CaptureQueriesContext(connection) as captured:
            record_adjustment(
                person=self.person_student, amount_iqd=500, reason_code="x",
            )
        self._assert_no_update_on_wallet_transaction(captured)


# ---------------------------------------------------------------------------
# Wallet lifecycle
# ---------------------------------------------------------------------------

class WalletLifecycleTests(FinanceBaseData):
    def test_suspend_then_reactivate(self):
        wallet = create_wallet(person=self.person_student)
        suspend_wallet(wallet)
        wallet.refresh_from_db()
        self.assertEqual(wallet.status, Wallet.Status.SUSPENDED)
        reactivate_wallet(wallet)
        wallet.refresh_from_db()
        self.assertEqual(wallet.status, Wallet.Status.ACTIVE)

    def test_invalid_transition_active_to_active(self):
        wallet = create_wallet(person=self.person_student)
        with self.assertRaises(ValidationError):
            validate_wallet_status_transition(
                current_status=Wallet.Status.ACTIVE, new_status=Wallet.Status.ACTIVE
            )

    def test_closed_is_terminal(self):
        wallet = create_wallet(person=self.person_student)
        close_wallet(wallet)
        wallet.refresh_from_db()
        self.assertEqual(wallet.status, Wallet.Status.CLOSED)
        with self.assertRaises(ValidationError):
            reactivate_wallet(wallet)

    def test_charge_on_suspended_wallet_rejected(self):
        wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        suspend_wallet(wallet)
        with self.assertRaises(ValidationError):
            create_charge(
                person=self.person_student, product_code="m", price_base_iqd=1000,
                source_module="meal",
            )

    def test_payment_on_closed_wallet_rejected(self):
        wallet = create_wallet(person=self.person_student)
        close_wallet(wallet)
        with self.assertRaises(ValidationError):
            record_payment(person=self.person_student, amount_iqd=1000)

    def test_close_wallet_zeroes_balance_via_adjustment(self):
        wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=5000)
        create_charge(
            person=self.person_student, product_code="m", price_base_iqd=2000,
            source_module="meal",
        )
        # Balance is now 3000. Closing should zero it via an adjustment.
        close_wallet(wallet)
        wallet.refresh_from_db()
        self.assertEqual(wallet.status, Wallet.Status.CLOSED)
        self.assertEqual(wallet.balance_iqd, 0)
        # The ledger must still contain all history.
        self.assertEqual(wallet.transactions.count(), 3)  # topup, debit, closing adjustment


# ---------------------------------------------------------------------------
# PriceList / PricingRule
# ---------------------------------------------------------------------------

class PricingTests(FinanceBaseData):
    def setUp(self):
        super().setUp()
        self.price_list = PriceList.objects.create(name="Default", currency="IQD")

    def test_pricelist_unique_name(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                PriceList.objects.create(name="Default")

    def test_pricelist_currency_uppercase_validation(self):
        pl = PriceList(name="Lower", currency="iqd")
        with self.assertRaises(ValidationError):
            pl.full_clean()

    def test_pricingrule_creation(self):
        rule = PricingRule.objects.create(
            price_list=self.price_list, product_code="meal_lunch", price_iqd=2500,
        )
        self.assertEqual(rule.price_iqd, 2500)
        self.assertTrue(rule.is_enabled)

    def test_pricingrule_unique_product_per_list(self):
        PricingRule.objects.create(
            price_list=self.price_list, product_code="meal_lunch", price_iqd=2500,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                PricingRule.objects.create(
                    price_list=self.price_list, product_code="meal_lunch", price_iqd=3000,
                )

    def test_pricingrule_negative_price_rejected(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                PricingRule.objects.create(
                    price_list=self.price_list, product_code="x", price_iqd=-1,
                )

    def test_resolve_price_no_rule_returns_none(self):
        self.assertIsNone(resolve_price(product_code="nope", price_list=self.price_list))

    def test_resolve_price_unscoped_rule(self):
        PricingRule.objects.create(
            price_list=self.price_list, product_code="meal_lunch", price_iqd=2500,
        )
        self.assertEqual(resolve_price(product_code="meal_lunch", price_list=self.price_list), 2500)

    def test_resolve_price_grade_scope_is_descriptive(self):
        # The model enforces one rule per (price_list, product_code); grade
        # is descriptive metadata on the rule, not a disambiguator.
        PricingRule.objects.create(
            price_list=self.price_list, product_code="tuition",
            price_iqd=8000, grade=self.grade,
        )
        self.assertEqual(resolve_price(product_code="tuition", price_list=self.price_list, grade=self.grade), 8000)
        self.assertEqual(resolve_price(product_code="tuition", price_list=self.price_list), 8000)

    def test_resolve_price_period_key_is_descriptive(self):
        PricingRule.objects.create(
            price_list=self.price_list, product_code="bus",
            price_iqd=900, period_key="term1",
        )
        self.assertEqual(
            resolve_price(product_code="bus", price_list=self.price_list, period_key="term1"),
            900,
        )

    def test_get_enabled_pricing_rule_ignores_disabled(self):
        PricingRule.objects.create(
            price_list=self.price_list, product_code="x", price_iqd=100, is_enabled=False,
        )
        self.assertIsNone(get_enabled_pricing_rule(product_code="x", price_list=self.price_list))

    def test_selectors_list_pricing_rules(self):
        PricingRule.objects.create(price_list=self.price_list, product_code="a", price_iqd=1)
        PricingRule.objects.create(price_list=self.price_list, product_code="b", price_iqd=2)
        self.assertEqual(list_pricing_rules(price_list=self.price_list).count(), 2)
        self.assertEqual(list_pricing_rules(price_list=self.price_list, product_code="a").count(), 1)

    def test_get_price_list_by_name(self):
        self.assertEqual(get_price_list_by_name("Default"), self.price_list)


# ---------------------------------------------------------------------------
# Database-level constraints
# ---------------------------------------------------------------------------

class DatabaseConstraintTests(FinanceBaseData):
    def test_wallet_negative_credit_limit_rejected(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Wallet.objects.create(person=self.person_student, credit_limit_iqd=-1)

    def test_wallet_balance_below_credit_limit_rejected(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Wallet.objects.create(
                    person=self.person_student, balance_iqd=-1000, credit_limit_iqd=500,
                )

    def test_charge_discount_greater_than_price_rejected_at_db(self):
        wallet = create_wallet(person=self.person_student)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Charge.objects.create(
                    wallet=wallet, person=self.person_student,
                    product_code="x", source_module="meal",
                    price_base_iqd=1000, discount_iqd=2000, final_charge_iqd=0,
                )

    def test_charge_final_not_price_minus_discount_rejected(self):
        wallet = create_wallet(person=self.person_student)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Charge.objects.create(
                    wallet=wallet, person=self.person_student,
                    product_code="x", source_module="meal",
                    price_base_iqd=1000, discount_iqd=0, final_charge_iqd=500,
                )

    def test_payment_zero_amount_rejected_at_db(self):
        wallet = create_wallet(person=self.person_student)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Payment.objects.create(
                    wallet=wallet, person=self.person_student, amount_iqd=0,
                    received_at=timezone.now(),
                )

    def test_refund_zero_amount_rejected_at_db(self):
        wallet = create_wallet(person=self.person_student)
        charge = Charge.objects.create(
            wallet=wallet, person=self.person_student, product_code="x",
            source_module="meal", price_base_iqd=1000, final_charge_iqd=1000,
            status=Charge.Status.SETTLED,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Refund.objects.create(
                    wallet=wallet, person=self.person_student,
                    original_charge=charge, amount_iqd=0, refunded_at=timezone.now(),
                )

    def test_adjustment_zero_amount_rejected_at_db(self):
        wallet = create_wallet(person=self.person_student)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Adjustment.objects.create(
                    wallet=wallet, person=self.person_student, amount_iqd=0,
                    reason_code="x",
                )

    def test_pricing_rule_negative_price_rejected_at_db(self):
        pl = PriceList.objects.create(name="P2")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                PricingRule.objects.create(price_list=pl, product_code="x", price_iqd=-5)


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

class SelectorTests(FinanceBaseData):
    def setUp(self):
        super().setUp()
        create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        self.charge, _ = create_charge(
            person=self.person_student, product_code="meal_lunch",
            price_base_iqd=3000, source_module="meal",
        )

    def test_balance_for(self):
        self.assertEqual(balance_for(person=self.person_student), 7000)

    def test_ledger_balance_for_matches(self):
        self.assertEqual(ledger_balance_for(person=self.person_student), balance_for(person=self.person_student))

    def test_transaction_history(self):
        qs = transaction_history(person=self.person_student)
        self.assertEqual(qs.count(), 2)
        self.assertEqual(qs.first().tx_type, WalletTransaction.TxType.DEBIT)

    def test_ledger_for_person(self):
        self.assertEqual(ledger_for_person(person=self.person_student).count(), 2)

    def test_charges_for_person(self):
        self.assertEqual(charges_for_person(person=self.person_student).count(), 1)
        self.assertEqual(
            charges_for_person(person=self.person_student, product_code="meal_lunch").count(), 1
        )
        self.assertEqual(
            charges_for_person(person=self.person_student, product_code="other").count(), 0
        )

    def test_refunds_for_charge_empty(self):
        self.assertEqual(refunds_for_charge(charge=self.charge).count(), 0)

    def test_payments_for_person(self):
        self.assertEqual(payments_for_person(person=self.person_student).count(), 1)

    def test_refunds_for_person(self):
        self.assertEqual(refunds_for_person(person=self.person_student).count(), 0)
        record_refund(original_charge=self.charge, amount_iqd=1000, reason_code="r")
        self.assertEqual(refunds_for_person(person=self.person_student).count(), 1)

    def test_adjustments_for_person(self):
        record_adjustment(person=self.person_student, amount_iqd=100, reason_code="x")
        self.assertEqual(adjustments_for_person(person=self.person_student).count(), 1)

    def test_check_balance(self):
        bal, sufficient = check_balance(person=self.person_student, amount_iqd=5000)
        self.assertEqual(bal, 7000)
        self.assertTrue(sufficient)
        bal, sufficient = check_balance(person=self.person_student, amount_iqd=99999)
        self.assertFalse(sufficient)

    def test_check_balance_no_wallet(self):
        Person.objects.create(code="P-NW", first_name="No", last_name="Wallet")
        no_wallet_person = Person.objects.get(code="P-NW")
        bal, sufficient = check_balance(person=no_wallet_person, amount_iqd=1)
        self.assertEqual(bal, 0)
        self.assertFalse(sufficient)


# ---------------------------------------------------------------------------
# Validators (unit)
# ---------------------------------------------------------------------------

class ValidatorUnitTests(TestCase):
    def test_validate_positive_amount_rejects_zero_and_negative(self):
        with self.assertRaises(ValidationError):
            validate_positive_amount(0)
        with self.assertRaises(ValidationError):
            validate_positive_amount(-1)

    def test_validate_charge_amounts_final_mismatch(self):
        with self.assertRaises(ValidationError):
            validate_charge_amounts(price_base_iqd=100, discount_iqd=10, final_charge_iqd=95)

    def test_validate_charge_amounts_discount_within_price_ok(self):
        validate_charge_amounts(price_base_iqd=100, discount_iqd=100, final_charge_iqd=0)

    def test_validate_no_duplicate_charge_reference_ignores_empty_reference(self):
        # No reference_id: never considered a duplicate.
        validate_no_duplicate_charge_reference(
            source_module="meal", reference_type="", reference_id=None,
            product_code="x",
        )

    def test_validate_source_module_required(self):
        with self.assertRaises(ValidationError):
            validate_source_module_required("")

    def test_validate_wallet_status_transition_closed_to_active_invalid(self):
        with self.assertRaises(ValidationError):
            validate_wallet_status_transition(
                current_status=Wallet.Status.CLOSED, new_status=Wallet.Status.ACTIVE
            )

    def test_validate_refund_reason_required(self):
        with self.assertRaises(ValidationError):
            validate_refund_reason_required("")

    def test_validate_adjustment_reason_required(self):
        with self.assertRaises(ValidationError):
            validate_adjustment_reason_required("  ")


# ===========================================================================
# Pre-resolved-amount services — charge() and refund()
#
# These tests cover the Meals-compatible boundary (meals_domain_architecture.md
# §5, §20): the caller passes a pre-resolved `amount_iqd`; Finance records
# the ledger row only and does NOT resolve pricing or call discounts.
# ===========================================================================


class ChargeServiceTests(FinanceBaseData):
    """Tests for the new ``charge()`` pre-resolved-amount service."""

    def setUp(self):
        super().setUp()
        self.wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)

    # --- happy path --------------------------------------------------

    def test_charge_debits_wallet(self):
        tx = charge(
            person=self.person_student,
            amount_iqd=2500,
            source_module="meals",
            reference_type="MealServiceEvent",
            reference_id=42,
            academic_year=self.year,
            description="meal_lunch base=2500 discount=0",
        )
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.DEBIT)
        self.assertEqual(tx.amount_iqd, -2500)
        self.assertEqual(tx.balance_before_iqd, 10000)
        self.assertEqual(tx.balance_after_iqd, 7500)
        self.assertEqual(tx.source_module, "meals")
        self.assertEqual(tx.reference_type, "MealServiceEvent")
        self.assertEqual(tx.reference_id, 42)
        self.wallet.refresh_from_db()
        self.assertEqual(self.wallet.balance_iqd, 7500)

    def test_charge_does_not_create_charge_row(self):
        # Unlike create_charge(), charge() must NOT create a Charge row.
        before = Charge.objects.count()
        charge(
            person=self.person_student,
            amount_iqd=1000,
            source_module="meals",
        )
        self.assertEqual(Charge.objects.count(), before)

    # --- zero amount -------------------------------------------------

    def test_charge_zero_amount_records_unpaid(self):
        tx = charge(
            person=self.person_student,
            amount_iqd=0,
            source_module="meals",
        )
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.UNPAID)
        self.assertEqual(tx.amount_iqd, 0)
        self.wallet.refresh_from_db()
        self.assertEqual(self.wallet.balance_iqd, 10000)

    # --- insufficient funds -----------------------------------------

    def test_charge_insufficient_funds_rejected(self):
        with self.assertRaises(ValidationError):
            charge(
                person=self.person_student,
                amount_iqd=100000,
                source_module="meals",
            )

    def test_charge_insufficient_funds_unpaid_allowed(self):
        tx = charge(
            person=self.person_student,
            amount_iqd=100000,
            source_module="meals",
            allow_unpaid=True,
        )
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.UNPAID)
        self.assertEqual(tx.amount_iqd, 0)
        self.wallet.refresh_from_db()
        self.assertEqual(self.wallet.balance_iqd, 10000)

    # --- credit limit -----------------------------------------------

    def test_charge_uses_credit_limit(self):
        # Drain wallet to zero, then charge within credit limit.
        record_adjustment(
            person=self.person_student, amount_iqd=-10000, reason_code="drain",
        )
        self.wallet.refresh_from_db()
        self.wallet.credit_limit_iqd = 5000
        self.wallet.save(update_fields=["credit_limit_iqd"])
        tx = charge(
            person=self.person_student,
            amount_iqd=2500,
            source_module="meals",
        )
        self.assertEqual(tx.tx_type, WalletTransaction.TxType.DEBIT)
        self.wallet.refresh_from_db()
        self.assertEqual(self.wallet.balance_iqd, -2500)

    def test_charge_beyond_credit_limit_rejected(self):
        record_adjustment(
            person=self.person_student, amount_iqd=-10000, reason_code="drain",
        )
        self.wallet.refresh_from_db()
        self.wallet.credit_limit_iqd = 1000
        self.wallet.save(update_fields=["credit_limit_iqd"])
        with self.assertRaises(ValidationError):
            charge(
                person=self.person_student,
                amount_iqd=2000,
                source_module="meals",
            )

    # --- validation --------------------------------------------------

    def test_charge_negative_amount_rejected(self):
        with self.assertRaises(ValidationError):
            charge(
                person=self.person_student,
                amount_iqd=-100,
                source_module="meals",
            )

    def test_charge_missing_source_module_rejected(self):
        with self.assertRaises(ValidationError):
            charge(
                person=self.person_student,
                amount_iqd=100,
                source_module="",
            )

    def test_charge_on_suspended_wallet_rejected(self):
        suspend_wallet(self.wallet)
        with self.assertRaises(ValidationError):
            charge(
                person=self.person_student,
                amount_iqd=100,
                source_module="meals",
            )

    def test_charge_on_closed_wallet_rejected(self):
        close_wallet(self.wallet)
        with self.assertRaises(ValidationError):
            charge(
                person=self.person_student,
                amount_iqd=100,
                source_module="meals",
            )

    # --- return type -------------------------------------------------

    def test_charge_returns_wallet_transaction_only(self):
        # The Meals architecture (§20.1) requires a WalletTransaction,
        # not a tuple.
        result = charge(
            person=self.person_student,
            amount_iqd=500,
            source_module="meals",
        )
        self.assertIsInstance(result, WalletTransaction)


class RefundServiceTests(FinanceBaseData):
    """Tests for the new ``refund()`` transaction-based service."""

    def setUp(self):
        super().setUp()
        self.wallet = create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=10000)
        self.original_tx = charge(
            person=self.person_student,
            amount_iqd=4000,
            source_module="meals",
            reference_type="MealServiceEvent",
            reference_id=7,
        )

    # --- full refund -------------------------------------------------

    def test_refund_full_amount_default(self):
        refund_tx = refund(
            person=self.person_student,
            original_transaction=self.original_tx,
            reason_code="supervisor_refund",
        )
        self.assertEqual(refund_tx.tx_type, WalletTransaction.TxType.REFUND)
        self.assertEqual(refund_tx.amount_iqd, 4000)
        self.assertTrue(refund_tx.is_reversal)
        self.assertEqual(refund_tx.reverses, self.original_tx)
        self.wallet.refresh_from_db()
        # 10000 - 4000 (charge) + 4000 (refund) = 10000
        self.assertEqual(self.wallet.balance_iqd, 10000)

    def test_refund_explicit_amount(self):
        refund_tx = refund(
            person=self.person_student,
            original_transaction=self.original_tx,
            amount_iqd=1000,
            reason_code="partial",
        )
        self.assertEqual(refund_tx.amount_iqd, 1000)
        self.wallet.refresh_from_db()
        # 10000 - 4000 + 1000 = 7000
        self.assertEqual(self.wallet.balance_iqd, 7000)

    # --- does NOT create a Refund model row --------------------------

    def test_refund_does_not_create_refund_row(self):
        # Unlike record_refund(), refund() must NOT create a Refund row.
        before = Refund.objects.count()
        refund(
            person=self.person_student,
            original_transaction=self.original_tx,
            reason_code="r",
        )
        self.assertEqual(Refund.objects.count(), before)

    # --- person mismatch --------------------------------------------

    def test_refund_person_mismatch_rejected(self):
        other = Person.objects.create(code="P-OTHER", first_name="O", last_name="O")
        with self.assertRaises(ValidationError):
            refund(
                person=other,
                original_transaction=self.original_tx,
                reason_code="r",
            )

    # --- validation --------------------------------------------------

    def test_refund_zero_amount_rejected(self):
        with self.assertRaises(ValidationError):
            refund(
                person=self.person_student,
                original_transaction=self.original_tx,
                amount_iqd=0,
                reason_code="r",
            )

    def test_refund_negative_amount_rejected(self):
        with self.assertRaises(ValidationError):
            refund(
                person=self.person_student,
                original_transaction=self.original_tx,
                amount_iqd=-100,
                reason_code="r",
            )

    def test_refund_non_transaction_rejected(self):
        with self.assertRaises(TypeError):
            refund(
                person=self.person_student,
                original_transaction="not a transaction",
                reason_code="r",
            )

    # --- wallet state ------------------------------------------------

    def test_refund_on_suspended_wallet_rejected(self):
        suspend_wallet(self.wallet)
        with self.assertRaises(ValidationError):
            refund(
                person=self.person_student,
                original_transaction=self.original_tx,
                reason_code="r",
            )

    # --- return type -------------------------------------------------

    def test_refund_returns_wallet_transaction_only(self):
        result = refund(
            person=self.person_student,
            original_transaction=self.original_tx,
            reason_code="r",
        )
        self.assertIsInstance(result, WalletTransaction)

    # --- reversal chain ----------------------------------------------

    def test_refund_reverses_links_to_original(self):
        refund_tx = refund(
            person=self.person_student,
            original_transaction=self.original_tx,
            reason_code="r",
        )
        self.assertEqual(refund_tx.reverses_id, self.original_tx.pk)
        # The original transaction is never mutated.
        self.original_tx.refresh_from_db()
        self.assertEqual(self.original_tx.amount_iqd, -4000)


class CheckBalanceSignatureTests(FinanceBaseData):
    """Verify the renamed ``amount_iqd`` parameter works by keyword."""

    def test_check_balance_keyword_amount_iqd(self):
        create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=5000)
        bal, sufficient = check_balance(
            person=self.person_student, amount_iqd=4000
        )
        self.assertEqual(bal, 5000)
        self.assertTrue(sufficient)

    def test_check_balance_keyword_amount_iqd_insufficient(self):
        create_wallet(person=self.person_student)
        record_payment(person=self.person_student, amount_iqd=1000)
        bal, sufficient = check_balance(
            person=self.person_student, amount_iqd=5000
        )
        self.assertEqual(bal, 1000)
        self.assertFalse(sufficient)
