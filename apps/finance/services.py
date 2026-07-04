"""Finance service layer.

All balance mutations happen **only** through this module. The
:class:`~apps.finance.models.WalletTransaction` ledger is the source of
truth; :attr:`Wallet.balance_iqd` is a cached projection updated atomically
with each new ledger row inside a single ``transaction.atomic`` block.

Other domains (meal, tuition, transport, ...) call these services; they
must never read ``Wallet.balance_iqd`` or create ``WalletTransaction``
rows directly.
"""

from __future__ import annotations

from typing import Optional

from django.db import transaction
from django.utils import timezone

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
from .validators import (
    validate_adjustment_reason_required,
    validate_charge_amounts,
    validate_charge_refundable,
    validate_no_duplicate_charge_reference,
    validate_no_duplicate_payment_reference,
    validate_nonzero_amount,
    validate_positive_amount,
    validate_refund_amount_within_charge,
    validate_refund_does_not_exceed_total,
    validate_refund_reason_required,
    validate_source_module_required,
    validate_sufficient_funds,
    validate_wallet_operational,
    validate_wallet_person_match,
    validate_wallet_status_transition,
)


# ---------------------------------------------------------------------------
# Wallet creation helpers
# ---------------------------------------------------------------------------

@transaction.atomic
def create_wallet(*, person, currency: str = "IQD", credit_limit_iqd: int = 0) -> Wallet:
    """Create a fresh, active wallet for a Person.

    Raises ``IntegrityError`` if a wallet already exists for this person
    (the OneToOneField enforces uniqueness).
    """
    currency = (currency or "IQD").strip().upper()
    return Wallet.objects.create(
        person=person,
        currency=currency,
        balance_iqd=0,
        status=Wallet.Status.ACTIVE,
        credit_limit_iqd=credit_limit_iqd,
    )


def get_or_create_wallet_for_person(
    *, person, currency: str = "IQD", credit_limit_iqd: int = 0
) -> tuple[Wallet, bool]:
    """Return ``(wallet, created)`` for the given Person."""
    wallet = Wallet.objects.filter(person=person).first()
    if wallet is not None:
        return wallet, False
    return create_wallet(
        person=person, currency=currency, credit_limit_iqd=credit_limit_iqd
    ), True


def get_or_create_wallet_for_student(
    *, student_profile, currency: str = "IQD", credit_limit_iqd: int = 0
) -> tuple[Wallet, bool]:
    """Convenience helper: wallet for a ``StudentProfile``'s Person."""
    return get_or_create_wallet_for_person(
        person=student_profile.person,
        currency=currency,
        credit_limit_iqd=credit_limit_iqd,
    )


def get_or_create_wallet_for_staff(
    *, staff_profile, currency: str = "IQD", credit_limit_iqd: int = 0
) -> tuple[Wallet, bool]:
    """Convenience helper: wallet for a ``StaffProfile``'s Person."""
    return get_or_create_wallet_for_person(
        person=staff_profile.person,
        currency=currency,
        credit_limit_iqd=credit_limit_iqd,
    )


# ---------------------------------------------------------------------------
# Core ledger operation
# ---------------------------------------------------------------------------

@transaction.atomic
def create_wallet_transaction(
    *,
    wallet: Wallet,
    tx_type: str,
    amount_iqd: int,
    source_module: str,
    reference_type: str = "",
    reference_id=None,
    reason_code: str = "",
    notes: str = "",
    reverses: Optional[WalletTransaction] = None,
    created_by=None,
    created_by_staff=None,
) -> WalletTransaction:
    """Append one immutable ledger row and update the wallet cache.

    This is the **single** entry point for ledger writes. All other
    services (``charge``, ``record_payment``, ``record_refund``,
    ``record_adjustment``) call this internally.

    Rules:
    * The wallet row is locked with ``select_for_update`` for the duration
      of the transaction, guaranteeing serial balance updates.
    * ``balance_before_iqd`` / ``balance_after_iqd`` are snapshotted on the
      new row.
    * The new balance is validated against ``credit_limit_iqd`` (the
      resulting balance may not drop below ``-credit_limit_iqd``) for
      tx types that affect the balance.
    * ``UNPAID`` rows have ``amount_iqd == 0`` and do not move the balance.
    """
    validate_source_module_required(source_module)
    if reverses is not None and not isinstance(reverses, WalletTransaction):
        raise TypeError("reverses must be a WalletTransaction instance")

    # Lock the wallet row to serialize concurrent ledger appends.
    wallet = (
        Wallet.objects.select_for_update()
        .select_related("person")
        .get(pk=wallet.pk)
    )

    balance_before = wallet.balance_iqd
    if tx_type == WalletTransaction.TxType.UNPAID:
        if amount_iqd != 0:
            from django.core.exceptions import ValidationError
            raise ValidationError(
                {"amount_iqd": "UNPAID transactions must have amount 0."}
            )
        balance_after = balance_before
    else:
        balance_after = balance_before + amount_iqd
        # Credit limit guard: balance may not drop below -credit_limit.
        if balance_after < -wallet.credit_limit_iqd:
            from django.core.exceptions import ValidationError
            raise ValidationError(
                "Transaction would push the wallet below its credit limit "
                f"(balance_after={balance_after}, "
                f"limit=-{wallet.credit_limit_iqd})."
            )

    tx = WalletTransaction.objects.create(
        wallet=wallet,
        person=wallet.person,
        tx_type=tx_type,
        amount_iqd=amount_iqd,
        balance_before_iqd=balance_before,
        balance_after_iqd=balance_after,
        source_module=source_module,
        reference_type=reference_type or "",
        reference_id=reference_id,
        reason_code=reason_code or "",
        notes=notes or "",
        is_reversal=reverses is not None,
        reverses=reverses,
        created_by=created_by,
        created_by_staff=created_by_staff,
    )

    if balance_after != balance_before:
        wallet.balance_iqd = balance_after
        wallet.save(update_fields=["balance_iqd", "updated_at"])
    return tx


# Alias recommended by the architecture document.
apply_wallet_transaction = create_wallet_transaction


# ---------------------------------------------------------------------------
# Balance check
# ---------------------------------------------------------------------------

def check_balance(*, person, amount: int = 0) -> tuple[int, bool]:
    """Return ``(current_balance, sufficient_for_amount)``.

    Reads the cached balance (a projection of the ledger). For an
    authoritative reconciliation use :func:`recompute_balance`.
    """
    wallet = Wallet.objects.filter(person=person).first()
    if wallet is None:
        return 0, False
    return wallet.balance_iqd, wallet.available_balance_iqd >= amount


# ---------------------------------------------------------------------------
# Charge
# ---------------------------------------------------------------------------

@transaction.atomic
def create_charge(
    *,
    person,
    product_code: str,
    price_base_iqd: int,
    source_module: str,
    reference_type: str = "",
    reference_id=None,
    academic_year=None,
    description: str = "",
    discount_iqd: int = 0,
    discount_reason_code: str = "",
    discount_notes: str = "",
    allow_unpaid: bool = False,
    created_by=None,
    created_by_staff=None,
) -> tuple[Charge, Optional[WalletTransaction]]:
    """Create a Charge and settle it against the wallet.

    If the wallet has sufficient funds (or credit), a ``DEBIT`` ledger row
    is appended and the charge becomes ``SETTLED``. Otherwise:

    * if ``allow_unpaid`` is True, an ``UNPAID`` row is appended (amount 0)
      and the charge becomes ``UNPAID``;
    * otherwise a ``ValidationError`` is raised for insufficient funds.

    Discounts are not resolved here; ``discount_iqd`` is a snapshot
    provided by the caller (the future ``apps.discounts`` domain). Until
    that domain exists, callers pass ``discount_iqd=0``.
    """
    product_code = (product_code or "").strip()
    source_module = (source_module or "").strip()
    final_charge_iqd = price_base_iqd - discount_iqd
    validate_charge_amounts(
        price_base_iqd=price_base_iqd,
        discount_iqd=discount_iqd,
        final_charge_iqd=final_charge_iqd,
    )
    validate_no_duplicate_charge_reference(
        source_module=source_module,
        reference_type=reference_type,
        reference_id=reference_id,
        product_code=product_code,
    )

    wallet, _ = get_or_create_wallet_for_person(person=person)
    validate_wallet_person_match(wallet=wallet, person=person)
    validate_wallet_operational(wallet)

    charge = Charge.objects.create(
        wallet=wallet,
        person=person,
        academic_year=academic_year,
        product_code=product_code,
        description=description or "",
        price_base_iqd=price_base_iqd,
        discount_iqd=discount_iqd,
        discount_reason_code=discount_reason_code or "",
        discount_notes=discount_notes or "",
        final_charge_iqd=final_charge_iqd,
        status=Charge.Status.PENDING,
        source_module=source_module,
        reference_type=reference_type or "",
        reference_id=reference_id,
    )

    if final_charge_iqd == 0:
        # Nothing to debit; settle immediately with no balance impact.
        tx = create_wallet_transaction(
            wallet=wallet,
            tx_type=WalletTransaction.TxType.UNPAID,
            amount_iqd=0,
            source_module=source_module,
            reference_type="Charge",
            reference_id=charge.pk,
            reason_code="zero_charge",
            created_by=created_by,
            created_by_staff=created_by_staff,
        )
        charge.settled_transaction = tx
        charge.status = Charge.Status.SETTLED
        charge.save(update_fields=["settled_transaction", "status", "updated_at"])
        return charge, tx

    if wallet.available_balance_iqd >= final_charge_iqd:
        tx = create_wallet_transaction(
            wallet=wallet,
            tx_type=WalletTransaction.TxType.DEBIT,
            amount_iqd=-final_charge_iqd,
            source_module=source_module,
            reference_type="Charge",
            reference_id=charge.pk,
            reason_code=product_code,
            created_by=created_by,
            created_by_staff=created_by_staff,
        )
        charge.settled_transaction = tx
        charge.status = Charge.Status.SETTLED
        charge.save(update_fields=["settled_transaction", "status", "updated_at"])
        return charge, tx

    if allow_unpaid:
        tx = create_wallet_transaction(
            wallet=wallet,
            tx_type=WalletTransaction.TxType.UNPAID,
            amount_iqd=0,
            source_module=source_module,
            reference_type="Charge",
            reference_id=charge.pk,
            reason_code=product_code,
            notes="Insufficient funds at charge time.",
            created_by=created_by,
            created_by_staff=created_by_staff,
        )
        charge.settled_transaction = tx
        charge.status = Charge.Status.UNPAID
        charge.save(update_fields=["settled_transaction", "status", "updated_at"])
        return charge, tx

    # Insufficient funds and unpaid not allowed: roll back the charge.
    from django.core.exceptions import ValidationError
    raise ValidationError(
        "Insufficient funds for charge and unpaid charges are not allowed."
    )


# ---------------------------------------------------------------------------
# Payment
# ---------------------------------------------------------------------------

@transaction.atomic
def record_payment(
    *,
    person,
    amount_iqd: int,
    method: str = Payment.Method.CASH,
    reference: str = "",
    received_at=None,
    received_by=None,
    notes: str = "",
    source_module: str = "manual_topup",
    created_by=None,
) -> tuple[Payment, WalletTransaction]:
    """Record money received into the wallet (a top-up)."""
    validate_positive_amount(amount_iqd)
    validate_no_duplicate_payment_reference(
        method=method, reference=reference
    )
    if received_at is None:
        received_at = timezone.now()

    wallet, _ = get_or_create_wallet_for_person(person=person)
    validate_wallet_person_match(wallet=wallet, person=person)
    validate_wallet_operational(wallet)

    # Create the Payment first so its PK is available as the ledger row's
    # reference_id at creation time. The ledger row is immutable, so we
    # must NOT create it first and then UPDATE it to set reference_id.
    payment = Payment.objects.create(
        wallet=wallet,
        person=person,
        amount_iqd=amount_iqd,
        method=method,
        reference=reference or "",
        settled_transaction=None,
        received_at=received_at,
        received_by=received_by,
        notes=notes or "",
    )

    tx = create_wallet_transaction(
        wallet=wallet,
        tx_type=WalletTransaction.TxType.TOPUP,
        amount_iqd=amount_iqd,
        source_module=source_module,
        reference_type="Payment",
        reference_id=payment.pk,
        reason_code=method,
        notes=reference or notes,
        created_by=created_by,
        created_by_staff=received_by,
    )

    # Link the Payment to its ledger row. Payment is not immutable, so
    # this UPDATE is permitted.
    payment.settled_transaction = tx
    payment.save(update_fields=["settled_transaction"])
    return payment, tx


# ---------------------------------------------------------------------------
# Refund
# ---------------------------------------------------------------------------

@transaction.atomic
def record_refund(
    *,
    original_charge: Charge,
    amount_iqd: int,
    reason_code: str,
    reason_notes: str = "",
    approved_by=None,
    refunded_at=None,
    created_by=None,
) -> tuple[Refund, WalletTransaction]:
    """Record a refund that reverses (part of) a prior Charge.

    A ``REFUND`` ledger row is appended with ``reverses`` pointing to the
    original charge's DEBIT transaction. The original charge becomes
    ``REVERSED`` only when fully refunded; partial refunds leave it
    ``SETTLED``.
    """
    validate_positive_amount(amount_iqd)
    validate_refund_reason_required(reason_code)
    validate_charge_refundable(charge=original_charge)
    validate_refund_amount_within_charge(
        charge=original_charge, amount_iqd=amount_iqd
    )
    validate_refund_does_not_exceed_total(
        charge=original_charge, amount_iqd=amount_iqd
    )
    if refunded_at is None:
        refunded_at = timezone.now()

    wallet = original_charge.wallet
    validate_wallet_person_match(wallet=wallet, person=original_charge.person)
    validate_wallet_operational(wallet)

    original_tx = original_charge.settled_transaction
    reverses_tx = original_tx if original_tx is not None else None

    # Create the Refund first so its PK is available as the ledger row's
    # reference_id at creation time. The ledger row is immutable, so we
    # must NOT create it first and then UPDATE it to set reference_id.
    refund = Refund.objects.create(
        wallet=wallet,
        person=original_charge.person,
        original_charge=original_charge,
        amount_iqd=amount_iqd,
        reason_code=reason_code,
        reason_notes=reason_notes or "",
        settled_transaction=None,
        approved_by=approved_by,
        refunded_at=refunded_at,
    )

    tx = create_wallet_transaction(
        wallet=wallet,
        tx_type=WalletTransaction.TxType.REFUND,
        amount_iqd=amount_iqd,
        source_module=original_charge.source_module,
        reference_type="Refund",
        reference_id=refund.pk,
        reason_code=reason_code,
        notes=reason_notes or "",
        reverses=reverses_tx,
        created_by=created_by,
        created_by_staff=approved_by,
    )

    # Link the Refund to its ledger row. Refund is not immutable, so this
    # UPDATE is permitted.
    refund.settled_transaction = tx
    refund.save(update_fields=["settled_transaction"])

    # Mark the original charge REVERSED only when fully refunded.
    total_refunded = sum(
        r.amount_iqd
        for r in Refund.objects.filter(original_charge=original_charge)
    )
    if total_refunded >= original_charge.final_charge_iqd:
        original_charge.status = Charge.Status.REVERSED
        original_charge.save(update_fields=["status", "updated_at"])

    return refund, tx


# ---------------------------------------------------------------------------
# Adjustment
# ---------------------------------------------------------------------------

@transaction.atomic
def record_adjustment(
    *,
    person,
    amount_iqd: int,
    reason_code: str,
    reason_notes: str = "",
    approved_by=None,
    source_module: str = "manual_adjustment",
    created_by=None,
) -> tuple[Adjustment, WalletTransaction]:
    """Record a manual accounting correction.

    ``amount_iqd`` is signed: positive credits the wallet, negative debits
    it. A debit adjustment is still subject to the credit-limit guard.
    """
    validate_nonzero_amount(amount_iqd)
    validate_adjustment_reason_required(reason_code)

    wallet, _ = get_or_create_wallet_for_person(person=person)
    validate_wallet_person_match(wallet=wallet, person=person)
    validate_wallet_operational(wallet)

    # Create the Adjustment first so its PK is available as the ledger
    # row's reference_id at creation time. The ledger row is immutable, so
    # we must NOT create it first and then UPDATE it to set reference_id.
    adjustment = Adjustment.objects.create(
        wallet=wallet,
        person=person,
        amount_iqd=amount_iqd,
        reason_code=reason_code,
        reason_notes=reason_notes or "",
        settled_transaction=None,
        approved_by=approved_by,
    )

    tx = create_wallet_transaction(
        wallet=wallet,
        tx_type=WalletTransaction.TxType.ADJUSTMENT,
        amount_iqd=amount_iqd,
        source_module=source_module,
        reference_type="Adjustment",
        reference_id=adjustment.pk,
        reason_code=reason_code,
        notes=reason_notes or "",
        created_by=created_by,
        created_by_staff=approved_by,
    )

    # Link the Adjustment to its ledger row. Adjustment is not immutable,
    # so this UPDATE is permitted.
    adjustment.settled_transaction = tx
    adjustment.save(update_fields=["settled_transaction"])
    return adjustment, tx


# ---------------------------------------------------------------------------
# Wallet lifecycle
# ---------------------------------------------------------------------------

@transaction.atomic
def suspend_wallet(wallet: Wallet) -> Wallet:
    validate_wallet_status_transition(
        current_status=wallet.status, new_status=Wallet.Status.SUSPENDED
    )
    wallet.status = Wallet.Status.SUSPENDED
    wallet.save(update_fields=["status", "updated_at"])
    return wallet


@transaction.atomic
def reactivate_wallet(wallet: Wallet) -> Wallet:
    validate_wallet_status_transition(
        current_status=wallet.status, new_status=Wallet.Status.ACTIVE
    )
    wallet.status = Wallet.Status.ACTIVE
    wallet.save(update_fields=["status", "updated_at"])
    return wallet


@transaction.atomic
def close_wallet(wallet: Wallet) -> Wallet:
    """Permanently close a wallet. The ledger remains queryable forever."""
    validate_wallet_status_transition(
        current_status=wallet.status, new_status=Wallet.Status.CLOSED
    )
    # Re-read the balance from the DB to avoid acting on a stale instance.
    wallet = Wallet.objects.select_for_update().get(pk=wallet.pk)
    if wallet.balance_iqd != 0:
        # Zero the balance via a final adjustment, leaving the ledger intact.
        record_adjustment(
            person=wallet.person,
            amount_iqd=-wallet.balance_iqd,
            reason_code="wallet_closure",
            reason_notes="Zeroing balance on wallet closure.",
            source_module="wallet_lifecycle",
        )
        wallet.refresh_from_db()
    wallet.status = Wallet.Status.CLOSED
    wallet.save(update_fields=["status", "updated_at"])
    return wallet


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

@transaction.atomic
def recompute_balance(wallet: Wallet) -> tuple[int, int, bool]:
    """Recompute the balance from the ledger and fix the cache if drifted.

    Returns ``(ledger_sum, cached_balance, was_drift)``. This is the
    reconciliation service recommended by the architecture document; it
    is intended to be run periodically (open question Q9).
    """
    wallet = Wallet.objects.select_for_update().get(pk=wallet.pk)
    ledger_sum = sum(
        WalletTransaction.objects.filter(wallet=wallet).values_list(
            "amount_iqd", flat=True
        )
    ) or 0
    cached = wallet.balance_iqd
    if ledger_sum != cached:
        wallet.balance_iqd = ledger_sum
        wallet.save(update_fields=["balance_iqd", "updated_at"])
        return ledger_sum, cached, True
    return ledger_sum, cached, False


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def resolve_price(
    *,
    product_code: str,
    price_list: Optional[PriceList] = None,
    grade=None,
    period_key: str = "",
) -> Optional[int]:
    """Return the list price for a product within a PriceList, or ``None``.

    Looks for an enabled rule matching the most specific scope first
    (grade + period_key), then progressively wider. Discounts are NOT
    applied here; they are resolved by the future ``apps.discounts``
    domain.
    """
    from .selectors import get_enabled_pricing_rule

    rule = get_enabled_pricing_rule(
        product_code=product_code,
        price_list=price_list,
        grade=grade,
        period_key=period_key,
    )
    if rule is None:
        return None
    return rule.price_iqd
