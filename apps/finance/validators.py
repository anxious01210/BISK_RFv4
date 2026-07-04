from __future__ import annotations

from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

from .models import (
    Adjustment,
    Charge,
    Payment,
    PricingRule,
    Refund,
    Wallet,
    WalletTransaction,
)


# ---------------------------------------------------------------------------
# Amount validation
# ---------------------------------------------------------------------------

def validate_positive_amount(amount_iqd: int) -> None:
    """A strictly positive amount (used for payments/refunds/top-ups)."""
    if amount_iqd is None or amount_iqd <= 0:
        raise ValidationError(_("Amount must be a positive integer."))


def validate_nonzero_amount(amount_iqd: int) -> None:
    """A non-zero signed amount (used for adjustments)."""
    if amount_iqd is None or amount_iqd == 0:
        raise ValidationError(_("Adjustment amount must be non-zero."))


def validate_non_negative_amount(amount_iqd: int) -> None:
    if amount_iqd is None or amount_iqd < 0:
        raise ValidationError(_("Amount cannot be negative."))


def validate_charge_amounts(
    *,
    price_base_iqd: int,
    discount_iqd: int = 0,
    final_charge_iqd: int | None = None,
) -> None:
    """Validate the price/discount/final triple for a Charge."""
    validate_non_negative_amount(price_base_iqd)
    validate_non_negative_amount(discount_iqd)
    if discount_iqd > price_base_iqd:
        raise ValidationError(
            _("Discount cannot exceed the base price.")
        )
    expected = price_base_iqd - discount_iqd
    if final_charge_iqd is None:
        final_charge_iqd = expected
    validate_non_negative_amount(final_charge_iqd)
    if final_charge_iqd != expected:
        raise ValidationError(
            _("Final charge must equal price base minus discount.")
        )


# ---------------------------------------------------------------------------
# Wallet status / lifecycle
# ---------------------------------------------------------------------------

WALLET_TRANSITIONS: dict[str, set[str]] = {
    Wallet.Status.ACTIVE: {Wallet.Status.SUSPENDED, Wallet.Status.CLOSED},
    Wallet.Status.SUSPENDED: {Wallet.Status.ACTIVE, Wallet.Status.CLOSED},
    Wallet.Status.CLOSED: set(),
}


def validate_wallet_status_transition(
    *, current_status: str, new_status: str
) -> None:
    allowed = WALLET_TRANSITIONS.get(current_status, set())
    if new_status not in allowed:
        raise ValidationError(
            _(
                "Invalid wallet status transition: %(from)s -> %(to)s."
                % {"from": current_status, "to": new_status}
            )
        )


def validate_wallet_operational(wallet: Wallet) -> None:
    """Wallet must be ACTIVE for charges, payments, refunds and adjustments."""
    if wallet.status != Wallet.Status.ACTIVE:
        raise ValidationError(
            _(
                "Wallet is not operational (status: %(status)s)."
                % {"status": wallet.status}
            )
        )


def validate_sufficient_funds(
    *, wallet: Wallet, amount_iqd: int
) -> None:
    """``amount_iqd`` is a positive charge magnitude."""
    if amount_iqd is None or amount_iqd <= 0:
        raise ValidationError(_("Charge amount must be positive."))
    if wallet.available_balance_iqd < amount_iqd:
        raise ValidationError(
            _(
                "Insufficient funds: available %(available)d, required %(required)d."
                % {
                    "available": wallet.available_balance_iqd,
                    "required": amount_iqd,
                }
            )
        )


# ---------------------------------------------------------------------------
# Wallet FK consistency
# ---------------------------------------------------------------------------

def validate_wallet_person_match(*, wallet: Wallet, person) -> None:
    if wallet.person_id != person.pk:
        raise ValidationError(
            _("Wallet does not belong to this person.")
        )


# ---------------------------------------------------------------------------
# Charge / Refund state rules
# ---------------------------------------------------------------------------

CHARGE_REFUNDABLE_STATUSES = {
    Charge.Status.SETTLED,
    Charge.Status.UNPAID,
}


def validate_charge_refundable(*, charge: Charge) -> None:
    if charge.status not in CHARGE_REFUNDABLE_STATUSES:
        raise ValidationError(
            _(
                "Charge is not refundable in its current status (%(status)s)."
                % {"status": charge.status}
            )
        )


def validate_refund_amount_within_charge(
    *, charge: Charge, amount_iqd: int
) -> None:
    """A single refund must not exceed the charge's final amount."""
    if amount_iqd > charge.final_charge_iqd:
        raise ValidationError(
            _(
                "Refund amount exceeds the original charge final amount."
            )
        )


def validate_refund_does_not_exceed_total(
    *, charge: Charge, amount_iqd: int
) -> None:
    """Sum of existing refunds + new refund must not exceed final charge."""
    already = sum(
        r.amount_iqd for r in Refund.objects.filter(original_charge=charge)
    )
    if already + amount_iqd > charge.final_charge_iqd:
        raise ValidationError(
            _(
                "Refund would exceed the original charge final amount "
                "(already refunded %(already)d, requested %(requested)d)."
                % {"already": already, "requested": amount_iqd}
            )
        )


# ---------------------------------------------------------------------------
# Duplicate / reference operations
# ---------------------------------------------------------------------------

def validate_no_duplicate_charge_reference(
    *,
    source_module: str,
    reference_type: str,
    reference_id,
    product_code: str,
    instance: Charge | None = None,
) -> None:
    """Avoid charging the same originating object twice for the same product.

    The reference is a generic triple (source_module, reference_type,
    reference_id) plus a product_code. A charge with an empty reference
    (manual entry) is never considered a duplicate.
    """
    if not reference_type or reference_id is None:
        return
    qs = Charge.objects.filter(
        source_module=source_module,
        reference_type=reference_type,
        reference_id=reference_id,
        product_code=product_code,
    )
    if instance is not None and instance.pk:
        qs = qs.exclude(pk=instance.pk)
    if qs.exists():
        raise ValidationError(
            _("A charge already exists for this reference and product.")
        )


def validate_no_duplicate_payment_reference(
    *,
    method: str,
    reference: str,
    instance: Payment | None = None,
) -> None:
    """Reject duplicate external payment references for card/bank transfers."""
    if not reference or method in (Payment.Method.CASH, Payment.Method.OTHER):
        return
    qs = Payment.objects.filter(method=method, reference=reference)
    if instance is not None and instance.pk:
        qs = qs.exclude(pk=instance.pk)
    if qs.exists():
        raise ValidationError(
            _("A payment with this external reference already exists.")
        )


# ---------------------------------------------------------------------------
# PriceList / PricingRule
# ---------------------------------------------------------------------------

def validate_pricing_rule_scope_unique(
    *, price_list, product_code, grade=None, period_key: str = "",
    instance: PricingRule | None = None,
) -> None:
    """A PriceList may have at most one enabled rule per product+scope.

    The model already enforces uniqueness of (price_list, product_code).
    This validator is an additional guard for the case where the unique
    constraint is relaxed in the future to allow overlapping scopes.
    """
    qs = PricingRule.objects.filter(
        price_list=price_list,
        product_code=product_code,
        grade=grade,
        period_key=period_key or "",
    )
    if instance is not None and instance.pk:
        qs = qs.exclude(pk=instance.pk)
    if qs.exists():
        raise ValidationError(
            _("A pricing rule for this product and scope already exists.")
        )


# ---------------------------------------------------------------------------
# Ledger immutability
# ---------------------------------------------------------------------------

def validate_transaction_not_mutated(instance: WalletTransaction) -> None:
    """A persisted WalletTransaction must never be edited."""
    if instance.pk:
        raise ValidationError(
            _("Wallet transactions are immutable; create a reversal instead.")
        )


def validate_adjustment_reason_required(reason_code: str) -> None:
    if not (reason_code or "").strip():
        raise ValidationError(_("Adjustment requires a reason code."))


def validate_refund_reason_required(reason_code: str) -> None:
    if not (reason_code or "").strip():
        raise ValidationError(_("Refund requires a reason code."))


def validate_source_module_required(source_module: str) -> None:
    if not (source_module or "").strip():
        raise ValidationError(_("Source module is required."))
