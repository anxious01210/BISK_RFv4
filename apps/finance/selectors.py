from __future__ import annotations

from typing import Optional

from django.db.models import QuerySet, Q, Sum

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


# ---------------------------------------------------------------------------
# Wallet reads
# ---------------------------------------------------------------------------

def get_wallet_for_person(person) -> Optional[Wallet]:
    return (
        Wallet.objects.select_related("person")
        .filter(person=person)
        .first()
    )


def list_wallets(
    *,
    status: Optional[str] = None,
    search: str = "",
) -> QuerySet[Wallet]:
    qs = Wallet.objects.select_related("person", "person__student_profile", "person__staff_profile")
    if status is not None:
        qs = qs.filter(status=status)
    search = (search or "").strip()
    if search:
        qs = qs.filter(
            Q(person__code__icontains=search)
            | Q(person__first_name__icontains=search)
            | Q(person__last_name__icontains=search)
        )
    return qs


# ---------------------------------------------------------------------------
# Balance / ledger reads
# ---------------------------------------------------------------------------

def balance_for(*, person) -> int:
    """Return the cached wallet balance for a Person (0 if no wallet)."""
    wallet = get_wallet_for_person(person)
    return wallet.balance_iqd if wallet is not None else 0


def ledger_balance_for(*, person) -> int:
    """Return the authoritative ledger sum for a Person.

    This recomputes from :class:`WalletTransaction` rows rather than
    reading the cache. Useful for reconciliation checks.
    """
    total = WalletTransaction.objects.filter(person=person).aggregate(
        s=Sum("amount_iqd")
    )["s"]
    return total or 0


def transaction_history(
    *, person=None, wallet=None, since=None, tx_type: Optional[str] = None
) -> QuerySet[WalletTransaction]:
    """Read-only transaction history, most recent first."""
    qs = WalletTransaction.objects.select_related(
        "wallet", "person", "created_by", "created_by_staff", "reverses"
    )
    if person is not None:
        qs = qs.filter(person=person)
    if wallet is not None:
        qs = qs.filter(wallet=wallet)
    if since is not None:
        qs = qs.filter(created_at__gte=since)
    if tx_type is not None:
        qs = qs.filter(tx_type=tx_type)
    return qs


def ledger_for_person(*, person) -> QuerySet[WalletTransaction]:
    return transaction_history(person=person)


def transaction_reversals_for(*, transaction: WalletTransaction) -> QuerySet[WalletTransaction]:
    return WalletTransaction.objects.filter(reverses=transaction)


# ---------------------------------------------------------------------------
# Charge reads
# ---------------------------------------------------------------------------

def charges_for_person(
    *, person, product_code: Optional[str] = None, status: Optional[str] = None
) -> QuerySet[Charge]:
    qs = Charge.objects.select_related(
        "wallet", "person", "academic_year", "settled_transaction"
    ).filter(person=person)
    if product_code is not None:
        qs = qs.filter(product_code=product_code)
    if status is not None:
        qs = qs.filter(status=status)
    return qs


def list_charges(
    *,
    status: Optional[str] = None,
    source_module: Optional[str] = None,
    product_code: Optional[str] = None,
    academic_year=None,
) -> QuerySet[Charge]:
    qs = Charge.objects.select_related(
        "wallet", "person", "academic_year", "settled_transaction"
    )
    if status is not None:
        qs = qs.filter(status=status)
    if source_module is not None:
        qs = qs.filter(source_module=source_module)
    if product_code is not None:
        qs = qs.filter(product_code=product_code)
    if academic_year is not None:
        qs = qs.filter(academic_year=academic_year)
    return qs


def refunds_for_charge(*, charge: Charge) -> QuerySet[Refund]:
    return Refund.objects.select_related(
        "wallet", "person", "original_charge", "settled_transaction", "approved_by"
    ).filter(original_charge=charge)


# ---------------------------------------------------------------------------
# Payment / Refund / Adjustment reads
# ---------------------------------------------------------------------------

def payments_for_person(*, person) -> QuerySet[Payment]:
    return Payment.objects.select_related(
        "wallet", "person", "settled_transaction", "received_by"
    ).filter(person=person)


def refunds_for_person(*, person) -> QuerySet[Refund]:
    return Refund.objects.select_related(
        "wallet", "person", "original_charge", "settled_transaction", "approved_by"
    ).filter(person=person)


def adjustments_for_person(*, person) -> QuerySet[Adjustment]:
    return Adjustment.objects.select_related(
        "wallet", "person", "settled_transaction", "approved_by"
    ).filter(person=person)


# ---------------------------------------------------------------------------
# Pricing reads
# ---------------------------------------------------------------------------

def list_price_lists(
    *, is_active: Optional[bool] = None, academic_year=None
) -> QuerySet[PriceList]:
    qs = PriceList.objects.select_related("academic_year")
    if is_active is not None:
        qs = qs.filter(is_active=is_active)
    if academic_year is not None:
        qs = qs.filter(academic_year=academic_year)
    return qs


def get_price_list_by_name(name: str) -> Optional[PriceList]:
    return PriceList.objects.filter(name=name).first()


def list_pricing_rules(
    *,
    price_list=None,
    product_code: Optional[str] = None,
    grade=None,
    is_enabled: Optional[bool] = None,
) -> QuerySet[PricingRule]:
    qs = PricingRule.objects.select_related("price_list", "grade")
    if price_list is not None:
        qs = qs.filter(price_list=price_list)
    if product_code is not None:
        qs = qs.filter(product_code=product_code)
    if grade is not None:
        qs = qs.filter(grade=grade)
    if is_enabled is not None:
        qs = qs.filter(is_enabled=is_enabled)
    return qs


def get_enabled_pricing_rule(
    *,
    product_code: str,
    price_list: Optional[PriceList] = None,
    grade=None,
    period_key: str = "",
) -> Optional[PricingRule]:
    """Return the enabled pricing rule for a product, or ``None``.

    The model enforces uniqueness of ``(price_list, product_code)``, so at
    most one rule exists per product per list. ``grade`` and ``period_key``
    are descriptive scope metadata stored on the rule; they are not used
    to disambiguate multiple rules for the same product (the unique
    constraint forbids that).

    If no ``price_list`` is given, the first active PriceList containing a
    matching enabled rule is used.
    """
    qs = PricingRule.objects.filter(
        product_code=product_code, is_enabled=True
    ).select_related("price_list", "grade")
    if price_list is not None:
        qs = qs.filter(price_list=price_list)
    else:
        qs = qs.filter(price_list__is_active=True)
    return qs.first()
