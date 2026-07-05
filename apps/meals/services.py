"""Meals-domain services — Phase 1.

Phase 1 implements only :func:`resolve_price`, a **pure** function: it
does not touch the wallet, does not call ``apps.finance``, does not
write any rows, and does not call ``apps.discounts`` (which does not
exist yet). It resolves the base price and override delta for a
``(person, meal_plan, meal_period, date)`` quadruple, following the
first-match-wins order in
``docs/architecture/meals_domain_architecture.md`` §14.1.

Phase 3 will add ``resolve_service`` (wallet-mode charging) on top of
``resolve_price``; Phase 3 also adds the ``finance.charge`` call with
the pre-resolved ``final_charge_iqd``. None of that is in scope here.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from django.db import transaction

from .models import MealPeriod, MealPeriodPrice, MealPersonPriceOverride, MealPlan
from .selectors import (
    active_override_for_date,
    get_period_price,
)
from .validators import (
    validate_effective_window,
    validate_non_negative_price,
    validate_period_price_unique,
    validate_person_override_unique,
)


# Resolution sources, matching meals_domain_architecture.md §14.1.
SOURCE_PERSON_OVERRIDE_PERIOD = "person_override_period"
SOURCE_PERSON_OVERRIDE = "person_override"
SOURCE_PERIOD_PRICE = "period_price"
SOURCE_DEFAULT = "default"
SOURCE_DEFAULT_ZERO = "default_zero"


@transaction.atomic
def create_meal_period(
    *,
    kind: str = MealPeriod.Kind.LUNCH,
    period_template_source: str = "",
    period_template_ref_id=None,
    label: str = "",
    is_active: bool = True,
    sort_order: int = 0,
) -> MealPeriod:
    """Create a MealPeriod. A generic template reference is optional."""
    from .validators import validate_meal_period_template_ref_consistency

    validate_meal_period_template_ref_consistency(
        period_template_source=period_template_source,
        period_template_ref_id=period_template_ref_id,
    )
    return MealPeriod.objects.create(
        kind=kind,
        period_template_source=(period_template_source or "").strip(),
        period_template_ref_id=period_template_ref_id,
        label=(label or "").strip(),
        is_active=is_active,
        sort_order=sort_order,
    )


@transaction.atomic
def create_meal_plan(
    *,
    name: str,
    kind: str = MealPlan.Kind.LUNCH,
    mode: str = MealPlan.Mode.DATE_RANGE,
    is_active: bool = True,
    allow_supervisor_confirm: bool = True,
    allow_supervisor_unconfirm: bool = False,
    allow_supervisor_refund: bool = False,
    require_reason_on_override: bool = True,
    require_reason_on_unconfirm: bool = False,
    require_reason_on_refund: bool = True,
    insufficient_funds_mode: str = MealPlan.InsufficientFundsMode.DENY,
    credit_limit_iqd=None,
    default_price_iqd: int = 0,
    notes: str = "",
) -> MealPlan:
    """Create a MealPlan (product definition)."""
    validate_non_negative_price(default_price_iqd)
    return MealPlan.objects.create(
        name=(name or "").strip(),
        kind=kind,
        mode=mode,
        is_active=is_active,
        allow_supervisor_confirm=allow_supervisor_confirm,
        allow_supervisor_unconfirm=allow_supervisor_unconfirm,
        allow_supervisor_refund=allow_supervisor_refund,
        require_reason_on_override=require_reason_on_override,
        require_reason_on_unconfirm=require_reason_on_unconfirm,
        require_reason_on_refund=require_reason_on_refund,
        insufficient_funds_mode=insufficient_funds_mode,
        credit_limit_iqd=credit_limit_iqd,
        default_price_iqd=default_price_iqd,
        notes=notes or "",
    )


@transaction.atomic
def create_period_price(
    *,
    meal_plan: MealPlan,
    meal_period: MealPeriod,
    price_iqd: int,
    is_enabled: bool = True,
    notes: str = "",
) -> MealPeriodPrice:
    """Create a MealPeriodPrice (list price for plan × period)."""
    validate_non_negative_price(price_iqd)
    validate_period_price_unique(
        meal_plan=meal_plan, meal_period=meal_period
    )
    return MealPeriodPrice.objects.create(
        meal_plan=meal_plan,
        meal_period=meal_period,
        price_iqd=price_iqd,
        is_enabled=is_enabled,
        notes=notes or "",
    )


@transaction.atomic
def create_person_override(
    *,
    person,
    meal_plan: MealPlan,
    price_iqd: int,
    meal_period: Optional[MealPeriod] = None,
    is_enabled: bool = True,
    reason_code: str = "",
    notes: str = "",
    effective_from=None,
    effective_until=None,
) -> MealPersonPriceOverride:
    """Create a MealPersonPriceOverride (per-Person granular price)."""
    validate_non_negative_price(price_iqd)
    validate_effective_window(
        effective_from=effective_from, effective_until=effective_until
    )
    validate_person_override_unique(
        person=person, meal_plan=meal_plan, meal_period=meal_period
    )
    return MealPersonPriceOverride.objects.create(
        person=person,
        meal_plan=meal_plan,
        meal_period=meal_period,
        price_iqd=price_iqd,
        is_enabled=is_enabled,
        reason_code=reason_code or "",
        notes=notes or "",
        effective_from=effective_from,
        effective_until=effective_until,
    )


# ---------------------------------------------------------------------------
# resolve_price — pure function (no wallet, no finance, no discounts)
# ---------------------------------------------------------------------------

def resolve_price(
    *,
    person,
    meal_plan: MealPlan,
    meal_period: MealPeriod,
    on_date: Optional[date] = None,
) -> tuple[int, int, str]:
    """Resolve the meal base price for a (person, plan, period, date).

    Returns ``(base_price_iqd, override_delta_iqd, source)``.

    * ``base_price_iqd`` — the resolved base amount to charge before any
      discount (Phase 3 will call ``apps.discounts`` to reduce it). For
      date-range plans this is ``0`` (paid entitlement, no per-service
      charge) per §13 step 4a; callers in Phase 3 will short-circuit on
      ``MealPlan.mode == DATE_RANGE`` before calling this function, but we
      also handle it defensively here.
    * ``override_delta_iqd`` — signed delta relative to the period list
      price, non-zero only when a per-Person override won (§14.1).
      ``+`` means the override is *cheaper* than the list price (a
      discount-style override); ``-`` means the override is *more
      expensive* (a surcharge-style override). ``0`` when no override won.
    * ``source`` — one of the ``SOURCE_*`` constants above.

    The function is **pure**: it reads pricing rows but performs no
    writes. Phase 3's ``resolve_service`` wraps this with the wallet
    charge call and the ``MealServiceEvent`` write.
    """
    if on_date is None:
        on_date = date.today()

    # Defensive: DATE_RANGE plans do not charge per service (§13 step 4a).
    # Phase 3 callers short-circuit before this; we mirror the rule here
    # for safety and for callers that resolve_price directly.
    if meal_plan.mode == MealPlan.Mode.DATE_RANGE:
        return 0, 0, "date_range_no_charge"

    # 1. Per-Person × Per-MealPeriod override (effective window).
    override = active_override_for_date(
        person=person,
        meal_plan=meal_plan,
        meal_period=meal_period,
        on_date=on_date,
        exact_period=True,
    )
    if override is not None:
        period_list = get_period_price(
            meal_plan=meal_plan, meal_period=meal_period, is_enabled=True
        )
        base = override.price_iqd
        # Per §14.1: delta = (period_list_price - override.price_iqd) only
        # when a period list price exists; 0 otherwise.
        if period_list is not None:
            delta = period_list.price_iqd - override.price_iqd
        else:
            delta = 0
        return base, delta, SOURCE_PERSON_OVERRIDE_PERIOD

    # 2. Per-Person × any-period override.
    override = active_override_for_date(
        person=person,
        meal_plan=meal_plan,
        meal_period=meal_period,
        on_date=on_date,
        exact_period=False,
    )
    if override is not None:
        period_list = get_period_price(
            meal_plan=meal_plan, meal_period=meal_period, is_enabled=True
        )
        base = override.price_iqd
        if period_list is not None:
            delta = period_list.price_iqd - override.price_iqd
        else:
            delta = 0
        return base, delta, SOURCE_PERSON_OVERRIDE

    # 3. Per-MealPeriod list price.
    period_price = get_period_price(
        meal_plan=meal_plan, meal_period=meal_period, is_enabled=True
    )
    if period_price is not None:
        return period_price.price_iqd, 0, SOURCE_PERIOD_PRICE

    # 4. Plan default.
    if meal_plan.default_price_iqd > 0:
        return meal_plan.default_price_iqd, 0, SOURCE_DEFAULT

    # 5. Zero (free meal / misconfiguration — flagged in logs).
    return 0, 0, SOURCE_DEFAULT_ZERO
