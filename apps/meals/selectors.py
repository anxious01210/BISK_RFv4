"""Meals-domain selectors — Phase 1.

Read-only helpers for querying meal pricing data. They are pure query
helpers; no business logic lives here. Pricing resolution (first-match
ordering) lives in :mod:`apps.meals.services`.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from django.db.models import Q, QuerySet

from .models import (
    MealPeriod,
    MealPeriodPrice,
    MealPersonPriceOverride,
    MealPlan,
)


# ---------------------------------------------------------------------------
# MealPeriod
# ---------------------------------------------------------------------------

def list_meal_periods(
    *,
    kind: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> QuerySet[MealPeriod]:
    qs = MealPeriod.objects.all()
    if kind is not None:
        qs = qs.filter(kind=kind)
    if is_active is not None:
        qs = qs.filter(is_active=is_active)
    return qs


def get_meal_period_by_id(meal_period_id) -> Optional[MealPeriod]:
    return MealPeriod.objects.filter(pk=meal_period_id).first()


def get_meal_period_by_template_ref(
    *,
    period_template_source: str,
    period_template_ref_id: int,
) -> Optional[MealPeriod]:
    return (
        MealPeriod.objects.filter(
            period_template_source=period_template_source,
            period_template_ref_id=period_template_ref_id,
        ).first()
    )


# ---------------------------------------------------------------------------
# MealPlan
# ---------------------------------------------------------------------------

def list_meal_plans(
    *,
    kind: Optional[str] = None,
    mode: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> QuerySet[MealPlan]:
    qs = MealPlan.objects.all()
    if kind is not None:
        qs = qs.filter(kind=kind)
    if mode is not None:
        qs = qs.filter(mode=mode)
    if is_active is not None:
        qs = qs.filter(is_active=is_active)
    return qs


def get_meal_plan_by_name(name: str) -> Optional[MealPlan]:
    return MealPlan.objects.filter(name=name).first()


def get_meal_plan_by_id(meal_plan_id) -> Optional[MealPlan]:
    return MealPlan.objects.filter(pk=meal_plan_id).first()


# ---------------------------------------------------------------------------
# MealPeriodPrice
# ---------------------------------------------------------------------------

def list_period_prices(
    *,
    meal_plan=None,
    meal_period=None,
    is_enabled: Optional[bool] = None,
) -> QuerySet[MealPeriodPrice]:
    qs = MealPeriodPrice.objects.select_related("meal_plan", "meal_period")
    if meal_plan is not None:
        qs = qs.filter(meal_plan=meal_plan)
    if meal_period is not None:
        qs = qs.filter(meal_period=meal_period)
    if is_enabled is not None:
        qs = qs.filter(is_enabled=is_enabled)
    return qs


def get_period_price(
    *,
    meal_plan,
    meal_period,
    is_enabled: bool = True,
) -> Optional[MealPeriodPrice]:
    return (
        MealPeriodPrice.objects.filter(
            meal_plan=meal_plan,
            meal_period=meal_period,
            is_enabled=is_enabled,
        ).first()
    )


# ---------------------------------------------------------------------------
# MealPersonPriceOverride
# ---------------------------------------------------------------------------

def list_person_overrides(
    *,
    person=None,
    meal_plan=None,
    meal_period=None,
    is_enabled: Optional[bool] = None,
) -> QuerySet[MealPersonPriceOverride]:
    qs = MealPersonPriceOverride.objects.select_related(
        "person", "meal_plan", "meal_period"
    )
    if person is not None:
        qs = qs.filter(person=person)
    if meal_plan is not None:
        qs = qs.filter(meal_plan=meal_plan)
    if meal_period is not None:
        qs = qs.filter(meal_period=meal_period)
    if is_enabled is not None:
        qs = qs.filter(is_enabled=is_enabled)
    return qs


def get_person_override(
    *,
    person,
    meal_plan,
    meal_period=None,
    is_enabled: bool = True,
) -> Optional[MealPersonPriceOverride]:
    """Return the enabled override for an exact (person, plan, period) triple.

    Pass ``meal_period=None`` to look up an any-period override
    (``meal_period`` IS NULL).
    """
    qs = MealPersonPriceOverride.objects.filter(
        person=person, meal_plan=meal_plan, is_enabled=is_enabled,
    )
    if meal_period is None:
        qs = qs.filter(meal_period__isnull=True)
    else:
        qs = qs.filter(meal_period=meal_period)
    return qs.first()


def active_override_for_date(
    *,
    person,
    meal_plan,
    meal_period,
    on_date: date,
    is_enabled: bool = True,
    exact_period: bool = True,
) -> Optional[MealPersonPriceOverride]:
    """Return the enabled, date-effective override matching the filter.

    ``exact_period=True`` returns only the per-period override.
    ``exact_period=False`` returns only the any-period override
    (``meal_period__isnull=True``). Effective-window filtering is applied:
    a bound is treated as unbounded when null.
    """
    qs = MealPersonPriceOverride.objects.filter(
        person=person,
        meal_plan=meal_plan,
        is_enabled=is_enabled,
    ).filter(
        Q(effective_from__isnull=True) | Q(effective_from__lte=on_date)
    ).filter(
        Q(effective_until__isnull=True) | Q(effective_until__gte=on_date)
    )
    if exact_period:
        qs = qs.filter(meal_period=meal_period)
    else:
        qs = qs.filter(meal_period__isnull=True)
    return qs.first()
