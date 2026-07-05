"""Meals-domain validators — Phase 1.

Validators here are defense-in-depth checks layered on top of the
database constraints declared in :mod:`apps.meals.models`. The DB
constraints are the source of truth; these functions exist to give
service-layer callers clear, translatable error messages without
relying on ``IntegrityError`` introspection.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

from .models import (
    MealPeriod,
    MealPeriodPrice,
    MealPersonPriceOverride,
    MealPlan,
)


# ---------------------------------------------------------------------------
# Amount validation
# ---------------------------------------------------------------------------

def validate_non_negative_price(price_iqd: int) -> None:
    """A meal price (list or override) must be non-negative."""
    if price_iqd is None or price_iqd < 0:
        raise ValidationError(_("Price must be a non-negative integer."))


def validate_effective_window(
    *,
    effective_from: Optional[date] = None,
    effective_until: Optional[date] = None,
) -> None:
    """When both effective bounds are set, ``until`` must be ≥ ``from``."""
    if effective_from and effective_until and effective_until < effective_from:
        raise ValidationError(
            {"effective_until": _("Effective until cannot be before effective from.")}
        )


# ---------------------------------------------------------------------------
# MealPeriod
# ---------------------------------------------------------------------------

def validate_meal_period_template_ref_consistency(
    *,
    period_template_source: str = "",
    period_template_ref_id=None,
) -> None:
    """A generic period-template reference must carry both source and id.

    A row with a ``ref_id`` but no ``source`` (or vice versa) is ambiguous
    and rejected. Both blank is allowed (an "unlinked" MealPeriod).
    """
    source = (period_template_source or "").strip()
    has_source = bool(source)
    has_ref = period_template_ref_id is not None
    if has_source != has_ref:
        raise ValidationError(
            "period_template_source and period_template_ref_id must both be "
            "set or both be blank."
        )


# ---------------------------------------------------------------------------
# MealPeriodPrice uniqueness
# ---------------------------------------------------------------------------

def validate_period_price_unique(
    *,
    meal_plan,
    meal_period,
    instance: MealPeriodPrice | None = None,
) -> None:
    """At most one price row per (plan, period).

    The model already enforces this via ``unique_together``; this
    validator is an explicit guard that yields a clear message.
    """
    qs = MealPeriodPrice.objects.filter(
        meal_plan=meal_plan, meal_period=meal_period
    )
    if instance is not None and instance.pk:
        qs = qs.exclude(pk=instance.pk)
    if qs.exists():
        raise ValidationError(
            _("A price for plan %(plan)s and period %(period)s already exists.")
            % {"plan": meal_plan, "period": meal_period}
        )


# ---------------------------------------------------------------------------
# MealPersonPriceOverride uniqueness
# ---------------------------------------------------------------------------

def validate_person_override_unique(
    *,
    person,
    meal_plan,
    meal_period=None,
    instance: MealPersonPriceOverride | None = None,
) -> None:
    """At most one override per (person, plan, period)."""
    qs = MealPersonPriceOverride.objects.filter(
        person=person, meal_plan=meal_plan, meal_period=meal_period
    )
    if instance is not None and instance.pk:
        qs = qs.exclude(pk=instance.pk)
    if qs.exists():
        raise ValidationError(
            _("An override for this person, plan, and period already exists.")
        )
