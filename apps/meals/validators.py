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


# ===========================================================================
# Phase 2 — Subscription / Exception / Eligibility validators
# ===========================================================================


def validate_subscription_dates(*, start_date, end_date) -> None:
    """``end_date`` must be ≥ ``start_date``."""
    if start_date is None or end_date is None:
        raise ValidationError(_("Subscription start_date and end_date are required."))
    if end_date < start_date:
        raise ValidationError(
            {"end_date": _("Subscription end_date cannot be before start_date.")}
        )


def validate_subscription_role(*, student, staff) -> None:
    """Exactly one of ``student`` / ``staff`` is set for non-guest
    subscriptions.

    A subscription with neither set is treated as a guest subscription
    (the future guest profile / exception flow). The Meals architecture
    §11 says exactly one is set for non-guest subscriptions; for Phase
    2 we accept either exactly one set OR both unset (guest), but
    reject both set.
    """
    if student is not None and staff is not None:
        raise ValidationError(
            _("A subscription cannot reference both student and staff profiles.")
        )


def validate_subscription_overlap(
    *,
    person,
    priority,
    start_date,
    end_date,
    status,
    exclude_pk=None,
) -> None:
    """Two ``ACTIVE`` (or ``FUTURE``) subscriptions for the same person
    with the same ``priority`` may not overlap in ``[start_date,
    end_date]``.

    Different-priority overlapping ``ACTIVE`` subscriptions are allowed
    (this is how primary + fallback coexist). Matches legacy
    ``MealSubscription.clean()`` at ``apps/attendance/models.py:610``
    and ``meals_domain_architecture.md`` §11.5.
    """
    from .models import MealSubscription

    overlapping_statuses = {
        MealSubscription.Status.ACTIVE,
        MealSubscription.Status.FUTURE,
        MealSubscription.Status.PAUSED,
    }
    if status not in overlapping_statuses:
        return

    qs = MealSubscription.objects.filter(
        person=person,
        priority=priority,
        status__in=overlapping_statuses,
        start_date__lte=end_date,
        end_date__gte=start_date,
    )
    if exclude_pk is not None:
        qs = qs.exclude(pk=exclude_pk)
    if qs.exists():
        raise ValidationError(
            _(
                "Overlapping ACTIVE/FUTURE/PAUSED subscription with the same "
                "priority (%(priority)d) already exists for this person in the "
                "selected date range."
            )
            % {"priority": priority}
        )


def validate_subscription_status_transition(*, current_status, new_status) -> None:
    """Validate a subscription lifecycle transition
    (``meals_domain_architecture.md`` §11.1)."""
    from .models import MealSubscription

    allowed = {
        MealSubscription.Status.FUTURE: {
            MealSubscription.Status.ACTIVE,
            MealSubscription.Status.CANCELLED,
            MealSubscription.Status.EXPIRED,
        },
        MealSubscription.Status.ACTIVE: {
            MealSubscription.Status.PAUSED,
            MealSubscription.Status.CANCELLED,
            MealSubscription.Status.EXPIRED,
        },
        MealSubscription.Status.PAUSED: {
            MealSubscription.Status.ACTIVE,
            MealSubscription.Status.CANCELLED,
            MealSubscription.Status.EXPIRED,
        },
        MealSubscription.Status.EXPIRED: set(),
        MealSubscription.Status.CANCELLED: set(),
    }
    permitted = allowed.get(current_status, set())
    if new_status not in permitted:
        raise ValidationError(
            _(
                "Invalid subscription status transition: %(from)s -> %(to)s."
                % {"from": current_status, "to": new_status}
            )
        )


def validate_exception_window(*, effective_date, end_date) -> None:
    """When ``end_date`` is set, it must be ≥ ``effective_date``."""
    if effective_date is None:
        raise ValidationError(_("MealException effective_date is required."))
    if end_date is not None and end_date < effective_date:
        raise ValidationError(
            {"end_date": _("MealException end_date cannot be before effective_date.")}
        )


def validate_eligibility_unique_for_person_date(
    *, person, date, instance=None
) -> None:
    """At most one canonical eligibility row per ``(person, date)``.

    The model enforces this via ``unique_together``; this validator
    yields a clear message without ``IntegrityError`` introspection.
    """
    from .models import MealEligibility

    qs = MealEligibility.objects.filter(person=person, date=date)
    if instance is not None and instance.pk:
        qs = qs.exclude(pk=instance.pk)
    if qs.exists():
        raise ValidationError(
            _("An eligibility row for person %(person)s on %(date)s already exists.")
            % {"person": person, "date": date}
        )
