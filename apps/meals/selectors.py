"""Meals-domain selectors — Phase 1 (pricing) and Phase 2
(subscriptions / exceptions / eligibility).

Read-only helpers for querying meal data. They are pure query helpers;
no business logic lives here. Pricing resolution (first-match ordering)
lives in :mod:`apps.meals.services`. Subscription lifecycle transitions
and eligibility resolution live in services as well.
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


# ===========================================================================
# Phase 2 — Subscription / Exception / Eligibility selectors
# ===========================================================================


# ---------------------------------------------------------------------------
# MealSubscription
# ---------------------------------------------------------------------------

def list_subscriptions(
    *,
    person=None,
    meal_plan=None,
    status=None,
    is_active_period: Optional[bool] = None,
    on_date=None,
) -> QuerySet[MealSubscription]:
    """Read-only subscription query.

    Pass ``on_date`` to filter to subscriptions covering that date
    (``start_date <= on_date <= end_date``). Pass
    ``is_active_period=True`` to also restrict to statuses considered
    "active for service" (ACTIVE/FUTURE/PAUSED); pass ``False`` to
    restrict to inactive statuses (EXPIRED/CANCELLED).
    """
    from .models import MealSubscription

    qs = MealSubscription.objects.select_related(
        "person", "student", "staff", "meal_plan", "academic_year"
    )
    if person is not None:
        qs = qs.filter(person=person)
    if meal_plan is not None:
        qs = qs.filter(meal_plan=meal_plan)
    if status is not None:
        qs = qs.filter(status=status)
    if on_date is not None:
        qs = qs.filter(start_date__lte=on_date, end_date__gte=on_date)
    if is_active_period is True:
        qs = qs.filter(
            status__in=[
                MealSubscription.Status.ACTIVE,
                MealSubscription.Status.FUTURE,
                MealSubscription.Status.PAUSED,
            ]
        )
    elif is_active_period is False:
        qs = qs.filter(
            status__in=[
                MealSubscription.Status.EXPIRED,
                MealSubscription.Status.CANCELLED,
            ]
        )
    return qs


def active_subscriptions_for(*, person, on_date) -> QuerySet[MealSubscription]:
    """Subscriptions that *could* grant a meal on ``on_date``.

    Returns ACTIVE/FUTURE/PAUSED subscriptions whose date range covers
    ``on_date``. Ordered by ``priority`` (lower = higher priority).
    """
    from .models import MealSubscription

    return (
        MealSubscription.objects.select_related(
            "person", "student", "staff", "meal_plan", "academic_year"
        )
        .filter(
            person=person,
            status__in=[
                MealSubscription.Status.ACTIVE,
                MealSubscription.Status.FUTURE,
                MealSubscription.Status.PAUSED,
            ],
            start_date__lte=on_date,
            end_date__gte=on_date,
        )
        .order_by("priority", "start_date", "id")
    )


def get_subscription_by_id(subscription_id) -> Optional[MealSubscription]:
    from .models import MealSubscription

    return (
        MealSubscription.objects.select_related(
            "person", "student", "staff", "meal_plan", "academic_year"
        )
        .filter(pk=subscription_id)
        .first()
    )


# ---------------------------------------------------------------------------
# MealException
# ---------------------------------------------------------------------------

def list_exceptions(
    *,
    person=None,
    kind=None,
    is_active: Optional[bool] = None,
    on_date=None,
) -> QuerySet:
    """Read-only exception query.

    Pass ``on_date`` to filter to exceptions covering that date:
    ``effective_date <= on_date`` AND (``end_date`` IS NULL OR
    ``on_date <= end_date``).
    """
    from .models import MealException

    qs = MealException.objects.select_related(
        "person", "meal_plan", "approved_by"
    )
    if person is not None:
        qs = qs.filter(person=person)
    if kind is not None:
        qs = qs.filter(kind=kind)
    if is_active is not None:
        qs = qs.filter(is_active=is_active)
    if on_date is not None:
        qs = qs.filter(effective_date__lte=on_date).filter(
            Q(end_date__isnull=True) | Q(end_date__gte=on_date)
        )
    return qs


def exceptions_for(*, person, on_date) -> QuerySet:
    """Active exceptions covering ``(person, on_date)``."""
    from .models import MealException

    return MealException.objects.select_related(
        "person", "meal_plan", "approved_by"
    ).filter(
        person=person,
        is_active=True,
        effective_date__lte=on_date,
    ).filter(
        Q(end_date__isnull=True) | Q(end_date__gte=on_date)
    )


# ---------------------------------------------------------------------------
# MealEligibility
# ---------------------------------------------------------------------------

def eligibility_for(*, person, on_date):
    """Return the canonical eligibility row for ``(person, on_date)`` or None."""
    from .models import MealEligibility

    return (
        MealEligibility.objects.select_related(
            "person", "student", "subscription", "meal_plan", "resolved_by"
        )
        .filter(person=person, date=on_date)
        .first()
    )


def list_eligibilities(
    *,
    person=None,
    on_date=None,
    decision=None,
) -> QuerySet:
    from .models import MealEligibility

    qs = MealEligibility.objects.select_related(
        "person", "student", "subscription", "meal_plan", "resolved_by"
    )
    if person is not None:
        qs = qs.filter(person=person)
    if on_date is not None:
        qs = qs.filter(date=on_date)
    if decision is not None:
        qs = qs.filter(decision=decision)
    return qs


# ===========================================================================
# Phase 3A — Service event / supervisor action selectors
# ===========================================================================


# ---------------------------------------------------------------------------
# MealServiceEvent
# ---------------------------------------------------------------------------

def list_service_events(
    *,
    person=None,
    student=None,
    staff=None,
    date=None,
    status=None,
    meal_plan=None,
    meal_period=None,
    section_code_snapshot=None,
) -> QuerySet:
    """Read-only service-event query.

    All filters optional. Ordered by ``(-date, -id)`` per the model
    Meta.
    """
    from .models import MealServiceEvent

    qs = MealServiceEvent.objects.select_related(
        "person",
        "student",
        "staff",
        "eligibility",
        "subscription",
        "meal_plan",
        "meal_period",
        "wallet_transaction",
        "wallet_refund_transaction",
        "served_by",
        "reversed_by",
    )
    if person is not None:
        qs = qs.filter(person=person)
    if student is not None:
        qs = qs.filter(student=student)
    if staff is not None:
        qs = qs.filter(staff=staff)
    if date is not None:
        qs = qs.filter(date=date)
    if status is not None:
        qs = qs.filter(status=status)
    if meal_plan is not None:
        qs = qs.filter(meal_plan=meal_plan)
    if meal_period is not None:
        qs = qs.filter(meal_period=meal_period)
    if section_code_snapshot is not None:
        qs = qs.filter(section_code_snapshot=section_code_snapshot)
    return qs


def service_events_for(*, person, date) -> QuerySet:
    """All service events for ``(person, date)``."""
    from .models import MealServiceEvent

    return MealServiceEvent.objects.select_related(
        "person",
        "student",
        "staff",
        "eligibility",
        "subscription",
        "meal_plan",
        "meal_period",
        "wallet_transaction",
        "wallet_refund_transaction",
    ).filter(person=person, date=date)


def service_events_for_section(*, section_code, date) -> QuerySet:
    """Service events for a section (by snapshot code) on a date."""
    from .models import MealServiceEvent

    return MealServiceEvent.objects.select_related(
        "person", "student", "meal_plan", "meal_period"
    ).filter(section_code_snapshot=section_code, date=date)


def get_service_event_by_id(service_event_id):
    """Single service-event lookup or ``None``."""
    from .models import MealServiceEvent

    return (
        MealServiceEvent.objects.select_related(
            "person",
            "student",
            "staff",
            "eligibility",
            "subscription",
            "meal_plan",
            "meal_period",
            "wallet_transaction",
            "wallet_refund_transaction",
            "served_by",
            "reversed_by",
        )
        .filter(pk=service_event_id)
        .first()
    )


def confirmed_meals_for(*, person, date) -> QuerySet:
    """Same-day confirmed meals for a person — used by the future
    same-day multi-meal discount context (§19.5)."""
    from .models import MealServiceEvent

    return MealServiceEvent.objects.filter(
        person=person,
        date=date,
        status=MealServiceEvent.Status.CONFIRMED,
    )


# ---------------------------------------------------------------------------
# MealSupervisorAction
# ---------------------------------------------------------------------------

def list_supervisor_actions(
    *,
    service_event=None,
    eligibility=None,
    action=None,
    performed_by=None,
) -> QuerySet:
    """Read-only supervisor-action query."""
    from .models import MealSupervisorAction

    qs = MealSupervisorAction.objects.select_related(
        "service_event",
        "eligibility",
        "performed_by",
        "performed_by_user",
    )
    if service_event is not None:
        qs = qs.filter(service_event=service_event)
    if eligibility is not None:
        qs = qs.filter(eligibility=eligibility)
    if action is not None:
        qs = qs.filter(action=action)
    if performed_by is not None:
        qs = qs.filter(performed_by=performed_by)
    return qs


def supervisor_actions_for_service_event(*, service_event) -> QuerySet:
    """All supervisor actions for a service event, newest first."""
    from .models import MealSupervisorAction

    return MealSupervisorAction.objects.select_related(
        "performed_by", "performed_by_user"
    ).filter(service_event=service_event)


def supervisor_actions_for_eligibility(*, eligibility) -> QuerySet:
    """All supervisor actions for an eligibility row, newest first."""
    from .models import MealSupervisorAction

    return MealSupervisorAction.objects.select_related(
        "performed_by", "performed_by_user"
    ).filter(eligibility=eligibility)


# ===========================================================================
# Phase 3B-1 — Service-resolution selectors
# ===========================================================================


def pending_service_event_for(
    *, person, date, meal_period=None
):
    """Return the most recent PENDING ``MealServiceEvent`` for
    ``(person, date, meal_period)`` or ``None``.

    ``resolve_service`` uses this to decide whether to create a new
    event or update an existing PENDING one. Terminal-status events are
    never returned (they are immutable — §12).
    """
    from .models import MealServiceEvent

    qs = MealServiceEvent.objects.filter(
        person=person,
        date=date,
        status=MealServiceEvent.Status.PENDING,
    )
    if meal_period is not None:
        qs = qs.filter(meal_period=meal_period)
    return qs.order_by("-id").first()


def service_event_referenced_by_eligibility(eligibility_id) -> bool:
    """Return True if any ``MealServiceEvent`` references the given
    eligibility row. Used by the freeze validator."""
    from .models import MealServiceEvent

    return MealServiceEvent.objects.filter(eligibility_id=eligibility_id).exists()
