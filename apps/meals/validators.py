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


# ===========================================================================
# Phase 3A — Service event validators
# ===========================================================================


def validate_service_event_role(*, student, staff) -> None:
    """Exactly one of ``student`` / ``staff`` is set for non-guest
    service events (§17). A guest event has neither set.

    Same rule as :func:`validate_subscription_role` but named for the
    service-event context so call sites read clearly.
    """
    if student is not None and staff is not None:
        raise ValidationError(
            _("A service event cannot reference both student and staff profiles.")
        )


def validate_service_event_snapshot_immutable(
    *, instance, updating_fields=None
) -> None:
    """A persisted ``MealServiceEvent`` in a terminal status must not
    have its snapshot fields mutated.

    Called from ``MealServiceEvent.clean()`` and from the future
    Phase 3B workflow services. The rule (§12): once ``status`` reaches
    a terminal state (CONFIRMED / DENIED / UNPAID / REFUNDED /
    VOIDED), the price / discount / charge / balance / academic /
    wallet-transaction-FK fields are immutable. Corrections create a
    *new* event (e.g. REFUNDED / VOIDED) referencing the original
    rather than mutating the confirmed row.

    ``instance`` must be a ``MealServiceEvent``. If it has a PK and is
    in a terminal status, the validator loads the persisted row from
    the database and compares each immutable snapshot field. If any
    snapshot field has been changed in-memory on ``instance``, a
    ``ValidationError`` is raised listing the mutated fields.

    ``updating_fields`` (optional) is the set of field names being
    updated by a ``save(update_fields=...)`` call. If supplied, only
    those fields are checked; this lets the workflow services update
    ``status`` / ``reason_*`` / ``reversed_*`` / ``wallet_refund_transaction``
    on a terminal event (the refund flow) without tripping the
    immutability guard on the *original* charge snapshot. The
    ``wallet_refund_transaction`` field is intentionally mutable on a
    CONFIRMED event (the refund flow sets it); it is immutable on a
    REFUNDED event.
    """
    from .models import (
        SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS,
        TERMINAL_SERVICE_EVENT_STATUSES,
        MealServiceEvent,
    )

    if not isinstance(instance, MealServiceEvent):
        raise TypeError("instance must be a MealServiceEvent.")

    # Only enforced on persisted instances in a terminal status.
    if not instance.pk:
        return
    if instance.status not in TERMINAL_SERVICE_EVENT_STATUSES:
        return

    # ``wallet_refund_transaction`` is set by the refund flow on a
    # CONFIRMED → REFUNDED transition; treat it as mutable on a
    # CONFIRMED event (the workflow will transition to REFUNDED in the
    # same transaction). On a REFUNDED event it is fully frozen.
    mutable_on_confirmed = set()
    if instance.status == "confirmed":
        mutable_on_confirmed = {"wallet_refund_transaction"}

    try:
        persisted = MealServiceEvent.objects.get(pk=instance.pk)
    except MealServiceEvent.DoesNotExist:
        return  # Nothing to compare against; let the save proceed.

    # Determine which fields to check.
    if updating_fields is not None:
        check_fields = set(updating_fields) & SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS
    else:
        check_fields = set(SERVICE_EVENT_IMMUTABLE_SNAPSHOT_FIELDS)

    check_fields -= mutable_on_confirmed

    mutated = []
    for field_name in sorted(check_fields):
        old = getattr(persisted, field_name, None)
        new = getattr(instance, field_name, None)
        if old != new:
            mutated.append(field_name)

    if mutated:
        raise ValidationError(
            {
                "status": _(
                    "Service event is in terminal status '%(status)s'; the "
                    "following snapshot fields are immutable and may not be "
                    "mutated: %(fields)s. Create a new corrective event "
                    "(e.g. REFUNDED / VOIDED) instead."
                    % {
                        "status": instance.status,
                        "fields": ", ".join(mutated),
                    }
                )
            }
        )


def validate_supervisor_action_target(*, service_event=None, eligibility=None) -> None:
    """A ``MealSupervisorAction`` must target exactly one of
    ``service_event`` / ``eligibility`` (or neither for a manual
    lookup). At most one is set; both set is ambiguous and rejected.
    """
    if service_event is not None and eligibility is not None:
        raise ValidationError(
            _("A supervisor action cannot target both a service event and an eligibility row.")
        )


# ===========================================================================
# Phase 3B-1 — Eligibility freezing
# ===========================================================================


def validate_eligibility_not_frozen(*, eligibility) -> None:
    """A ``MealEligibility`` row that is referenced by a
    ``MealServiceEvent`` is **frozen** and must not be mutated.

    Per ``meals_domain_architecture.md`` §12.1: "Recalculable before
    service; the row may be recomputed/overwritten by
    ``resolve_eligibility`` at any time **before** a
    ``MealServiceEvent`` references it. Frozen once referenced: once a
    ``MealServiceEvent`` references the eligibility, the row is frozen
    and must not be mutated. Further corrections are audited through
    ``MealSupervisorAction`` records against the (immutable) service
    event."

    This validator is called by ``resolve_eligibility`` before upserting
    an existing row. If the row is already referenced by a service
    event, the validator raises ``ValidationError`` and the eligibility
    is left unchanged. ``resolve_service`` sets ``MealServiceEvent.eligibility``
    *after* resolving eligibility, so the freeze takes effect on the
    next ``resolve_eligibility`` call for the same ``(person, date)``.
    """
    from .models import MealServiceEvent

    if eligibility is None or not eligibility.pk:
        return  # Nothing to freeze-check on an unsaved row.

    referenced = MealServiceEvent.objects.filter(eligibility=eligibility).exists()
    if referenced:
        raise ValidationError(
            _(
                "Eligibility for person %(person)s on %(date)s is frozen "
                "(referenced by a MealServiceEvent). Further corrections must "
                "be audited via MealSupervisorAction against the service event."
            )
            % {"person": eligibility.person, "date": eligibility.date}
        )
