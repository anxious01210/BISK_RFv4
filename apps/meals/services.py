"""Meals-domain services — Phase 1 (pricing), Phase 2 (subscriptions
+ eligibility), and Phase 3B-1 (service resolution with wallet
charging).

Phase 1 implements :func:`resolve_price`, a **pure** function: it
does not touch the wallet, does not call ``apps.finance``, does not
write any rows, and does not call ``apps.discounts`` (which does not
exist yet). It resolves the base price and override delta for a
``(person, meal_plan, meal_period, date)`` quadruple, following the
first-match-wins order in
``docs/architecture/meals_domain_architecture.md`` §14.1.

Phase 2 adds the subscription lifecycle services (``create_subscription``
/ ``pause_subscription`` / ``resume_subscription`` /
``cancel_subscription``) and the eligibility resolver
``resolve_eligibility``. ``resolve_eligibility`` follows §13 step 4a
(DATE_RANGE primary grants eligibility with no wallet charge) and
§13 step 5 (exception match) and §13 step 6 (default NOT_ELIGIBLE).

Phase 3B-1 adds :func:`resolve_service` — the full §13 service
resolver. It composes ``resolve_eligibility`` + ``resolve_price`` +
``finance.check_balance`` / ``finance.charge`` and writes the
immutable :class:`MealServiceEvent` with price + financial snapshots.
DATE_RANGE plans produce a no-charge CONFIRMED event; WALLET plans
produce a CONFIRMED (DEBIT) or UNPAID or DENIED event per the
``MealPlan.insufficient_funds_mode`` policy. Eligibility is frozen
once a service event references it (§12.1).

Out of scope (Phase 3B-2+):
* ``confirm`` / ``deny`` / ``unconfirm`` / ``refund`` / ``void``
  supervisor workflow services (each writing a
  ``MealSupervisorAction`` audit row).
* ``finance.refund`` calls (refund flow).
* Academic-presence / absence gates (§13 step 7) — needs attendance.
* Supervisor dashboard views.
* Legacy / data migration.
* Discounts.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from .models import (
    MealEligibility,
    MealPeriod,
    MealPeriodPrice,
    MealPersonPriceOverride,
    MealPlan,
    MealServiceEvent,
    MealSubscription,
    MealSupervisorAction,
)
from .selectors import (
    active_override_for_date,
    active_subscriptions_for,
    eligibility_for,
    exceptions_for,
    get_period_price,
    pending_service_event_for,
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


# ===========================================================================
# Phase 2 — Subscriptions, Exceptions, Eligibility
# ===========================================================================

# Eligibility reason codes (mirrors meals_domain_architecture.md §13).
REASON_NO_SUBSCRIPTION = "no_subscription"
REASON_DATE_RANGE_NO_CHARGE = "date_range_no_charge"
REASON_ONE_TIME_ELIGIBLE = "one_time_eligible"
REASON_GUEST_ELIGIBLE = "guest_eligible"
REASON_TEMPORARY_DENY = "temporary_deny"
REASON_PAUSED_SUBSCRIPTION = "subscription_paused"


@transaction.atomic
def create_subscription(
    *,
    person,
    meal_plan=None,
    start_date,
    end_date,
    academic_year=None,
    plan_type: str = "monthly",
    priority: int = 1,
    source: str = "manual",
    student=None,
    staff=None,
    status=None,
    notes: str = "",
) -> MealSubscription:
    """Create a :class:`MealSubscription`.

    Validates role (exactly one of ``student`` / ``staff`` for
    non-guest subscriptions), date validity, and the same-priority
    overlap rule (§11.5).
    """
    from .models import MealSubscription
    from .validators import (
        validate_subscription_dates,
        validate_subscription_overlap,
        validate_subscription_role,
    )

    validate_subscription_dates(start_date=start_date, end_date=end_date)
    validate_subscription_role(student=student, staff=staff)
    # Default status: ACTIVE if start_date <= today <= end_date, else FUTURE.
    # Caller may override via the ``status`` parameter.
    if status is None:
        today = date.today()
        if start_date > today:
            status = MealSubscription.Status.FUTURE
        else:
            status = MealSubscription.Status.ACTIVE
    validate_subscription_overlap(
        person=person,
        priority=priority,
        start_date=start_date,
        end_date=end_date,
        status=status,
    )
    return MealSubscription.objects.create(
        person=person,
        student=student,
        staff=staff,
        meal_plan=meal_plan,
        academic_year=academic_year,
        status=status,
        start_date=start_date,
        end_date=end_date,
        plan_type=plan_type,
        source=source,
        priority=priority,
        notes=notes or "",
    )


def _transition_subscription(
    subscription: MealSubscription,
    *,
    new_status: str,
) -> MealSubscription:
    from .validators import validate_subscription_status_transition

    validate_subscription_status_transition(
        current_status=subscription.status, new_status=new_status
    )
    subscription.status = new_status
    subscription.save(update_fields=["status", "updated_at"])
    return subscription


@transaction.atomic
def pause_subscription(
    *, subscription: MealSubscription, changed_by=None
) -> MealSubscription:
    """Transition a subscription from ACTIVE → PAUSED."""
    return _transition_subscription(
        subscription, new_status=MealSubscription.Status.PAUSED
    )


@transaction.atomic
def resume_subscription(
    *, subscription: MealSubscription, changed_by=None
) -> MealSubscription:
    """Transition a subscription from PAUSED → ACTIVE."""
    return _transition_subscription(
        subscription, new_status=MealSubscription.Status.ACTIVE
    )


@transaction.atomic
def cancel_subscription(
    *,
    subscription: MealSubscription,
    reason_code: str = "",
    changed_by=None,
) -> MealSubscription:
    """Transition a subscription to CANCELLED.

    The row is **never** deleted (§11.3 historical preservation); it
    remains forever for historical FK validity.
    """
    if reason_code:
        subscription.notes = (subscription.notes or "")
        if subscription.notes and not subscription.notes.endswith(" "):
            subscription.notes += " "
        subscription.notes += f"[cancelled: {reason_code}]"
    return _transition_subscription(
        subscription, new_status=MealSubscription.Status.CANCELLED
    )


@transaction.atomic
def grant_one_time_permission(
    *,
    person,
    effective_date,
    meal_plan=None,
    kind=None,
    reason_code: str = "",
    reason_notes: str = "",
    approved_by=None,
) -> "MealException":
    """Grant a one-time meal permission/denial for a single date (or a
    bounded window).

    Defaults to ``kind=ONE_TIME_ELIGIBLE`` for a single date. Use
    ``end_date`` (via the model directly) for multi-day exceptions.
    """
    from .models import MealException

    if kind is None:
        kind = MealException.Kind.ONE_TIME_ELIGIBLE
    return MealException.objects.create(
        person=person,
        kind=kind,
        effective_date=effective_date,
        end_date=None,
        meal_plan=meal_plan,
        reason_code=reason_code or "",
        reason_notes=reason_notes or "",
        approved_by=approved_by,
        is_active=True,
    )


def resolve_eligibility(*, person, on_date, resolved_by=None) -> "MealEligibility":
    """Resolve and persist the canonical eligibility for ``(person, on_date)``.

    Implements §13 steps 1–6 and 8 of the resolver, restricted to the
    Phase 2 scope:

    1. Find candidate subscriptions (ACTIVE/FUTURE/PAUSED covering on_date).
    2. Filter by period applicability — skipped (no meal_period in scope).
    3. Sort by priority (lower = higher priority).
    4. Evaluate in priority order (§13 step 4a only in Phase 2):
       - DATE_RANGE primary: ELIGIBLE with ``final_charge_iqd = 0``
         (``reason_code = date_range_no_charge``). Stop.
       - WALLET plan: in Phase 2 we **cannot** charge the wallet (no
         finance integration). The WALLET plan is recognized as a valid
         subscription but the eligibility decision is NOT_ELIGIBLE
         with ``reason_code = "wallet_mode_not_implemented"`` so that
         service events are not created until Phase 3. (Phase 3 will
         resolve the price + check_balance + finance.charge here and
         return ELIGIBLE.)
    5. No subscription granted eligibility → check MealException rows
       (ONE_TIME_ELIGIBLE / GUEST_ELIGIBLE → ELIGIBLE;
        TEMPORARY_DENY → OVERRIDDEN_DENIED).
    6. Default: NOT_ELIGIBLE (``reason_code = no_subscription``).
    7. Academic-presence/absence gates — NOT implemented in Phase 2
       (academics enrollment exists but absence comes from attendance,
       which is not integrated yet). Deferred to Phase 3.
    8. Write/upsert the :class:`MealEligibility` row.

    This function **writes** exactly one row per ``(person, on_date)``.
    It is the only writer for ``MealEligibility`` in Phase 2. The row
    is recalculable until a future ``MealServiceEvent`` references it;
    once referenced, it is frozen (enforced in Phase 3).

    ``resolve_eligibility`` is safe to call for display — its only side
    effect is the eligibility-row upsert.
    """
    from .models import MealEligibility, MealException, MealSubscription

    # ---------------------------------------------------------------
    # 1–3. Candidate subscriptions, ordered by priority.
    # ---------------------------------------------------------------
    candidates = list(
        active_subscriptions_for(person=person, on_date=on_date)
    )

    decision = MealEligibility.Decision.NOT_ELIGIBLE
    reason_code = REASON_NO_SUBSCRIPTION
    reason_notes = ""
    winning_subscription = None
    winning_plan = None

    # ---------------------------------------------------------------
    # 4. Evaluate in priority order (Phase 2: only DATE_RANGE grants
    #    eligibility; WALLET mode defers to Phase 3).
    # ---------------------------------------------------------------
    for sub in candidates:
        if sub.meal_plan is None:
            # Defensive: a subscription without a plan can't drive
            # eligibility in Phase 2.
            continue
        if sub.status == MealSubscription.Status.PAUSED:
            # A paused subscription never grants eligibility (§11.1).
            # It remains a candidate for reporting but does not win.
            continue
        if sub.meal_plan.mode == MealPlan.Mode.DATE_RANGE:
            # 4a. Date-range primary grants eligibility with no charge.
            decision = MealEligibility.Decision.ELIGIBLE
            reason_code = REASON_DATE_RANGE_NO_CHARGE
            reason_notes = "Date-range plan covers the meal; no wallet charge."
            winning_subscription = sub
            winning_plan = sub.meal_plan
            break
        if sub.meal_plan.mode == MealPlan.Mode.WALLET:
            # 4b. Wallet mode — Phase 3B territory (resolve_price +
            # finance.check_balance + finance.charge). The eligibility
            # decision is ELIGIBLE (the subscription entitles the person
            # to a meal); the *charging* decision (CONFIRMED vs UNPAID
            # vs DENIED on insufficient funds) is made by
            # ``resolve_service`` based on ``MealPlan.insufficient_funds_mode``.
            decision = MealEligibility.Decision.ELIGIBLE
            reason_code = "wallet_subscription"
            reason_notes = (
                "Wallet-mode subscription; charging resolved by resolve_service."
            )
            winning_subscription = sub
            winning_plan = sub.meal_plan
            break

    # ---------------------------------------------------------------
    # 5. No subscription granted eligibility → check exceptions.
    # ---------------------------------------------------------------
    if decision == MealEligibility.Decision.NOT_ELIGIBLE:
        exceptions = list(
            exceptions_for(person=person, on_date=on_date)
        )
        # TEMPORARY_DENY wins over ONE_TIME_ELIGIBLE / GUEST_ELIGIBLE
        # (matches §13 step 5 ordering: deny overrides eligible when
        # both exist on the same date — defensive).
        deny = next(
            (e for e in exceptions
             if e.kind == MealException.Kind.TEMPORARY_DENY),
            None,
        )
        if deny is not None:
            decision = MealEligibility.Decision.OVERRIDDEN_DENIED
            reason_code = REASON_TEMPORARY_DENY
            reason_notes = deny.reason_notes or "Temporary deny exception."
        else:
            eligible_ex = next(
                (e for e in exceptions if e.kind in (
                    MealException.Kind.ONE_TIME_ELIGIBLE,
                    MealException.Kind.GUEST_ELIGIBLE,
                )),
                None,
            )
            if eligible_ex is not None:
                decision = MealEligibility.Decision.ELIGIBLE
                if eligible_ex.kind == MealException.Kind.ONE_TIME_ELIGIBLE:
                    reason_code = REASON_ONE_TIME_ELIGIBLE
                else:
                    reason_code = REASON_GUEST_ELIGIBLE
                reason_notes = eligible_ex.reason_notes or ""
                winning_plan = eligible_ex.meal_plan

    # ---------------------------------------------------------------
    # 8. Upsert the canonical eligibility row.
    # ---------------------------------------------------------------
    from .validators import (
        validate_eligibility_not_frozen,
        validate_eligibility_unique_for_person_date,
    )

    existing = eligibility_for(person=person, on_date=on_date)
    if existing is None:
        validate_eligibility_unique_for_person_date(
            person=person, date=on_date
        )
        return MealEligibility.objects.create(
            person=person,
            student=getattr(person, "student_profile", None),
            date=on_date,
            decision=decision,
            subscription=winning_subscription,
            meal_plan=winning_plan,
            reason_code=reason_code,
            reason_notes=reason_notes,
            resolved_by=resolved_by,
        )
    # Freeze check: once a MealServiceEvent references this eligibility
    # row, it must not be mutated (§12.1). Corrections are audited via
    # MealSupervisorAction against the service event.
    validate_eligibility_not_frozen(eligibility=existing)
    # Update in place (the row is recalculable until referenced by a
    # MealServiceEvent — Phase 3B-1's resolve_service sets that FK).
    existing.decision = decision
    existing.subscription = winning_subscription
    existing.meal_plan = winning_plan
    existing.reason_code = reason_code
    existing.reason_notes = reason_notes
    existing.resolved_by = resolved_by
    existing.save(update_fields=[
        "decision", "subscription", "meal_plan",
        "reason_code", "reason_notes", "resolved_by",
    ])
    return existing


# ===========================================================================
# Phase 3B-1 — resolve_service (full §13 resolver with wallet charging)
# ===========================================================================

# Service-event reason codes (mirrors meals_domain_architecture.md §13).
REASON_SERVICE_NO_ELIGIBILITY = "no_eligibility"
REASON_SERVICE_DATE_RANGE_NO_CHARGE = "date_range_no_charge"
REASON_SERVICE_WALLET_CHARGED = "wallet_charged"
REASON_SERVICE_WALLET_UNPAID = "wallet_unpaid"
REASON_SERVICE_INSUFFICIENT_FUNDS = "insufficient_funds"
REASON_SERVICE_WALLET_ZERO_CHARGE = "wallet_zero_charge"

# Source-module tag used on finance.charge calls and on the
# WalletTransaction rows it creates.
MEALS_SOURCE_MODULE = "meals"


def _resolve_student_or_staff(person):
    """Return ``(student, staff)`` for a Person based on attached
    profiles. Used by ``resolve_service`` to populate the snapshot FKs
    on the service event."""
    student = getattr(person, "student_profile", None)
    staff = getattr(person, "staff_profile", None)
    return student, staff


def _build_service_event_kwargs(
    *,
    person,
    on_date,
    meal_period,
    eligibility,
    subscription,
    meal_plan,
    academic_year,
    created_by,
):
    """Build the common kwargs for a MealServiceEvent create/update."""
    student, staff = _resolve_student_or_staff(person)
    meal_period_label = ""
    if meal_period is not None:
        meal_period_label = meal_period.label or ""
    return {
        "person": person,
        "student": student,
        "staff": staff,
        "date": on_date,
        "eligibility": eligibility,
        "subscription": subscription,
        "meal_plan": meal_plan,
        "meal_period": meal_period,
        "meal_period_label_snapshot": meal_period_label,
        "served_by": created_by,
    }


@transaction.atomic
def resolve_service(
    *,
    person,
    on_date,
    meal_period=None,
    academic_year=None,
    recognition_event=None,
    created_by=None,
) -> MealServiceEvent:
    """Resolve meal service for ``(person, on_date, meal_period)``.

    Implements §13 steps 1–8 of the resolver, composing:

    * :func:`resolve_eligibility` — the eligibility decision.
    * :func:`resolve_price` — the price-resolution (base + override
      + discount = 0 stub until ``apps.discounts`` exists).
    * :func:`finance.check_balance` / :func:`finance.charge` — wallet
      charging for WALLET-mode plans.

    Behaviour:

    * **DATE_RANGE plan wins** → CONFIRMED service event with
      ``final_charge_iqd = 0`` and no wallet call.
    * **WALLET plan wins** → resolve price, then:
        - ``insufficient_funds_mode=DENY`` and insufficient → DENIED.
        - ``insufficient_funds_mode=ALLOW_UNPAID`` and insufficient →
          UNPAID (no ``wallet_transaction`` FK; ``final_charge_iqd``
          snapshotted but no balance impact).
        - ``insufficient_funds_mode=ALLOW_NEGATIVE`` and within credit
          limit → CONFIRMED with DEBIT (``finance.charge``).
        - sufficient → CONFIRMED with DEBIT (``finance.charge``).
    * **Exception (ONE_TIME / GUEST) wins** → if the exception has a
      wallet-mode ``meal_plan``, the wallet path runs; otherwise
      CONFIRMED with no charge.
    * **NOT_ELIGIBLE** → DENIED service event.
    * A ``MealSupervisorAction(action=CONFIRM or DENY)`` audit row is
      written in the same transaction as the state change.
    * The :class:`MealEligibility` row is referenced (FK) by the
      service event, freezing it (§12.1). Subsequent
      ``resolve_eligibility`` calls for the same ``(person, date)``
      will raise ``ValidationError``.

    If a PENDING service event already exists for
    ``(person, date, meal_period)``, it is updated in place; otherwise
    a new one is created. Terminal-status events are never mutated
    (§12 immutability).

    Returns the :class:`MealServiceEvent`.
    """
    # ---------------------------------------------------------------
    # Step 1: resolve eligibility (writes/upserts the eligibility row).
    # ---------------------------------------------------------------
    eligibility = resolve_eligibility(
        person=person, on_date=on_date, resolved_by=created_by
    )

    # Locate or create the service event shell (PENDING).
    event = pending_service_event_for(
        person=person, date=on_date, meal_period=meal_period
    )

    common_kwargs = _build_service_event_kwargs(
        person=person,
        on_date=on_date,
        meal_period=meal_period,
        eligibility=eligibility,
        subscription=eligibility.subscription,
        meal_plan=eligibility.meal_plan,
        academic_year=academic_year,
        created_by=created_by,
    )

    # ---------------------------------------------------------------
    # Step 2: NOT_ELIGIBLE / OVERRIDDEN_DENIED → DENIED event.
    # ---------------------------------------------------------------
    if eligibility.decision in (
        MealEligibility.Decision.NOT_ELIGIBLE,
        MealEligibility.Decision.OVERRIDDEN_DENIED,
    ):
        return _finalize_denied_event(
            event=event,
            common_kwargs=common_kwargs,
            eligibility=eligibility,
            reason_code=eligibility.reason_code or REASON_SERVICE_NO_ELIGIBILITY,
            reason_notes=eligibility.reason_notes,
            recognition_event=recognition_event,
            created_by=created_by,
        )

    # ---------------------------------------------------------------
    # Step 3: ELIGIBLE. Decide based on the winning plan's mode.
    # ---------------------------------------------------------------
    meal_plan = eligibility.meal_plan

    # No plan attached (e.g. a guest exception without a plan) →
    # CONFIRMED with no charge.
    if meal_plan is None or meal_plan.mode == MealPlan.Mode.DATE_RANGE:
        return _finalize_date_range_event(
            event=event,
            common_kwargs=common_kwargs,
            eligibility=eligibility,
            meal_plan=meal_plan,
            reason_code=REASON_SERVICE_DATE_RANGE_NO_CHARGE,
            reason_notes="Date-range plan covers the meal; no wallet charge.",
            recognition_event=recognition_event,
            created_by=created_by,
        )

    # ---------------------------------------------------------------
    # Step 4: WALLET mode — resolve price, then charge the wallet.
    # ---------------------------------------------------------------
    return _resolve_wallet_mode(
        event=event,
        common_kwargs=common_kwargs,
        person=person,
        on_date=on_date,
        meal_period=meal_period,
        meal_plan=meal_plan,
        eligibility=eligibility,
        academic_year=academic_year,
        recognition_event=recognition_event,
        created_by=created_by,
    )


# ---------------------------------------------------------------------------
# Internal helpers for resolve_service
# ---------------------------------------------------------------------------


def _create_or_update_pending_event(
    *,
    event,
    common_kwargs,
    recognition_event=None,
    status=MealServiceEvent.Status.PENDING,
    price_base_iqd=0,
    price_override_iqd=0,
    discount_iqd=0,
    final_charge_iqd=0,
    price_resolution_source="",
    wallet_balance_before_iqd=0,
    wallet_balance_after_iqd=0,
    wallet_transaction=None,
    wallet_refund_transaction=None,
    reason_code="",
    reason_notes="",
    served_at=None,
):
    """Create a new MealServiceEvent or update an existing PENDING one
    in place. Terminal events are never mutated (immutability — §12)."""
    fields = {
        **common_kwargs,
        "status": status,
        "price_base_iqd": price_base_iqd,
        "price_override_iqd": price_override_iqd,
        "discount_iqd": discount_iqd,
        "final_charge_iqd": final_charge_iqd,
        "price_resolution_source": price_resolution_source,
        "wallet_balance_before_iqd": wallet_balance_before_iqd,
        "wallet_balance_after_iqd": wallet_balance_after_iqd,
        "wallet_transaction": wallet_transaction,
        "wallet_refund_transaction": wallet_refund_transaction,
        "reason_code": reason_code,
        "reason_notes": reason_notes,
        "served_at": served_at,
        "recognition_event": recognition_event,
    }

    if event is None:
        return MealServiceEvent.objects.create(**fields)

    # Update the PENDING event in place. Per §12, PENDING events may be
    # freely mutated (only terminal statuses are frozen).
    for key, value in fields.items():
        setattr(event, key, value)
    event.save(update_fields=list(fields.keys()))
    return event


def _finalize_denied_event(
    *,
    event,
    common_kwargs,
    eligibility,
    reason_code,
    reason_notes,
    recognition_event=None,
    created_by=None,
) -> MealServiceEvent:
    """Write a DENIED service event + a MealSupervisorAction(DENY) row."""
    ev = _create_or_update_pending_event(
        event=event,
        common_kwargs=common_kwargs,
        recognition_event=recognition_event,
        status=MealServiceEvent.Status.DENIED,
        reason_code=reason_code,
        reason_notes=reason_notes,
    )
    MealSupervisorAction.objects.create(
        service_event=ev,
        action=MealSupervisorAction.Action.DENY,
        reason_code=reason_code,
        reason_notes=reason_notes,
        performed_by_user=created_by,
    )
    return ev


def _finalize_date_range_event(
    *,
    event,
    common_kwargs,
    eligibility,
    meal_plan,
    reason_code,
    reason_notes,
    recognition_event=None,
    created_by=None,
) -> MealServiceEvent:
    """Write a CONFIRMED no-charge service event for a DATE_RANGE plan
    + a MealSupervisorAction(CONFIRM) row."""
    ev = _create_or_update_pending_event(
        event=event,
        common_kwargs=common_kwargs,
        recognition_event=recognition_event,
        status=MealServiceEvent.Status.CONFIRMED,
        price_base_iqd=0,
        price_override_iqd=0,
        discount_iqd=0,
        final_charge_iqd=0,
        price_resolution_source=reason_code,
        reason_code=reason_code,
        reason_notes=reason_notes,
        served_at=timezone.now(),
    )
    MealSupervisorAction.objects.create(
        service_event=ev,
        action=MealSupervisorAction.Action.CONFIRM,
        reason_code=reason_code,
        reason_notes=reason_notes,
        performed_by_user=created_by,
    )
    return ev


def _resolve_wallet_mode(
    *,
    event,
    common_kwargs,
    person,
    on_date,
    meal_period,
    meal_plan,
    eligibility,
    academic_year,
    recognition_event=None,
    created_by=None,
) -> MealServiceEvent:
    """Resolve a WALLET-mode plan: price → check_balance → charge.

    Implements §13 step 4b."""
    # 4b.1 Resolve the price (pure function — no wallet calls).
    if meal_period is not None:
        base, override_delta, source = resolve_price(
            person=person,
            meal_plan=meal_plan,
            meal_period=meal_period,
            on_date=on_date,
        )
    else:
        # No meal_period supplied — fall back to the plan default.
        base = meal_plan.default_price_iqd
        override_delta = 0
        source = SOURCE_DEFAULT if base > 0 else SOURCE_DEFAULT_ZERO

    # Until apps.discounts exists, discount_iqd = 0 (§19.6).
    discount_iqd = 0
    final_charge_iqd = max(0, base - discount_iqd)

    # 4b.2 Check balance via finance (no wallet mutation).
    from apps.finance.services import check_balance as finance_check_balance

    balance, sufficient = finance_check_balance(
        person=person, amount_iqd=final_charge_iqd
    )

    insufficient_funds_mode = meal_plan.insufficient_funds_mode

    # 4b.3 Decide per MealPlan.insufficient_funds_mode.
    if final_charge_iqd == 0:
        # Zero charge — confirm with no wallet call.
        return _finalize_wallet_zero_charge(
            event=event,
            common_kwargs=common_kwargs,
            base=base,
            override_delta=override_delta,
            discount_iqd=discount_iqd,
            final_charge_iqd=0,
            source=source,
            recognition_event=recognition_event,
            created_by=created_by,
        )

    if sufficient or (
        insufficient_funds_mode == MealPlan.InsufficientFundsMode.ALLOW_NEGATIVE
    ):
        # Sufficient funds (or ALLOW_NEGATIVE within credit limit) → charge.
        return _finalize_wallet_charged(
            event=event,
            common_kwargs=common_kwargs,
            person=person,
            meal_plan=meal_plan,
            academic_year=academic_year,
            base=base,
            override_delta=override_delta,
            discount_iqd=discount_iqd,
            final_charge_iqd=final_charge_iqd,
            source=source,
            balance_before=balance,
            recognition_event=recognition_event,
            created_by=created_by,
        )

    # Insufficient funds.
    if insufficient_funds_mode == MealPlan.InsufficientFundsMode.ALLOW_UNPAID:
        return _finalize_wallet_unpaid(
            event=event,
            common_kwargs=common_kwargs,
            base=base,
            override_delta=override_delta,
            discount_iqd=discount_iqd,
            final_charge_iqd=final_charge_iqd,
            source=source,
            balance_before=balance,
            recognition_event=recognition_event,
            created_by=created_by,
        )

    # DENY mode (default) → DENIED event.
    return _finalize_denied_event(
        event=event,
        common_kwargs=common_kwargs,
        eligibility=eligibility,
        reason_code=REASON_SERVICE_INSUFFICIENT_FUNDS,
        reason_notes=(
            f"Insufficient funds: balance {balance}, required {final_charge_iqd}."
        ),
        recognition_event=recognition_event,
        created_by=created_by,
    )


def _finalize_wallet_charged(
    *,
    event,
    common_kwargs,
    person,
    meal_plan,
    academic_year,
    base,
    override_delta,
    discount_iqd,
    final_charge_iqd,
    source,
    balance_before,
    recognition_event=None,
    created_by=None,
) -> MealServiceEvent:
    """Charge the wallet via finance.charge and write a CONFIRMED
    service event with the wallet-transaction FK + balance snapshots."""
    from apps.finance.services import charge as finance_charge

    # 4b.3.1 Charge (DEBIT). If insufficient (ALLOW_NEGATIVE beyond
    # limit), finance.charge raises ValidationError — we let it
    # propagate so the caller sees the finance-layer rejection.
    tx = finance_charge(
        person=person,
        amount_iqd=final_charge_iqd,
        source_module=MEALS_SOURCE_MODULE,
        reference_type="MealServiceEvent",
        reference_id=None,  # set below after the event is persisted
        academic_year=academic_year,
        description=f"meal_lunch base={base} discount={discount_iqd}",
        created_by=created_by,
    )

    ev = _create_or_update_pending_event(
        event=event,
        common_kwargs=common_kwargs,
        recognition_event=recognition_event,
        status=MealServiceEvent.Status.CONFIRMED,
        price_base_iqd=base,
        price_override_iqd=override_delta,
        discount_iqd=discount_iqd,
        final_charge_iqd=final_charge_iqd,
        price_resolution_source=source,
        wallet_balance_before_iqd=tx.balance_before_iqd,
        wallet_balance_after_iqd=tx.balance_after_iqd,
        wallet_transaction=tx,
        reason_code=REASON_SERVICE_WALLET_CHARGED,
        reason_notes=f"Charged {final_charge_iqd} IQD (tx #{tx.pk}).",
        served_at=timezone.now(),
    )

    # The ledger row's reference_id could not be set to the event PK at
    # charge time (the event didn't exist yet). The ledger row is
    # immutable, so we cannot update it now; the reverse relationship
    # (MealServiceEvent.wallet_transaction → WalletTransaction) is the
    # canonical link. Finance's generic reference_type="MealServiceEvent"
    # + reference_id=None is acceptable for the audit trail.

    MealSupervisorAction.objects.create(
        service_event=ev,
        action=MealSupervisorAction.Action.CONFIRM,
        reason_code=REASON_SERVICE_WALLET_CHARGED,
        reason_notes=f"Wallet charged {final_charge_iqd} IQD.",
        performed_by_user=created_by,
    )
    return ev


def _finalize_wallet_unpaid(
    *,
    event,
    common_kwargs,
    base,
    override_delta,
    discount_iqd,
    final_charge_iqd,
    source,
    balance_before,
    recognition_event=None,
    created_by=None,
) -> MealServiceEvent:
    """Write an UNPAID service event (insufficient funds allowed)."""
    ev = _create_or_update_pending_event(
        event=event,
        common_kwargs=common_kwargs,
        recognition_event=recognition_event,
        status=MealServiceEvent.Status.UNPAID,
        price_base_iqd=base,
        price_override_iqd=override_delta,
        discount_iqd=discount_iqd,
        final_charge_iqd=final_charge_iqd,
        price_resolution_source=source,
        wallet_balance_before_iqd=balance_before,
        wallet_balance_after_iqd=balance_before,  # no balance impact
        wallet_transaction=None,
        reason_code=REASON_SERVICE_WALLET_UNPAID,
        reason_notes=(
            f"Insufficient funds allowed as UNPAID; intended charge {final_charge_iqd} IQD."
        ),
    )
    MealSupervisorAction.objects.create(
        service_event=ev,
        action=MealSupervisorAction.Action.CONFIRM,
        reason_code=REASON_SERVICE_WALLET_UNPAID,
        reason_notes="Insufficient funds; recorded as UNPAID.",
        performed_by_user=created_by,
    )
    return ev


def _finalize_wallet_zero_charge(
    *,
    event,
    common_kwargs,
    base,
    override_delta,
    discount_iqd,
    final_charge_iqd,
    source,
    recognition_event=None,
    created_by=None,
) -> MealServiceEvent:
    """Write a CONFIRMED service event for a zero-charge wallet plan
    (e.g. default_price_iqd=0). No wallet call."""
    ev = _create_or_update_pending_event(
        event=event,
        common_kwargs=common_kwargs,
        recognition_event=recognition_event,
        status=MealServiceEvent.Status.CONFIRMED,
        price_base_iqd=base,
        price_override_iqd=override_delta,
        discount_iqd=discount_iqd,
        final_charge_iqd=final_charge_iqd,
        price_resolution_source=source,
        wallet_balance_before_iqd=0,
        wallet_balance_after_iqd=0,
        wallet_transaction=None,
        reason_code=REASON_SERVICE_WALLET_ZERO_CHARGE,
        reason_notes="Wallet plan with zero charge; no wallet call.",
        served_at=timezone.now(),
    )
    MealSupervisorAction.objects.create(
        service_event=ev,
        action=MealSupervisorAction.Action.CONFIRM,
        reason_code=REASON_SERVICE_WALLET_ZERO_CHARGE,
        reason_notes="Zero charge; no wallet call.",
        performed_by_user=created_by,
    )
    return ev


# ===========================================================================
# Phase 3B-2 — Supervisor workflow services
#
# Each operation:
#   1. Locks the MealServiceEvent row (select_for_update).
#   2. Validates the status transition + MealPlan supervisor-policy flags.
#   3. Calls finance services where needed (refund, void-with-reversal).
#   4. Transitions the event status.
#   5. Writes a MealSupervisorAction audit row in the same transaction.
#
# Immutable snapshot fields are never mutated (enforced by
# validate_service_event_snapshot_immutable). The only mutable snapshot
# field on a CONFIRMED event is wallet_refund_transaction (the refund
# flow sets it before transitioning to REFUNDED/VOIDED).
# ===========================================================================

# Reason codes for supervisor actions.
REASON_SUPERVISOR_CONFIRM = "supervisor_confirm"
REASON_SUPERVISOR_DENY = "supervisor_deny"
REASON_SUPERVISOR_UNCONFIRM = "supervisor_unconfirm"
REASON_SUPERVISOR_REFUND = "supervisor_refund"
REASON_SUPERVISOR_VOID = "supervisor_void"
REASON_SUPERVISOR_OVERRIDE_ELIGIBLE = "supervisor_override_eligible"
REASON_SUPERVISOR_OVERRIDE_DENIED = "supervisor_override_denied"


def _lock_service_event(service_event: MealServiceEvent) -> MealServiceEvent:
    """Re-load and lock the MealServiceEvent row for the duration of
    the transaction. Prevents concurrent supervisor actions from
    racing on the same event.

    Note: ``select_related`` is intentionally omitted because
    PostgreSQL does not allow ``FOR UPDATE`` on the nullable side of
    an outer join (several meals FKs are nullable). Callers that need
    related objects should access them via the locked instance's
    already-loaded FKs (Django lazy-loads them safely).
    """
    return (
        MealServiceEvent.objects.select_for_update()
        .get(pk=service_event.pk)
    )


def _write_audit(
    *,
    service_event,
    action: str,
    reason_code: str,
    reason_notes: str = "",
    performed_by=None,
    performed_by_user=None,
) -> MealSupervisorAction:
    """Write a MealSupervisorAction audit row."""
    return MealSupervisorAction.objects.create(
        service_event=service_event,
        action=action,
        reason_code=reason_code or "",
        reason_notes=reason_notes or "",
        performed_by=performed_by,
        performed_by_user=performed_by_user,
    )


@transaction.atomic
def confirm_service_event(
    *,
    service_event: MealServiceEvent,
    performed_by=None,
    performed_by_user=None,
    reason_code: str = "",
    reason_notes: str = "",
) -> MealServiceEvent:
    """Confirm a PENDING or UNPAID service event.

    For a PENDING wallet-mode event: this is where the supervisor
    manually confirms the charge. ``finance.charge()`` is called if
    the event has no ``wallet_transaction`` yet (the charge was not
    settled by ``resolve_service`` — e.g. an UNPAID event the
    supervisor is now resolving after a manual payment).

    For a PENDING DATE_RANGE event: transitions directly to CONFIRMED
    (no wallet call).

    Enforces ``MealPlan.allow_supervisor_confirm``.
    """
    from .validators import (
        validate_service_event_status_transition,
        validate_supervisor_can_confirm,
        validate_supervisor_override_reason,
    )

    ev = _lock_service_event(service_event)
    meal_plan = ev.meal_plan

    validate_service_event_status_transition(
        current_status=ev.status, new_status=MealServiceEvent.Status.CONFIRMED
    )
    validate_supervisor_can_confirm(meal_plan=meal_plan, reason_code=reason_code)
    validate_supervisor_override_reason(
        meal_plan=meal_plan, reason_code=reason_code
    )

    # If the event was UNPAID (intended charge snapshotted but no debit),
    # attempt the charge now — the supervisor may have topped up the wallet.
    if (
        ev.status == MealServiceEvent.Status.UNPAID
        and ev.wallet_transaction is None
        and ev.final_charge_iqd > 0
    ):
        from apps.finance.services import charge as finance_charge

        tx = finance_charge(
            person=ev.person,
            amount_iqd=ev.final_charge_iqd,
            source_module=MEALS_SOURCE_MODULE,
            reference_type="MealServiceEvent",
            reference_id=ev.pk,
            description=f"supervisor_confirm base={ev.price_base_iqd}",
            created_by=performed_by_user,
        )
        # wallet_transaction is being populated for the first time (was None).
        # Use update_fields to narrow the immutability check.
        ev.wallet_transaction = tx
        ev.wallet_balance_before_iqd = tx.balance_before_iqd
        ev.wallet_balance_after_iqd = tx.balance_after_iqd
        ev.save(update_fields=[
            "wallet_transaction",
            "wallet_balance_before_iqd",
            "wallet_balance_after_iqd",
        ])

    ev.status = MealServiceEvent.Status.CONFIRMED
    ev.reason_code = reason_code or REASON_SUPERVISOR_CONFIRM
    ev.reason_notes = reason_notes or ""
    if not ev.served_at:
        ev.served_at = timezone.now()
    ev.save(update_fields=["status", "reason_code", "reason_notes", "served_at", "updated_at"])

    _write_audit(
        service_event=ev,
        action=MealSupervisorAction.Action.CONFIRM,
        reason_code=reason_code or REASON_SUPERVISOR_CONFIRM,
        reason_notes=reason_notes,
        performed_by=performed_by,
        performed_by_user=performed_by_user,
    )
    return ev


@transaction.atomic
def deny_service_event(
    *,
    service_event: MealServiceEvent,
    performed_by=None,
    performed_by_user=None,
    reason_code: str = "",
    reason_notes: str = "",
) -> MealServiceEvent:
    """Deny a PENDING service event.

    Enforces ``MealPlan.allow_supervisor_confirm`` (deny is the
    inverse of confirm — gated by the same flag).
    """
    from .validators import (
        validate_service_event_status_transition,
        validate_supervisor_can_confirm,
        validate_supervisor_override_reason,
    )

    ev = _lock_service_event(service_event)
    meal_plan = ev.meal_plan

    validate_service_event_status_transition(
        current_status=ev.status, new_status=MealServiceEvent.Status.DENIED
    )
    # Deny uses the confirm flag (supervisor has authority to decide).
    validate_supervisor_can_confirm(meal_plan=meal_plan, reason_code=reason_code)
    validate_supervisor_override_reason(
        meal_plan=meal_plan, reason_code=reason_code
    )

    ev.status = MealServiceEvent.Status.DENIED
    ev.reason_code = reason_code or REASON_SUPERVISOR_DENY
    ev.reason_notes = reason_notes or ""
    ev.save(update_fields=["status", "reason_code", "reason_notes", "updated_at"])

    _write_audit(
        service_event=ev,
        action=MealSupervisorAction.Action.DENY,
        reason_code=reason_code or REASON_SUPERVISOR_DENY,
        reason_notes=reason_notes,
        performed_by=performed_by,
        performed_by_user=performed_by_user,
    )
    return ev


@transaction.atomic
def unconfirm_service_event(
    *,
    service_event: MealServiceEvent,
    performed_by=None,
    performed_by_user=None,
    reason_code: str = "",
    reason_notes: str = "",
) -> MealServiceEvent:
    """Unconfirm a CONFIRMED service event.

    Reverses the wallet charge (if any) via ``finance.refund()``, then
    transitions the event back to PENDING. The ``wallet_refund_transaction``
    FK is set (mutable on CONFIRMED per the immutability validator).

    Enforces ``MealPlan.allow_supervisor_unconfirm`` and
    ``MealPlan.require_reason_on_unconfirm``.
    """
    from .validators import (
        validate_service_event_status_transition,
        validate_supervisor_can_unconfirm,
    )

    ev = _lock_service_event(service_event)
    meal_plan = ev.meal_plan

    validate_service_event_status_transition(
        current_status=ev.status, new_status=MealServiceEvent.Status.PENDING
    )
    validate_supervisor_can_unconfirm(meal_plan=meal_plan, reason_code=reason_code)

    # Reverse the wallet charge if one exists.
    if ev.wallet_transaction is not None:
        from apps.finance.services import refund as finance_refund

        refund_tx = finance_refund(
            person=ev.person,
            original_transaction=ev.wallet_transaction,
            reason_code=reason_code or REASON_SUPERVISOR_UNCONFIRM,
            approved_by=performed_by,
            created_by=performed_by_user,
        )
        # wallet_refund_transaction is mutable on CONFIRMED.
        ev.wallet_refund_transaction = refund_tx

    ev.status = MealServiceEvent.Status.PENDING
    ev.reason_code = reason_code or REASON_SUPERVISOR_UNCONFIRM
    ev.reason_notes = reason_notes or ""
    ev.save(update_fields=[
        "status", "reason_code", "reason_notes",
        "wallet_refund_transaction", "updated_at",
    ])

    _write_audit(
        service_event=ev,
        action=MealSupervisorAction.Action.UNCONFIRM,
        reason_code=reason_code or REASON_SUPERVISOR_UNCONFIRM,
        reason_notes=reason_notes,
        performed_by=performed_by,
        performed_by_user=performed_by_user,
    )
    return ev


@transaction.atomic
def refund_service_event(
    *,
    service_event: MealServiceEvent,
    amount_iqd=None,
    performed_by=None,
    performed_by_user=None,
    reason_code: str = "",
    reason_notes: str = "",
) -> MealServiceEvent:
    """Refund a CONFIRMED (charged) service event.

    Calls ``finance.refund()`` to reverse the original DEBIT, stores
    the refund ``WalletTransaction`` FK on
    ``MealServiceEvent.wallet_refund_transaction``, and transitions
    the event to REFUNDED.

    Enforces ``MealPlan.allow_supervisor_refund`` and
    ``MealPlan.require_reason_on_refund``.
    """
    from .validators import (
        validate_service_event_status_transition,
        validate_supervisor_can_refund,
    )

    ev = _lock_service_event(service_event)
    meal_plan = ev.meal_plan

    validate_service_event_status_transition(
        current_status=ev.status, new_status=MealServiceEvent.Status.REFUNDED
    )
    validate_supervisor_can_refund(meal_plan=meal_plan, reason_code=reason_code)

    if ev.wallet_transaction is None:
        raise ValidationError(
            "Cannot refund a service event with no wallet transaction."
        )

    from apps.finance.services import refund as finance_refund

    refund_tx = finance_refund(
        person=ev.person,
        original_transaction=ev.wallet_transaction,
        amount_iqd=amount_iqd,
        reason_code=reason_code or REASON_SUPERVISOR_REFUND,
        approved_by=performed_by,
        created_by=performed_by_user,
    )
    # wallet_refund_transaction is mutable on CONFIRMED.
    ev.wallet_refund_transaction = refund_tx
    ev.status = MealServiceEvent.Status.REFUNDED
    ev.reason_code = reason_code or REASON_SUPERVISOR_REFUND
    ev.reason_notes = reason_notes or ""
    ev.reversed_at = timezone.now()
    ev.reversed_by = performed_by_user
    ev.save(update_fields=[
        "status", "reason_code", "reason_notes",
        "wallet_refund_transaction", "reversed_at", "reversed_by",
        "updated_at",
    ])

    _write_audit(
        service_event=ev,
        action=MealSupervisorAction.Action.REFUND,
        reason_code=reason_code or REASON_SUPERVISOR_REFUND,
        reason_notes=reason_notes,
        performed_by=performed_by,
        performed_by_user=performed_by_user,
    )
    return ev


@transaction.atomic
def void_service_event(
    *,
    service_event: MealServiceEvent,
    performed_by=None,
    performed_by_user=None,
    reason_code: str = "",
    reason_notes: str = "",
) -> MealServiceEvent:
    """Void a service event.

    For a CONFIRMED event with a wallet charge: reverses the charge
    via ``finance.refund()`` before transitioning to VOIDED. The
    ``wallet_refund_transaction`` FK is set.

    For PENDING / UNPAID / DENIED events: transitions directly to
    VOIDED with no wallet call.

    Enforces ``MealPlan.allow_supervisor_refund`` when a charge
    reversal is needed (voiding a charged event is effectively a
    refund). For non-charged voids, no supervisor-policy flag is
    checked (voiding a pending event is an admin correction).
    """
    from .validators import (
        validate_service_event_status_transition,
        validate_supervisor_can_refund,
    )

    ev = _lock_service_event(service_event)
    meal_plan = ev.meal_plan

    validate_service_event_status_transition(
        current_status=ev.status, new_status=MealServiceEvent.Status.VOIDED
    )

    # If the event has a wallet charge, reverse it before voiding.
    if ev.wallet_transaction is not None:
        # Voiding a charged event is effectively a refund — enforce
        # the refund policy flags.
        validate_supervisor_can_refund(meal_plan=meal_plan, reason_code=reason_code)

        from apps.finance.services import refund as finance_refund

        refund_tx = finance_refund(
            person=ev.person,
            original_transaction=ev.wallet_transaction,
            reason_code=reason_code or REASON_SUPERVISOR_VOID,
            approved_by=performed_by,
            created_by=performed_by_user,
        )
        # wallet_refund_transaction is mutable on CONFIRMED.
        ev.wallet_refund_transaction = refund_tx

    ev.status = MealServiceEvent.Status.VOIDED
    ev.reason_code = reason_code or REASON_SUPERVISOR_VOID
    ev.reason_notes = reason_notes or ""
    ev.reversed_at = timezone.now()
    ev.reversed_by = performed_by_user
    ev.save(update_fields=[
        "status", "reason_code", "reason_notes",
        "wallet_refund_transaction", "reversed_at", "reversed_by",
        "updated_at",
    ])

    _write_audit(
        service_event=ev,
        action=MealSupervisorAction.Action.VOID,
        reason_code=reason_code or REASON_SUPERVISOR_VOID,
        reason_notes=reason_notes,
        performed_by=performed_by,
        performed_by_user=performed_by_user,
    )
    return ev


@transaction.atomic
def supervisor_override(
    *,
    person,
    on_date,
    decision: str,
    reason_code: str = "",
    reason_notes: str = "",
    performed_by=None,
    performed_by_user=None,
    meal_plan=None,
) -> MealEligibility:
    """Override the eligibility decision for ``(person, on_date)``.

    Writes a ``MealSupervisorAction(action=OVERRIDE_ELIGIBLE or
    OVERRIDE_DENIED)`` audit row, then upserts the
    :class:`MealEligibility` row with the overridden decision.

    If the eligibility row is already frozen (referenced by a
    ``MealServiceEvent``), the override is rejected — corrections must
    be audited against the service event, not the eligibility row.

    Enforces ``MealPlan.require_reason_on_override``.
    """
    from .models import MealEligibility
    from .validators import (
        validate_eligibility_not_frozen,
        validate_supervisor_override_reason,
    )

    # Resolve the meal_plan from an existing eligibility row if not supplied.
    existing = eligibility_for(person=person, on_date=on_date)
    effective_plan = meal_plan or (existing.meal_plan if existing else None)

    validate_supervisor_override_reason(
        meal_plan=effective_plan, reason_code=reason_code
    )

    # The eligibility must not be frozen.
    if existing is not None:
        validate_eligibility_not_frozen(eligibility=existing)

    # Map the override decision to the MealEligibility.Decision enum.
    if decision in (
        MealEligibility.Decision.OVERRIDDEN_ELIGIBLE,
        MealEligibility.Decision.OVERRIDDEN_DENIED,
    ):
        override_decision = decision
    elif decision.lower() in ("eligible", "override_eligible"):
        override_decision = MealEligibility.Decision.OVERRIDDEN_ELIGIBLE
    elif decision.lower() in ("denied", "override_denied"):
        override_decision = MealEligibility.Decision.OVERRIDDEN_DENIED
    else:
        raise ValidationError(
            {"decision": f"Unknown override decision: {decision!r}."}
        )

    action = (
        MealSupervisorAction.Action.OVERRIDE_ELIGIBLE
        if override_decision == MealEligibility.Decision.OVERRIDDEN_ELIGIBLE
        else MealSupervisorAction.Action.OVERRIDE_DENIED
    )

    default_reason = (
        REASON_SUPERVISOR_OVERRIDE_ELIGIBLE
        if override_decision == MealEligibility.Decision.OVERRIDDEN_ELIGIBLE
        else REASON_SUPERVISOR_OVERRIDE_DENIED
    )

    # Upsert the eligibility row.
    if existing is None:
        existing = MealEligibility.objects.create(
            person=person,
            student=getattr(person, "student_profile", None),
            date=on_date,
            decision=override_decision,
            meal_plan=effective_plan,
            reason_code=reason_code or default_reason,
            reason_notes=reason_notes or "",
            resolved_by=performed_by_user,
        )
    else:
        existing.decision = override_decision
        existing.meal_plan = effective_plan
        existing.reason_code = reason_code or default_reason
        existing.reason_notes = reason_notes or ""
        existing.resolved_by = performed_by_user
        existing.save(update_fields=[
            "decision", "meal_plan", "reason_code", "reason_notes",
            "resolved_by",
        ])

    # Write the audit row against the eligibility (not a service event).
    MealSupervisorAction.objects.create(
        eligibility=existing,
        action=action,
        reason_code=reason_code or default_reason,
        reason_notes=reason_notes or "",
        performed_by=performed_by,
        performed_by_user=performed_by_user,
    )
    return existing
