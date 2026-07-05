"""Meals-domain services — Phase 1 (pricing) and Phase 2 (subscriptions
+ eligibility).

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
It does **not** perform wallet-mode charging (§13 step 4b) — that is
Phase 3's ``resolve_service``.

Out of scope (Phase 3+):
* ``MealServiceEvent`` / ``MealSupervisorAction`` /
  ``resolve_service`` / wallet charging / finance calls /
  supervisor workflow / legacy migration / discounts.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from django.db import transaction

from .models import MealPeriod, MealPeriodPrice, MealPersonPriceOverride, MealPlan, MealSubscription
from .selectors import (
    active_override_for_date,
    active_subscriptions_for,
    eligibility_for,
    exceptions_for,
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
            # 4b. Wallet mode — Phase 3 territory (resolve_price +
            # finance.check_balance + finance.charge). In Phase 2 we
            # cannot charge, so we do NOT grant eligibility; the next
            # candidate (e.g. a lower-priority DATE_RANGE fallback) is
            # evaluated. If no other candidate grants eligibility, the
            # final decision stays NOT_ELIGIBLE with this reason.
            if decision == MealEligibility.Decision.NOT_ELIGIBLE:
                reason_code = "wallet_mode_not_implemented"
                reason_notes = (
                    "Wallet-mode charging is implemented in Phase 3; "
                    "no eligibility granted in Phase 2 for wallet plans."
                )
            # Continue to next candidate.
            continue

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
    from .validators import validate_eligibility_unique_for_person_date

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
    # Update in place (the row is recalculable until referenced by a
    # future MealServiceEvent — Phase 3 will enforce freezing).
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
