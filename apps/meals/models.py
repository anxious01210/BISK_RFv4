"""Meals domain models — Phase 1 (pricing & period foundation)
and Phase 2 (subscriptions, exceptions, eligibility).

Phase 1 scope:
* :class:`MealPeriod` — a meal-serving window (wraps a generic period
  template reference until ``apps.scheduler.PeriodTemplate`` exists).
* :class:`MealPlan` — a reusable meal product definition (mode, policy
  flags, default price). Migration target of legacy
  ``attendance.MealProfile``.
* :class:`MealPeriodPrice` — the list price for a ``(MealPlan,
  MealPeriod)`` pair. Migration target of legacy
  ``attendance.MealProfilePeriod``.
* :class:`MealPersonPriceOverride` — a granular per-Person price
  override, optionally scoped to a specific ``MealPeriod``.

Phase 2 scope:
* :class:`MealSubscription` — a Person's dated entitlement to meals
  under a :class:`MealPlan` (date-range vs wallet). Supports students
  and staff, and primary + fallback combinations.
* :class:`MealException` — a one-time / temporary meal permission or
  denial (not a subscription).
* :class:`MealEligibility` — the canonical daily eligibility snapshot
  for ``(person, date)``. Recalculable until a future
  ``MealServiceEvent`` references it, then frozen.

Out of scope (later phases):
* ``MealServiceEvent``, ``MealSupervisorAction``, ``resolve_service``,
  wallet charging, supervisor workflow, legacy migration, discounts.
"""

from django.conf import settings
from django.db import models


class MealPeriod(models.Model):
    """A meal-serving window.

    This is the meal-domain declaration of *which* timetable periods
    serve meals and for *which* meal kind (lunch / breakfast / snack).
    It wraps a generic period-template reference rather than carrying a
    typed FK, because ``apps.scheduler.PeriodTemplate`` does not exist
    yet (the legacy ``attendance.PeriodTemplate`` is still the source of
    truth). See ``meals_domain_architecture.md`` §6.1, §9.

    Until ``apps.scheduler`` owns ``PeriodTemplate``, ``MealPeriod``
    carries the period identity as a generic
    ``(period_template_source, period_template_ref_id)`` pair plus a
    free-form ``label``. This keeps ``apps.meals`` from importing
    ``apps.attendance`` models.
    """

    class Kind(models.TextChoices):
        LUNCH = "lunch", "Lunch"
        BREAKFAST = "breakfast", "Breakfast"
        SNACK = "snack", "Snack"

    kind = models.CharField(
        max_length=20,
        choices=Kind.choices,
        default=Kind.LUNCH,
        db_index=True,
    )
    # Generic fallback used during the migration window only.
    # Typed FK to ``scheduler.PeriodTemplate`` is added once that model
    # exists; until then this pair carries the legacy period identity
    # without forcing ``apps.meals`` to import ``apps.attendance``.
    period_template_source = models.CharField(
        max_length=20,
        blank=True,
        default="",
        help_text=(
            "Source tag for the generic period-template reference "
            "(e.g. 'attendance'). Blank when no period template is linked."
        ),
    )
    period_template_ref_id = models.PositiveBigIntegerField(
        null=True,
        blank=True,
        help_text=(
            "Generic period-template reference id. Used together with "
            "period_template_source to identify the underlying timetable "
            "period without a typed FK."
        ),
    )
    label = models.CharField(
        max_length=64,
        blank=True,
        default="",
        help_text="Optional display override, e.g. 'First lunch block'.",
    )
    is_active = models.BooleanField(default=True, db_index=True)
    sort_order = models.PositiveSmallIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["kind", "sort_order", "id"]
        indexes = [
            models.Index(fields=["kind", "is_active"]),
            models.Index(fields=["period_template_source", "period_template_ref_id"]),
        ]
        constraints = [
            # Enforce one MealPeriod per (kind, period_template_source,
            # period_template_ref_id) when a generic reference is present.
            # NULL ref_id rows are allowed and not deduped (PostgreSQL
            # treats NULL as distinct in a unique constraint; a
            # conditional UniqueConstraint is used to keep the rule tight
            # for linked rows only).
            models.UniqueConstraint(
                fields=["kind", "period_template_source", "period_template_ref_id"],
                condition=models.Q(period_template_ref_id__isnull=False),
                name="meal_period_unique_kind_and_template_ref",
            ),
        ]

    def __str__(self) -> str:
        if self.label:
            return f"{self.get_kind_display()} — {self.label}"
        return f"{self.get_kind_display()} (#{self.pk})"

    def clean(self):
        super().clean()
        source = (self.period_template_source or "").strip()
        has_source = bool(source)
        has_ref = self.period_template_ref_id is not None
        if has_source != has_ref:
            from django.core.exceptions import ValidationError
            raise ValidationError(
                "period_template_source and period_template_ref_id must "
                "both be set or both be blank."
            )


class MealPlan(models.Model):
    """A reusable meal product definition.

    A named plan, its meal kind, its mode (date-range vs wallet),
    supervisor-override policy, insufficient-funds policy, and default
    price. This is the migration target for legacy
    ``attendance.MealProfile`` (see ``meals_domain_architecture.md``
    §10, §21.2).

    A ``MealPlan`` is a *product definition* — it does not reference
    ``Person``, ``StudentProfile``, or ``StaffProfile``. Per-period and
    per-Person pricing live on :class:`MealPeriodPrice` and
    :class:`MealPersonPriceOverride` respectively.
    """

    class Kind(models.TextChoices):
        LUNCH = "lunch", "Lunch"
        BREAKFAST = "breakfast", "Breakfast"
        SNACK = "snack", "Snack"

    class Mode(models.TextChoices):
        DATE_RANGE = "date_range", "Date-range"
        WALLET = "wallet", "Wallet"

    class InsufficientFundsMode(models.TextChoices):
        DENY = "deny", "Deny"
        ALLOW_UNPAID = "allow_unpaid", "Allow unpaid"
        ALLOW_NEGATIVE = "allow_negative", "Allow negative"

    name = models.CharField(max_length=100, unique=True)
    kind = models.CharField(
        max_length=20,
        choices=Kind.choices,
        default=Kind.LUNCH,
        db_index=True,
    )
    mode = models.CharField(
        max_length=20,
        choices=Mode.choices,
        default=Mode.DATE_RANGE,
        db_index=True,
    )
    is_active = models.BooleanField(default=True, db_index=True)

    # Supervisor-override policy (migrated from legacy MealProfile).
    allow_supervisor_confirm = models.BooleanField(default=True)
    allow_supervisor_unconfirm = models.BooleanField(default=False)
    allow_supervisor_refund = models.BooleanField(default=False)
    require_reason_on_override = models.BooleanField(default=True)
    require_reason_on_unconfirm = models.BooleanField(default=False)
    require_reason_on_refund = models.BooleanField(default=True)

    # Insufficient-funds policy (wallet mode only).
    insufficient_funds_mode = models.CharField(
        max_length=20,
        choices=InsufficientFundsMode.choices,
        default=InsufficientFundsMode.DENY,
    )
    credit_limit_iqd = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text=(
            "Only used when mode=Wallet and insufficient-funds mode "
            "= Allow negative. Blank = no negative limit."
        ),
    )

    # Default base price used when no MealPeriodPrice or
    # MealPersonPriceOverride matches (§18).
    default_price_iqd = models.IntegerField(
        default=0,
        help_text="Fallback base price (IQD) when no period or person override matches.",
    )

    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        indexes = [
            models.Index(fields=["kind", "is_active"]),
            models.Index(fields=["mode", "is_active"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(default_price_iqd__gte=0),
                name="meal_plan_default_price_non_negative",
            ),
        ]

    def __str__(self) -> str:
        return self.name


class MealPeriodPrice(models.Model):
    """The list price of a meal under a :class:`MealPlan` for a given
    :class:`MealPeriod`.

    Migration target for legacy ``attendance.MealProfilePeriod``. See
    ``meals_domain_architecture.md`` §16.

    ``price_iqd`` is the **list price** (the sticker price before any
    per-Person override or discount). It is owned by Meals because it
    is a meal-product fact, not a ledger fact.
    """

    meal_plan = models.ForeignKey(
        MealPlan,
        on_delete=models.CASCADE,
        related_name="period_prices",
    )
    meal_period = models.ForeignKey(
        MealPeriod,
        on_delete=models.CASCADE,
        related_name="plan_prices",
    )
    price_iqd = models.IntegerField(default=0)
    is_enabled = models.BooleanField(default=True, db_index=True)
    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [["meal_plan", "meal_period"]]
        ordering = ["meal_plan__name", "meal_period__sort_order"]
        indexes = [
            models.Index(fields=["meal_plan", "is_enabled"]),
            models.Index(fields=["meal_period", "is_enabled"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(price_iqd__gte=0),
                name="meal_period_price_non_negative",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.meal_plan} / {self.meal_period} = {self.price_iqd}"


class MealPersonPriceOverride(models.Model):
    """A granular price override for a specific :class:`~apps.identity.models.Person`
    under a :class:`MealPlan`, optionally scoped to a specific
    :class:`MealPeriod`.

    See ``meals_domain_architecture.md`` §15. ``meal_period`` may be
    null, in which case the override applies to *all* periods of the
    plan. The resolver picks the per-period override first (§14.1).

    Like :class:`MealPeriodPrice`, ``price_iqd`` is a meal-product fact
    owned by Meals; it is snapshotted (as ``price_override_iqd``) on the
    future immutable ``MealServiceEvent`` when an override wins.
    """

    person = models.ForeignKey(
        "identity.Person",
        on_delete=models.CASCADE,
        related_name="meal_price_overrides",
    )
    meal_plan = models.ForeignKey(
        MealPlan,
        on_delete=models.CASCADE,
        related_name="person_price_overrides",
    )
    meal_period = models.ForeignKey(
        MealPeriod,
        on_delete=models.CASCADE,
        related_name="person_price_overrides",
        null=True,
        blank=True,
        help_text="Blank = applies to all periods of this plan.",
    )
    price_iqd = models.IntegerField(default=0)
    is_enabled = models.BooleanField(default=True, db_index=True)
    reason_code = models.CharField(max_length=32, blank=True, default="")
    notes = models.CharField(max_length=200, blank=True, default="")
    effective_from = models.DateField(null=True, blank=True)
    effective_until = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["person", "meal_plan", "meal_period__sort_order"]
        indexes = [
            models.Index(fields=["person", "is_enabled"]),
            models.Index(fields=["meal_plan", "is_enabled"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(price_iqd__gte=0),
                name="meal_person_price_override_non_negative",
            ),
            # One override per (person, plan, period). PostgreSQL treats
            # NULL as distinct in a plain unique_together, so a person
            # may legitimately hold both an any-period override and a
            # per-period override for the same plan; the resolver picks
            # the per-period one first (§14.1).
            models.UniqueConstraint(
                fields=["person", "meal_plan", "meal_period"],
                name="meal_person_override_unique_per_plan_period",
            ),
            # An effective window, when both bounds are set, must be valid.
            models.CheckConstraint(
                check=(
                    models.Q(effective_until__isnull=True)
                    | models.Q(effective_from__isnull=True)
                    | models.Q(effective_until__gte=models.F("effective_from"))
                ),
                name="meal_person_override_effective_window_valid",
            ),
        ]

    def __str__(self) -> str:
        period_part = self.meal_period.label or f"#{self.meal_period_id}" if self.meal_period_id else "any period"
        return f"{self.person} / {self.meal_plan} / {period_part} = {self.price_iqd}"


# ===========================================================================
# Phase 2 — Subscriptions, Exceptions, Eligibility
# ===========================================================================


class MealSubscription(models.Model):
    """A Person's entitlement to meals under a :class:`MealPlan` for a
    date range.

    Migration target for legacy ``attendance.MealSubscription``. See
    ``meals_domain_architecture.md`` §11.

    Supports both students and staff (exactly one of ``student`` /
    ``staff`` is set for a non-guest subscription; ``person`` is always
    set), both date-range and wallet plans, and primary + fallback
    combinations via ``priority`` (lower number = higher priority =
    evaluated first by the resolver).

    The subscription stores **no** grade/section/name fields — those
    are read from the current enrollment/placement or snapshotted on
    the future immutable ``MealServiceEvent``.
    """

    class Status(models.TextChoices):
        FUTURE = "future", "Future"
        ACTIVE = "active", "Active"
        PAUSED = "paused", "Paused"
        EXPIRED = "expired", "Expired"
        CANCELLED = "cancelled", "Cancelled"

    person = models.ForeignKey(
        "identity.Person",
        on_delete=models.CASCADE,
        related_name="meal_subscriptions",
    )
    student = models.ForeignKey(
        "identity.StudentProfile",
        on_delete=models.CASCADE,
        related_name="meal_subscriptions",
        null=True,
        blank=True,
    )
    staff = models.ForeignKey(
        "identity.StaffProfile",
        on_delete=models.CASCADE,
        related_name="meal_subscriptions",
        null=True,
        blank=True,
    )
    meal_plan = models.ForeignKey(
        MealPlan,
        on_delete=models.PROTECT,
        related_name="subscriptions",
        null=True,
        blank=True,
    )
    academic_year = models.ForeignKey(
        "academics.AcademicYear",
        on_delete=models.PROTECT,
        related_name="meal_subscriptions",
        null=True,
        blank=True,
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
        db_index=True,
    )
    start_date = models.DateField(db_index=True)
    end_date = models.DateField(db_index=True)
    plan_type = models.CharField(
        max_length=20,
        default="monthly",
        help_text="Reporting label (annual/monthly/other). Does not drive charging.",
    )
    source = models.CharField(max_length=40, default="manual", db_index=True)
    priority = models.PositiveSmallIntegerField(
        default=1,
        db_index=True,
        help_text=(
            "Order number. Lower = higher priority = evaluated first. "
            "1=primary, 2=fallback. Same-priority overlapping ACTIVE "
            "subscriptions for the same person are rejected."
        ),
    )
    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["person", "priority", "start_date", "id"]
        indexes = [
            models.Index(fields=["person", "status", "start_date", "end_date"]),
            models.Index(fields=["student", "status"]),
            models.Index(fields=["staff", "status"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(end_date__gte=models.F("start_date")),
                name="meal_subscription_end_after_start",
            ),
            models.CheckConstraint(
                check=models.Q(priority__gte=1),
                name="meal_subscription_priority_positive",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.person} {self.meal_plan} [{self.start_date} → {self.end_date}] (p{self.priority})"

    def clean(self):
        super().clean()
        from .validators import validate_subscription_dates, validate_subscription_role

        validate_subscription_dates(start_date=self.start_date, end_date=self.end_date)
        validate_subscription_role(student=self.student, staff=self.staff)


class MealException(models.Model):
    """A one-time / temporary meal permission or denial.

    A one-time permission is **not** a :class:`MealSubscription`; it is
    a ``MealException(kind=ONE_TIME_ELIGIBLE, effective_date=<date>)``.
    The eligibility resolver checks exceptions after subscriptions, so a
    person without any subscription can still be eligible for one date
    via an exception. See ``meals_domain_architecture.md`` §11.2.
    """

    class Kind(models.TextChoices):
        ONE_TIME_ELIGIBLE = "one_time_eligible", "One-time eligible"
        TEMPORARY_DENY = "temporary_deny", "Temporary deny"
        GUEST_ELIGIBLE = "guest_eligible", "Guest eligible"

    person = models.ForeignKey(
        "identity.Person",
        on_delete=models.CASCADE,
        related_name="meal_exceptions",
    )
    kind = models.CharField(
        max_length=30,
        choices=Kind.choices,
        db_index=True,
    )
    effective_date = models.DateField(db_index=True)
    end_date = models.DateField(
        null=True,
        blank=True,
        help_text="Optional. Blank = single-day exception (effective_date only).",
    )
    meal_plan = models.ForeignKey(
        MealPlan,
        on_delete=models.SET_NULL,
        related_name="exceptions",
        null=True,
        blank=True,
    )
    reason_code = models.CharField(max_length=32, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    approved_by = models.ForeignKey(
        "identity.StaffProfile",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="approved_meal_exceptions",
    )
    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-effective_date", "-id"]
        indexes = [
            models.Index(fields=["person", "effective_date"]),
            models.Index(fields=["kind", "is_active"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=(
                    models.Q(end_date__isnull=True)
                    | models.Q(end_date__gte=models.F("effective_date"))
                ),
                name="meal_exception_end_after_effective",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.person} {self.kind} on {self.effective_date}"

    def clean(self):
        super().clean()
        from .validators import validate_exception_window

        validate_exception_window(
            effective_date=self.effective_date, end_date=self.end_date
        )


class MealEligibility(models.Model):
    """The canonical daily eligibility snapshot for ``(person, date)``.

    See ``meals_domain_architecture.md`` §12.1. Exactly one row per
    ``(person, date)`` (enforced by ``unique_together``).

    Recalculable until a future ``MealServiceEvent`` references it; once
    referenced, the row is frozen and corrections are audited via the
    future ``MealSupervisorAction`` model (Phase 3+).
    """

    class Decision(models.TextChoices):
        ELIGIBLE = "eligible", "Eligible"
        NOT_ELIGIBLE = "not_eligible", "Not eligible"
        OVERRIDDEN_ELIGIBLE = "overridden_eligible", "Overridden eligible"
        OVERRIDDEN_DENIED = "overridden_denied", "Overridden denied"

    person = models.ForeignKey(
        "identity.Person",
        on_delete=models.CASCADE,
        related_name="meal_eligibilities",
    )
    student = models.ForeignKey(
        "identity.StudentProfile",
        on_delete=models.CASCADE,
        related_name="meal_eligibilities",
        null=True,
        blank=True,
    )
    date = models.DateField(db_index=True)
    decision = models.CharField(
        max_length=30,
        choices=Decision.choices,
        db_index=True,
    )
    subscription = models.ForeignKey(
        MealSubscription,
        on_delete=models.SET_NULL,
        related_name="eligibilities",
        null=True,
        blank=True,
    )
    meal_plan = models.ForeignKey(
        MealPlan,
        on_delete=models.SET_NULL,
        related_name="eligibilities",
        null=True,
        blank=True,
    )
    grade_code_snapshot = models.CharField(max_length=32, blank=True, default="")
    section_code_snapshot = models.CharField(max_length=32, blank=True, default="")
    absence_reason = models.CharField(max_length=40, blank=True, default="")
    reason_code = models.CharField(max_length=40, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    resolved_at = models.DateTimeField(auto_now_add=True)
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="resolved_meal_eligibilities",
    )

    class Meta:
        unique_together = [["person", "date"]]
        ordering = ["-date"]
        indexes = [
            models.Index(fields=["date", "decision"]),
            models.Index(fields=["person", "date"]),
            models.Index(fields=["student", "date"]),
        ]

    def __str__(self) -> str:
        return f"{self.person} on {self.date} → {self.decision}"
