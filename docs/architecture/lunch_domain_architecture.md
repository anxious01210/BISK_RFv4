# Lunch Domain Architecture — BISK_RFv4

Date: 2026-07-03
Branch: feature/person-architecture
Version: 1.0 Draft — pending review
Status: Architecture design only. No code, models, or migrations are produced by this document.

> ⚠️ **SUPERSEDED — TRANSITIONAL / LEGACY DOCUMENT (2026-07-04)**
>
> This document is **superseded** for the future meals domain by:
>
> → `docs/architecture/meals_domain_architecture.md` (v1.1, authoritative)
>
> The meals document is the **authoritative** design for the future meal
> domain. It reconciles this document with the finalized business
> requirements and the decided architecture. The two documents disagree
> on the following decided points; **the meals document wins** in every
> case:
>
> | Topic | This (lunch) document | Authoritative meals v1.1 decision |
> |---|---|---|
> | App name | `apps.meal` (singular) | **`apps.meals`** (plural) |
> | Plan model name | `MealPlan` (with `MealProfile` alias) | **`MealPlan`** is the future model; legacy `attendance.MealProfile` maps to it. `MealProfile` is legacy only. |
> | Entity naming | `Lunch*` (e.g. `LunchSubscription`, `LunchServiceEvent`) | **`Meal*`** (e.g. `MealSubscription`, `MealServiceEvent`) — student + staff + multiple meal kinds are first-class |
> | Price resolution ownership | List price in `apps.meal`; **final charge** computed by a "finance/pricing pipeline" (§9) | **Meals owns price resolution** entirely (base + per-Person override + discount composition); Finance records pre-resolved charges only |
> | Discount caller | "apps.meal calls finance.resolve_discount" (§9.1); finance.charge resolves discounts | **Meals** calls `apps.discounts` directly during `resolve_price`; Finance never calls discounts on the meal code path |
> | `finance.charge` signature | `finance.charge(..., price_base_iqd=...)` resolving pricing | `finance.charge(*, person, amount_iqd, ...)` recording a **pre-resolved** amount; Finance does not resolve prices |
> | Date-range vs wallet | Mode on `MealPlan`; both use date ranges (implicitly) | Same — clarified explicitly: both modes may have date ranges; date-range = entitlement (no debit), wallet = per-service charge while active (§10.1) |
> | Primary / fallback + priority | Priority field; same-priority overlap rejected | Same — clarified: lower number = higher priority = evaluated first; primary-only, fallback-only, both, multi-wallet (§11.4, §11.5) |
> | Resolver flow | Spread across eligibility + pricing | **Single `resolve_service` flow** (meals doc §13): candidates → period filter → sort by priority → date-range primary → wallet fallback → price → discounts → finance.charge |
> | Staff meals | Not addressed | **First-class** (`MealSubscription.staff`, `StaffProfile`, staff eligibility/pricing — meals doc §17) |
> | Same-day multi-meal discounts | Not addressed | **Supported** via `same_day_meals` context (meals doc §19.5) |
>
> This lunch document is retained as **historical context** for the
> legacy `apps.attendance` lunch stack and as a record of the design
> evolution. **Do not implement from this document.** Implement from
> `meals_domain_architecture.md` v1.1. Where any statement here
> conflicts with the meals document, the meals document is correct.
>
> The reconciled finance boundary is documented in
> `finance_domain_architecture.md` v1.1 (Finance records pre-resolved
> charges only; Finance does not resolve meal prices or call discounts
> for meals). The cross-app integration rules are in
> `docs/development/DOMAIN_INTEGRATION_GUIDE.md`.

---

## 1. Purpose

This document designs the **Lunch domain** for BISK_RFv4 — meal plans,
lunch subscriptions, daily lunch eligibility, lunch attendance/service
events, supervisor workflow, and the integration boundaries with the
wallet, discount, and AI-attendance subsystems.

The current implementation lives scattered inside `apps.attendance`
(`MealSubscription`, `MealProfile`, `MealProfilePeriod`, `MealRecord`),
all currently FK-keyed to the legacy `attendance.Student`. This document
designs a clean `apps.meal` domain that:

- Anchors to `Person` / `StudentProfile` (never to legacy `Student`).
- Reads academic context (grade/section/roster) from `apps.academics`
  rather than duplicating it.
- Preserves historical lunch records (eligibility snapshots, service
  events, supervisor actions) immutably.
- Delegates charging to the wallet/finance domain and recognition to the
  AI-attendance domain — it does **not** embed either engine.

This is a **design document**, not an implementation order. Each entity
below is a blueprint for a future, separately-approved implementation step.

### Scope

| In scope | Out of scope |
|---|---|
| MealPlan, LunchSubscription, LunchEligibility, LunchAttendance / LunchServiceEvent, LunchSupervisorAction, LunchException | Wallet model design (see wallet/finance docs) |
| Subscription lifecycle and one-time permission | Discount engine internals |
| Daily eligibility calculation, snapshotting, manual overrides | AI face-recognition engine internals |
| Supervisor dashboard workflow and audit | Portal / mobile API design |
| Integration boundaries with wallet, discount, AI attendance | Menu/kitchen inventory management |
| Recommended model boundaries and implementation order | Multi-tenant activation |

### Constraints respected

- Do **not** recreate the old `Student` model. Lunch references
  `StudentProfile` (and `Person`), never legacy `attendance.Student`.
- Lunch references `StudentProfile` and the **active/current enrollment**
  where grade/section/roster context is needed — it does not duplicate
  student name, grade, or section fields unless snapshotting is explicitly
  justified (see Section 4 / Section 6).
- Lunch **preserves historical records**: eligibility snapshots and
  service events are append-only and never overwritten.
- Lunch **must not contain wallet/finance calculations directly**; it calls
  into the wallet domain via a service boundary.
- Lunch **must not contain AI recognition engine logic directly**; it
  consumes recognition events as inputs.
- Uses the `services.py` / `selectors.py` / `validators.py` pattern
  established by `apps.identity` and `apps.academics`.
- Follows `education_domain_architecture.md`,
  `academics_domain_architecture.md`, `person_identity_architecture.md`,
  `PROJECT_ARCHITECTURE.md`, and `AI_DEVELOPMENT_GUIDE.md`.

---

## 2. Design Principles

| # | Principle | Application to Lunch |
|---|---|---|
| 1 | **Identity before role** | Lunch anchors to `Person` / `StudentProfile`. A subscription belongs to a Person (via profile), not to a legacy Student row. |
| 2 | **Academic context is read, not copied** | Grade/section/roster come from `apps.academics` enrollment + current `StudentEnrollmentSectionPlacement`. Lunch stores at most a *snapshot* on an immutable historical record, never a live duplicated field. |
| 3 | **History is append-only** | Service events and supervisor actions are immutable once written; corrections create new rows, they do not mutate prior rows. `LunchEligibility` is the canonical daily decision/snapshot for a `(student, date)`; it may be recalculated before a `LunchServiceEvent` exists, but becomes frozen once a service event references it. |
| 4 | **Workflow as state machine** | Subscription and lunch-attendance use explicit `status` enums with documented valid transitions. No scattered `is_paid`/`is_served`/`is_refunded` booleans. |
| 5 | **Money is auditable** | Every charge and refund is a permanent `WalletTransaction` created by the wallet domain. Lunch records the *result* (amount, balance before/after, transaction FK) as a snapshot, never computes balance itself. |
| 6 | **Service-layer business logic** | Eligibility resolution, subscription lifecycle, and supervisor actions live in `services.py`; queries live in `selectors.py`; invariants live in `validators.py`. Not in views, templates, or admin actions. |
| 7 | **Domain ownership first** | `apps.meal` owns MealPlan/LunchSubscription/LunchEligibility/LunchServiceEvent/LunchSupervisorAction/LunchException. Wallet owns Wallet/WalletTransaction. Finance owns Discount. Attendance owns recognition events. `apps.meal` depends on identity + academics + (via service boundary) wallet/finance/attendance. |
| 8 | **display_code in templates** | Supervisor dashboard and reports render `person.display_code`, never legacy `h_code`. |
| 9 | **Incremental delivery** | Each entity is independently deployable. The meal app can be introduced while the legacy `MealSubscription`/`MealRecord` continue to work via the dual-FK migration pattern. |
| 10 | **No duplicated identity** | No `h_code`, no `first_name`/`last_name`/`grade`/`section` live fields on Lunch entities. Names are read via `subscription.student.person.full_name`; grade/section via `enrollment.grade`/`enrollment.section` or a snapshot on an immutable record. |
| 11 | **Snapshot only when justified** | A snapshot is allowed only on an immutable historical record (e.g. `LunchServiceEvent`) where reproducing the value later from live data would be impossible or incorrect (the student's grade/section at service time, the price charged, the balance before charge). |
| 12 | **Backward compatibility always** | Legacy `MealSubscription`/`MealRecord` keep working during migration. New models are additive; legacy FKs are removed only after verification. |

---

## 3. Proposed App / Module Name

```
apps/meal/
├── __init__.py
├── apps.py
├── models.py        # MealPlan, LunchSubscription, LunchEligibility,
│                    # LunchServiceEvent, LunchSupervisorAction, LunchException
├── admin.py
├── services.py      # create_subscription, pause, cancel, resolve_eligibility,
│                    # record_service_event, supervisor_override, ...
├── selectors.py     # active_subscriptions_for, eligible_students_for_date,
│                    # service_events_for_section, ...
├── validators.py    # subscription overlap, eligibility window, capacity, ...
└── migrations/
```

**App label:** `meal` (Django app). Python module path: `apps.meal`.

> Naming note: the broader ERP sketch uses `apps/meal/` for the meal
> domain. This document keeps `meal` as the app label and uses `Lunch*`
> entity names because the school's operational term is "lunch". Other
> meal types (breakfast, snack) can be added later as `MealPlan.kind`
> values without renaming the app. App-name finalization is open question
> Q1.

### Dependency direction

```
apps.meal ──depends on──► apps.identity   (StudentProfile, Person)
apps.meal ──depends on──► apps.academics  (AcademicYear, Section, Enrollment, current placement)
apps.meal ──calls (service boundary)──► apps.wallet   (charge / refund / balance check)
apps.meal ──calls (service boundary)──► apps.finance  (discount resolution, if active)
apps.meal ──consumes──► apps.attendance  (recognition events as input only)
```

`apps.meal` must **not** import wallet/finance/attendance model classes
directly into its own models beyond a typed FK to `Person`/`StudentProfile`
and `AcademicYear`/`Section`. Where a wallet transaction must be linked,
the FK points to the wallet domain's model (`apps.wallet.WalletTransaction`),
and the *creation* of that row is performed by a wallet service call, not
by meal code.

A dependency `wallet → meal` is **not** allowed. A dependency
`attendance → meal` is **not** allowed (recognition produces a Person-identified
event; meal reads it, not the reverse).

---

## 4. Core Entities

All designs below are **conceptual blueprints**. No migration is produced by
this document. Field types are indicative; exact choices are made at
implementation time.

### 4.1 MealPlan

- **Owner:** `apps.meal`
- **Purpose:** A reusable meal offering definition (the "product"): a named
  plan, its kind, pricing/policy profile, and supervisor-override policy.
  This is the migration target for the legacy `MealProfile` /
  `MealProfilePeriod` concepts, renamed to avoid confusion with profile
  classes.
- **Conceptual fields:**

```python
class MealPlan(models.Model):
    class Kind(models.TextChoices):
        LUNCH = "lunch", "Lunch"
        BREAKFAST = "breakfast", "Breakfast"
        SNACK = "snack", "Snack"

    class Mode(models.TextChoices):
        DATE_RANGE = "date_range", "Date-range"
        WALLET = "wallet", "Wallet"  # per-service charge

    class InsufficientFundsMode(models.TextChoices):
        DENY = "deny", "Deny"
        ALLOW_UNPAID = "allow_unpaid", "Allow unpaid"
        ALLOW_NEGATIVE = "allow_negative", "Allow negative"

    name = models.CharField(max_length=100, unique=True)
    kind = models.CharField(max_length=20, choices=Kind.choices, default=Kind.LUNCH)
    mode = models.CharField(max_length=20, choices=Mode.choices, default=Mode.DATE_RANGE)
    is_active = models.BooleanField(default=True)
    # policy flags (migrated from MealProfile)
    allow_supervisor_confirm = models.BooleanField(default=True)
    allow_supervisor_unconfirm = models.BooleanField(default=False)
    allow_supervisor_refund = models.BooleanField(default=False)
    require_reason_on_override = models.BooleanField(default=True)
    require_reason_on_unconfirm = models.BooleanField(default=False)
    require_reason_on_refund = models.BooleanField(default=True)
    insufficient_funds_mode = models.CharField(
        max_length=20, choices=InsufficientFundsMode.choices,
        default=InsufficientFundsMode.DENY,
    )
    credit_limit_iqd = models.PositiveIntegerField(null=True, blank=True)
    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

- **Per-period pricing** is kept on a separate `MealPlanPeriodPrice`
  (`meal_plan` FK, `period_template` FK, `price_iqd`, `is_enabled`),
  migrating `MealProfilePeriod`. Pricing is owned by the meal domain as
  the *list price*; the **final charge** after discounts is computed by
  the finance/pricing pipeline (Section 9), not by `MealPlan`.
- **Rules:**
  - `MealPlan` is admin-configurable.
  - It does **not** reference `Person` or `StudentProfile`. It is a product
    definition.
  - Supervisor-override flags live here because they are product policy,
    not per-student data.

### 4.2 LunchSubscription

- **Owner:** `apps.meal`
- **Purpose:** A Person's entitlement to meals under a `MealPlan` for a
  date range (the migration target for legacy `MealSubscription`).
- **Conceptual fields:**

```python
class LunchSubscription(models.Model):
    class Status(models.TextChoices):
        FUTURE = "future", "Future"
        ACTIVE = "active", "Active"
        PAUSED = "paused", "Paused"
        EXPIRED = "expired", "Expired"
        CANCELLED = "cancelled", "Cancelled"

    student = models.ForeignKey(
        "identity.StudentProfile", on_delete=models.CASCADE,
        related_name="lunch_subscriptions",
    )
    meal_plan = models.ForeignKey(
        MealPlan, on_delete=models.PROTECT, related_name="subscriptions",
        null=True, blank=True,
    )
    academic_year = models.ForeignKey(
        "academics.AcademicYear", on_delete=models.PROTECT,
        related_name="lunch_subscriptions", null=True, blank=True,
    )
    status = models.CharField(max_length=20, choices=Status.choices,
                              default=Status.ACTIVE, db_index=True)
    start_date = models.DateField(db_index=True)
    end_date = models.DateField(db_index=True)
    plan_type = models.CharField(max_length=20, default="monthly")  # reporting
    source = models.CharField(max_length=40, default="manual", db_index=True)
    priority = models.PositiveSmallIntegerField(default=1, db_index=True)
    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

- **Relationships:**
  - `student` → `StudentProfile` (FK CASCADE). The subscription belongs
    to a Person via their student profile, never to legacy `Student`.
  - `meal_plan` → `MealPlan` (PROTECT).
  - `academic_year` → `AcademicYear` (PROTECT, nullable until the
    academics app is rolled out; then year-scoping is recommended).
- **Rules:**
  - Overlap validation lives in `validators.py`: two `ACTIVE` subscriptions
    for the same student with the **same priority** may not overlap in
    `[start_date, end_date]`. Different priorities may overlap (primary +
    fallback), mirroring legacy behavior.
  - `status` is the single source of truth for the subscription's
    lifecycle (see Section 5).
  - The subscription stores **no** grade/section/name fields. Grade/section
    at service time are read from the current enrollment/placement or
    snapshotted on the immutable `LunchServiceEvent` (Section 4.4).

### 4.3 LunchEligibility

- **Owner:** `apps.meal`
- **Purpose:** The **canonical daily eligibility decision/snapshot** for a
  `StudentProfile` on a given date. It records *whether* the student was
  eligible for lunch that day and *why* (which subscription, which plan,
  whether overridden), as of the moment eligibility was resolved.
- **Lifecycle:** An eligibility row may be **recalculated** (re-resolved
  and overwritten) at any time **before** a `LunchServiceEvent` references
  it — e.g. when a subscription is added late, an absence is corrected, or
  academic context changes. Once a `LunchServiceEvent` references it, the
  row is **frozen** and must not be mutated; further corrections happen
  through `LunchSupervisorAction` audit records against the service
  event, not by rewriting the eligibility.
- **Conceptual fields:**

```python
class LunchEligibility(models.Model):
    class Decision(models.TextChoices):
        ELIGIBLE = "eligible", "Eligible"
        NOT_ELIGIBLE = "not_eligible", "Not eligible"
        OVERRIDDEN_ELIGIBLE = "overridden_eligible", "Overridden eligible"
        OVERRIDDEN_DENIED = "overridden_denied", "Overridden denied"

    student = models.ForeignKey(
        "identity.StudentProfile", on_delete=models.CASCADE,
        related_name="lunch_eligibilities",
    )
    date = models.DateField(db_index=True)
    decision = models.CharField(max_length=30, choices=Decision.choices,
                                db_index=True)
    subscription = models.ForeignKey(
        LunchSubscription, on_delete=models.SET_NULL,
        related_name="eligibilities", null=True, blank=True,
    )
    meal_plan = models.ForeignKey(
        MealPlan, on_delete=models.SET_NULL,
        related_name="eligibilities", null=True, blank=True,
    )
    # snapshot of academic context at resolution time
    # (recalculable until a LunchServiceEvent references this row)
    grade_code_snapshot = models.CharField(max_length=32, blank=True, default="")
    section_code_snapshot = models.CharField(max_length=32, blank=True, default="")
    absence_reason = models.CharField(max_length=40, blank=True, default="")
    # why this decision was reached
    reason_code = models.CharField(max_length=40, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    resolved_at = models.DateTimeField(auto_now_add=True)
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name="resolved_lunch_eligibilities",
    )

    class Meta:
        unique_together = [("student", "date")]
        ordering = ["-date"]
        indexes = [
            models.Index(fields=["date", "decision"]),
            models.Index(fields=["student", "date"]),
        ]
```

- **Rules:**
  - `unique_together = [("student", "date")]`: exactly one canonical
    eligibility row per student per day. This row is the authoritative
    daily decision.
  - **Recalculable before service:** the row may be recomputed/overwritten
    by `resolve_eligibility` at any time **before** a `LunchServiceEvent`
    references it (e.g. late subscription, corrected absence, changed
    academic context).
  - **Frozen once referenced:** once a `LunchServiceEvent` references the
    eligibility, the row is frozen and must not be mutated. Subsequent
    corrections are audited through `LunchSupervisorAction` records against
    the service event (Section 7), not by rewriting the eligibility.
  - `grade_code_snapshot` / `section_code_snapshot` are **justified
    snapshots**: the student's grade/section at resolution time, preserved
    for historical accuracy even if the student later changes sections or
    the enrollment is archived.
  - Eligibility **does not** store wallet balance or price; those live on
    the service event snapshot (Section 4.4) and the wallet domain.

### 4.4 LunchServiceEvent (LunchAttendance)

- **Owner:** `apps.meal`
- **Purpose:** The immutable record that a student was actually served
  lunch on a date (the migration target for legacy `MealRecord`, freed
  from its 1:1 coupling to `AttendanceRecord`). This is the "lunch
  attendance" entity.
- **Conceptual fields:**

```python
class LunchServiceEvent(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        CONFIRMED = "confirmed", "Confirmed"
        DENIED = "denied", "Denied"
        UNPAID = "unpaid", "Unpaid"      # wallet mode, insufficient funds allowed
        REFUNDED = "refunded", "Refunded"
        VOIDED = "voided", "Voided"

    student = models.ForeignKey(
        "identity.StudentProfile", on_delete=models.CASCADE,
        related_name="lunch_service_events",
    )
    date = models.DateField(db_index=True)
    eligibility = models.ForeignKey(
        LunchEligibility, on_delete=models.PROTECT,
        related_name="service_events", null=True, blank=True,
    )
    subscription = models.ForeignKey(
        LunchSubscription, on_delete=models.SET_NULL,
        related_name="service_events", null=True, blank=True,
    )
    meal_plan = models.ForeignKey(
        MealPlan, on_delete=models.SET_NULL,
        related_name="service_events", null=True, blank=True,
    )
    # optional link to the recognition that triggered the service
    recognition_event = models.ForeignKey(
        "attendance.AttendanceEvent", on_delete=models.SET_NULL,
        related_name="lunch_service_events", null=True, blank=True,
    )
    status = models.CharField(max_length=20, choices=Status.choices,
                              default=Status.PENDING, db_index=True)
    # immutable financial snapshot (justified: historical record)
    price_base_iqd = models.IntegerField(default=0)
    discount_iqd = models.IntegerField(default=0)
    final_charge_iqd = models.IntegerField(default=0)
    wallet_balance_before_iqd = models.IntegerField(default=0)
    wallet_balance_after_iqd = models.IntegerField(default=0)
    wallet_transaction = models.ForeignKey(
        "wallet.WalletTransaction", on_delete=models.PROTECT,
        related_name="lunch_service_events", null=True, blank=True,
    )
    wallet_refund_transaction = models.ForeignKey(
        "wallet.WalletTransaction", on_delete=models.PROTECT,
        related_name="lunch_refund_events", null=True, blank=True,
    )
    # immutable academic snapshot at service time
    grade_code_snapshot = models.CharField(max_length=32, blank=True, default="")
    section_code_snapshot = models.CharField(max_length=32, blank=True, default="")
    reason_code = models.CharField(max_length=32, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    served_at = models.DateTimeField(null=True, blank=True, db_index=True)
    served_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name="served_lunch_events",
    )
    reversed_at = models.DateTimeField(null=True, blank=True)
    reversed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name="reversed_lunch_events",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-date", "-id"]
        indexes = [
            models.Index(fields=["date", "status"]),
            models.Index(fields=["student", "date"]),
            models.Index(fields=["section_code_snapshot", "date"]),
            models.Index(fields=["meal_plan", "date"]),
        ]
```

- **Rules:**
  - Once `status` reaches `CONFIRMED` (or any terminal state), the
    financial and academic snapshot fields are **immutable**. Corrections
    create a new event (e.g. a `REFUNDED`/`VOIDED` event referencing the
    original) rather than mutating the confirmed row.
  - The event stores *snapshots* of price/discount/charge/balance because
    those values are the historical truth of what happened at service
    time; they cannot be reconstructed from live wallet/plan data later.
  - The wallet transaction FK is created by the **wallet domain**; meal
    only stores the resulting reference.
  - `recognition_event` is an optional link to the AI-attendance event
    that triggered the service. Meal does not run recognition; it
    consumes the event.

### 4.5 LunchSupervisorAction

- **Owner:** `apps.meal`
- **Purpose:** An audit record of every supervisor action — confirm,
  unconfirm, deny, refund, override eligibility, void — against a
  `LunchServiceEvent` or `LunchEligibility`.
- **Conceptual fields:**

```python
class LunchSupervisorAction(models.Model):
    class Action(models.TextChoices):
        CONFIRM = "confirm", "Confirm"
        UNCONFIRM = "unconfirm", "Unconfirm"
        DENY = "deny", "Deny"
        REFUND = "refund", "Refund"
        VOID = "void", "Void"
        OVERRIDE_ELIGIBLE = "override_eligible", "Override (eligible)"
        OVERRIDE_DENIED = "override_denied", "Override (denied)"
        MANUAL_LOOKUP = "manual_lookup", "Manual lookup"

    service_event = models.ForeignKey(
        LunchServiceEvent, on_delete=models.CASCADE,
        related_name="supervisor_actions", null=True, blank=True,
    )
    eligibility = models.ForeignKey(
        LunchEligibility, on_delete=models.CASCADE,
        related_name="supervisor_actions", null=True, blank=True,
    )
    action = models.CharField(max_length=30, choices=Action.choices, db_index=True)
    reason_code = models.CharField(max_length=32, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    performed_by = models.ForeignKey(
        "identity.StaffProfile", on_delete=models.SET_NULL,
        null=True, blank=True, related_name="lunch_supervisor_actions",
    )
    performed_at = models.DateTimeField(auto_now_add=True)
    # optional: the auth.User who performed the action (login account)
    performed_by_user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name="lunch_supervisor_actions",
    )

    class Meta:
        ordering = ["-performed_at"]
        indexes = [
            models.Index(fields=["service_event", "performed_at"]),
            models.Index(fields=["action", "performed_at"]),
        ]
```

- **Rules:**
  - Every mutating supervisor operation writes a `LunchSupervisorAction`
    row, in the same transaction as the state change. This is the audit
    trail required by the "money is auditable" and "history is
    append-only" principles.
  - `performed_by` references `StaffProfile` (the business identity of
    the supervisor); `performed_by_user` references `auth.User` (the login
    account). Either may be null depending on how the action was
    performed (API vs. admin vs. dashboard).

### 4.6 LunchException

- **Owner:** `apps.meal`
- **Purpose:** A scoped, dated exception that grants or denies lunch for
  a student outside the normal subscription rules — e.g. a one-time
  permission, a guest, a temporary medical exception.
- **Conceptual fields:**

```python
class LunchException(models.Model):
    class Kind(models.TextChoices):
        ONE_TIME_ELIGIBLE = "one_time_eligible", "One-time eligible"
        TEMPORARY_DENY = "temporary_deny", "Temporary deny"
        GUEST_ELIGIBLE = "guest_eligible", "Guest eligible"

    student = models.ForeignKey(
        "identity.StudentProfile", on_delete=models.CASCADE,
        related_name="lunch_exceptions", null=True, blank=True,
    )
    # for guest_eligible when there is no StudentProfile yet
    person = models.ForeignKey(
        "identity.Person", on_delete=models.CASCADE,
        related_name="lunch_exceptions", null=True, blank=True,
    )
    kind = models.CharField(max_length=30, choices=Kind.choices, db_index=True)
    effective_date = models.DateField(db_index=True)
    end_date = models.DateField(null=True, blank=True)
    meal_plan = models.ForeignKey(
        MealPlan, on_delete=models.SET_NULL, null=True, blank=True,
    )
    reason_code = models.CharField(max_length=32, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    approved_by = models.ForeignKey(
        "identity.StaffProfile", on_delete=models.SET_NULL,
        null=True, blank=True, related_name="approved_lunch_exceptions",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-effective_date"]
        indexes = [
            models.Index(fields=["student", "effective_date", "is_active"]),
            models.Index(fields=["person", "effective_date", "is_active"]),
        ]
```

- **Rules:**
  - A `ONE_TIME_ELIGIBLE` exception is the implementation of "one-time
    lunch permission" (Section 5): it grants eligibility for a single
    date without creating a subscription.
  - Either `student` or `person` is set (not both). `student` for
    enrolled students; `person` for guests/staff who have no
    `StudentProfile`.
  - Exceptions are evaluated by the eligibility resolver **after**
    subscriptions and **before** the default denial.

### 4.7 Relationship to StudentProfile

- `LunchSubscription.student`, `LunchEligibility.student`,
  `LunchServiceEvent.student`, `LunchException.student` all FK to
  `identity.StudentProfile`.
- Student identity (name, code, photo) is read via
  `student.person.full_name` / `student.person.display_code` /
  `student.person.photo`. **No** name/code fields are duplicated on meal
  entities.
- "Is this person a student entitled to lunch?" combines
  `PersonRole(person, student)` (timeless role) with an active
  `LunchSubscription` (dated entitlement).

### 4.8 Relationship to Academics Enrollment

- Lunch reads academic context from `apps.academics`:
  - The **current roster** of a section on a date is
    `apps.academics.selectors.current_enrollments_in_section(section=...)`
    (enrollments with a current `StudentEnrollmentSectionPlacement`).
  - A student's current grade/section on a date is read from their active
    `StudentEnrollment` for the `AcademicYear` containing that date.
- Lunch stores grade/section only as **snapshots** on immutable records
  (`LunchEligibility.grade_code_snapshot`,
  `LunchServiceEvent.section_code_snapshot`), justified because the
  student may later transfer sections or the enrollment may be archived.
- `LunchSubscription.academic_year` optionally scopes a subscription to a
  year (nullable until academics is rolled out).

### 4.9 Relationship to SectionPlacement / current roster

- The supervisor dashboard's "expected today" roster for a section is
  derived by joining the section's **current placements** (from academics)
  to active lunch subscriptions for the date.
- Meal does **not** re-implement roster logic. It calls
  `current_enrollments_in_section` and filters by subscription status.
- Closed placements (withdrawn/graduated/repeated students) are excluded
  from the current roster, so a withdrawn student does not appear as
  "expected" even if their subscription is technically still `ACTIVE` by
  date. The eligibility resolver treats "no current placement" as a
  strong signal (open decision Q6 — whether it's a hard deny or a
  warning).

---

## 5. Subscription Lifecycle

Modeled as a state machine. Each transition is a service method in
`apps.meal.services`.

### 5.1 Status values

| Status | Meaning |
|---|---|
| `future` | Subscription's `start_date` has not yet arrived. |
| `active` | Current date is within `[start_date, end_date]` and not paused/cancelled. |
| `paused` | Temporarily suspended (e.g. illness, trip) within the active window; no eligibility granted while paused. |
| `expired` | `end_date` has passed; no longer grants eligibility. |
| `cancelled` | Terminated before `end_date` by an admin/supervisor. |

### 5.2 Valid transitions

```
future ──(start_date reached)──► active
active ──(pause)──► paused
paused ──(resume)──► active
active/paused ──(cancel)──► cancelled
active/paused/future ──(end_date passed)──► expired
```

### 5.3 One-time lunch permission

A one-time permission is **not** a `LunchSubscription`. It is a
`LunchException(kind=ONE_TIME_ELIGIBLE, effective_date=<date>)`. The
eligibility resolver checks exceptions after subscriptions, so a student
without any subscription can still be eligible for one date via an
exception. This avoids polluting the subscription model with single-day
rows and keeps the "subscription = date range" contract clean.

### 5.4 Lifecycle service methods (conceptual)

```python
# apps/meal/services.py

@transaction.atomic
def create_subscription(*, student, meal_plan, start_date, end_date,
                        academic_year=None, plan_type="monthly",
                        priority=1, source="manual") -> LunchSubscription: ...

@transaction.atomic
def activate_future_subscriptions(*, as_of) -> int: ...      # future -> active by date

@transaction.atomic
def pause_subscription(*, subscription, changed_by=None) -> LunchSubscription: ...

@transaction.atomic
def resume_subscription(*, subscription, changed_by=None) -> LunchSubscription: ...

@transaction.atomic
def cancel_subscription(*, subscription, reason_code="", changed_by=None) -> LunchSubscription: ...

@transaction.atomic
def expire_due_subscriptions(*, as_of) -> int: ...           # active/paused -> expired by date

@transaction.atomic
def grant_one_time_permission(*, student, effective_date, meal_plan=None,
                              reason_code="", approved_by=None) -> LunchException: ...
```

### 5.5 Historical preservation

- Subscriptions are **never deleted**. Cancelled/expired rows remain for
  historical reporting and for `LunchEligibility.subscription` /
  `LunchServiceEvent.subscription` FKs to remain valid.
- A cancelled subscription's `[start_date, end_date]` and `status` history
  are preserved; only the `status` field transitions.
- If subscription metadata (plan, priority) must change mid-window, the
  recommended pattern is to cancel the current subscription and create a
  new one for the remainder, so history is append-only. (Open decision
  Q4 — whether in-place edits are allowed with an audit record.)

---

## 6. Daily Lunch Eligibility

### 6.1 How eligibility is calculated

Eligibility for a `(student, date)` is resolved by
`resolve_eligibility(student, date)` in `apps.meal.services`, in this
order:

1. **Academic presence check.** Does the student have a current
   `StudentEnrollmentSectionPlacement` for the `AcademicYear` containing
   `date`? If not, eligibility is `NOT_ELIGIBLE` with
   `reason_code="no_current_placement"` (subject to Q6).
2. **Absence check.** Is the student marked absent that day (from the
   attendance domain)? If yes and no override exists, eligibility is
   `NOT_ELIGIBLE` with `reason_code="absent"`. (The absence signal is
   read from attendance, not computed by meal.)
3. **Subscription match.** Is there an `ACTIVE` subscription whose
   `[start_date, end_date]` contains `date`? If yes, eligibility is
   `ELIGIBLE`, referencing the subscription and its `meal_plan`.
4. **Exception match.** Is there an active `LunchException` covering
   `date`? If yes, eligibility is `ELIGIBLE` (or
   `OVERRIDDEN_ELIGIBLE`/`OVERRIDDEN_DENIED` depending on kind),
   referencing the exception.
5. **Default.** `NOT_ELIGIBLE` with `reason_code="no_subscription"`.

### 6.2 How eligibility is snapshotted

- The result of `resolve_eligibility` is written as the canonical
  `LunchEligibility` row for `(student, date)` (`unique_together=("student","date")`).
- The row snapshots `grade_code_snapshot` and `section_code_snapshot`
  from the current enrollment/placement at resolution time.
- **Recalculable before service:** as long as no `LunchServiceEvent`
  references the row, `resolve_eligibility` may recompute and overwrite it
  (e.g. a late subscription, a corrected absence, a section transfer).
- **Frozen once referenced:** once a `LunchServiceEvent` references the
  eligibility, the row is frozen. A later `resolve_eligibility` for the
  same `(student, date)` must not mutate the frozen row; it returns the
  existing frozen decision. (Open decision Q9 — whether to allow a
  superseding row with a `superseded` flag, or simply block. The
  recommended initial approach is a single canonical row per
  `(student, date)` that is immutable once a service event exists.)
- `LunchServiceEvent` itself remains immutable historical truth; it is
  never rewritten when eligibility is later discussed.

### 6.3 How manual overrides work

- A supervisor override calls `supervisor_override(student, date, decision,
  reason_code, performed_by)`, which:
  1. Writes a `LunchSupervisorAction(action=OVERRIDE_ELIGIBLE|OVERRIDE_DENIED)`
     — this audit record is mandatory and is created **before** any
     eligibility change is applied.
  2. If no `LunchServiceEvent` references the eligibility yet, updates the
     canonical `LunchEligibility` row with the overridden decision (this is
     a permitted recalculation before freezing).
  3. If a `LunchServiceEvent` already references the eligibility, the
     eligibility is frozen and is **not** mutated; the override is
     recorded only as a `LunchSupervisorAction` against the (immutable)
     service event. The service event's status may transition via the
     supervisor action, but its financial/academic snapshots are not
     rewritten.
- Overrides require a `reason_code` when `meal_plan.require_reason_on_override`
  is true (policy on the plan).

### 6.4 How absence affects eligibility

- Absence is a **signal in**, read from the attendance domain
  (`AttendanceRecord.status` for that student on that date). Meal does
  not compute attendance.
- An absent student is `NOT_ELIGIBLE` unless a supervisor override or a
  `LunchException` grants eligibility anyway.
- The absence reason is recorded on `LunchEligibility.absence_reason` for
  auditability.

---

## 7. Supervisor Dashboard Workflow

The dashboard is a **view layer** in `apps.meal` (or a thin
`apps.lunch_supervisor` app that owns no models). It composes selectors
from `meal`, `academics`, `wallet`, and `attendance`.

### 7.1 Workflow steps

```
1. Section/date selection
   - Supervisor picks a Section (year-scoped) and a date.
   - Roster: current_enrollments_in_section(section)  [from academics]
   - Expected: filter roster by active subscriptions + exceptions for date

2. Student lookup (three input modes)
   a. QR/manual lookup: scan/enter StudentProfile.code or Person.code.
      - apps.identity.selectors.get_person_by_display_code resolves it.
   b. AI recognition input: a recent AttendanceEvent (face match) for a
      Person in this section's current roster. Meal reads the event via
      attendance's selector; it does not run recognition.
   c. Roster tap: select a student from the expected list directly.

3. Eligibility decision
   - resolve_eligibility(student, date) -> LunchEligibility
   - Display decision + reason + meal_plan + snapshot grade/section.

4. Balance warning placeholder
   - If meal_plan.mode == WALLET, call wallet.check_balance(person) (a
     wallet-domain service). Show a warning if insufficient and
     insufficient_funds_mode == DENY/ALLOW_UNPAID/ALLOW_NEGATIVE.
   - This is a *placeholder*: the wallet integration is via a service
     boundary (Section 8). Meal does not read Wallet.balance directly in
     its models; the dashboard calls the wallet service.

5. Override actions
   - Confirm / unconfirm / deny / refund / void / override eligibility.
   - Each action calls the corresponding meal service, which writes a
     LunchSupervisorAction and transitions the LunchServiceEvent status.

6. Audit history
   - For each student/date, show the LunchServiceEvent + all
     LunchSupervisorAction rows + the LunchEligibility snapshot.
   - History is read-only; corrections create new rows.
```

### 7.2 Dashboard composition

| Data | Source |
|---|---|
| Roster (current placements) | `apps.academics.selectors.current_enrollments_in_section` |
| Active subscriptions for date | `apps.meal.selectors.active_subscriptions_for_date` |
| Exceptions for date | `apps.meal.selectors.exceptions_for_date` |
| Eligibility snapshot | `apps.meal.selectors.eligibility_for` |
| Service events for section/date | `apps.meal.selectors.service_events_for_section` |
| Wallet balance (placeholder) | `apps.wallet.selectors.balance_for` (service boundary) |
| Recognition input | `apps.attendance.selectors.recent_events_for_section` |

### 7.3 Auth rule

Access is granted to `PersonRole(person, staff)` with a `meal_supervisor`
role metadata (or a future `RoleType` `meal_supervisor`), combined with a
Django Group for system access to the dashboard view. See `person_identity_architecture.md`
§18 (Django Groups vs PersonRole).

---

## 8. Future Wallet / Finance Integration

### 8.1 Integration boundary

- `apps.meal` does **not** compute balances or create `WalletTransaction`
  rows directly. It calls into `apps.wallet` services:
  - `wallet.charge(person, amount, *, reason, ref) -> WalletTransaction`
  - `wallet.refund(person, amount, *, original_transaction, reason) -> WalletTransaction`
  - `wallet.check_balance(person) -> (balance, sufficient_for_amount)`
- The meal service records the **result** (amount, balance before/after,
  transaction FK) as an immutable snapshot on `LunchServiceEvent`.

### 8.2 Charges

- When a `LunchServiceEvent` is confirmed and `meal_plan.mode == WALLET`:
  1. Meal computes the **base price** from `MealPlanPeriodPrice` for the
     relevant period.
  2. Meal calls finance to resolve discounts (Section 9), obtaining
     `discount_iqd` and `final_charge_iqd`.
  3. Meal calls `wallet.charge(person, final_charge_iqd, reason="lunch",
     ref=service_event)`.
  4. The returned `WalletTransaction` FK and the balance snapshots are
     stored on the service event.

### 8.3 Refunds

- A supervisor refund (if `meal_plan.allow_supervisor_refund`) calls
  `wallet.refund(...)` and stores the refund transaction FK on
  `LunchServiceEvent.wallet_refund_transaction`, transitioning the event
  to `REFUNDED`.

### 8.4 Balance checks

- The dashboard's balance warning calls `wallet.check_balance`. The
  decision to allow/deny service on insufficient funds follows
  `meal_plan.insufficient_funds_mode`, which is **meal-domain policy**
  (how the meal product reacts to a balance state), not wallet logic.

### 8.5 Payment source

- The wallet is the single payment source. Meal never accepts cash/card
  fields; those belong to a future `apps.finance.Receipt` if needed.

### 8.6 Pending integration boundaries

- Until `apps.wallet` is extracted and exposes the service boundary, meal
  charges are **disabled** (or run in a "dry-run"/`UNPAID` mode that
  records the intended charge without calling the wallet). This is the
  migration safety valve (Section 13).

---

## 9. Future Discount Integration

### 9.1 Where discount logic lives

- Discount **rules, profiles, and assignment resolution** live in
  `apps.finance` (`DiscountProfile`, `DiscountRule`, `DiscountAssignment`),
  scoped by `Person` and `AcademicYear`, per `education_domain_architecture.md`.
- `apps.meal` calls `finance.resolve_discount(person, base_price,
  academic_year, context) -> (discount_iqd, final_charge_iqd,
  applied_profiles)` and stores the result as a snapshot on the
  `LunchServiceEvent`.

### 9.2 Discount types

| Type | Source | Meal handling |
|---|---|---|
| Fixed amount | `DiscountProfile.type=fixed` | Finance returns `discount_iqd = min(value, base)`. |
| Percentage | `DiscountProfile.type=percentage` | Finance returns `discount_iqd = base * value / 100`. |
| Subsidy / scholarship | `DiscountAssignment` to `Person` or `Family` | Finance resolves; meal only records the snapshot. |
| Priority/exclusive | `DiscountProfile.priority` | Finance decides stacking/exclusivity (open question in academics/education docs). |

### 9.3 Where discount logic should NOT live

- **Not** on `MealPlan`: a plan holds a list price, not discount rules.
- **Not** on `LunchSubscription`: a subscription is an entitlement, not a
  pricing rule.
- **Not** on `LunchServiceEvent` as live logic: the event stores only the
  *snapshot* of the resolved discount.
- **Not** in the supervisor dashboard: the dashboard calls the meal
  service, which calls the finance service; the dashboard never computes
  discounts.

---

## 10. Future AI Attendance Integration

### 10.1 Recognition events as input

- `apps.attendance` produces `AttendanceEvent` rows for a `Person` (after
  the FK migration to Person). Meal consumes these as **inputs only**:
  `LunchServiceEvent.recognition_event` → `AttendanceEvent` (SET_NULL).
- Meal never imports the recognition engine, model weights, or camera
  logic.

### 10.2 Confidence

- The `AttendanceEvent` carries a `score` (confidence). The meal
  supervisor workflow may surface it ("recognized at 0.87") but does not
  threshold it — that is the attendance domain's `RecognitionSettings`
  policy.

### 10.3 Supervisor confirmation

- A recognition event creates a `LunchServiceEvent` in `PENDING` status.
  The supervisor confirms or denies it via a `LunchSupervisorAction`,
  transitioning the event to `CONFIRMED`/`DENIED`. This mirrors the
  legacy `MealRecord.status` flow but decoupled from a 1:1
  `AttendanceRecord` link.

### 10.4 Unknown person queue

- When recognition produces no match (unknown face), the attendance
  domain records an unidentified event. The supervisor dashboard may
  show an "unknown person" queue, but resolving it (assigning a Person)
  is an **attendance/identity** action, not a meal action. Meal only
  processes events that already resolve to a `Person`.

---

## 11. Recommended Model Boundaries

### Belongs in `apps.meal`

| Belongs in meal | Why |
|---|---|
| `MealPlan`, `MealPlanPeriodPrice` | Product/pricing definition + list price. |
| `LunchSubscription` | Per-student dated entitlement. |
| `LunchEligibility` | Canonical daily eligibility decision/snapshot (recalculable until a service event references it, then frozen). |
| `LunchServiceEvent` | Immutable service/attendance record with financial + academic snapshots. |
| `LunchSupervisorAction` | Audit trail of supervisor actions. |
| `LunchException` | One-time/temporary permission. |
| Subscription lifecycle, eligibility resolution, supervisor-action services | Meal business logic. |

### Does NOT belong in `apps.meal`

| Does NOT belong in meal | Where it belongs |
|---|---|
| Wallet balance, transactions, charges logic | `apps.wallet` (meal calls it) |
| Discount profiles/rules/assignment resolution | `apps.finance` (meal calls it) |
| Recognition engine, cameras, embeddings, scores | `apps.attendance` |
| Grade/Section/Enrollment/Placement models | `apps.academics` |
| Student identity (name/code/photo/DOB) | `apps.identity` |
| Period templates/occurrences (timetable) | `apps.scheduler` |
| Menu/kitchen inventory | future `apps.kitchen` (out of scope) |
| Invoice/receipt/payment method | `apps.finance` |

### Boundary justification

- Placing wallet balance on `LunchServiceEvent` as a **live field** would
  couple meal to wallet and break the append-only contract (balance
  changes would mutate historical records). Hence only a **snapshot** is
  allowed.
- Placing discount rules on `MealPlan` would couple meal to finance and
  duplicate the discount engine. Hence meal calls finance and records the
  result.
- Placing roster/grade/section as live fields on meal entities would
  duplicate academics and break history when a student transfers. Hence
  only snapshots on immutable records.

---

## 12. What Should NOT Be Implemented Yet

| Item | Why deferred |
|---|---|
| The `apps.meal` app itself | Design-only. Implementation needs a separate approved task following Section 15. |
| Wallet charge integration (live) | Until `apps.wallet` is extracted and exposes a service boundary, charges run in dry-run/`UNPAID` mode. |
| Discount resolution (live) | Until `apps.finance` discount engine exists, `discount_iqd=0` and `final_charge_iqd=base`. |
| AI recognition event consumption (live) | Until `AttendanceEvent.person` migration is done, recognition input is manual/QR only. |
| `LunchSupervisorAction.performed_by` (StaffProfile) population | Requires supervisor role/assignment modeling; can start with `performed_by_user` (auth.User) only. |
| Guest/person-based exceptions (non-student) | Defer until a non-student lunch use case exists; start with `student`-keyed exceptions. |
| Multi-meal-kind support (breakfast/snack) | Start with `kind=LUNCH` only; other kinds are additive later. |
| Eligibility superseding/multiple-rows-per-date | Start with one canonical `(student, date)` row that is recalculable until a service event references it, then frozen. No `superseded` flag/multiple rows. |
| Family-level billing/exceptions | Defer until guardian/family domain is built. |
| Removing legacy `MealSubscription`/`MealRecord` | Keep through the dual-FK migration phase; remove only after verification. |

---

## 13. Migration Safety Notes

When the meal app is eventually implemented, these rules apply. They are
**not** actions for this document.

### General rules

- **Additive first.** The meal app creates new tables only. It does not
  alter `StudentProfile`, `Person`, or `AcademicYear`/`Section`. Legacy
  `MealSubscription`/`MealRecord` remain.
- **Dual-FK pattern** for migrating off legacy `Student`:
  1. Add `student` FK to `StudentProfile` (nullable) on the new meal
     models.
  2. Data migration: populate `student_id` from
     `legacy_meal_subscription.student.migrated_to` (the StudentProfile).
  3. Enforce NOT NULL.
  4. Flip read paths.
  5. Later release: drop legacy `MealSubscription`/`MealRecord`.
- **PROTECT/SET_NULL** on meal FKs so historical service events survive
  plan/subscription deactivation.
- **One logical feature per migration.** Do not combine the meal app's
  `0001_initial` with unrelated schema changes.
- **Run `python manage.py makemigrations --check --dry-run` and
  `python manage.py check`** after each step.
- **Tests run under the CREATEDB-enabled role** (`TEST_DB_USER` /
  `TEST_DB_PASSWORD`), already configured in `bisk/settings.py`. New
  migrations must pass `python manage.py test apps.meal` (and
  `apps.identity`, `apps.academics`) before merge.
- **Backup the production database** before backfilling eligibility/
  service-event snapshots from legacy `MealRecord`.

### Snapshot immutability

- Once a `LunchServiceEvent` is confirmed, its financial and academic
  snapshot fields must not be edited by any code path. Enforce in
  `clean()` and in service methods (only `status`, `reason_*`, and
  refund/reversal fields are updatable post-confirmation).
- Refunds/voids create new `LunchSupervisorAction` rows and link a
  `wallet_refund_transaction`; they do not overwrite the original
  charge snapshot.

### Rollback

- Because the meal app is purely additive (new tables only), rollback
  means un-applying its migrations and removing the app from
  `INSTALLED_APPS`. No existing table is altered, so rollback is safe
  before the legacy removal step.

---

## 14. Open Questions / Decisions Needed

1. **App label: `meal` vs `lunch`.** The ERP sketch uses `apps/meal/`; the
   school's operational term is "lunch". Recommendation: `meal` app label
   with `Lunch*` entity names (supports future breakfast/snack). Confirm.

2. **Eligibility re-resolution after a service event.** This is settled
   by design (Section 6.2): once a `LunchServiceEvent` references the
   eligibility, the row is frozen and `resolve_eligibility` must not
   rewrite it; corrections are audited via `LunchSupervisorAction`. The
   remaining open question is whether to *also* allow a superseding
   `LunchEligibility` row with a `superseded` flag for forensic history
   (see Q9), or keep strictly one canonical frozen row. Recommendation:
   one canonical frozen row.

3. **"No current placement" handling.** Should a student with no current
   `StudentEnrollmentSectionPlacement` (withdrawn/graduated) be a hard
   `NOT_ELIGIBLE`, or a warning that still allows a supervisor override?
   Affects the eligibility resolver's first step.

4. **Subscription in-place edits vs cancel-and-recreate.** Are mid-window
   plan/priority changes allowed with an audit record, or must they
   cancel-and-recreate? Recommendation: cancel-and-recreate for history
   purity; confirm.

5. **`LunchException.person` for guests.** Should guest exceptions require
   a `Person` with a `PersonRole(guest)`, or allow any `Person`? Affects
   authorization checks on the dashboard.

6. **One-time permission and capacity.** Does a `ONE_TIME_ELIGIBLE`
   exception count against any section/meal capacity? (Meal plans have no
   inherent capacity; this may be moot, but confirm.)

7. **Supervisor identity: StaffProfile vs auth.User.** Should
   `LunchSupervisorAction.performed_by` (StaffProfile) be required, or is
   `performed_by_user` (auth.User) sufficient until supervisor
   assignments are modeled? Recommendation: start with
   `performed_by_user` required, `performed_by` optional.

8. **Wallet integration dry-run mode.** Should the initial meal
   implementation support a `WALLET_DRY_RUN` setting that records
   intended charges as `UNPAID` without calling the wallet? Affects
   migration safety.

9. **Multiple `LunchEligibility` rows per date.** The canonical row is
   recalculable before a service event exists and frozen afterward. Allow
   a `superseded` flag to preserve prior recalculation history, or enforce
   a single row that is simply overwritten until frozen? Recommendation:
   single canonical row (overwrite-until-frozen); the `LunchSupervisorAction`
   audit trail captures override history.

10. **Eligibility for non-enrolled students with active subscriptions.** A
    student who is between enrollments (e.g. admitted but not yet
    enrolled for the year) may have an active subscription by date.
    Eligible or not? Tied to Q3.

11. **Meal plan period price currency.** Confirm IQD integer fields are
    correct vs. Decimal. Legacy uses `IntegerField`; finance docs may
    prefer `DecimalField`. Align with finance domain decision.

12. **AI recognition event link cardinality.** Can multiple
    `LunchServiceEvent`s link to one `AttendanceEvent` (e.g. lunch +
    snack), or is it 1:1? Affects the FK design.

---

## 15. Recommended Implementation Order

Each step is a **separate, approved task**. This document implements
nothing.

| Step | Deliverable | Depends on | Notes |
|---|---|---|---|
| 0 | Identity + Academics foundations | — | **DONE (identity)** / **DESIGNED (academics)**. Meal needs `StudentProfile`, `AcademicYear`, `Section`, current placement. |
| 1 | `apps.meal` app skeleton (`apps.py`, empty `models.py`, `INSTALLED_APPS`) | identity, academics | Additive; no models yet. |
| 2 | `MealPlan` + `MealPlanPeriodPrice` models + admin + seed from legacy `MealProfile`/`MealProfilePeriod` | step 1 | Product/pricing definition. |
| 3 | `LunchSubscription` model (FK to `StudentProfile`, optional `AcademicYear`) + admin + services (`create_subscription`, lifecycle transitions) + validators (overlap) | step 2 | Migrates legacy `MealSubscription`. |
| 4 | `LunchException` model + `grant_one_time_permission` service | step 3 | One-time permission support. |
| 5 | `LunchEligibility` model + `resolve_eligibility` service + selectors (`eligibility_for`, `eligible_students_for_date`) | steps 3, 4, academics | Daily eligibility snapshot. |
| 6 | `LunchServiceEvent` model + `record_service_event` service (dry-run/`UNPAID` mode, no live wallet) + selectors | step 5 | Immutable service record; no live charges yet. |
| 7 | `LunchSupervisorAction` model + supervisor-action services (`confirm`, `unconfirm`, `deny`, `refund`, `void`, `override`) | step 6 | Audit trail. |
| 8 | Supervisor dashboard view layer (roster, lookup, eligibility, override, audit history) | steps 5–7, academics selectors | Owns no models. |
| 9 | Wallet integration: replace dry-run with `wallet.charge`/`refund` calls | step 6, `apps.wallet` extraction | Live charges. |
| 10 | Discount integration: call `finance.resolve_discount` for `final_charge_iqd` | step 9, `apps.finance` | Live discounts. |
| 11 | AI attendance integration: consume `AttendanceEvent.person` as `recognition_event` input | step 6, attendance FK migration to Person | Recognition-driven service events. |
| 12 | (Later phase) Legacy `MealSubscription`/`MealRecord` removal after verification | steps 3–11 verified | Drop legacy tables. |

### Ordering rationale

- `MealPlan` (step 2) precedes `LunchSubscription` (step 3) because
  subscriptions reference plans.
- `LunchEligibility` (step 5) precedes `LunchServiceEvent` (step 6)
  because a service event references the eligibility; that reference is
  what freezes the eligibility row.
- Wallet (step 9), discount (step 10), and AI (step 11) integrations are
  deliberately late and parallelizable; the meal domain functions in
  dry-run/manual mode until then.
- Legacy removal (step 12) is last and conditional on full verification,
  preserving backward compatibility throughout.

---

## Appendix A — Lunch Domain Relationship Diagram (text)

```
                         auth.User (login identity)
                              │ optional O2O
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                            Person                                 │
│  code  = global ERP identifier                                   │
│  display_code → StudentProfile.code / StaffProfile.code / code    │
└───────────────┬───────────────────────────────────────────────────┘
                │ O2O            │ O2O
   ┌────────────┴───────┐   ┌────┴────────────┐
   │ StudentProfile    │   │ StaffProfile    │
   │ code (student op) │   │ code (staff op) │
   └────────┬───────────┘   └────────┬────────┘
            │                          │ (supervisor)
            │                          │
   ┌────────┴──────────────────┐       │
   │ LunchSubscription         │       │
   │  student ──► StudentProfile│      │
   │  meal_plan ──► MealPlan   │       │
   │  academic_year ──► AcademicYear  │
   └────────┬──────────────────┘       │
            │                          │
   ┌────────┴──────────────────┐       │
   │ LunchEligibility (daily)  │       │
   │  student, date (unique)   │       │
   │  subscription ──► LunchSubscription
   │  grade/section snapshot   │       │
   └────────┬──────────────────┘       │
            │                          │
   ┌────────┴──────────────────┐       │
   │ LunchServiceEvent         │◄──────┘ LunchSupervisorAction
   │  student, date            │       │  (performed_by ──► StaffProfile)
   │  eligibility ──► LunchEligibility│
   │  recognition_event ──► AttendanceEvent (optional, AI input)
   │  wallet_transaction ──► WalletTransaction (created by wallet domain)
   │  price/discount/charge/balance snapshots
   └───────────────────────────┘

LunchException ──► StudentProfile | Person  (one-time/temporary)
MealPlan ──► MealPlanPeriodPrice ──► PeriodTemplate (scheduler)

AcademicYear ──► Section ──► StudentEnrollmentSectionPlacement (current roster)
                            └─► StudentEnrollment ──► StudentProfile
```

All arrows point from the depending module to the owning module. Meal owns
the Lunch* entities; identity owns Person/StudentProfile/StaffProfile;
academics owns AcademicYear/Section/Enrollment/Placement; wallet owns
WalletTransaction; attendance owns AttendanceEvent.

---

## Appendix B — Snapshot Justification Summary

| Snapshot field | On entity | Why justified |
|---|---|---|
| `grade_code_snapshot` | `LunchEligibility`, `LunchServiceEvent` | Student may transfer sections or the enrollment may be archived; the grade at resolution/service time cannot be reconstructed. On `LunchEligibility` the snapshot is recalculable until a service event references it, then frozen. On `LunchServiceEvent` it is immutable historical truth. |
| `section_code_snapshot` | `LunchEligibility`, `LunchServiceEvent` | Same; current placement may be closed after service. |
| `price_base_iqd` | `LunchServiceEvent` | Plan prices change over time; the list price at service time is immutable historical truth. |
| `discount_iqd` | `LunchServiceEvent` | Discount rules/assignments change; the applied discount is immutable historical truth. |
| `final_charge_iqd` | `LunchServiceEvent` | The amount actually charged; immutable for audit. |
| `wallet_balance_before_iqd` / `after_iqd` | `LunchServiceEvent` | Wallet balance changes continuously; the balance at charge time is immutable historical truth. |

No live (non-snapshot) identity, academic, wallet, or discount fields exist
on meal entities other than the FKs needed to reference the owning rows.

---

## Appendix C — Document Hierarchy

```
docs/development/PROJECT_ARCHITECTURE.md            (standards)
docs/development/AI_DEVELOPMENT_GUIDE.md            (AI workflow)
docs/architecture/person_identity_architecture.md   (identity foundation — implemented)
docs/architecture/erp_foundation_architecture.md    (top-level ERP blueprint)
docs/architecture/education_domain_architecture.md  (education domain — broad)
docs/architecture/academics_domain_architecture.md  (academics domain — enrollment/section)
docs/architecture/lunch_domain_architecture.md      (THIS document — lunch domain)
```

This document is consistent with all of the above. Where they describe
implemented models, this document references them; where they describe
future models, this document narrows the lunch subset and adds lifecycle,
eligibility, supervisor-workflow, integration-boundary, safety, and
open-question detail specific to the lunch domain.

---

End of document.
