# Meals Domain Architecture — BISK_RFv4

Date: 2026-07-04 (revised 2026-07-04)
Branch: feature/person-architecture
Version: 1.1 Draft — pending review
Status: Architecture design only. No code, models, or migrations are produced by this document.

---

## 1. Purpose

This document designs the **Meals domain** (`apps.meals`) for BISK_RFv4 —
meal plans, meal periods, meal subscriptions, per-Person and per-MealPeriod
pricing, daily meal eligibility, meal service events, supervisor workflow,
and the integration boundaries with the Identity, Finance, Attendance, and
AI-Recognition subsystems.

The current implementation lives scattered inside `apps.attendance`
(`MealSubscription`, `MealProfile`, `MealProfilePeriod`, `MealRecord`,
`Wallet`, `WalletTransaction`, `DiscountProfile`, `DiscountRule`), all
currently FK-keyed to the legacy `attendance.Student`. This document
designs a clean `apps.meals` domain that:

- Anchors to `Person` / `StudentProfile` / `StaffProfile` (never to legacy
  `Student`), so both students and staff can be served meals.
- **Owns meal subscriptions and price resolution.** Meal-period pricing,
  per-Person pricing overrides, and the composition of the final charge
  (base price → per-Person override → discount) all live in `apps.meals`.
- Delegates **money movement** (wallet debit/credit, ledger append, balance
  mutation) exclusively to `apps.finance` services. Meals never reads or
  writes `Wallet.balance`, never creates `WalletTransaction` rows, and
  never computes `balance_before`/`balance_after`.
- Delegates **recognition** to `apps.attendance` and consumes recognition
  events as inputs only. Meals does not run recognition, does not own
  cameras or embeddings, and does not gate pricing on confidence scores.
- Preserves **historical records immutably**: eligibility snapshots and
  service events are append-only; corrections create new rows, never
  mutate prior rows.

This is a **design document**, not an implementation order. Each entity
below is a blueprint for a future, separately-approved implementation step.

### Decided architecture (this document is the authoritative expression)

The following decisions are **settled** for this document and are not open
questions (see §26 for the full decided/open list):

| # | Decision |
|---|---|
| 1 | **App name is `apps.meals`** (plural). It supports breakfast, lunch, snack, and any future meal kind. "Lunch" is one meal kind, not the app name. |
| 2 | **The future model name is `MealPlan`.** Legacy `attendance.MealProfile` maps to `MealPlan` during migration. `MealProfile` is used in this document **only** when discussing the legacy model. |
| 3 | **Meals owns price resolution.** Base price, per-Person overrides, per-period overrides, and final-charge composition all live in `apps.meals`. |
| 4 | **Finance records pre-resolved charges only.** `finance.charge` receives a pre-resolved `amount_iqd` from Meals; it does not look up meal prices, does not resolve per-Person overrides, and does not call the discount service. |
| 5 | **Granular pricing per Person and per MealPeriod belongs to Meals.** A person-specific price and a period-specific price are meal-product facts, not ledger facts. |
| 6 | **Wallet mutations happen only through Finance services.** Meals calls `finance.charge` / `finance.refund` / `finance.check_balance`; it never touches the wallet model directly. |
| 7 | **Attendance does not own pricing.** Attendance produces recognition events; it never resolves or snapshots a meal price. |
| 8 | **AI recognition does not own pricing.** Recognition produces a Person-identified event; pricing is resolved downstream by Meals. |
| 9 | **Legacy behavior is preserved through migration.** Legacy `MealSubscription`, `MealProfile`, `MealProfilePeriod`, `MealRecord`, `Wallet`, `WalletTransaction`, `DiscountProfile`, `DiscountRule` keep working until the dual-FK migration is verified and the legacy tables are explicitly dropped. |

> Relationship to prior documents: This document **supersedes the pricing
> placement** described in `lunch_domain_architecture.md` (which deferred
> `MealPlanPeriodPrice` list-price ownership to a future Finance
> `PricingRule`), `finance_domain_architecture.md` §4.7 / §7.1 / §8 (which
> placed `PricingRule`/`PriceList` in Finance and had `finance.charge` call
> the discount service internally), and `DOMAIN_INTEGRATION_GUIDE.md` §10 /
> §13.1 / §13.3 (which describe `finance.charge` resolving pricing and
> discounts). Under the decided architecture above, **list-price ownership
> and price resolution move to Meals**, and `finance.charge` records a
> pre-resolved amount. The deviation is documented in the review package
> `architecture_review.md`; a follow-up docs task should reconcile the three
> prior documents with this one.

### Scope

| In scope | Out of scope |
|---|---|
| MealPeriod, MealPlan, MealSubscription, MealServiceEvent, MealSupervisorAction, MealException | Wallet / ledger model design (see `finance_domain_architecture.md`) |
| Per-Person and per-MealPeriod pricing, default pricing, overrides | Discount rule/profile/assignment internals (future `apps.discounts`) |
| Daily eligibility calculation, snapshotting, manual overrides | AI face-recognition engine internals (see `attendance_*` docs) |
| Supervisor dashboard workflow and audit | Portal / mobile API design |
| Integration boundaries with Identity, Finance, Attendance, AI Recognition | Menu/kitchen inventory management |
| Recommended model boundaries and implementation order | Multi-tenant activation |

### Constraints respected

- Do **not** recreate the old `Student` model. Meals references
  `StudentProfile` and `StaffProfile` (and `Person`), never legacy
  `attendance.Student`.
- Meals references `StudentProfile` / `StaffProfile` and the **active
  enrollment** where grade/section/roster context is needed — it does not
  duplicate student name, grade, or section fields unless snapshotting on
  an immutable historical record is explicitly justified (§12, §21).
- Meals **preserves historical records**: eligibility snapshots and
  service events are append-only and never overwritten.
- Meals **must not contain wallet/finance calculations directly**; it calls
  into `apps.finance` via a service boundary (§20).
- Meals **must not contain AI recognition engine logic directly**; it
  consumes recognition events as inputs (§6, §7).
- Meals **must not import** `apps.attendance`, `apps.finance`, or
  `apps.discounts` model classes into its own models beyond typed FKs to
  `Person`/`StudentProfile`/`StaffProfile`/`AcademicYear`/`Section` and a
  typed FK to `wallet.WalletTransaction` (the reverse of the finance
  generic reference — see §20).
- Uses the `services.py` / `selectors.py` / `validators.py` pattern
  established by `apps.identity` and `apps.academics`, per
  `DOMAIN_INTEGRATION_GUIDE.md` §3.
- Follows `education_domain_architecture.md`,
  `academics_domain_architecture.md`, `person_identity_architecture.md`,
  `finance_domain_architecture.md`, `lunch_domain_architecture.md`,
  `DOMAIN_INTEGRATION_GUIDE.md`, `PROJECT_ARCHITECTURE.md`, and
  `AI_DEVELOPMENT_GUIDE.md`.

---

## 2. Legacy behavior summary

The legacy meal/lunch stack lives in `apps/attendance/models.py`. It is
FK-keyed to `attendance.Student` and tightly coupled to
`AttendanceRecord`. The relevant legacy models:

| Legacy model | Purpose | Key fields |
|---|---|---|
| `MealSubscription` | A student's dated entitlement to meals under a `MealProfile`. | `student` (FK→Student), `meal_profile`, `plan_type` (annual/monthly/other), `status` (active/cancelled/expired), `start_date`, `end_date`, `priority`, `source` (manual/dashboard_postpaid). |
| `MealProfile` | A named pricing/policy profile (the legacy "plan"). | `name`, `mode` (date_range/wallet), `discount_profile`, `insufficient_funds_mode` (deny/allow_unpaid/allow_negative), `credit_limit_iqd`, supervisor-override flags, `require_reason_on_*`. |
| `MealProfilePeriod` | Per-period list price attached to a `MealProfile`. | `meal_profile`, `period_template` (FK→`attendance.PeriodTemplate`), `price_iqd`, `is_enabled`. `unique_together=("meal_profile","period_template")`. |
| `MealRecord` | The immutable per-occurrence service/charge record. 1:1 with `AttendanceRecord`. | `attendance_record` (OneToOne), `meal_subscription`, `meal_profile`, `eligible_at_time`, `mode_snapshot`, `status` (pending/confirmed/denied/unpaid/refunded/voided), `price_base_iqd`, `discount_iqd`, `final_charge_iqd`, `wallet_balance_before_iqd`, `wallet_balance_after_iqd`, `wallet_transaction`, `wallet_refund_transaction`, `reason_code`, `confirmed_at/by`, `reversed_at/by`. |
| `Wallet` / `WalletTransaction` | Prepaid balance and append-only ledger, FK→Student. | See `finance_domain_architecture.md`. |
| `DiscountProfile` / `DiscountRule` | Discount profiles with fixed/percent rules, optionally scoped to a `PeriodTemplate`. | `rule_type` (fixed/percent), `value_iqd`, `value_percent`, `period_template`, `min_same_day_confirmed_meals`, `priority`. |
| `PeriodTemplate` | Daily period definition (start/end time, weekdays mask, grace minutes). | Owned by `apps.attendance` today; the migration target is `apps.scheduler` (or a `MealPeriod` wrapper in `apps.meals` — §9). |

### Legacy behavior to preserve

1. **Date-range and wallet modes.** A subscription is either a paid
   date-range entitlement (no per-service charge) or a per-service wallet
   charge. `MealProfile.mode` selects the behavior; `MealRecord.mode_snapshot`
   records which mode produced a given service event.
2. **Priority-based subscription overlap.** Multiple `ACTIVE` subscriptions
   may overlap for the same student **only across different priorities**
   (1=primary, 2=fallback). Same-priority overlap is blocked in
   `MealSubscription.clean()` (legacy `apps/attendance/models.py:610`).
3. **Per-period pricing.** `MealProfilePeriod.price_iqd` sets the list price
   per `PeriodTemplate`. A blank `period_template` on a `DiscountRule` means
   "all periods".
4. **Insufficient-funds policy.** `insufficient_funds_mode` decides whether
   to deny, record as unpaid, or allow a negative balance (down to
   `credit_limit_iqd`).
5. **Supervisor actions.** Confirm / unconfirm / deny / refund / void, gated
   by `allow_supervisor_*` and `require_reason_on_*` flags on the profile.
6. **Immutable financial snapshot on the service record.** `MealRecord`
   stores `price_base_iqd`, `discount_iqd`, `final_charge_iqd`,
   `wallet_balance_before_iqd`, `wallet_balance_after_iqd`, and the
   `wallet_transaction` FK. These are historical truth, not live values.
7. **1:1 coupling of `MealRecord` to `AttendanceRecord`.** This is a legacy
   constraint, not a desired future. The migration target
   (`MealServiceEvent`) decouples service events from attendance records
   (§12) while keeping an optional `recognition_event` FK.

### Legacy behavior to abandon (with migration care)

- The 1:1 `MealRecord.attendance_record` coupling (replaced by an optional
  `MealServiceEvent.recognition_event` FK).
- FK-to-`Student` everywhere (replaced by `Person` / `StudentProfile` /
  `StaffProfile`).
- Pricing scattered across `MealProfile` + `MealProfilePeriod` +
  `DiscountRule.period_template` (consolidated into the Meals pricing
  stack: §14–§18).
- `Wallet` / `WalletTransaction` living inside `apps.attendance` (moved to
  `apps.finance`; Meals only reads a snapshot FK).
- The `MealProfile` model name (renamed to `MealPlan` in the new domain;
  legacy `MealProfile` maps to `MealPlan` during migration — §10, §21).

---

## 3. Domain boundaries

```
apps.meals ──depends on──► apps.identity    (Person, StudentProfile, StaffProfile)
apps.meals ──depends on──► apps.academics   (AcademicYear, Section, Enrollment, current placement)
apps.meals ──depends on──► apps.scheduler   (PeriodTemplate / PeriodOccurrence; or MealPeriod — §9)
apps.meals ──calls (service boundary)──► apps.finance    (charge / refund / check_balance)
apps.meals ──calls (service boundary)──► apps.discounts   (discount resolution; future — §19)
apps.meals ──consumes──► apps.attendance    (recognition events as input only — §6, §7)
```

### What Meals owns

| Owns | Why |
|---|---|
| `MealPeriod` (or wrapping of `scheduler.PeriodTemplate`) | The meal-serving window is a meal-product concept (which periods serve lunch / breakfast / snack), not a ledger concept. |
| `MealPlan` | Product definition: kind, mode, supervisor policy, insufficient-funds policy, default price. (Migration target of legacy `MealProfile`.) |
| `MealPeriodPrice` (per `MealPlan` × `MealPeriod`) | The list price per period — a meal-product fact. |
| `MealPersonPriceOverride` (per `Person` × `MealPlan`/`MealPeriod`) | Per-Person granular pricing — a meal-product fact (§15). |
| `MealSubscription` | Per-Person dated entitlement (date-range or wallet). |
| `MealEligibility` | Canonical daily eligibility snapshot. |
| `MealServiceEvent` | Immutable service/charge record with financial + academic snapshots. |
| `MealSupervisorAction` | Audit trail of supervisor actions. |
| `MealException` | One-time / temporary permission or denial. |
| `resolve_service(...)`, `resolve_price(...)`, subscription lifecycle, eligibility resolution, supervisor-action services | Meal business logic. |

### What Meals does NOT own

| Does NOT own | Where it belongs |
|---|---|
| Wallet balance, transactions, ledger math | `apps.finance` (Meals calls it — §20) |
| Discount profiles/rules/assignment resolution | future `apps.discounts` (Meals calls it — §19) |
| Recognition engine, cameras, embeddings, scores | `apps.attendance` |
| Grade/Section/Enrollment/Placement models | `apps.academics` |
| Student/Staff identity (name/code/photo/DOB) | `apps.identity` |
| Period templates/occurrences (timetable) | `apps.scheduler` (Meals may wrap them as `MealPeriod` — §9) |
| Menu/kitchen inventory | future `apps.kitchen` (out of scope) |
| Invoice/receipt/payment method | `apps.finance` |

### Dependency rules

- A dependency `finance → meals` is **not allowed**. Finance records money
  for any source module; it does not import meals models.
- A dependency `attendance → meals` is **not allowed**. Recognition produces
  a Person-identified event; Meals reads it, not the reverse.
- A dependency `discounts → meals` is **not allowed**. Discounts resolve
  against `(Person, product_code, academic_year, context)`; Meals calls the
  result.
- Meals may import `apps.identity`, `apps.academics`, and (optionally)
  `apps.scheduler` model classes for typed FKs. Meals may **not** import
  `apps.finance`, `apps.discounts`, or `apps.attendance` model classes
  except for the single typed `wallet.WalletTransaction` FK stored on
  `MealServiceEvent` (the reverse of Finance's generic reference — §20).

---

## 4. Relationship with Identity

- All meal entities that attach to a person FK to `identity.Person`,
  `identity.StudentProfile`, or `identity.StaffProfile` — never to legacy
  `attendance.Student`.
- `MealSubscription.person` → `Person` (the canonical anchor). A
  subscription belongs to a Person; whether that Person is a student or
  staff is answered by `PersonRole`, not by the subscription.
- `MealPersonPriceOverride.person` → `Person` (per-Person pricing — §15).
- `MealEligibility`, `MealServiceEvent`, `MealException` carry a `person`
  FK to `Person`; for students they also carry a `student` FK to
  `StudentProfile` for fast student-scoped queries and academic snapshot
  joins; for staff they may carry a `staff` FK to `StaffProfile` (§17).
- Person identity (name, code, photo) is read via `person.full_name` /
  `person.display_code` / `person.photo`. **No** `h_code`,
  `first_name`/`last_name` are duplicated on meal entities.
- "Is this person entitled to a meal today?" combines:
  - `PersonRole(person, student)` (timeless role) **or**
    `PersonRole(person, staff)` (§17), with
  - an active `MealSubscription` (dated entitlement), **or**
  - an active `MealException` covering the date.

---

## 5. Relationship with Finance

> Decided architecture: **Finance records money only.** Meals owns price
> resolution. This is the key integration boundary.

### 5.1 The boundary

- `apps.finance` owns `Wallet`, `WalletTransaction`, and the ledger math.
  It exposes a service boundary (§20):
  ```python
  finance.charge(*, person, amount_iqd, source_module, reference_type="",
                 reference_id=None, academic_year=None, description="") -> WalletTransaction
  finance.refund(*, person, original_transaction, amount_iqd=None,
                 reason_code="", approved_by=None) -> WalletTransaction
  finance.check_balance(*, person, amount_iqd=0) -> tuple[int, bool]
  ```
- `finance.charge` receives a **pre-resolved `amount_iqd`** from Meals. It
  does **not** look up meal prices, does **not** call the discount service,
  and does **not** resolve per-Person overrides. It only:
  1. validates the wallet is active and funds/credit suffice (or returns an
     UNPAID transaction per finance policy),
  2. appends a signed `WalletTransaction` row,
  3. updates `Wallet.balance_iqd` atomically inside the same transaction,
  4. returns the `WalletTransaction` (with `balance_before`/`after`
     snapshots) to the caller.
- Meals stores the returned `WalletTransaction` FK and the
  `balance_before`/`balance_after` snapshots on the immutable
  `MealServiceEvent` (§12).

### 5.2 Why price resolution lives in Meals, not Finance

- The list price of a meal period, a per-Person override, and a per-period
  override are **meal-product facts**: they describe what the school is
  selling, to whom, at which window. They change with the meal program, not
  with the ledger.
- Putting them in Finance would couple Finance to meal-product concepts
  (`MealPeriod`, `MealPlan`, student-vs-staff pricing) and break the
  "Finance records money only" invariant.
- Discounts remain a **separate** future domain (`apps.discounts`) that
  Meals calls during final-charge composition (§19). Finance never calls
  the discount service under this architecture.

### 5.3 What Meals must NOT do

- Read or write `Wallet.balance_iqd`.
- Create `WalletTransaction` rows.
- Compute `balance_before_iqd` / `balance_after_iqd`.
- Accept cash/card/payment-method fields (those belong to Finance
  `Payment`).
- Import `apps.finance` models except the `WalletTransaction` FK stored on
  `MealServiceEvent`.

### 5.4 Reconciliation

- The `MealServiceEvent.wallet_transaction` FK + `wallet_balance_before/after`
  snapshot is Meals' auditable copy. The authoritative ledger row lives in
  `apps.finance`. Reconciliation = "every confirmed `MealServiceEvent` has
  exactly one `wallet_transaction` and its `final_charge_iqd` equals the
  transaction's signed `amount_iqd` magnitude." Drift is a bug, not a
  balance to silently overwrite.

---

## 6. Relationship with Attendance

- `apps.attendance` produces `AttendanceEvent` rows for a `Person` (after
  the Person-FK migration). Meals consumes these as **inputs only**:
  `MealServiceEvent.recognition_event` → `attendance.AttendanceEvent`
  (SET_NULL).
- Attendance does **not** own meal pricing, meal eligibility, or meal
  service records. The legacy `MealRecord.attendance_record` 1:1 coupling
  is abandoned; the migration target is an **optional** recognition link.
- Attendance does **not** write `MealServiceEvent` rows. The supervisor
  workflow (or a meal-domain consumer of recognition) creates the
  `MealServiceEvent` in `PENDING` status and then confirms/denies it (§7,
  §12).
- Absence is a **signal in**: Meals reads `AttendanceRecord.status` for
  the person/date to factor absence into eligibility (§13). Meals does not
  compute attendance.

### 6.1 Legacy `PeriodTemplate` ownership

- Legacy `PeriodTemplate` lives in `apps.attendance`. The migration target
  is `apps.scheduler` (period templates / occurrences as a generic
  timetable primitive). Until `apps.scheduler` exists, Meals may wrap the
  legacy `attendance.PeriodTemplate` in a `MealPeriod` join model (§9) so
  that Meals never imports `apps.attendance` models directly — the
  `MealPeriod` row carries a generic `period_template_id` (integer) plus a
  `source` tag, and is migrated to a typed FK once `apps.scheduler` lands.

---

## 7. Relationship with AI Recognition

- AI recognition produces a Person-identified `AttendanceEvent` with a
  confidence `score`. Meals consumes the event as the trigger for a
  `MealServiceEvent` in `PENDING` status; the supervisor confirms or
  denies it (§12).
- Meals **does not** run recognition, **does not** import recognition
  engine code, **does not** own cameras or embeddings, and **does not**
  threshold confidence. Confidence thresholds are attendance-domain
  `RecognitionSettings` policy.
- Meals may **display** the confidence ("recognized at 0.87") on the
  supervisor dashboard for situational awareness, but the deny/confirm
  decision is a supervisor action, not an automatic function of the score.
- "Unknown person" queues (recognition with no Person match) are an
  **attendance/identity** responsibility. Meals only processes events that
  already resolve to a `Person`.
- Pricing is **never** a function of recognition. The recognition event
  triggers the service event; the price is resolved from
  `MealPlan`/`MealPeriod`/`MealPersonPriceOverride` + discounts (§14).

---

## 8. Proposed models

All designs below are **conceptual blueprints**. No migration is produced
by this document. Field types are indicative; exact choices are made at
implementation time. Integer IQD amounts follow the legacy convention
(matching `finance_domain_architecture.md` Q11 — stay integer until a
fractional-currency requirement appears).

```
apps/meals/
├── __init__.py
├── apps.py
├── models.py        # MealPeriod, MealPlan, MealPeriodPrice,
│                    # MealPersonPriceOverride, MealSubscription,
│                    # MealEligibility, MealServiceEvent,
│                    # MealSupervisorAction, MealException
├── admin.py
├── services.py      # create_subscription, pause, cancel, resolve_eligibility,
│                    # resolve_service, resolve_price, record_service_event,
│                    # supervisor_override, ...
├── selectors.py     # active_subscriptions_for, eligible_persons_for_date,
│                    # service_events_for_section, price_for, ...
├── validators.py    # subscription overlap, eligibility window, pricing conflicts, ...
└── migrations/
```

**App label:** `meals` (Django app). Python module path: `apps.meals`.

> Naming: the app label is `meals` (plural); entity names use `Meal*`
> (not `Lunch*`) to make student + staff support and multiple meal kinds
> (breakfast, lunch, snack) first-class. The plan model is `MealPlan`
> (decided; migration target of legacy `MealProfile`).

### Entity map

| Entity | Section | Owner | Immutable? |
|---|---|---|---|
| `MealPeriod` | §9 | meals | configurable |
| `MealPlan` | §10 | meals | configurable |
| `MealPeriodPrice` | §10, §16 | meals | configurable (snapshot on service event) |
| `MealPersonPriceOverride` | §15 | meals | configurable (snapshot on service event) |
| `MealSubscription` | §11 | meals | append-only status transitions |
| `MealEligibility` | §12.1 | meals | recalculable until a service event references it, then frozen |
| `MealServiceEvent` | §12 | meals | immutable after confirmation |
| `MealSupervisorAction` | §12.2 | meals | append-only |
| `MealException` | §11.2 | meals | append-only |

---

## 9. MealPeriod

- **Owner:** `apps.meals`
- **Purpose:** A meal-serving window — the linkage between a generic
  timetable period (`scheduler.PeriodTemplate`, or legacy
  `attendance.PeriodTemplate`) and the meal domain. A `MealPeriod` says
  "this period template is a *meal* period (e.g. lunch block 1)", with
  optional meal-kind scoping.
- **Why a wrapper instead of reusing `PeriodTemplate` directly:**
  - `PeriodTemplate` is a generic timetable primitive owned by
    `apps.scheduler` (migration target) / `apps.attendance` (legacy). Not
    every period is a meal period. `MealPeriod` is the meal-domain
    declaration of which periods serve meals and for which meal kind.
  - It keeps `apps.meals` from importing `apps.attendance` models: the
    `MealPeriod` carries the period identity (a typed FK to
    `scheduler.PeriodTemplate` once it exists, or a generic
    `period_template_id` + `source` until then).
- **Conceptual fields:**

```python
class MealPeriod(models.Model):
    class Kind(models.TextChoices):
        LUNCH = "lunch", "Lunch"
        BREAKFAST = "breakfast", "Breakfast"
        SNACK = "snack", "Snack"

    kind = models.CharField(max_length=20, choices=Kind.choices,
                            default=Kind.LUNCH, db_index=True)
    # Typed FK once apps.scheduler exists. Until then, a generic
    # reference (no apps.attendance import in apps.meals models).
    period_template = models.ForeignKey(
        "scheduler.PeriodTemplate", on_delete=models.PROTECT,
        related_name="meal_periods", null=True, blank=True,
    )
    # Generic fallback used during the migration window only.
    period_template_source = models.CharField(max_length=20, blank=True, default="")
    period_template_ref_id = models.PositiveBigIntegerField(null=True, blank=True)
    label = models.CharField(max_length=64, blank=True, default="",
                             help_text="Optional display override, e.g. 'First lunch block'.")
    is_active = models.BooleanField(default=True, db_index=True)
    sort_order = models.PositiveSmallIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["kind", "sort_order", "id"]
        indexes = [
            models.Index(fields=["kind", "is_active"]),
        ]
```

- **Rules:**
  - One `MealPeriod` per `(kind, period_template)` is recommended
    (enforced by a unique constraint at implementation time once the FK
    target is fixed).
  - `MealPeriod` does **not** store a price. The price is on
    `MealPeriodPrice` (§10, §16), keyed by `MealPlan` × `MealPeriod`.
  - `MealPeriod` is referenced by `MealPeriodPrice` and by
    `MealPersonPriceOverride` (§15). The `MealServiceEvent` records the
    `MealPeriod` snapshot at service time.

---

## 10. MealPlan

- **Owner:** `apps.meals`
- **Purpose:** A reusable meal offering definition (the "product"): a
  named plan, its meal kind, mode (date-range vs wallet), supervisor
  override policy, insufficient-funds policy, and default price. This is
  the migration target for legacy `attendance.MealProfile`.
- **Naming decision (settled):** The future model name is **`MealPlan`**.
  Legacy `attendance.MealProfile` maps to `MealPlan` during migration
  (§21). `MealProfile` is used in this document **only** when referring to
  the legacy model.
- **Conceptual fields:**

```python
class MealPlan(models.Model):
    class Kind(models.TextChoices):
        LUNCH = "lunch", "Lunch"
        BREAKFAST = "breakfast", "Breakfast"
        SNACK = "snack", "Snack"

    class Mode(models.TextChoices):
        DATE_RANGE = "date_range", "Date-range"   # paid entitlement, no per-service charge
        WALLET = "wallet", "Wallet"               # per-service wallet charge while active

    class InsufficientFundsMode(models.TextChoices):
        DENY = "deny", "Deny"
        ALLOW_UNPAID = "allow_unpaid", "Allow unpaid"
        ALLOW_NEGATIVE = "allow_negative", "Allow negative"

    name = models.CharField(max_length=100, unique=True)
    kind = models.CharField(max_length=20, choices=Kind.choices, default=Kind.LUNCH)
    mode = models.CharField(max_length=20, choices=Mode.choices, default=Mode.DATE_RANGE)
    is_active = models.BooleanField(default=True)
    # policy flags (migrated from legacy MealProfile)
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
    # Optional: a default base price used when no MealPeriodPrice or
    # MealPersonPriceOverride matches (§18). Integer IQD.
    default_price_iqd = models.IntegerField(default=0)
    notes = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

### 10.1 Mode semantics (date-range vs wallet)

A `MealPlan` has a **mode** that decides how a subscription under it is
charged. **Both modes may use a date range on the subscription**
(`start_date`/`end_date`, §11). The mode determines what the date range
*means*:

| Mode | Date range meaning | Charging |
|---|---|---|
| `DATE_RANGE` | The subscription is a **paid entitlement** valid within `[start_date, end_date]`. | **No per-service wallet debit.** The plan was paid upfront (or is free); a service event snapshots `final_charge_iqd = 0` and `price_resolution_source = "date_range_no_charge"`. |
| `WALLET` | The subscription is **active** within `[start_date, end_date]`; outside the range it does not grant meals. | **Per-service wallet charge** while the subscription is active. Each confirmed `MealServiceEvent` resolves a price (§14), calls discounts (§19), and asks Finance to debit the wallet (§20). |

- A `WALLET` plan with `start_date = -∞` / `end_date = +∞` (or a very wide
  range) is the common "standing wallet subscription" case — it is active
  every day and charges per service.
- A `DATE_RANGE` plan is the common "monthly/annual paid lunch" case — it
  grants meals for the range with no per-service charge.
- The two modes are **not** mutually exclusive per person: a person may
  hold a `DATE_RANGE` subscription (primary) and a `WALLET` subscription
  (fallback) simultaneously, resolved by priority (§11, §13).

### 10.2 Per-period and per-Person pricing attachments

- **Per-period pricing** lives on a separate `MealPeriodPrice`
  (`meal_plan` FK, `meal_period` FK, `price_iqd`, `is_enabled`), the
  migration target for legacy `MealProfilePeriod` (§16).
- **Per-Person pricing** lives on `MealPersonPriceOverride`
  (`person` FK, `meal_plan` FK, optional `meal_period` FK, `price_iqd`,
  `is_enabled`) — §15.
- The plan carries a `default_price_iqd` used as the fallback when no
  more-specific price row matches (§18).

### 10.3 Rules

- `MealPlan` is admin-configurable. It does **not** reference `Person`,
  `StudentProfile`, or `StaffProfile`. It is a product definition.
- Supervisor-override flags live here because they are product policy,
  not per-person data.
- `MealProfile` (legacy) → `MealPlan` migration is described in §21.2.

> Historical note: an earlier draft of this document and
> `lunch_domain_architecture.md` used `MealProfile` and `MealPlan` as
> competing names for the same concept. The decision is now settled:
> `MealPlan` is the future model; `MealProfile` is legacy only.

---

## 11. MealSubscription

- **Owner:** `apps.meals`
- **Purpose:** A Person's entitlement to meals under a `MealPlan` for a
  date range (the migration target for legacy `MealSubscription`).
  Supports both students and staff (§17), both date-range and wallet plans
  (§10.1), and primary + fallback combinations (§11.4).
- **Conceptual fields:**

```python
class MealSubscription(models.Model):
    class Status(models.TextChoices):
        FUTURE = "future", "Future"
        ACTIVE = "active", "Active"
        PAUSED = "paused", "Paused"
        EXPIRED = "expired", "Expired"
        CANCELLED = "cancelled", "Cancelled"

    person = models.ForeignKey(
        "identity.Person", on_delete=models.CASCADE,
        related_name="meal_subscriptions",
    )
    student = models.ForeignKey(
        "identity.StudentProfile", on_delete=models.CASCADE,
        related_name="meal_subscriptions", null=True, blank=True,
    )
    staff = models.ForeignKey(
        "identity.StaffProfile", on_delete=models.CASCADE,
        related_name="meal_subscriptions", null=True, blank=True,
    )
    meal_plan = models.ForeignKey(
        MealPlan, on_delete=models.PROTECT, related_name="subscriptions",
        null=True, blank=True,
    )
    academic_year = models.ForeignKey(
        "academics.AcademicYear", on_delete=models.PROTECT,
        related_name="meal_subscriptions", null=True, blank=True,
    )
    status = models.CharField(max_length=20, choices=Status.choices,
                              default=Status.ACTIVE, db_index=True)
    start_date = models.DateField(db_index=True)
    end_date = models.DateField(db_index=True)
    plan_type = models.CharField(max_length=20, default="monthly")  # reporting
    source = models.CharField(max_length=40, default="manual", db_index=True)
    # priority IS the order number. Lower number = higher priority =
    # evaluated first by the resolver. 1 = primary, 2 = fallback, etc.
    priority = models.PositiveSmallIntegerField(default=1, db_index=True)
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
```

- **Relationships:**
  - `person` → `Person` (FK CASCADE) — the canonical anchor.
  - `student` → `StudentProfile` (FK CASCADE, nullable) — set for student
    subscriptions; used for fast student-scoped queries and academic
    snapshot joins.
  - `staff` → `StaffProfile` (FK CASCADE, nullable) — set for staff
    subscriptions (§17). Exactly one of `student` / `staff` is set for a
    non-guest subscription; `person` is always set.
  - `meal_plan` → `MealPlan` (PROTECT).
  - `academic_year` → `AcademicYear` (PROTECT, nullable until academics is
    rolled out; then year-scoping is recommended).
- **Rules:**
  - Overlap validation lives in `validators.py` (§11.5).
  - `status` is the single source of truth for the subscription lifecycle
    (the state machine below).
  - The subscription stores **no** grade/section/name fields. Grade/section
    at service time are read from the current enrollment/placement or
    snapshotted on the immutable `MealServiceEvent` (§12).

### 11.1 Subscription lifecycle (state machine)

| Status | Meaning |
|---|---|
| `future` | `start_date` has not yet arrived. |
| `active` | Current date is within `[start_date, end_date]` and not paused/cancelled. |
| `paused` | Temporarily suspended within the active window; no eligibility granted while paused. |
| `expired` | `end_date` has passed; no longer grants eligibility. |
| `cancelled` | Terminated before `end_date` by an admin/supervisor. |

Valid transitions:

```
future ──(start_date reached)──► active
active ──(pause)──► paused
paused ──(resume)──► active
active/paused ──(cancel)──► cancelled
active/paused/future ──(end_date passed)──► expired
```

### 11.2 One-time meal permission

- A one-time permission is **not** a `MealSubscription`. It is a
  `MealException(kind=ONE_TIME_ELIGIBLE, effective_date=<date>)`. The
  eligibility resolver checks exceptions after subscriptions, so a person
  without any subscription can still be eligible for one date via an
  exception. This avoids polluting the subscription model with single-day
  rows and keeps the "subscription = date range" contract clean.
- `MealException` conceptual fields:

```python
class MealException(models.Model):
    class Kind(models.TextChoices):
        ONE_TIME_ELIGIBLE = "one_time_eligible", "One-time eligible"
        TEMPORARY_DENY = "temporary_deny", "Temporary deny"
        GUEST_ELIGIBLE = "guest_eligible", "Guest eligible"

    person = models.ForeignKey("identity.Person", on_delete=models.CASCADE,
                               related_name="meal_exceptions")
    kind = models.CharField(max_length=30, choices=Kind.choices, db_index=True)
    effective_date = models.DateField(db_index=True)
    end_date = models.DateField(null=True, blank=True)
    meal_plan = models.ForeignKey(MealPlan, on_delete=models.SET_NULL,
                                  null=True, blank=True)
    reason_code = models.CharField(max_length=32, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    approved_by = models.ForeignKey("identity.StaffProfile", on_delete=models.SET_NULL,
                                    null=True, blank=True,
                                    related_name="approved_meal_exceptions")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

### 11.3 Historical preservation

- Subscriptions are **never deleted**. Cancelled/expired rows remain for
  historical reporting and for `MealEligibility.subscription` /
  `MealServiceEvent.subscription` FKs to remain valid.
- A cancelled subscription's `[start_date, end_date]` and `status` history
  are preserved; only the `status` field transitions.
- Mid-window plan/priority changes should use **cancel-and-recreate** so
  history is append-only (open question — whether in-place edits with an
  audit record are allowed; recommendation: cancel-and-recreate).

### 11.4 Primary / fallback semantics

A person may hold **multiple simultaneous `ACTIVE` subscriptions** under
different `MealPlan` instances (and different modes). The resolver (§13)
uses `priority` to decide which subscription wins for a given
`(person, date, meal_period)`:

| Role | Typical `priority` | Typical `MealPlan.mode` | Behavior |
|---|---|---|---|
| **Primary** | `1` | `DATE_RANGE` | Paid entitlement; if it covers the meal, the service is eligible with **no wallet charge**. |
| **Fallback** | `2` (or higher) | `WALLET` | Per-service charge; evaluated only if the primary does not cover the meal (or if no primary exists). |

A person's configuration may be any of:

- **Primary only** — a single `DATE_RANGE` subscription; meals are
  covered by the entitlement, no per-service charge.
- **Fallback only** — a single `WALLET` subscription; every service is
  charged to the wallet.
- **Both primary and fallback** — a `DATE_RANGE` subscription (priority 1)
  **and** a `WALLET` subscription (priority 2). The resolver tries the
  primary first; if the primary covers the meal, no charge. If the primary
  does **not** cover the meal (e.g. the meal kind or period is outside the
  primary plan, or the primary is paused/expired), the resolver falls back
  to the wallet subscription and charges per service.
- **Multiple wallet plans** — e.g. a breakfast wallet plan and a lunch
  wallet plan, each scoped by `MealPlan.kind` and/or `MealPeriodPrice` for
  the relevant `MealPeriod`. Priority orders them; the resolver picks the
  first that applies to the requested meal period.

The `MealPlan.kind` and the `MealPeriodPrice` / `MealPersonPriceOverride`
rows further scope which subscription applies to which `MealPeriod`
(breakfast vs lunch). The resolver (§13) filters candidate subscriptions
by kind and period before sorting by priority.

### 11.5 Priority / order semantics (decided)

- `priority` **is** the order number. **Lower number = higher priority =
  evaluated first** (matching legacy `MealSubscription.priority`, where
  "lower number wins").
- Primary date-range subscriptions are usually `priority = 1`; fallback
  wallet subscriptions are usually `priority = 2`. Admins may use other
  numbers (3, 4, …) for additional fallback tiers.
- **Same-priority overlapping `ACTIVE` subscriptions for the same person
  are rejected** by the validator (mirrors legacy
  `MealSubscription.clean()` at `apps/attendance/models.py:610`).
- **Different-priority overlapping `ACTIVE` subscriptions for the same
  person are allowed** (this is how primary + fallback coexist).
- Overlap is defined on `[start_date, end_date]`. Two subscriptions with
  the same `priority` whose `[start_date, end_date]` ranges intersect and
  which are both `ACTIVE` (or `FUTURE`→`ACTIVE`) are a validation error.
- The resolver (§13) sorts candidate subscriptions by `priority`
  ascending and evaluates them in that order; the first that grants
  eligibility wins.

---

## 12. MealServiceEvent

- **Owner:** `apps.meals`
- **Purpose:** The immutable record that a person was actually served a
  meal on a date (the migration target for legacy `MealRecord`, freed from
  its 1:1 coupling to `AttendanceRecord`). This is the "meal attendance"
  entity, and the canonical financial-snapshot record for a meal charge.
- **Conceptual fields:**

```python
class MealServiceEvent(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        CONFIRMED = "confirmed", "Confirmed"
        DENIED = "denied", "Denied"
        UNPAID = "unpaid", "Unpaid"          # wallet mode, insufficient funds allowed
        REFUNDED = "refunded", "Refunded"
        VOIDED = "voided", "Voided"

    person = models.ForeignKey("identity.Person", on_delete=models.CASCADE,
                               related_name="meal_service_events")
    student = models.ForeignKey("identity.StudentProfile", on_delete=models.CASCADE,
                                related_name="meal_service_events",
                                null=True, blank=True)
    staff = models.ForeignKey("identity.StaffProfile", on_delete=models.CASCADE,
                              related_name="meal_service_events",
                              null=True, blank=True)
    date = models.DateField(db_index=True)
    eligibility = models.ForeignKey("MealEligibility", on_delete=models.PROTECT,
                                    related_name="service_events",
                                    null=True, blank=True)
    subscription = models.ForeignKey(MealSubscription, on_delete=models.SET_NULL,
                                     related_name="service_events",
                                     null=True, blank=True)
    meal_plan = models.ForeignKey(MealPlan, on_delete=models.SET_NULL,
                                  related_name="service_events",
                                  null=True, blank=True)
    meal_period = models.ForeignKey(MealPeriod, on_delete=models.SET_NULL,
                                    related_name="service_events",
                                    null=True, blank=True)
    # optional link to the recognition that triggered the service
    recognition_event = models.ForeignKey("attendance.AttendanceEvent",
                                          on_delete=models.SET_NULL,
                                          related_name="meal_service_events",
                                          null=True, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices,
                              default=Status.PENDING, db_index=True)
    # immutable price-resolution snapshot (justified: historical record)
    price_base_iqd = models.IntegerField(default=0)          # from MealPeriodPrice / default
    price_override_iqd = models.IntegerField(default=0)     # from MealPersonPriceOverride
    discount_iqd = models.IntegerField(default=0)           # from apps.discounts (§19)
    final_charge_iqd = models.IntegerField(default=0)       # base - override - discount (≥0)
    price_resolution_source = models.CharField(max_length=32, blank=True, default="",
        help_text="Which rule won: 'period_price' / 'person_override' / 'default' / 'date_range_no_charge'.")
    # immutable financial snapshot
    wallet_balance_before_iqd = models.IntegerField(default=0)
    wallet_balance_after_iqd = models.IntegerField(default=0)
    wallet_transaction = models.ForeignKey("wallet.WalletTransaction",
                                            on_delete=models.PROTECT,
                                            related_name="meal_service_events",
                                            null=True, blank=True)
    wallet_refund_transaction = models.ForeignKey("wallet.WalletTransaction",
                                                  on_delete=models.PROTECT,
                                                  related_name="meal_refund_events",
                                                  null=True, blank=True)
    # immutable academic snapshot at service time (students only)
    grade_code_snapshot = models.CharField(max_length=32, blank=True, default="")
    section_code_snapshot = models.CharField(max_length=32, blank=True, default="")
    # meal-period snapshot (defensive against later MealPeriod edits)
    meal_period_label_snapshot = models.CharField(max_length=64, blank=True, default="")
    reason_code = models.CharField(max_length=32, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    served_at = models.DateTimeField(null=True, blank=True, db_index=True)
    served_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                   null=True, blank=True,
                                   related_name="served_meal_events")
    reversed_at = models.DateTimeField(null=True, blank=True)
    reversed_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                     null=True, blank=True,
                                     related_name="reversed_meal_events")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-date", "-id"]
        indexes = [
            models.Index(fields=["date", "status"]),
            models.Index(fields=["person", "date"]),
            models.Index(fields=["student", "date"]),
            models.Index(fields=["section_code_snapshot", "date"]),
            models.Index(fields=["meal_plan", "date"]),
        ]
```

- **Rules:**
  - Once `status` reaches `CONFIRMED` (or any terminal state), the price,
    discount, charge, balance, and academic snapshot fields are
    **immutable**. Corrections create a new event (e.g. a `REFUNDED`/
    `VOIDED` event referencing the original) rather than mutating the
    confirmed row.
  - The event stores *snapshots* of `price_base_iqd`,
    `price_override_iqd`, `discount_iqd`, `final_charge_iqd`,
    `wallet_balance_before_iqd`/`after`, and the `wallet_transaction` FK
    because those values are the historical truth of what happened at
    service time; they cannot be reconstructed from live wallet/plan data
    later.
  - `price_resolution_source` records which pricing rule won, for audit.
  - The `wallet_transaction` FK is created by the **finance domain**;
    Meals only stores the resulting reference (§20).
  - `recognition_event` is an optional link to the AI-attendance event
    that triggered the service. Meals does not run recognition.
  - Exactly one of `student` / `staff` is set for non-guest service events
    (§17); `person` is always set.

### 12.1 MealEligibility (canonical daily snapshot)

```python
class MealEligibility(models.Model):
    class Decision(models.TextChoices):
        ELIGIBLE = "eligible", "Eligible"
        NOT_ELIGIBLE = "not_eligible", "Not eligible"
        OVERRIDDEN_ELIGIBLE = "overridden_eligible", "Overridden eligible"
        OVERRIDDEN_DENIED = "overridden_denied", "Overridden denied"

    person = models.ForeignKey("identity.Person", on_delete=models.CASCADE,
                               related_name="meal_eligibilities")
    student = models.ForeignKey("identity.StudentProfile", on_delete=models.CASCADE,
                                related_name="meal_eligibilities",
                                null=True, blank=True)
    date = models.DateField(db_index=True)
    decision = models.CharField(max_length=30, choices=Decision.choices, db_index=True)
    subscription = models.ForeignKey(MealSubscription, on_delete=models.SET_NULL,
                                     related_name="eligibilities",
                                     null=True, blank=True)
    meal_plan = models.ForeignKey(MealPlan, on_delete=models.SET_NULL,
                                  related_name="eligibilities",
                                  null=True, blank=True)
    grade_code_snapshot = models.CharField(max_length=32, blank=True, default="")
    section_code_snapshot = models.CharField(max_length=32, blank=True, default="")
    absence_reason = models.CharField(max_length=40, blank=True, default="")
    reason_code = models.CharField(max_length=40, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    resolved_at = models.DateTimeField(auto_now_add=True)
    resolved_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                    null=True, blank=True,
                                    related_name="resolved_meal_eligibilities")

    class Meta:
        unique_together = [("person", "date")]
        ordering = ["-date"]
        indexes = [
            models.Index(fields=["date", "decision"]),
            models.Index(fields=["person", "date"]),
            models.Index(fields=["student", "date"]),
        ]
```

- `unique_together = [("person", "date")]`: exactly one canonical
  eligibility row per person per day.
- **Recalculable before service:** the row may be recomputed/overwritten
  by `resolve_eligibility` at any time **before** a `MealServiceEvent`
  references it.
- **Frozen once referenced:** once a `MealServiceEvent` references the
  eligibility, the row is frozen and must not be mutated. Further
  corrections are audited through `MealSupervisorAction` records against
  the (immutable) service event.

### 12.2 MealSupervisorAction (audit trail)

```python
class MealSupervisorAction(models.Model):
    class Action(models.TextChoices):
        CONFIRM = "confirm", "Confirm"
        UNCONFIRM = "unconfirm", "Unconfirm"
        DENY = "deny", "Deny"
        REFUND = "refund", "Refund"
        VOID = "void", "Void"
        OVERRIDE_ELIGIBLE = "override_eligible", "Override (eligible)"
        OVERRIDE_DENIED = "override_denied", "Override (denied)"
        MANUAL_LOOKUP = "manual_lookup", "Manual lookup"

    service_event = models.ForeignKey(MealServiceEvent, on_delete=models.CASCADE,
                                      related_name="supervisor_actions",
                                      null=True, blank=True)
    eligibility = models.ForeignKey(MealEligibility, on_delete=models.CASCADE,
                                    related_name="supervisor_actions",
                                    null=True, blank=True)
    action = models.CharField(max_length=30, choices=Action.choices, db_index=True)
    reason_code = models.CharField(max_length=32, blank=True, default="")
    reason_notes = models.CharField(max_length=200, blank=True, default="")
    performed_by = models.ForeignKey("identity.StaffProfile", on_delete=models.SET_NULL,
                                     null=True, blank=True,
                                     related_name="meal_supervisor_actions")
    performed_by_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                           null=True, blank=True,
                                           related_name="meal_supervisor_actions")
    performed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-performed_at"]
        indexes = [
            models.Index(fields=["service_event", "performed_at"]),
            models.Index(fields=["action", "performed_at"]),
        ]
```

- Every mutating supervisor operation writes a `MealSupervisorAction` row
  in the same transaction as the state change. This is the audit trail
  required by the "money is auditable" and "history is append-only"
  principles.
- `performed_by` (StaffProfile) is the business identity of the
  supervisor; `performed_by_user` (auth.User) is the login account. Either
  may be null. Recommendation: start with `performed_by_user` required,
  `performed_by` optional until supervisor-assignment modeling exists.

---

## 13. Meal service resolver flow

> This section is the heart of the **Meals owns meal service resolution**
> principle. The resolver decides, for a given `(person, date,
> meal_period)`, **whether** the person is served and **how much** (if
> anything) is charged. It runs entirely in `apps.meals`; Finance only
  records the resulting amount.

### 13.1 Resolver steps

`resolve_service(person, date, meal_period, *, academic_year=None,
recognition_event=None, created_by=None)` in `apps.meals.services` runs
the following steps, in order:

1. **Find candidate subscriptions.**
   `MealSubscription` rows where `person = person`, `status = ACTIVE`,
   `start_date <= date <= end_date`, and `meal_plan.is_active = True`.
   Optionally filter by `meal_plan.kind == meal_period.kind` (a lunch
   subscription does not cover a breakfast period). This is the universe
   of subscriptions that *could* grant this meal.

2. **Filter by period applicability.**
   Keep only subscriptions whose `meal_plan` has a `MealPeriodPrice` (or a
   `default_price_iqd`) for `meal_period`, **or** whose `meal_plan.mode ==
   DATE_RANGE` and the plan's `kind` matches `meal_period.kind`. (A
   date-range plan covers the period by kind; a wallet plan covers the
   period if a price can be resolved for it — §14.)

3. **Sort by priority.**
   Sort candidate subscriptions by `priority` **ascending** (lower number
   = higher priority = evaluated first). Ties on `priority` are a
   validation error and should not occur (§11.5); if they do, the
   resolver rejects the lower-`id` one defensively and logs.

4. **Evaluate in priority order (first that grants eligibility wins):**

   For each candidate subscription (lowest `priority` first):

   - **4a. Date-range primary.** If `meal_plan.mode == DATE_RANGE`:
     - The plan covers this meal (kind matches and the subscription is
       active on `date`). The service is **ELIGIBLE with no wallet
       charge**. `final_charge_iqd = 0`,
       `price_resolution_source = "date_range_no_charge"`. **Stop** —
       this is the canonical "primary date-range covers it" outcome.
     - (A date-range plan that does *not* cover this meal — e.g. wrong
       kind, or period not in the plan — falls through to the next
       candidate.)

   - **4b. Wallet fallback (or wallet primary, if no date-range exists).**
     If `meal_plan.mode == WALLET`:
     1. Resolve the price (§14): `base`, `override`, `discount`,
        `final_charge_iqd`.
     2. Call `finance.check_balance(person, amount_iqd=final_charge_iqd)`
        to obtain `(balance, sufficient)`.
     3. Decide per `meal_plan.insufficient_funds_mode`:
        - `DENY` and not sufficient → this subscription does **not**
          grant the meal; **continue** to the next candidate (next
          priority). If none grant, eligibility is `NOT_ELIGIBLE` with
          `reason_code="insufficient_funds"`.
        - `ALLOW_UNPAID` and not sufficient → ELIGIBLE with
          `status = UNPAID`; the service event records the intended
          `final_charge_iqd` but no `wallet_transaction` is created.
          **Stop**.
        - `ALLOW_NEGATIVE` (down to `credit_limit_iqd`) and within
          limit → proceed to charge.
        - sufficient → proceed to charge.
     4. Charge: call `finance.charge(person=person,
        amount_iqd=final_charge_iqd, source_module="meals",
        reference_type="MealServiceEvent", reference_id=<pk>,
        academic_year=academic_year, description=...)`. Store the
        returned `WalletTransaction` FK and `balance_before`/`after`
        snapshots. **Stop**.

5. **No subscription granted eligibility.** Check `MealException` rows
   covering `(person, date)`:
   - `ONE_TIME_ELIGIBLE` / `GUEST_ELIGIBLE` → ELIGIBLE (no charge unless
     the exception's `meal_plan` is a wallet plan, in which case step 4b
     runs against the exception's plan).
   - `TEMPORARY_DENY` → `OVERRIDDEN_DENIED`.

6. **Default.** `NOT_ELIGIBLE` with `reason_code="no_subscription"`.

7. **Academic-presence and absence gates (students only).** Before step
   4, the resolver applies the student gates from §17.2: a student
   without a current `StudentEnrollmentSectionPlacement` is
   `NOT_ELIGIBLE` (`reason_code="no_current_placement"`); an absent
   student is `NOT_ELIGIBLE` (`reason_code="absent"`) unless an
   override/exception grants eligibility. Staff skip these gates (§17.2).

8. **Write the `MealEligibility` row** for `(person, date)` (canonical,
   recalculable until a `MealServiceEvent` references it — §12.1) and the
   `MealServiceEvent` (in `PENDING` until confirmed, then `CONFIRMED` /
   `UNPAID` / `DENIED`), plus the `MealSupervisorAction` audit row in the
   same transaction.

### 13.2 Resolver invariants

- The resolver is **deterministic**: given the same
  `(person, date, meal_period)` and the same subscription/price rows, it
  produces the same decision and the same `final_charge_iqd`.
- The resolver is **pure** for the decision/price (steps 1–6); the only
  side effects are the eligibility-row write (step 8), the service-event
  write (step 8), and the Finance charge call (step 4b). It is safe to
  call `resolve_price` / `resolve_eligibility` for display without
  charging.
- The resolver **never** reads `Wallet.balance` directly; it uses
  `finance.check_balance`.
- The resolver **never** computes discounts; it calls `apps.discounts`
  during step 4b.1 (§19).
- The resolver passes the **pre-resolved** `final_charge_iqd` to
  `finance.charge`; Finance records the ledger only (§5, §20).

### 13.3 Worked trace (primary + fallback)

A student with a `DATE_RANGE` lunch subscription (priority 1) and a
`WALLET` breakfast subscription (priority 2) requests lunch on a school
day:

1. Candidates: both subscriptions match `person`/`status=ACTIVE`/date.
2. Period filter: the lunch subscription's `kind=LUNCH` matches the
   requested lunch `MealPeriod`; the breakfast subscription's
   `kind=BREAKFAST` does not. Only the lunch subscription remains.
3. Sort: one candidate.
4. Evaluate: `meal_plan.mode == DATE_RANGE` → ELIGIBLE, `final_charge=0`,
   no wallet call. Stop.

The same student requests breakfast:

1. Candidates: both subscriptions match person/status/date.
2. Period filter: the breakfast wallet subscription's `kind=BREAKFAST`
   matches; the lunch subscription's `kind=LUNCH` does not. Only the
   breakfast subscription remains.
3. Sort: one candidate.
4. Evaluate: `meal_plan.mode == WALLET` → resolve price →
   `finance.check_balance` → `finance.charge(final_charge_iqd)`. Stop.

This is Example A in §25.

---

## 14. Meal price resolution

> `resolve_price` is the price component of the resolver (§13 step 4b.1).
> It is a pure function of `(person, meal_plan, meal_period, date,
> academic_year)` and returns the **base price**, the **override price**
> (if any), the **discount** (via `apps.discounts`), and the **final
> charge**. Finance receives `final_charge_iqd` as a pre-resolved amount
> (§20).

### 14.1 Resolution order

`resolve_price(person, meal_plan, meal_period, date, academic_year)`
in `apps.meals.services` resolves in this order (first match wins):

1. **Per-Person × Per-MealPeriod override.**
   `MealPersonPriceOverride(person=person, meal_plan=meal_plan,
   meal_period=meal_period, is_enabled=True)` and `effective_from`/`until`
   covers `date`. If found, `base = override.price_iqd`,
   `source = "person_override_period"`.
2. **Per-Person × any-period override.**
   `MealPersonPriceOverride(person=person, meal_plan=meal_plan,
   meal_period__isnull=True, is_enabled=True)`. If found,
   `base = override.price_iqd`, `source = "person_override"`.
3. **Per-MealPeriod list price.**
   `MealPeriodPrice(meal_plan=meal_plan, meal_period=meal_period,
   is_enabled=True)`. If found, `base = price.price_iqd`,
   `source = "period_price"`.
4. **Plan default.**
   `meal_plan.default_price_iqd`. `source = "default"`.
5. **Zero.** If none of the above match and `default_price_iqd` is 0,
   `base = 0`, `source = "default_zero"` (free meal / misconfiguration —
   flagged in logs).

`price_override_iqd` is non-zero only when a per-Person override won
(steps 1–2); it records the *difference* the override makes relative to
the period list price, for audit:

```
price_override_iqd =
    (period_list_price - override.price_iqd)  if override won and a period price exists
    0                                          otherwise
```

(Signed so that a discounted override is positive and a surcharged
override is negative — implementation choice; alternative is to store the
absolute override price. The snapshot on `MealServiceEvent` stores both
`price_base_iqd` (the resolved base, i.e. what will actually be charged
before discounts) and `price_override_iqd` for audit clarity. See open
question in §26.)

### 14.2 Discount composition

After the base is resolved, Meals calls the discount service boundary
(§19):

```python
discount_iqd, final_charge_iqd, applied = discounts.resolve(
    person=person,
    price_base_iqd=base,
    product_code="meal_lunch",   # or meal_breakfast / meal_snack
    academic_year=academic_year,
    context={"meal_period": meal_period_id, "meal_plan": meal_plan_id, "date": date},
)
```

- Until `apps.discounts` exists, `discount_iqd = 0` and
  `final_charge_iqd = base`.
- `final_charge_iqd = max(0, base - discount_iqd)` (discounts never push
  the charge below zero; the discount domain enforces this and Meals
  re-validates defensively).
- The resolved `base`, `discount_iqd`, `final_charge_iqd`, and
  `price_resolution_source` are snapshotted on the immutable
  `MealServiceEvent` (§12).

### 14.3 Where resolution happens

- `resolve_price` is called by `resolve_service` (§13 step 4b.1) inside
  `apps.meals.services`, **not** by views, admin actions, or the
  supervisor dashboard directly.
- For `MealPlan.mode == DATE_RANGE` subscriptions, the resolver short-
  circuits with `final_charge_iqd = 0` and
  `price_resolution_source = "date_range_no_charge"` (§13 step 4a). No
  per-service charge and no discount call.
- For `MealPlan.mode == WALLET`, the resolved `final_charge_iqd` is
  passed to `finance.charge(...)` (§20).

---

## 15. Per-Person pricing

- **Owner:** `apps.meals`
- **Purpose:** A granular price override for a specific `Person` under a
  `MealPlan`, optionally scoped to a specific `MealPeriod`. This is how
  the school expresses "this student pays 750 IQD for lunch instead of
  the 1000 IQD list price" or "this staff member pays 500 IQD for the
  first lunch block only."
- **Conceptual fields:**

```python
class MealPersonPriceOverride(models.Model):
    person = models.ForeignKey("identity.Person", on_delete=models.CASCADE,
                               related_name="meal_price_overrides")
    meal_plan = models.ForeignKey(MealPlan, on_delete=models.CASCADE,
                                  related_name="person_price_overrides")
    meal_period = models.ForeignKey(MealPeriod, on_delete=models.CASCADE,
                                    related_name="person_price_overrides",
                                    null=True, blank=True,
                                    help_text="Blank = applies to all periods of this plan.")
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
        unique_together = [["person", "meal_plan", "meal_period"]]
        indexes = [
            models.Index(fields=["person", "is_enabled"]),
            models.Index(fields=["meal_plan", "is_enabled"]),
        ]
```

- **Rules:**
  - `unique_together = [["person", "meal_plan", "meal_period"]]` enforces
    one override per `(person, plan, period)`. `meal_period` may be
    null (applies to all periods of the plan); the unique constraint in
    PostgreSQL treats NULL as distinct, so a person may have both an
    any-period override and a per-period override; the resolver picks the
    per-period one first (§14.1).
  - `effective_from` / `effective_until` scope the override in time (e.g.
    a temporary discount for one term). The resolver checks
    `effective_from <= date <= effective_until` (treating nulls as
    unbounded).
  - The override is **snapshotted** on the immutable `MealServiceEvent`
    via `price_override_iqd` (§12, §14.1); later edits to the override do
    not rewrite historical events.
  - Per-Person overrides do **not** stack with per-period prices; the
    resolver uses first-match-wins (§14.1). Stacking is a discount-domain
    concern (§19).

### 15.1 Per-Person pricing for staff

- Staff may have `MealPersonPriceOverride` rows just like students
  (§17). Staff overrides may use a different `MealPlan` (e.g.
  "Staff lunch") with its own list prices and overrides.
- The resolver is identical for students and staff — it operates on
  `Person`, not on `StudentProfile` / `StaffProfile`. The role distinction
  only matters for eligibility (academic-presence check is student-only —
  §17.2) and for academic snapshot fields (blank for staff).

---

## 16. Per-MealPeriod pricing

- **Owner:** `apps.meals`
- **Purpose:** The list price of a meal under a `MealPlan` for a given
  `MealPeriod`. This is the migration target for legacy
  `MealProfilePeriod` (currently keyed to `attendance.PeriodTemplate`).
- **Conceptual fields:**

```python
class MealPeriodPrice(models.Model):
    meal_plan = models.ForeignKey(MealPlan, on_delete=models.CASCADE,
                                  related_name="period_prices")
    meal_period = models.ForeignKey(MealPeriod, on_delete=models.CASCADE,
                                    related_name="plan_prices")
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
```

- **Rules:**
  - `unique_together = [["meal_plan", "meal_period"]]` enforces one
    price per `(plan, period)`.
  - `price_iqd` is the **list price** (the "sticker price" before any
    per-Person override or discount). It is owned by Meals because it is
    a meal-product fact, not a ledger fact.
  - A missing `MealPeriodPrice` row (no row for a given period) falls
    back to `MealPlan.default_price_iqd` (§18).
  - The list price is **snapshotted** on the immutable `MealServiceEvent`
    via `price_base_iqd` (when it wins) or used to compute
    `price_override_iqd` (when a per-Person override wins) — §14.1.

### 16.1 Migration from legacy `MealProfilePeriod`

- Legacy `MealProfilePeriod` is keyed by `(meal_profile,
  period_template)`. Migration steps (executed in §21.2):
  1. Create `MealPeriod` rows from the legacy `PeriodTemplate` rows that
     are referenced by any `MealProfilePeriod` (kind = LUNCH by default).
  2. Create `MealPeriodPrice` rows from legacy `MealProfilePeriod`,
     mapping `period_template` → `MealPeriod`, `meal_profile` →
     `MealPlan`, and copying `price_iqd` / `is_enabled` / `notes`.
  3. Verify counts and total price sum match.
  4. Flip read paths. Keep legacy rows until verification is complete.

---

## 17. Student and Staff support

> The decided architecture anchors meal entitlement and pricing to
> `Person`, so both students and staff can be served meals under the same
> resolver. The role distinction is handled by `PersonRole` and by which
> plan (`student`-facing vs `staff`-facing) the subscription uses.

### 17.1 Person-anchored entitlement

- `MealSubscription.person` is always set. `MealSubscription.student` is
  set for student subscriptions; `MealSubscription.staff` is set for
  staff subscriptions. Exactly one of `student` / `staff` is set for a
  non-guest subscription.
- `MealPersonPriceOverride.person` is the only person FK; it works
  identically for students and staff (§15.1).
- `MealServiceEvent.person` is always set; `student` / `staff` are
  snapshotted for fast role-scoped queries.

### 17.2 Eligibility differences

- The resolver (§13) runs the same steps for students and staff **except**
  the academic-presence and absence gates (step 7):
  - **Students:** the resolver checks for a current
    `StudentEnrollmentSectionPlacement` for the academic year containing
    `date`. Withdrawn/graduated students are `NOT_ELIGIBLE` (open
    decision: hard deny vs. warning — recommendation: hard deny with
    supervisor override allowed). Absent students are `NOT_ELIGIBLE`
    unless an override/exception grants eligibility.
  - **Staff:** the resolver checks for an active `StaffProfile` (employment
    status active on `date`). No academic placement is required. Staff
    have no "absent" signal that denies meals unless explicitly modeled in
    a future HR domain (open question — §26).
- Subscription match (resolver step 4) and exception match (step 5) are
  identical for both roles — they operate on `MealSubscription.person`.

### 17.3 Pricing differences

- Pricing is **role-agnostic** at the resolver level: it operates on
  `(person, meal_plan, meal_period)`. The role distinction is expressed
  by which `MealPlan` the subscription uses (e.g. "Student lunch" vs
  "Staff lunch"), each with its own `MealPeriodPrice` list and its own
  `MealPersonPriceOverride` rows.
- A school may choose to give staff a free meal by setting
  `MealPlan.default_price_iqd = 0` for the staff plan, or by granting a
  100% discount via `apps.discounts` (§19). Both approaches produce
  `final_charge_iqd = 0` and an audit trail.

### 17.4 Academic snapshots

- `MealServiceEvent.grade_code_snapshot` / `section_code_snapshot` are
  blank for staff service events (no academic placement). Staff
  snapshots may include a `department` / `role` snapshot in a future
  expansion (open question — not required for v1).

---

## 18. Default pricing vs overrides

> The pricing stack is a **fall-through ladder**. The first matching run
> wins; later runs are not stacked. This keeps the resolver predictable
> and auditable.

### 18.1 The ladder

| Priority | Source | Model | Scope |
|---|---|---|---|
| 1 (highest) | Per-Person × Per-Period override | `MealPersonPriceOverride` (with `meal_period`) | one person, one period |
| 2 | Per-Person × any-period override | `MealPersonPriceOverride` (no `meal_period`) | one person, all periods |
| 3 | Per-Period list price | `MealPeriodPrice` | one plan, one period |
| 4 | Plan default | `MealPlan.default_price_iqd` | one plan, all periods |
| 5 (lowest) | Zero | — | free / misconfiguration |

### 18.2 Why first-match-wins (not stacking)

- Stacking base prices (e.g. "period price minus a per-person discount")
  would couple Meals to discount semantics. Discounts are a separate
  domain (§19) and are applied **after** the base is resolved, not
  layered into the base.
- A per-Person override is an **alternative** list price for that person,
  not a delta on the period price. The audit captures both the override
  price and the period list price (via `price_override_iqd`, §14.1) so
  the historical "what would the list price have been" is preserved.
- Stacked/computed discounts (percentage, scholarship, subsidy, same-day
  multi-meal) are exclusively the discount domain's job (§19).

### 18.3 Default price semantics

- `MealPlan.default_price_iqd` is the **catch-all** base when no
  `MealPeriodPrice` matches. It is also the natural place to express a
  flat-rate plan ("all periods 1000 IQD") without creating a
  `MealPeriodPrice` row per period.
- `default_price_iqd = 0` means "free meal unless an override or period
  price says otherwise." This is the recommended default for staff plans
  and for subsidized student plans.
- Setting `default_price_iqd = 0` and `MealPeriodPrice` rows to nonzero
  values is a valid configuration: the period prices win for those
  periods, and any period without a `MealPeriodPrice` row is free. The
  resolver's step 4 fallback makes this explicit.

### 18.4 Disabled rows

- `MealPeriodPrice.is_enabled = False` and
  `MealPersonPriceOverride.is_enabled = False` cause the resolver to skip
  the row and fall through. This lets admins "turn off" an override
  without deleting it (preserving history). `effective_from` /
  `effective_until` on overrides provide time-scoping on top of
  `is_enabled`.

---

## 19. Future Discounts integration

> Discounts remain a **separate** future domain (`apps.discounts`). Meals
> calls the discount service boundary during final-charge composition;
> Meals does not own discount profiles, rules, or assignments, and
> Finance never calls the discount service under this architecture (the
> deviation from `finance_domain_architecture.md` §8.2 and
> `DOMAIN_INTEGRATION_GUIDE.md` §13.3 is documented in the review
> package).

### 19.1 Where discount logic lives

- Discount **profiles, rules, and assignment resolution** live in
  `apps.discounts` (`DiscountProfile`, `DiscountRule`,
  `DiscountAssignment`), scoped by `Person` and `AcademicYear`, per
  `education_domain_architecture.md`. These are the migration targets for
  legacy `attendance.DiscountProfile` / `attendance.DiscountRule`.
- `apps.meals` calls `discounts.resolve(...)` and stores the result as a
  snapshot on `MealServiceEvent`.

### 19.2 The service boundary

```python
# apps/discounts/services.py  (future)
def resolve(*, person, price_base_iqd, product_code, academic_year,
            context) -> tuple[int, int, list[dict]]:
    """Return (discount_iqd, final_charge_iqd, applied_profiles)."""
```

- Meals passes the **resolved base price** (from §14.1) and the meal
  context (`meal_period`, `meal_plan`, `date`, and a same-day meal
  history summary for multi-meal discounts — §19.5).
- The discount domain decides stacking/exclusivity, fixed-vs-percent,
  subsidy/scholarship, `min_same_day_confirmed_meals` conditions
  (migrated from legacy `DiscountRule`), and same-day multi-meal rules
  (§19.5).
- Meals stores `discount_iqd` and `final_charge_iqd` on the immutable
  `MealServiceEvent` (§12). It does **not** store discount-profile IDs or
  rule IDs as live FKs on the service event (snapshots only — a rule
  rename must not corrupt historical rows).

### 19.3 Discount types (resolved by the discount domain)

| Type | Resolved by |
|---|---|
| Fixed amount | `apps.discounts` (`discount_iqd = min(value, base)`) |
| Percentage | `apps.discounts` (`discount_iqd = base * value_percent / 100`) |
| Subsidy / scholarship | `apps.discounts` via `DiscountAssignment` to Person/Family/Grade |
| Stacking / exclusive | `apps.discounts` (`DiscountProfile.priority` + stacking policy) |
| Per-period discount | `apps.discounts` scoped by `context["meal_period"]` (migrates legacy `DiscountRule.period_template`) |
| Same-day multi-meal | `apps.discounts` using `context["same_day_meals"]` (e.g. breakfast + lunch triggers a discount — §19.5) |

### 19.4 Where discount logic should NOT live

- **Not** on `MealPlan`: a plan holds a list/default price, not discount
  rules.
- **Not** on `MealSubscription`: a subscription is an entitlement, not a
  pricing rule.
- **Not** on `MealServiceEvent` as live logic: the event stores only the
  *snapshot* of the resolved discount.
- **Not** in `apps.finance`: Finance records the pre-resolved
  `final_charge_iqd`; it does not call the discount service under this
  architecture.
- **Not** in the supervisor dashboard: the dashboard calls the meal
  service, which calls the discount service; the dashboard never computes
  discounts.

### 19.5 Same-day multi-meal discounts

A future discount rule type the architecture must support is a
**same-day multi-meal discount**: e.g. "breakfast + lunch on the same day
→ 500 IQD off the total" or "second meal of the day is 50% off." The
resolver supports this by passing a `same_day_meals` summary in the
discount context:

```python
context = {
    "meal_period": meal_period_id,
    "meal_plan": meal_plan_id,
    "date": date,
    "same_day_meals": [
        # list of (meal_period_kind, meal_plan_id, final_charge_iqd, status)
        # for already-confirmed meals for this person on this date
    ],
}
```

- The discount domain decides whether the current meal qualifies given
  the same-day history (e.g. "this is the second meal today").
- Meals supplies the same-day history via a selector
  (`confirmed_meals_for(person, date)`); it does **not** compute the
  discount.
- Example E in §25 illustrates this flow.

### 19.6 Until `apps.discounts` exists

- `discount_iqd = 0` and `final_charge_iqd = base` for all service
  events. The discount call is a no-op stub in `apps.meals.services`
  (guarded by an `apps.discounts` availability check or a feature flag).
- Legacy `attendance.DiscountProfile` / `DiscountRule` keep working in
  the legacy read path until `apps.discounts` is built and migrated
  (§21).

---

## 20. Finance integration

> The integration boundary is one-way: Meals calls Finance; Finance never
> calls Meals. Meals owns price resolution; Finance records money only.

### 20.1 The service boundary Meals calls

```python
# apps/finance/services.py  (see finance_domain_architecture.md §7.3)
def check_balance(*, person, amount_iqd: int = 0) -> tuple[int, bool]:
    """Return (current_balance, sufficient_for_amount)."""

def charge(*, person, amount_iqd: int, source_module: str,
           reference_type: str = "", reference_id: int | None = None,
           academic_year=None, description: str = "") -> WalletTransaction:
    """Append a signed DEBIT (or UNPAID) row and update the wallet balance
    atomically. `amount_iqd` is the pre-resolved final charge from the
    caller (Meals). Finance does not resolve pricing or discounts."""

def refund(*, person, original_transaction, amount_iqd: int | None = None,
           reason_code: str = "", approved_by=None) -> WalletTransaction:
    """Append a REFUND row that reverses the original DEBIT."""
```

> Deviation from `finance_domain_architecture.md` §7.1 / §8.2 and
> `DOMAIN_INTEGRATION_GUIDE.md` §10 / §13.3: under the decided
> architecture, `finance.charge` takes a **pre-resolved `amount_iqd`**
> and does **not** call the discount service internally. Price resolution
> (base + override) and discount composition are Meals' responsibility
> (§14, §19). Finance's `Charge` model (if implemented) still snapshots
> `price_base_iqd` / `discount_iqd` / `final_charge_iqd` — Meals passes
> them as part of the `reference` / `description` so Finance's audit
> record is self-contained — but Finance does not *compute* them.

### 20.2 Charge flow (wallet mode)

When the resolver (§13) reaches step 4b for a `WALLET` plan:

1. Meals resolves the price (§14): `base`, `override`, `discount`,
   `final_charge_iqd`.
2. Meals calls `finance.check_balance(person=person,
   amount_iqd=final_charge_iqd)` to decide whether to proceed, deny, or
   record as unpaid, based on `MealPlan.insufficient_funds_mode`
   (meal-domain policy on a finance-returned state).
3. Meals calls `finance.charge(person=person,
   amount_iqd=final_charge_iqd, source_module="meals",
   reference_type="MealServiceEvent", reference_id=service_event.pk,
   academic_year=academic_year, description=...)`.
4. Finance appends a DEBIT (or UNPAID) `WalletTransaction` and returns
   it.
5. Meals stores the returned `WalletTransaction` FK and the
   `balance_before`/`balance_after` snapshots on the immutable
   `MealServiceEvent`, transitions the event to `CONFIRMED` (or `UNPAID`).
6. The `MealSupervisorAction(action=CONFIRM, ...)` audit row is written
   in the same transaction.

### 20.3 Refund flow

- A supervisor refund (if `MealPlan.allow_supervisor_refund`) calls
  `finance.refund(person=..., original_transaction=<the DEBIT>,
  amount_iqd=<full or partial>, reason_code=..., approved_by=...)`.
- The returned refund `WalletTransaction` is stored on
  `MealServiceEvent.wallet_refund_transaction`, and the event transitions
  to `REFUNDED`. A `MealSupervisorAction(action=REFUND, ...)` audit row
  is written in the same transaction.

### 20.4 Balance checks (dashboard placeholder)

- The supervisor dashboard calls `finance.check_balance` to display a
  balance warning. The decision to allow/deny service on insufficient
  funds follows `MealPlan.insufficient_funds_mode`, which is
  **meal-domain policy** (how the meal product reacts to a balance state
  returned by finance), not finance logic.

### 20.5 What Meals stores on the service event

- `wallet_transaction` (FK to `wallet.WalletTransaction`, PROTECT) — the
  DEBIT row for the charge.
- `wallet_refund_transaction` (FK, PROTECT) — the REFUND row, if any.
- `wallet_balance_before_iqd` / `wallet_balance_after_iqd` — snapshots
  from the returned transaction.
- `final_charge_iqd` — the pre-resolved amount Meals asked Finance to
  debit.
- These are **snapshots**; the authoritative ledger row lives in
  `apps.finance`. Reconciliation is one-way (§5.4).

### 20.6 Until `apps.finance` exists

- Charges run in a **dry-run / `UNPAID` mode**: Meals resolves the price,
  records the `MealServiceEvent` with `final_charge_iqd` snapshotted, sets
  `status = UNPAID`, and writes a `MealSupervisorAction(action=CONFIRM,
  reason_code="finance_not_available")` row. No `wallet_transaction` FK
  is set. This is the migration safety valve (§21).

---

## 21. Migration strategy

When the meals app is implemented, these rules apply. They are **not**
actions for this document; they are the safety plan for a future,
separately-approved implementation task.

### 21.1 General rules

- **Additive first.** The meals app creates new tables only. It does not
  alter `Person`, `StudentProfile`, `StaffProfile`, or
  `AcademicYear` / `Section`. Legacy `MealSubscription`, `MealProfile`,
  `MealProfilePeriod`, `MealRecord`, `Wallet`, `WalletTransaction`,
  `DiscountProfile`, `DiscountRule` remain.
- **Dual-FK pattern** for migrating off legacy `Student`:
  1. Add `person` (and `student` / `staff`) FKs (nullable) to the new
     meals models.
  2. Data migration: populate `person_id` from
     `legacy_meal_subscription.student.migrated_to.person` (via
     `StudentProfile`).
  3. Enforce NOT NULL on `person`.
  4. Flip read paths.
  5. Later release: drop legacy `MealSubscription` / `MealProfile` /
     `MealProfilePeriod` / `MealRecord`.
- **PROTECT / SET_NULL** on meals FKs so historical service events survive
  plan/subscription/period deactivation.
- **One logical feature per migration.** Do not combine the meals app's
  `0001_initial` with unrelated schema changes.
- Run `python manage.py makemigrations --check --dry-run` and
  `python manage.py check` after each step.
- Tests run under the CREATEDB-enabled role (`TEST_DB_USER` /
  `TEST_DB_PASSWORD`), already configured in `bisk/settings.py`. New
  migrations must pass `python manage.py test apps.meals` (and
  `apps.identity`, `apps.academics`) before merge.
- **Back up the production database** before backfilling eligibility /
  service-event snapshots from legacy `MealRecord`.

### 21.2 Plan / pricing migration (legacy `MealProfile` → `MealPlan`)

1. **Create `MealPeriod` rows** from the legacy `PeriodTemplate` rows
   referenced by any `MealProfilePeriod`. Kind = `LUNCH` by default. Map
   `period_template` → `MealPeriod` (typed FK once `apps.scheduler`
   exists; generic `period_template_ref_id` +
   `period_template_source="attendance"` until then).
2. **Create `MealPlan` rows** from legacy `MealProfile`, copying `name`,
   `mode`, `insufficient_funds_mode`, `credit_limit_iqd`, supervisor
   flags, `notes`. Set `default_price_iqd = 0` initially (or the most
   common `MealProfilePeriod.price_iqd` for the profile, as a fallback).
   The legacy `MealProfile.discount_profile` FK is **not** copied to
   `MealPlan`; discounts migrate to `apps.discounts` separately (§19.6).
3. **Create `MealPeriodPrice` rows** from legacy `MealProfilePeriod`,
   mapping `period_template` → `MealPeriod` and `meal_profile` →
   `MealPlan`, copying `price_iqd` / `is_enabled` / `notes`. Verify count
   and `SUM(price_iqd)` match per plan.
4. **Create `MealPersonPriceOverride` rows** only if legacy per-student
   pricing exists (it does not in the current schema — this is a new
   capability). Skip if not applicable.
5. **Flip the price resolver read path** from legacy
   `MealProfilePeriod.price_iqd` to `MealPeriodPrice.price_iqd`. Keep
   legacy rows until verification.
6. **Snapshot on new service events**: once `MealServiceEvent` is live,
   all new events snapshot from the new pricing stack. Historical
   `MealRecord` rows keep their existing snapshots unchanged.

### 21.3 Subscription / eligibility / service-event migration

1. **Create `MealSubscription` rows** from legacy `MealSubscription`,
   mapping `student` → `person` / `student` via `StudentProfile`. Copy
   `meal_profile` → `meal_plan` (mapped to new `MealPlan`), `plan_type`,
   `status` (legacy active → `ACTIVE`, cancelled → `CANCELLED`, expired →
   `EXPIRED`; introduce `FUTURE`/`PAUSED` only for new rows), `start_date`,
   `end_date`, `priority`, `source`, `notes`.
2. **Backfill `MealServiceEvent` rows** from legacy `MealRecord`:
   - `person` / `student` from `meal_record.attendance_record.student`
     via `StudentProfile`.
   - `date` from `attendance_record.period.date`.
   - `meal_plan` from `meal_record.meal_profile` (mapped to new
     `MealPlan`).
   - `meal_period` from `attendance_record.period.template` (mapped to
     `MealPeriod`).
   - `status` from `meal_record.status` (direct map).
   - `price_base_iqd`, `discount_iqd`, `final_charge_iqd` from the
     `meal_record` snapshots (do **not** recompute — they are historical
     truth).
   - `wallet_balance_before/after_iqd` from `meal_record` snapshots.
   - `wallet_transaction` from `meal_record.wallet_transaction` (mapped
     to the finance-domain `WalletTransaction` once finance migration
     runs; until then, store a generic reference).
   - `recognition_event` from `meal_record.attendance_record` (mapped to
     the attendance-domain `AttendanceEvent` once attendance migration
     runs; until then, leave null and store the legacy
     `attendance_record_id` in `reason_notes` for traceability).
   - `grade_code_snapshot` / `section_code_snapshot` from the legacy
     `student` grade/section at the record date (reconstructed from
     academics enrollment, or blank if unavailable).
3. **Backfill `MealEligibility` rows** for dates with `MealRecord` rows:
   `decision = ELIGIBLE` (or `OVERRIDDEN_ELIGIBLE` if the record was a
   supervisor override), `subscription` / `meal_plan` from the record,
   `grade_code_snapshot` / `section_code_snapshot` from the record. Dates
   without a `MealRecord` are **not** backfilled retroactively (no
   historical eligibility decision exists for them); they are computed
   going forward.
4. **Verify**:
   - `MealServiceEvent` count == legacy `MealRecord` count.
   - `SUM(final_charge_iqd)` matches per month.
   - Every `MealServiceEvent` with `status=CONFIRMED` has a
     `wallet_transaction` (or a `reason_code="finance_not_available"`
     during the dry-run phase).
5. **Flip read paths**: supervisor dashboard, reports, finance
   reconciliation all read from the new models.
6. **Later release**: drop legacy `MealSubscription`, `MealProfile`,
   `MealProfilePeriod`, `MealRecord` (only after verification and a full
   backup).

### 21.4 Snapshot immutability

- Once a `MealServiceEvent` is confirmed, its financial and academic
  snapshot fields must not be edited by any code path. Enforce in
  `clean()` and in service methods (only `status`, `reason_*`, and
  refund/reversal fields are updatable post-confirmation).
- Refunds/voids create new `MealSupervisorAction` rows and link a
  `wallet_refund_transaction`; they do not overwrite the original charge
  snapshot.

### 21.5 Rollback

- Because the meals app is purely additive (new tables only), rollback
  means un-applying its migrations and removing the app from
  `INSTALLED_APPS`. No existing table is altered, so rollback is safe
  before the legacy removal step.
- The plan migration (§21.2) and service-event backfill (§21.3) are
  **data** migrations; their rollback is "drop the new tables and
  re-point reads at the legacy tables." The legacy tables remain intact
  throughout.

---

## 22. Service / Selector / Validator architecture

> Meals follows the `services.py` / `selectors.py` / `validators.py`
> pattern established by `apps.identity` and `apps.academics` and required
> by `DOMAIN_INTEGRATION_GUIDE.md` §3. Business logic lives in services;
> queries live in selectors; invariants live in validators. Views,
> templates, admin actions, and the supervisor dashboard call services
> and selectors; they do not embed business logic.

### 22.1 `apps/meals/services.py`

```python
@transaction.atomic
def create_subscription(*, person, meal_plan, start_date, end_date,
                        academic_year=None, plan_type="monthly",
                        priority=1, source="manual",
                        student=None, staff=None) -> MealSubscription: ...

@transaction.atomic
def activate_future_subscriptions(*, as_of) -> int: ...      # future -> active by date

@transaction.atomic
def pause_subscription(*, subscription, changed_by=None) -> MealSubscription: ...

@transaction.atomic
def resume_subscription(*, subscription, changed_by=None) -> MealSubscription: ...

@transaction.atomic
def cancel_subscription(*, subscription, reason_code="", changed_by=None) -> MealSubscription: ...

@transaction.atomic
def expire_due_subscriptions(*, as_of) -> int: ...           # active/paused -> expired by date

@transaction.atomic
def grant_one_time_permission(*, person, effective_date, meal_plan=None,
                              reason_code="", approved_by=None) -> MealException: ...

def resolve_eligibility(*, person, date) -> MealEligibility: ...
# Steps: academic-presence (student) / staff-active (staff) -> absence
#        -> subscription match -> exception match -> default NOT_ELIGIBLE.

def resolve_service(*, person, date, meal_period, academic_year=None,
                    recognition_event=None, created_by=None) -> MealServiceEvent: ...
# Full resolver (§13): find candidate subscriptions -> sort by priority
# -> evaluate date-range primary -> evaluate wallet fallback -> resolve
# price -> discounts -> finance.charge -> snapshot -> write audit.

def resolve_price(*, person, meal_plan, meal_period, date,
                  academic_year=None) -> PriceResolution: ...
# Pure price resolution (§14). Calls apps.discounts if available.

@transaction.atomic
def record_service_event(*, person, date, meal_period=None, meal_plan=None,
                         subscription=None, recognition_event=None,
                         created_by=None) -> MealServiceEvent: ...
# Resolves eligibility + price, creates a PENDING event, snapshots price.

@transaction.atomic
def confirm_service_event(*, service_event, performed_by=None,
                          reason_code="") -> MealServiceEvent: ...
# For wallet mode: resolve_price -> finance.check_balance -> finance.charge
# -> snapshot -> transition to CONFIRMED (or UNPAID).
# Writes a MealSupervisorAction(CONFIRM).

@transaction.atomic
def unconfirm_service_event(*, service_event, performed_by=None,
                            reason_code="") -> MealServiceEvent: ...

@transaction.atomic
def deny_service_event(*, service_event, performed_by=None,
                       reason_code="") -> MealServiceEvent: ...

@transaction.atomic
def refund_service_event(*, service_event, amount_iqd=None, performed_by=None,
                         reason_code="") -> MealServiceEvent: ...
# Calls finance.refund; stores wallet_refund_transaction; transitions REFUNDED.

@transaction.atomic
def void_service_event(*, service_event, performed_by=None,
                      reason_code="") -> MealServiceEvent: ...

@transaction.atomic
def supervisor_override(*, person, date, decision, reason_code="",
                        performed_by=None) -> MealEligibility: ...
# Writes MealSupervisorAction(OVERRIDE_*) before mutating eligibility
# (only if no MealServiceEvent references the eligibility).
```

- All mutating services are `@transaction.atomic` and write the
  `MealSupervisorAction` audit row in the same transaction as the state
  change.
- `resolve_service`, `resolve_price`, and `resolve_eligibility` are pure
  for the decision/price (no side effects beyond the eligibility-row
  write); they are safe to call from views and the dashboard for display.
- `resolve_service` is the single entry point for meal service
  resolution; it composes `resolve_eligibility` + `resolve_price` +
  Finance calls.

### 22.2 `apps/meals/selectors.py`

```python
def active_subscriptions_for(*, person, date) -> QuerySet[MealSubscription]: ...
def active_subscriptions_for_date(*, date, meal_plan=None) -> QuerySet[MealSubscription]: ...
def exceptions_for(*, person, date) -> QuerySet[MealException]: ...
def eligibility_for(*, person, date) -> MealEligibility | None: ...
def service_events_for(*, person, date) -> QuerySet[MealServiceEvent]: ...
def service_events_for_section(*, section, date) -> QuerySet[MealServiceEvent]: ...
def price_for(*, person, meal_plan, meal_period, date,
              academic_year=None) -> PriceResolution: ...
def expected_roster_for(*, section, date) -> QuerySet: ...
# Composes academics.current_enrollments_in_section with active subscriptions.
def confirmed_meals_for(*, person, date) -> QuerySet[MealServiceEvent]: ...
# Same-day meal history for multi-meal discount context (§19.5).
```

- Selectors are read-only. They never mutate state, never call finance
  charge/refund, and never write audit rows.

### 22.3 `apps/meals/validators.py`

```python
def validate_subscription_overlap(*, person, priority, start_date, end_date,
                                  exclude_pk=None) -> None: ...
# Two ACTIVE subscriptions for the same person with the same priority may
# not overlap in [start_date, end_date].

def validate_subscription_role(*, student, staff) -> None: ...
# Exactly one of student / staff is set for non-guest subscriptions.

def validate_subscription_dates(*, start_date, end_date) -> None: ...
# end_date >= start_date.

def validate_eligibility_window(*, person, date) -> None: ...
def validate_price_override_uniqueness(*, person, meal_plan, meal_period,
                                       exclude_pk=None) -> None: ...
def validate_period_price_uniqueness(*, meal_plan, meal_period,
                                     exclude_pk=None) -> None: ...
def validate_final_charge_nonnegative(*, final_charge_iqd) -> None: ...
def validate_service_event_immutable(*, service_event, updating_fields) -> None: ...
# Raises if a frozen service event's snapshot fields are being mutated.
```

- Validators raise `ValidationError` and are called from `clean()` and
  from service methods. They encode the invariants, not the business
  flow.

### 22.4 `apps/meals/admin.py`

- Read-only admin for `MealServiceEvent` (post-confirmation),
  `MealEligibility` (post-reference), `MealSupervisorAction` (always).
- Editable admin for `MealPlan`, `MealPeriod`, `MealPeriodPrice`,
  `MealPersonPriceOverride`, `MealSubscription` (status transitions via
  custom actions that call services), `MealException`.
- Custom admin actions for `confirm` / `deny` / `refund` / `void` that
  call the corresponding service (never mutate the model directly in an
  admin action).

---

## 23. Anti-patterns

These are explicitly forbidden under the decided architecture. Reviewers
should reject any PR that introduces them.

### 23.1 Pricing anti-patterns

- **Putting `PricingRule` / `PriceList` in `apps.finance`.** Finance
  records money only. Meal-product pricing lives in `apps.meals`.
- **Putting `Wallet.balance` reads in `apps.meals` models.** Meals calls
  `finance.check_balance`; it never reads `Wallet.balance_iqd` directly.
- **Computing `final_charge_iqd` inside `finance.charge`.** Finance
  receives a pre-resolved amount; it does not resolve prices or
  discounts.
- **Stacking `MealPeriodPrice` + `MealPersonPriceOverride` additively.**
  The resolver is first-match-wins (§18.2). Stacking is a discount-domain
  concern.
- **Storing discount-profile/rule FKs on `MealServiceEvent` as live
  FKs.** Only snapshots belong on the immutable event; a rule rename must
  not corrupt historical rows.
- **Putting discount rules on `MealPlan` or `MealSubscription`.** A plan
  holds a list/default price; a subscription is an entitlement. Discount
  rules live in `apps.discounts`.
- **Thresholding price by recognition confidence.** Pricing is never a
  function of recognition (§7).
- **Treating a `DATE_RANGE` plan as a per-service charge, or a `WALLET`
  plan as a free entitlement.** Mode determines charging (§10.1); the
  resolver enforces it.

### 23.2 Ownership anti-patterns

- **A typed FK from `WalletTransaction` to `MealServiceEvent`.** This
  creates a `finance → meals` circular dependency. Finance uses a generic
  `reference_type`/`reference_id`; Meals holds the reverse typed FK.
- **A dependency `finance → meals`, `attendance → meals`, or
  `discounts → meals`.** All forbidden (§3).
- **Importing `apps.attendance` models into `apps.meals` models.** Meals
  consumes recognition events via an optional typed FK to
  `attendance.AttendanceEvent` (the one allowed attendance import) — or,
  during the migration window, a generic reference. Meals never imports
  `attendance.Student`, `PeriodTemplate`, or `MealRecord`.
- **Recreating the legacy `Student` model.** Meals references
  `Person` / `StudentProfile` / `StaffProfile`, never `attendance.Student`.
- **Duplicating identity fields (`h_code`, `first_name`, `last_name`,
  `grade`, `section`) on meal entities as live fields.** Names/grades are
  read via `person.full_name` / enrollment; only justified snapshots on
  immutable records are allowed.
- **Naming the future model `MealProfile`.** The decided future name is
  `MealPlan`. `MealProfile` is legacy only (§10).

### 23.3 Lifecycle anti-patterns

- **Editing a confirmed `MealServiceEvent`'s financial/academic
  snapshot.** Corrections create new events (REFUNDED/VOIDED) referencing
  the original; the original is immutable.
- **Deleting a `MealSubscription` to "end" it.** Subscriptions are
  cancelled, not deleted; the row remains forever for historical FK
  validity.
- **Rewriting a `MealEligibility` row after a `MealServiceEvent`
  references it.** The row is frozen; corrections are audited via
  `MealSupervisorAction`.
- **Scattered `is_paid` / `is_served` / `is_refunded` booleans.** Use the
  `status` enum on `MealServiceEvent` and the `MealSupervisorAction`
  audit trail.
- **Mixing integer IQD and `Decimal` money fields.** Stay integer until a
  fractional-currency requirement appears (match
  `finance_domain_architecture.md` Q11).
- **Creating `WalletTransaction` rows from meals code.** Only
  `apps.finance.services` may append ledger rows.
- **Allowing same-priority overlapping `ACTIVE` subscriptions.** The
  validator rejects them (§11.5); the resolver would be non-deterministic
  otherwise.

### 23.4 Migration anti-patterns

- **Combining the meals app's `0001_initial` with unrelated schema
  changes.** One logical feature per migration.
- **Dropping legacy `MealSubscription` / `MealProfile` /
  `MealProfilePeriod` / `MealRecord` before verification.** Keep through
  the dual-FK migration phase; drop only after verification and a full
  backup.
- **Recomputing historical `final_charge_iqd` during backfill.**
  Historical snapshots are copied as-is from legacy `MealRecord`; they are
  not recomputed from current pricing rules.
- **Committing secrets, `.env`, DB files, media, pycache, virtualenvs, or
  local backups** (per `AGENTS.md`).

---

## 24. Recommended implementation phases

Each phase is a **separate, approved task**. This document implements
nothing. Phases are ordered so that each phase is independently
deployable and the legacy stack keeps working throughout.

| Phase | Deliverable | Depends on | Notes |
|---|---|---|---|
| 0 | Identity + Academics foundations | — | **DONE (identity)** / **DESIGNED (academics)**. Meals needs `Person`, `StudentProfile`, `StaffProfile`, `AcademicYear`, `Section`. |
| 1 | `apps.meals` app skeleton (`apps.py`, empty `models.py`, `INSTALLED_APPS`) | phase 0 | Additive; no models yet. |
| 2 | `MealPeriod` model + admin (wraps `scheduler.PeriodTemplate` or generic reference) | phase 1 | Meal-serving window declaration. |
| 3 | `MealPlan` model + admin (migrates legacy `MealProfile` policy flags) | phase 1 | Product definition. |
| 4 | `MealPeriodPrice` model + admin (migrates legacy `MealProfilePeriod`) + `MealPlan.default_price_iqd` | phases 2–3 | Per-period list price. |
| 5 | `MealPersonPriceOverride` model + admin | phases 2–3 | Per-Person granular pricing (new capability). |
| 6 | `resolve_price(...)` service + `price_for(...)` selector + pricing validators | phases 4–5 | The price-resolution engine (§14). Calls `apps.discounts` stub (no-op). |
| 7 | `MealSubscription` model + admin + lifecycle services (`create` / `pause` / `cancel` / `expire`) + overlap validator (same-priority overlap rejected) | phase 3 | Per-Person dated entitlement, student + staff, primary + fallback. |
| 8 | `MealException` model + admin + `grant_one_time_permission` service | phase 7 | One-time/temporary permissions. |
| 9 | `MealEligibility` model + `resolve_eligibility(...)` service + `eligibility_for(...)` selector | phases 7–8 | Canonical daily eligibility snapshot. |
| 10 | `MealServiceEvent` model + admin (read-only post-confirmation) + `record_service_event` / `confirm` / `deny` / `unconfirm` / `void` services | phases 6, 9 | Immutable service/charge record with price + academic snapshots. Charges run in dry-run/`UNPAID` mode (no `apps.finance` yet). |
| 11 | `MealSupervisorAction` model + audit-row writing in all supervisor services | phase 10 | Audit trail. |
| 12 | `resolve_service(...)` full resolver (§13): candidate subscriptions → sort by priority → date-range primary → wallet fallback → price → discounts → finance.charge | phases 6–11 | The integrated resolver. |
| 13 | Supervisor dashboard view (compose `meals` + `academics` + `finance` selectors; no business logic in the view) | phase 12 | View layer only. |
| 14 | Finance integration: `apps.finance` exposes `charge` / `refund` / `check_balance` taking pre-resolved `amount_iqd`; Meals calls them; replace dry-run with live charges (`discount_iqd=0`) | phases 12, `apps.finance` | Live meal billing without discounts. |
| 15 | Wallet-mode refund flow (`refund_service_event` calls `finance.refund`) | phase 14 | Refunds. |
| 16 | Data migration: backfill `MealPlan` / `MealPeriodPrice` / `MealSubscription` / `MealServiceEvent` / `MealEligibility` from legacy tables (§21.2–§21.3) | phases 6–12 verified | One-time data migration with verification. |
| 17 | Flip read paths: dashboard, reports, finance reconciliation read from new models | phase 16 verified | Legacy tables kept. |
| 18 | (Later phase) `apps.discounts` domain: `DiscountProfile` / `DiscountRule` / `DiscountAssignment` migrated from `apps.attendance` + `discounts.resolve(...)` service boundary, including same-day multi-meal rules | Finance + Meals foundations verified | Discount resolution; Meals `resolve_price` calls it. |
| 19 | Meals ↔ discounts integration: `resolve_price` calls `discounts.resolve(...)` with `same_day_meals` context, snapshots `discount_iqd` on `MealServiceEvent` | phase 18 | Live discounts in meal billing, including same-day multi-meal. |
| 20 | (Later phase) Staff meal plans, staff-specific `MealPeriodPrice` / `MealPersonPriceOverride` rows, staff eligibility rules | phases 7–12 | First-class staff support (models support it from phase 7; this phase adds operational staff plans). |
| 21 | (Later phase) Guest / `MealException.person`-only exceptions (non-student, non-staff) | phase 8 | Guest meals. |
| 22 | (Later phase) Multi-meal-kind support (breakfast / snack) — already modeled via `MealPlan.kind` / `MealPeriod.kind`; this phase adds operational plans/periods/prices for non-lunch kinds | phases 2–12 | Breakfast / snack as first-class. |
| 23 | (Later phase) Legacy `MealSubscription` / `MealProfile` / `MealProfilePeriod` / `MealRecord` removal | phases 16–17 verified | Drop legacy meal tables. |
| 24 | (Later phase) Legacy `DiscountProfile` / `DiscountRule` removal | phase 18 verified | Drop legacy discount tables. |

### Ordering rationale

- Phases 2–6 (periods, plan, prices, overrides, resolver) come first
  because price resolution is the heart of the Meals-owns-pricing
  architecture and is independently testable without subscriptions or
  service events.
- Phases 7–9 (subscriptions, exceptions, eligibility) build the
  entitlement layer on top of the pricing layer.
- Phase 10 (service events) integrates pricing + eligibility into the
  immutable record. Charges run in dry-run/`UNPAID` mode until
  `apps.finance` is available, so phase 10 is deployable without finance.
- Phase 12 (full resolver) integrates the date-range-primary →
  wallet-fallback flow (§13) once the component services exist.
- Phase 14 (finance integration) is the point at which charges go live;
  it ships **without** discounts (`discount_iqd=0`).
- Phase 16 (data migration) is a one-time backfill executed only after
  the new models are verified by phase 15.
- Phase 18 (discounts domain) is built only after the Finance + Meals
  foundation is verified, per the deferral rule in §19.6. It includes
  same-day multi-meal discount rules (§19.5).
- Phase 19 (Meals ↔ discounts integration) is the point at which
  `resolve_price` begins calling `discounts.resolve(...)` with the
  `same_day_meals` context; until then, the discount call is a no-op stub.
- Legacy removal (phases 23–24) is last and conditional on full
  verification; meal legacy and discount legacy are removed independently.

---

## 25. Worked examples

These examples illustrate the resolver (§13) and pricing stack (§14–§18)
on the configurations required by the business. All examples assume the
resolver runs at service time for a `(person, date, meal_period)` triple.

### Example A — Student has a lunch date-range plan and a breakfast wallet fallback

**Setup:**
- `MealPlan "Student Lunch Monthly"`: `kind=LUNCH`, `mode=DATE_RANGE`,
  `default_price_iqd=0`.
- `MealSubscription S1`: `person=Alice` (student), `meal_plan="Student
  Lunch Monthly"`, `start_date=2026-09-01`, `end_date=2026-09-30`,
  `priority=1`, `status=ACTIVE`.
- `MealPlan "Student Breakfast Wallet"`: `kind=BREAKFAST`, `mode=WALLET`.
  `MealPeriodPrice`: breakfast period → 1000 IQD.
- `MealSubscription S2`: `person=Alice`, `meal_plan="Student Breakfast
  Wallet"`, `start_date=2026-09-01`, `end_date=2026-12-31`,
  `priority=2`, `status=ACTIVE`.

**Scenario A1 — Alice requests lunch on 2026-09-10:**
1. Candidates: S1 and S2 (both active, both cover the date).
2. Period filter: the requested `MealPeriod` is a lunch period. S1's
   `kind=LUNCH` matches; S2's `kind=BREAKFAST` does not. Only S1 remains.
3. Sort: one candidate.
4. Evaluate: S1 `mode=DATE_RANGE` → ELIGIBLE, `final_charge_iqd=0`,
   `price_resolution_source="date_range_no_charge"`. **No wallet call.**
   Stop.

**Scenario A2 — Alice requests breakfast on 2026-09-10:**
1. Candidates: S1 and S2.
2. Period filter: requested period is breakfast. S2 `kind=BREAKFAST`
   matches; S1 `kind=LUNCH` does not. Only S2 remains.
3. Sort: one candidate.
4. Evaluate: S2 `mode=WALLET` → `resolve_price`:
   - Step 1 (per-Person × period override): none.
   - Step 2 (per-Person × any-period): none.
   - Step 3 (per-period list price): `MealPeriodPrice` = 1000 IQD.
     `base=1000`, `source="period_price"`.
   - Discounts: `apps.discounts` not built → `discount_iqd=0`,
     `final_charge_iqd=1000`.
   - `finance.check_balance(Alice, 1000)` → sufficient.
   - `finance.charge(Alice, amount_iqd=1000, source_module="meals",
     reference_type="MealServiceEvent", ...)`.
   - Snapshot on `MealServiceEvent`: `price_base_iqd=1000`,
     `final_charge_iqd=1000`, `wallet_balance_before/after` from the
     returned transaction. Stop.

**Outcome:** Lunch is covered by the entitlement (no charge); breakfast is
charged to the wallet at the list price. This is the **primary +
fallback** configuration.

### Example B — Student has no date-range plan but has a wallet plan for both breakfast and lunch

**Setup:**
- `MealPlan "Student Wallet Meals"`: `kind=LUNCH`, `mode=WALLET`.
  `MealPeriodPrice`: lunch period → 1000 IQD. (`kind=LUNCH` but
  `MealPeriodPrice` rows may exist for any `MealPeriod` linked to this
  plan; for breakfast a separate plan or the same plan with a breakfast
  `MealPeriodPrice` is used. In this example, two wallet plans, one per
  kind, for clarity.)
- `MealPlan "Student Breakfast Wallet"`: `kind=BREAKFAST`, `mode=WALLET`.
  `MealPeriodPrice`: breakfast → 800 IQD.
- `MealSubscription S1`: `person=Bob`, `meal_plan="Student Breakfast
  Wallet"`, `priority=1`, `status=ACTIVE`, wide date range.
- `MealSubscription S2`: `person=Bob`, `meal_plan="Student Wallet Meals"
  (lunch)`, `priority=2`, `status=ACTIVE`, wide date range.

**Scenario B1 — Bob requests breakfast:**
1. Candidates: S1 and S2.
2. Period filter: breakfast period. S1 `kind=BREAKFAST` matches; S2
   `kind=LUNCH` does not. Only S1.
3. Evaluate: S1 `mode=WALLET` → `resolve_price` → `MealPeriodPrice`=800
   → `final_charge_iqd=800` → `finance.charge(800)`. Stop.

**Scenario B2 — Bob requests lunch:**
1. Candidates: S1 and S2.
2. Period filter: lunch period. S2 `kind=LUNCH` matches; S1 does not.
3. Evaluate: S2 `mode=WALLET` → `MealPeriodPrice`=1000 →
   `final_charge_iqd=1000` → `finance.charge(1000)`. Stop.

**Outcome:** This is the **fallback-only** configuration (no date-range
plan); each meal kind is charged at its list price to the wallet. Two
wallet plans coexist (different `kind`); priorities order them, but the
period filter selects the right one regardless.

### Example C — Staff has a wallet-based staff lunch plan

**Setup:**
- `MealPlan "Staff Lunch Wallet"`: `kind=LUNCH`, `mode=WALLET`,
  `default_price_iqd=500` (staff rate). `MealPeriodPrice`: lunch period →
  500 IQD. `insufficient_funds_mode=ALLOW_NEGATIVE`,
  `credit_limit_iqd=1000`.
- `MealSubscription S1`: `person=Carol` (staff, `staff=StaffProfile`),
  `meal_plan="Staff Lunch Wallet"`, `priority=1`, `status=ACTIVE`, wide
  date range.

**Scenario C1 — Carol requests lunch:**
1. Candidates: S1.
2. Period filter: lunch period; `kind=LUNCH` matches.
3. Evaluate: S1 `mode=WALLET` → `resolve_price`:
   - `MealPeriodPrice`=500 → `base=500`.
   - Discounts: none → `final_charge_iqd=500`.
   - `finance.check_balance(Carol, 500)` → not sufficient (balance 0),
     but `insufficient_funds_mode=ALLOW_NEGATIVE` and within
     `credit_limit_iqd=1000` → proceed.
   - `finance.charge(Carol, 500)` → wallet goes to -500.
   - Snapshot. Stop.

**Outcome:** Staff are charged at the staff rate (500 IQD), and the staff
plan allows a negative balance up to the credit limit. This is the
**staff wallet** configuration.

### Example D — Student has a date-range lunch plan but uses wallet for extra breakfast

**Setup:**
- `MealPlan "Student Lunch Monthly"`: `kind=LUNCH`, `mode=DATE_RANGE`.
- `MealSubscription S1`: `person=Dave`, `meal_plan="Student Lunch
  Monthly"`, `priority=1`, `status=ACTIVE`, September range.
- `MealPlan "Student Breakfast Wallet"`: `kind=BREAKFAST`, `mode=WALLET`.
  `MealPeriodPrice`: breakfast → 800 IQD.
- `MealSubscription S2`: `person=Dave`, `meal_plan="Student Breakfast
  Wallet"`, `priority=2`, `status=ACTIVE`, wide range.

**Scenario D1 — Dave requests lunch on 2026-09-15:**
1. Candidates: S1, S2.
2. Period filter: lunch. S1 matches; S2 does not.
3. Evaluate: S1 `mode=DATE_RANGE` → ELIGIBLE, `final_charge_iqd=0`. No
   wallet call. Stop.

**Scenario D2 — Dave requests breakfast on 2026-09-15:**
1. Candidates: S1, S2.
2. Period filter: breakfast. S2 matches; S1 does not.
3. Evaluate: S2 `mode=WALLET` → `MealPeriodPrice`=800 →
   `final_charge_iqd=800` → `finance.charge(800)`. Stop.

**Scenario D3 — Dave requests lunch on 2026-10-15 (after the date-range
plan expired):**
1. Candidates: S1 (`status=EXPIRED` by now — filtered out), S2.
2. Period filter: lunch. S2's `kind=BREAKFAST` does not match a lunch
   period. No candidate remains.
3. No subscription grants eligibility. Exceptions: none.
4. Default: `NOT_ELIGIBLE`, `reason_code="no_subscription"`.

> If Dave also had a `WALLET` lunch fallback (priority 3), step 4 would
> evaluate it and charge per service. This shows the **primary expires →
> fallback** behavior.

**Outcome:** Date-range covers lunch in-term; breakfast is an extra
wallet charge; after the date-range plan expires, lunch is no longer
covered (unless a wallet fallback exists).

### Example E — Breakfast + lunch same day triggers a future same-day discount rule

**Setup (after `apps.discounts` is built — phase 19):**
- `MealPlan "Student Breakfast Wallet"`: `kind=BREAKFAST`, `mode=WALLET`.
  `MealPeriodPrice`: breakfast → 1000 IQD.
- `MealPlan "Student Lunch Wallet"`: `kind=LUNCH`, `mode=WALLET`.
  `MealPeriodPrice`: lunch → 1000 IQD.
- `MealSubscription S1`: `person=Eve`, `meal_plan="Student Breakfast
  Wallet"`, `priority=1`, `status=ACTIVE`.
- `MealSubscription S2`: `person=Eve`, `meal_plan="Student Lunch Wallet"`,
  `priority=2`, `status=ACTIVE`.
- `apps.discounts` rule: "Same-day multi-meal: if a person has one
  confirmed breakfast and one confirmed lunch on the same day, the
  second meal gets 500 IQD off."

**Scenario E1 — Eve requests breakfast at 07:30:**
1. Candidates: S1 (breakfast), S2 (lunch). Period filter: breakfast → S1.
2. Evaluate: S1 `mode=WALLET` → `resolve_price` → `MealPeriodPrice`=1000.
3. Discounts: `discounts.resolve(person=Eve, price_base=1000,
   product_code="meal_breakfast", context={..., "same_day_meals": []})`
   → no prior meals today → `discount_iqd=0`, `final_charge_iqd=1000`.
4. `finance.charge(1000)`. `MealServiceEvent` confirmed for breakfast.
   Stop.

**Scenario E2 — Eve requests lunch at 12:00 the same day:**
1. Candidates: S1, S2. Period filter: lunch → S2.
2. Evaluate: S2 `mode=WALLET` → `resolve_price` → `MealPeriodPrice`=1000.
3. Discounts: Meals builds `same_day_meals` via
   `selectors.confirmed_meals_for(Eve, date)` →
   `[(BREAKFAST, "Student Breakfast Wallet", 1000, CONFIRMED)]`.
   `discounts.resolve(person=Eve, price_base=1000,
   product_code="meal_lunch", context={..., "same_day_meals": [...]})`
   → rule matches "second meal of the day" → `discount_iqd=500`,
   `final_charge_iqd=500`.
4. `finance.check_balance(Eve, 500)` → sufficient.
5. `finance.charge(500)`. Snapshot: `price_base_iqd=1000`,
   `discount_iqd=500`, `final_charge_iqd=500`. Stop.

**Outcome:** The same-day multi-meal discount is resolved by
`apps.discounts` using the `same_day_meals` context that Meals supplies.
Meals does **not** compute the discount; it supplies the history and
snapshots the result. This is the **future same-day multi-meal
discount** configuration (depends on phase 19).

---

## 26. Open questions / decisions

### 26.1 Decided (this document)

| # | Decision | Effect |
|---|---|---|
| D1 | **App name is `apps.meals`** (plural). | Used consistently throughout. |
| D2 | **Future model name is `MealPlan`.** | Legacy `MealProfile` maps to `MealPlan` (§10, §21.2). `MealProfile` is legacy-only. |
| D3 | **Legacy `MealProfile` maps to `MealPlan`.** | Migration in §21.2. |
| D4 | **Meals owns price resolution.** | Base price, per-Person overrides, per-period overrides, and final-charge composition live in `apps.meals` (§14). |
| D5 | **Finance records pre-resolved charges only.** | `finance.charge` takes a pre-resolved `amount_iqd`; it does not resolve prices or call discounts (§5, §20). |
| D6 | **Granular pricing per Person and per MealPeriod belongs to Meals.** | `MealPersonPriceOverride` and `MealPeriodPrice` are meal-product facts (§15, §16). |
| D7 | **Wallet mutations only through Finance services.** | Meals calls `finance.charge` / `refund` / `check_balance` (§20). |
| D8 | **Attendance / AI recognition do not own pricing.** | Recognition events are inputs only (§6, §7). |
| D9 | **Date-range and wallet plans may both have date ranges.** | Date-range = entitlement, no per-service debit; wallet = per-service charge while active (§10.1, §11). |
| D10 | **Priority/order semantics: lower number = higher priority = evaluated first; same-priority overlap rejected; different-priority overlap allowed.** | §11.5, resolver §13. |
| D11 | **Discount call lives in Meals' `resolve_price`, not in `finance.charge`.** | §14.2, §19. Deviation from `finance_domain_architecture.md` §8.2 and `DOMAIN_INTEGRATION_GUIDE.md` §13.3; follow-up docs reconciliation needed. |

### 26.2 Open (genuinely unresolved)

1. **`MealPeriod` vs reusing `scheduler.PeriodTemplate` directly.** This
   document recommends a `MealPeriod` wrapper so Meals does not import
   `apps.attendance` and so not every period is a meal period. Confirm
   once `apps.scheduler` is designed.
2. **`price_override_iqd` semantics.** Store the signed difference (period
   list − override) or the absolute override price? Recommendation: store
   both `price_base_iqd` (the resolved base, i.e. what is charged before
   discounts) and `price_override_iqd` as the signed difference for audit
   clarity (§14.1).
3. **"No current placement" handling for students.** Hard `NOT_ELIGIBLE`
   or warning with supervisor override? Recommendation: hard deny with
   override allowed.
4. **Subscription in-place edits vs cancel-and-recreate.** Recommendation:
   cancel-and-recreate for history purity.
5. **Staff absence signal.** Should staff have an absence-like signal that
   denies meals (e.g. from a future HR domain)? Defer until HR exists.
6. **Supervisor identity: `StaffProfile` vs `auth.User`.** Start with
   `performed_by_user` (auth.User) required, `performed_by` (StaffProfile)
   optional.
7. **`MealException.person`-only (guest) exceptions.** Require a
   `PersonRole(guest)` or allow any `Person`? Defer to phase 21.
8. **Eligibility re-resolution after a service event.** Settled: one
   canonical frozen row per `(person, date)` (§12.1). Open: whether to
   *also* allow a superseding row with a `superseded` flag for forensic
   history. Recommendation: one canonical frozen row.
9. **Per-Person override time-scoping.** Are `effective_from` /
   `effective_until` required, or is `is_enabled` enough?
   Recommendation: keep both; `is_enabled` for on/off,
   `effective_from`/`until` for term-scoped overrides.
10. **`MealServiceEvent.wallet_transaction` typed FK target app label.**
    Confirm that `apps.wallet` (or `apps.finance`) exposes
    `WalletTransaction` as the FK target, and that the FK is `PROTECT`.
    Reconcile with `finance_domain_architecture.md` app-label choice.
11. **Staff academic/department snapshot.** Should `MealServiceEvent` add
    a `department`/`role` snapshot for staff service events? Defer until
    a reporting requirement appears (not required for v1).
12. **Same-day multi-meal discount context shape.** Is
    `same_day_meals = [(kind, plan_id, final_charge, status), ...]`
    sufficient, or does `apps.discounts` need richer context (e.g.
    timestamps, period ids)? Confirm when `apps.discounts` is designed
    (phase 18).

---

## 27. What should NOT be implemented yet

| Item | Why deferred |
|---|---|
| The `apps.meals` app itself | Design-only. Implementation needs a separate approved task per phase (§24). |
| Wallet charge integration (live) | Until `apps.finance` is extracted and exposes the service boundary, charges run in dry-run/`UNPAID` mode (phase 10 → phase 14). |
| Discount resolution (live) | Until `apps.discounts` exists, `discount_iqd=0` and `final_charge_iqd=base` (phase 18 → phase 19). |
| Same-day multi-meal discounts (live) | Depends on `apps.discounts` (phase 18) and the `same_day_meals` context (§19.5). |
| AI recognition event consumption (live) | Until `AttendanceEvent.person` migration is done, recognition input is manual/QR only. |
| `MealSupervisorAction.performed_by` (StaffProfile) population | Start with `performed_by_user` (auth.User) only. |
| Guest / person-only exceptions (non-student, non-staff) | Defer to phase 21. |
| Multi-meal-kind support (breakfast / snack) operational plans | Models support it from phase 2 (`MealPlan.kind` / `MealPeriod.kind`); operational plans/periods/prices for non-lunch kinds are phase 22. |
| Family-level billing / exceptions | Defer until guardian/family domain is built. |
| Removing legacy `MealSubscription` / `MealProfile` / `MealProfilePeriod` / `MealRecord` | Keep through dual-FK migration; remove only after verification (phase 23). |
| Removing legacy `DiscountProfile` / `DiscountRule` | Keep until `apps.discounts` is built and migrated (phase 24). |
| Typed FKs from `WalletTransaction` to meals models | Forbidden by design (circular ownership); use the reverse typed FK on `MealServiceEvent`. |
| Stacking `MealPeriodPrice` + `MealPersonPriceOverride` | Forbidden (first-match-wins — §18.2). Stacking is a discount-domain concern. |
| Naming the future model `MealProfile` | Decided: future name is `MealPlan`. `MealProfile` is legacy only. |

---

End of document.
