# Domain Integration Guide — BISK_RFv4

Date: 2026-07-03
Branch: feature/person-architecture
Version: 1.0
Status: Engineering guide. No code, models, or migrations changed.

---

## 1. Purpose

This guide defines **how BISK_RFv4 domains communicate during
implementation**. It establishes the rules for cross-app calls, service
boundaries, dependency direction, adapters, and audit records so that new
apps (`apps.meals`, `apps.finance`, `apps.discounts`, etc.) integrate
cleanly with the existing `apps.identity` and `apps.academics` foundations
without creating circular dependencies or coupling to legacy code.

It complements:
- `docs/development/AI_DEVELOPMENT_GUIDE.md`
- `docs/development/PROJECT_ARCHITECTURE.md`
- `docs/architecture/education_domain_architecture.md`
- `docs/architecture/academics_domain_architecture.md`
- `docs/architecture/finance_domain_architecture.md`
- `docs/architecture/lunch_domain_architecture.md`
- `docs/architecture/attendance_migration_safety_plan.md`
- `docs/architecture/legacy_lunch_wallet_attendance_investigation.md`

---

## 2. Core Principle: Domain Ownership

Each domain owns its models, its business logic, and its migration
history. Other domains interact with it through **well-defined boundaries**,
not by reaching into its internals.

| Domain | Owns | Depends on (read/service only) |
|---|---|---|
| `apps.identity` | Person, StudentProfile, StaffProfile, RoleType, PersonRole | (nothing) |
| `apps.academics` | AcademicYear, SchoolLevel, Grade, Section, StudentEnrollment, SectionPlacement | identity |
| `apps.meals` | MealPlan, LunchSubscription, LunchEligibility, LunchServiceEvent, LunchSupervisorAction, LunchException | identity, academics |
| `apps.finance` | Wallet, WalletTransaction, Charge, Payment, Refund, Adjustment, PricingRule | identity, academics |
| `apps.discounts` (future) | DiscountProfile, DiscountRule, DiscountAssignment | identity, academics |
| `apps.attendance` | AttendanceRecord, AttendanceEvent, FaceEmbedding, RecognitionSettings | identity |
| `apps.scheduler` | PeriodTemplate, PeriodOccurrence | (standalone) |

**Rule:** if domain A depends on domain B, then domain B must **never**
depend on domain A. Dependencies are one-way.

---

## 3. Service / Selectors / Validators Pattern

Every domain app follows the same internal structure (established by
`apps.identity` and `apps.academics`):

```
apps/<domain>/
├── models.py        # Owned models only
├── services.py      # Business logic that mutates state (write path)
├── selectors.py     # Read-only queries (read path)
├── validators.py    # Invariants and validation rules
├── admin.py         # Django admin configuration
└── migrations/      # Clean, app-owned migration history
```

| Layer | Responsibility | Allowed to |
|---|---|---|
| `models.py` | Field definitions, `clean()`, properties, `__str__` | Validate self, reference other apps via FK only |
| `services.py` | Create/update/delete operations, state transitions, cross-domain calls | Mutate own models, call other domains' services |
| `selectors.py` | Queries, lookups, aggregations | Read only; never mutate |
| `validators.py` | Reusable validation functions | Raise `ValidationError`; never mutate |
| `admin.py` | Admin UI | Call services; never inline business logic |

**Views and templates** call services and selectors. They never contain
business logic or direct cross-model mutation.

---

## 4. Cross-App Communication Rules

| Rule | Enforcement |
|---|---|
| **Call services, not models.** Domain A calls `domain_b.services.some_function()`, not `domain_b.models.SomeModel.objects.create(...)`. | The owning domain's service encapsulates invariants. |
| **Read via selectors.** Domain A calls `domain_b.selectors.some_query()` for cross-domain reads. | Selectors are the stable read API. |
| **FKs point to owning models.** A meals model may FK to `identity.StudentProfile`, but it must not FK to `finance.WalletTransaction` (use a generic reference instead). | Prevents migration-level coupling. |
| **Migration dependencies are one-way.** `apps.meals` migrations may depend on `apps.identity` and `apps.academics`, never on `apps.finance` or `apps.attendance`. | Isolates migration histories. |
| **No circular imports.** If A imports B, B must not import A. | Python-level enforcement. |

---

## 5. Avoiding Circular Dependencies

| Technique | When to use |
|---|---|
| **Service boundary** | A calls `finance.charge(person, amount)` — finance imports nothing from A. |
| **Generic reference** (`source_module` + `reference_type` + `reference_id`) | When a ledger row must reference the originating object without a typed FK. Finance stores `source_module="meals", reference_type="LunchServiceEvent", reference_id=42`. |
| **Reverse typed FK** | The calling domain stores the FK to the owning domain's model (e.g. `LunchServiceEvent.wallet_transaction` → `finance.WalletTransaction`). The owning domain stores only the generic reference. |
| **String FK references** | Use `"identity.StudentProfile"` (string) in FK definitions to avoid import-time coupling. |
| **Signal/event (last resort)** | Only when a truly decoupled notification is needed and a service call is impractical. Prefer explicit service calls. |

---

## 6. When Direct Model Imports Are Acceptable

| Acceptable | Example |
|---|---|
| **FK field definition** referencing another app's model via string. | `student = ForeignKey("identity.StudentProfile", ...)` |
| **Type hints** in service signatures. | `def enroll_student(*, student: StudentProfile, ...)` |
| **`select_related` / `prefetch_related`** across FKs. | `qs.select_related("student__person")` |
| **Reading** another domain's model in a selector when no service exists yet. | `Section.objects.filter(...)` from a meals selector (read-only). |

| Not acceptable | Why |
|---|---|
| **`OtherAppModel.objects.create(...)`** from a different domain. | Bypasses the owning domain's services and invariants. |
| **`other_app_model.field = x; other_app_model.save()`** from a different domain. | Direct mutation; no audit, no validation. |
| **Importing another domain's `models.py` at module top-level in a way that creates a cycle.** | Circular import crash. |

---

## 7. When to Use Services Instead of Direct Model Mutation

| Scenario | Use service | Why |
|---|---|---|
| Creating a wallet transaction | `finance.charge(...)` | Ensures ledger immutability, balance snapshot, atomicity. |
| Confirming a lunch service event | `meals.record_service_event(...)` | Ensures eligibility snapshot, placement link, audit row. |
| Transferring a student's section | `academics.transfer_section(...)` | Closes old placement, creates new placement, preserves history. |
| Withdrawing an enrollment | `academics.withdraw(...)` | Closes placement, sets status, records date. |
| Creating a person | `identity.create_person(...)` | Validates code uniqueness, normalizes names. |
| Top-up / refund / adjustment | `finance.top_up(...)` / `finance.refund(...)` / `finance.adjust(...)` | Atomic ledger + balance update. |

**Golden rule:** if an operation mutates state across more than one model,
or has invariants (uniqueness, balance, status transition, audit), it
belongs in a service.

---

## 8. Legacy Adapter Strategy

The legacy `apps.attendance` lunch/wallet/dashboard code works and must not
be broken. New domains coexist with it via **adapters**, not rewrites.

| Legacy code | Adapter |
|---|---|
| `_resolve_effective_meal_setup(student, day, period)` | Wrap as `meals.resolve_eligibility(student_profile, date)` — calls the legacy function internally. |
| `confirm_record` / `reverse_record` wallet logic | Extract into `finance.charge()` / `finance.refund()` services. Views call services instead of mutating `wallet.balance_iqd` directly. Same behavior, service boundary. |
| `recalc_meal_flags_for_students` | Wrap as `meals.sync_has_meal_flag()` adapter. Signal keeps calling it until enrollment-based eligibility replaces it. |
| `ingest_match(h_code, ...)` | Add `ingest_match_by_person(person, ...)` adapter that resolves the legacy `Student` via `StudentProfile.legacy_student`. |
| `Student.h_code` / `Student.full_name()` template accessors | Add a legacy-adapter property on `Person`/`StudentProfile` so templates switch to `person.display_code` incrementally. |

**Principles:**
- Adapters wrap; they do not rewrite.
- Legacy behavior is preserved exactly.
- New code calls the adapter, not the legacy function directly.
- The legacy function is replaced only after all callers use the adapter and the legacy path is verified dead.

---

## 9. Event / History / Audit Records

| Record type | Owner | Immutability |
|---|---|---|
| `WalletTransaction` | finance | Immutable after creation. Corrections create reversal rows. |
| `LunchServiceEvent` | meals | Immutable once confirmed. Financial/academic snapshots frozen. |
| `LunchEligibility` | meals | Recalculable until a `LunchServiceEvent` references it, then frozen. |
| `LunchSupervisorAction` | meals | Append-only audit trail. Never updated or deleted. |
| `StudentEnrollmentSectionPlacement` | academics | Append-only. Transfers close old + create new. Never deleted. |
| `StudentEnrollment` | academics | Status transitions only; never deleted. Historical grade/section snapshots preserved. |

**Rules:**
- Every mutating supervisor/financial action writes an audit row in the
  same transaction.
- Snapshots (price, discount, balance, grade, section) on immutable records
  are justified because they cannot be reconstructed from live data later.
- No audit record is ever `UPDATE`d or `DELETE`d.

---

## 10. Finance Integration Rules

| Rule | Detail |
|---|---|
| Meals calls `finance.charge(person, product_code, price_base, source_module="meals", reference_type="LunchServiceEvent", reference_id=...)`. | Finance resolves pricing, calls discounts (if available), checks funds, appends ledger row, returns `(Charge, WalletTransaction)`. |
| Meals calls `finance.check_balance(person, amount)` for dashboard warnings. | Returns `(balance, sufficient)`. |
| Meals calls `finance.refund(person, original_charge, amount, reason)` for reversals. | Finance creates the refund transaction. |
| Meals **never** reads `Wallet.balance_iqd` directly. | Balance is finance-owned. |
| Meals **never** creates `WalletTransaction` rows. | Ledger is finance-owned. |
| Meals stores the returned `WalletTransaction` FK + balance snapshots on `LunchServiceEvent`. | Immutable historical truth. |
| Finance stores `source_module="meals"` + generic reference on the transaction. | No typed FK back to meals (avoids circular dependency). |
| Discounts are **not** owned by finance. | Finance calls `discounts.resolve(...)` (future `apps.discounts`) via service boundary. Until then, `discount_iqd=0`. |

---

## 11. Meals Integration Rules

| Rule | Detail |
|---|---|
| App name is `apps.meals` (plural). | Supports breakfast, lunch, snack, etc. "Lunch" is a specific meal period. |
| Meals FK to `identity.StudentProfile` / `identity.Person`. | Never to `attendance.Student`. |
| Meals reads academic context from `apps.academics` selectors. | `current_enrollments_in_section`, `active_enrollment_for`, etc. |
| Meals does not duplicate grade/section/name fields. | Reads via `student.person.full_name` / `enrollment.grade` / `enrollment.section`. Snapshots allowed only on immutable records. |
| Meals eligibility resolver calls `finance.check_balance` for wallet-mode plans. | But the insufficient-funds *policy* (`deny`/`allow_unpaid`/`allow_negative`) is meals-domain policy, not finance logic. |
| Meals consumes `AttendanceEvent` as input only. | `LunchServiceEvent.recognition_event` → `AttendanceEvent` (SET_NULL). Meals never runs recognition. |
| Meals migrations depend on `apps.identity` + `apps.academics` only. | Never on `apps.attendance` or `apps.finance`. |

---

## 12. Attendance / AI Integration Rules

| Rule | Detail |
|---|---|
| `apps.attendance` produces `AttendanceEvent` / `AttendanceRecord` for a `Person` (after FK migration). | Recognition engine, cameras, embeddings stay in attendance. |
| Meals consumes recognition events as **input only**. | `LunchServiceEvent.recognition_event` FK (SET_NULL). |
| Meals **never** imports recognition engine, camera, or embedding code. | Service boundary only. |
| Confidence (`score`) is surfaced in the dashboard but **not** thresholded by meals. | Thresholding is `RecognitionSettings` policy in attendance. |
| Unknown-person queue resolution is an attendance/identity action, not a meals action. | Meals only processes events that resolve to a `Person`. |
| `apps.attendance` migrations are **frozen** (drift not reconciled yet). | New apps must not depend on attendance migrations. See `attendance_migration_safety_plan.md`. |

---

## 13. Examples

### 13.1 Meals asking Finance to charge wallet

```python
# apps/meals/services.py
from apps.finance import services as finance_services

def confirm_service_event(*, service_event, confirmed_by):
    ...
    if service_event.meal_plan.mode == "WALLET":
        charge, tx = finance_services.charge(
            person=service_event.student.person,
            product_code="meal_lunch",
            price_base_iqd=price,
            source_module="meals",
            reference_type="LunchServiceEvent",
            reference_id=service_event.pk,
            academic_year=current_academic_year,
        )
        service_event.wallet_transaction = tx
        service_event.wallet_balance_before_iqd = tx.balance_before_iqd
        service_event.wallet_balance_after_iqd = tx.balance_after_iqd
    ...
```

### 13.2 Attendance sending recognition event to Meals

```python
# Meals does NOT pull from attendance. Meals reads existing events.
# apps/meals/selectors.py
from apps.attendance.models import AttendanceEvent

def pending_recognition_for_section(*, section, date):
    """Return today's attendance events for students in this section
    that have not yet been linked to a LunchServiceEvent."""
    enrollment_ids = current_enrollments_in_section(section=section).values_list("id", flat=True)
    person_ids = StudentProfile.objects.filter(
        enrollments__in=enrollment_ids
    ).values_list("person_id", flat=True)
    return (
        AttendanceEvent.objects
        .filter(person_id__in=person_ids, ts__date=date)
        .exclude(lunch_service_events__isnull=False)
        .order_by("-ts")
    )
```

### 13.3 Finance consuming Discounts later

```python
# apps/finance/services.py
def charge(*, person, product_code, price_base_iqd, source_module, ...):
    # Step 1: resolve discount (if discounts domain exists)
    try:
        from apps.discounts import services as discount_services
        discount_result = discount_services.resolve(
            person=person,
            price_base=price_base_iqd,
            product_code=product_code,
            academic_year=academic_year,
        )
        discount_iqd = discount_result.discount_iqd
    except ImportError:
        # discounts domain not built yet — no discount
        discount_iqd = 0

    final_charge = price_base_iqd - discount_iqd
    # ... create Charge, WalletTransaction, update Wallet.balance ...
```

### 13.4 Legacy attendance wrapped by adapter

```python
# apps/meals/legacy_adapter.py
from apps.attendance.views import _resolve_effective_meal_setup

def resolve_eligibility(*, student_profile, date, period_template=None):
    """Adapter: wraps the legacy eligibility resolver.
    New code calls this; legacy dashboard keeps calling the original."""
    legacy_student = student_profile.legacy_student
    if legacy_student is None:
        return {"sub": None, "profile": None, "blocked": True, "block_reason": "no_legacy_student"}
    return _resolve_effective_meal_setup(legacy_student, date, period_template)
```

---

## 14. Anti-Patterns

| Anti-pattern | Why it's wrong | Do instead |
|---|---|---|
| **Mutating `Wallet.balance_iqd` from a view.** | Bypasses the ledger, no audit, no atomicity. | Call `finance.charge(...)` / `finance.refund(...)`. |
| **Putting finance logic inside Meals.** | Couples meals to wallet internals; breaks domain ownership. | Meals calls `finance.charge(...)`; finance owns the ledger. |
| **Putting AI recognition logic inside Meals.** | Couples meals to camera/embedding engine; meals is not a recognition system. | Meals consumes `AttendanceEvent` as input only. |
| **Duplicating student fields (name, grade, section) on meal/finance models.** | Diverges from identity source of truth; stale data. | Reference `StudentProfile` / `Person` via FK; read via `person.full_name` / `enrollment.grade`. |
| **Creating `WalletTransaction` rows from meals.** | Ledger is finance-owned; meals must not write ledger rows. | Meals calls `finance.charge(...)` which creates the transaction. |
| **FK-ing from finance to `LunchServiceEvent`.** | Creates `finance → meals` circular dependency. | Use generic reference (`source_module` + `reference_id`); meals holds the reverse typed FK. |
| **Importing `apps.attendance.models` in a new app's `models.py`.** | Couples to the drifted attendance migration graph. | Use string FK references or service-boundary calls only. |
| **Putting discount rule logic on `WalletTransaction` or `Charge`.** | Couples ledger to discount-rule versions; breaks immutability. | Discounts live in `apps.discounts`; finance consumes the result and snapshots `discount_iqd`. |
| **Using `h_code` in new code.** | Legacy identifier; new code uses `person.display_code`. | Use `person.display_code` / `StudentProfile.code`. |
| **Editing the legacy attendance migration graph.** | Risky; squash was fake-applied; DB ≠ migration state. | Freeze attendance migrations; build new apps separately. |

---

## 15. Recommended Implementation Checklist

Before starting any cross-domain integration:

- [ ] Confirm the dependency direction is one-way (no cycles).
- [ ] Confirm the calling app's migrations do not depend on the called app's migrations (unless the called app is `identity` or `academics`).
- [ ] Confirm the calling app uses **services** for mutations, **selectors** for reads.
- [ ] Confirm no direct `OtherAppModel.objects.create/update/delete` from outside the owning domain.
- [ ] Confirm no `Wallet.balance_iqd` mutation outside `apps.finance.services`.
- [ ] Confirm no `WalletTransaction` creation outside `apps.finance.services`.
- [ ] Confirm no recognition-engine imports in `apps.meals`.
- [ ] Confirm no `h_code` references in new code (use `person.display_code`).
- [ ] Confirm no duplicated identity fields (name/grade/section) on new models (use FK + read, or justified snapshot on immutable record).
- [ ] Confirm audit records are written in the same transaction as the state change.
- [ ] Confirm immutable records (transactions, service events, placements) are never `UPDATE`d or `DELETE`d.
- [ ] Confirm legacy attendance migrations are not touched.
- [ ] Run `python manage.py check` and `python manage.py makemigrations --check --dry-run`.
- [ ] Run the relevant app's tests under the test-DB role.

---

End of document.
