# Academics Domain Architecture — BISK_RFv4

Date: 2026-07-03
Branch: feature/person-architecture
Version: 1.0 Draft — pending review
Status: Architecture design only. No code, models, or migrations are produced by this document.

---

## 1. Purpose

This document designs the **Academics domain** for BISK_RFv4 — specifically
the `AcademicYear`, `Grade`, `Section`, and `Enrollment` entities — **before**
any of them are implemented.

It is the focused, deeper counterpart to the broader
`docs/architecture/education_domain_architecture.md`, which described the
education domain at a high level. This document narrows in on the academic
structure itself: how a year, a grade, a section, and a student's enrollment
relate to the already-implemented identity foundation (`Person`,
`StudentProfile`, `StaffProfile`, `RoleType`, `PersonRole`).

The goal is to settle the academic structure so that future modules —
attendance, lunch, wallet, discounts, reports, AI attendance — have a stable,
year-scoped anchor to reference.

### Scope

| In scope | Out of scope |
|---|---|
| `AcademicYear`, `Grade`, `Section`, `StudentEnrollment` design | Implementing these models now |
| Homeroom / adviser relationship via `StaffProfile` | Full timetable (Period/Lecture/Classroom) |
| Enrollment lifecycle (active, future, transfer, withdrawal, graduation, repeat, history) | Assessment, marks, report cards |
| How academics supports other domains | Guardian / family design (see education_domain doc) |
| Recommended model boundaries | App extraction / mechanical refactor |
| Migration safety and open questions | Multi-tenant activation (Organization/School/Campus) |

### Constraints respected

- Do **not** recreate the old `attendance.Student` model. Student identity
  lives in `Person` + `StudentProfile`.
- Academics references `StudentProfile` (and `StaffProfile`), never
  duplicate student identity fields.
- Finance, lunch, and attendance fields are **not** placed directly on
  `Enrollment` unless architecturally justified (see Section 7).
- Historical enrollments are preserved; they are never overwritten
  carelessly when a student transfers, withdraws, or repeats a year.
- This document follows `education_domain_architecture.md`,
  `person_identity_architecture.md`, `PROJECT_ARCHITECTURE.md`, and
  `AI_DEVELOPMENT_GUIDE.md`.

---

## 2. Design Principles

Inherited from the project standards and the identity foundation.

| # | Principle | Application to Academics |
|---|---|---|
| 1 | **Identity before role** | `AcademicYear`, `Grade`, `Section` are structural; they do not own identity. `Enrollment` links to `StudentProfile`, which shares its PK with `Person`. |
| 2 | **AcademicYear is foundational** | Every year-specific fact (which grade, which section, which adviser) lives on an enrollment scoped to an `AcademicYear`. Until `AcademicYear` exists, the transitional `StudentProfile.grade`/`homeroom` fields remain the interim source of truth. |
| 3 | **Year-scoped, not Person-scoped** | Grade and section are **not** fields on `Person` or `StudentProfile`. A student's grade changes every year; that change is a new enrollment row, not an update. |
| 4 | **History is append-only** | An enrollment is a historical record of "this student was in this grade/section in this year." Transfers, withdrawals, and repeats create or close enrollments; they do not delete or silently overwrite prior ones. |
| 5 | **Workflow as state machine** | Enrollment uses a `status` enum with documented valid transitions. No scattered `is_active`/`is_withdrawn`/`is_graduated` booleans. |
| 6 | **Service-layer business logic** | Enrollment creation, transfer, withdrawal, and graduation are service methods, not admin actions or view logic. |
| 7 | **Backward compatibility always** | `StudentProfile.grade` and `StudentProfile.homeroom` remain as transitional mirrors while enrollments are introduced. They are backfilled from the active enrollment and removed only after verification. |
| 8 | **Domain ownership first** | Academics owns `AcademicYear`, `Grade`, `Section`, `StudentEnrollment`. Attendance, lunch, wallet, and discounts depend on academics; academics does not depend on them. |
| 9 | **display_code in templates** | Reports and dashboards render `person.display_code`, never legacy `h_code`. Enrollment lists are keyed by `StudentProfile.code`. |
| 10 | **Incremental delivery** | The academic app is introduced in one phase; each entity is independently deployable. No big-bang. |
| 11 | **Configuration over hard-coding** | Grades and sections are admin-configurable. Status enums and the enrollment state machine remain code-controlled. |
| 12 | **No duplicated identity** | No new `h_code`, no `first_name`/`last_name` on `Enrollment`. The student's name is always read via `enrollment.student.person.full_name`. |

---

## 3. Proposed App / Module Name

```
apps/academics/
├── __init__.py
├── apps.py
├── models.py        # AcademicYear, Grade, Section, StudentEnrollment
├── admin.py
├── services.py      # enroll_student, transfer_section, withdraw, graduate, repeat_year
├── selectors.py     # active_enrollment_for, enrollments_for_year, students_in_section
├── validators.py    # overlap, capacity, status-transition validation
└── migrations/
```

**App label:** `academics` (Django app). Python module path: `apps.academics`.

> Naming note: the broader ERP doc uses `apps/academic/` (singular) in its
> long-term sketch. This document proposes `apps/academics/` (plural) to
  match Django convention (e.g. `django.contrib.auth`). This is an open
  decision (see Section 10, Q1); either is acceptable as long as it is
  consistent.

### Dependency direction

```
apps.academics ──depends on──► apps.identity   (StudentProfile, StaffProfile, Person)
apps.academics ──depended on by──► attendance, meal, wallet, finance, reporting
```

Academics must **not** import from `apps.attendance`, `apps.wallet`, or
`apps.meal`. Those modules reference academics (e.g.
`MealSubscription.academic_year`); the reverse is forbidden to avoid
circular ownership.

---

## 4. Core Entities

All designs below are **conceptual blueprints**. No migration is produced by
this document. Field types are indicative; exact choices are made at
implementation time.

### 4.1 AcademicYear

- **Owner:** `apps.academics`
- **Purpose:** The temporal scope for all year-specific facts.
- **Conceptual fields:**

```python
class AcademicYear(models.Model):
    name = models.CharField(max_length=100)        # "2026-2027"
    code = models.CharField(max_length=32, unique=True)  # "2026-27"
    start_date = models.DateField()
    end_date = models.DateField()
    is_active = models.BooleanField(default=False)  # exactly one active per school
    school = models.ForeignKey(
        "schools.School", on_delete=models.CASCADE, null=True, blank=True,
        related_name="academic_years",
    )  # nullable until multi-tenant exists

    class Meta:
        ordering = ["-start_date"]
        constraints = [
            # end_date must be on or after start_date
            models.CheckConstraint(
                check=models.Q(end_date__gte=models.F("start_date")),
                name="academic_year_end_after_start",
            ),
        ]
```

- **Rules:**
  - `code` is the natural key used by other models' FKs and by imports.
  - At most one row with `is_active=True` per school. Enforced in
    `clean()`/`save()` (or via a partial unique constraint when multi-tenant
    exists) — not via a hard DB constraint that would complicate seeding.
  - `school` is nullable now; it becomes required when multi-tenant is
    activated. Do **not** add `school` to `Person`; year-scoping belongs on
    enrollment/subscription/assignment, per the identity architecture.

### 4.2 Grade

- **Owner:** `apps.academics`
- **Purpose:** A year-group level (e.g., "Grade 1", "Grade 10", "KG 2").
- **Conceptual fields:**

```python
class SchoolLevel(models.Model):
    name = models.CharField(max_length=100)   # "Primary", "Secondary"
    code = models.CharField(max_length=32, unique=True)
    order = models.PositiveSmallIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["order"]


class Grade(models.Model):
    name = models.CharField(max_length=100)    # "Grade 1"
    code = models.CharField(max_length=32, unique=True)  # "G1"
    level = models.ForeignKey(
        SchoolLevel, on_delete=models.PROTECT, related_name="grades",
        null=True, blank=True,
    )
    order = models.PositiveSmallIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["order"]
```

- **Rules:**
  - `Grade` is a reusable, admin-configurable entity. It is **not** a field
    on `Person` or `StudentProfile`.
  - `code` is the natural key for imports and FK references.
  - `level` (FK to `SchoolLevel`) is optional; some schools may not use
    levels. `SchoolLevel` groups grades for reporting (Primary vs.
    Secondary).
  - `Grade` itself is not year-scoped. "Grade 1 in 2026-27" is expressed
    through `StudentEnrollment(grade=..., academic_year=...)`, not by
    creating a new `Grade` row each year.

### 4.3 Section

- **Owner:** `apps.academics`
- **Purpose:** A real class section for **one academic year** — e.g.,
  "Grade 1 / Section A / 2026-27". A `Section` is a year-scoped instance, not
  a timeless label. Each academic year the school creates a fresh set of
  `Section` rows for the grades it offers that year.
- **Conceptual fields:**

```python
class Section(models.Model):
    name = models.CharField(max_length=100)     # "Section A"
    code = models.CharField(max_length=32)      # "A"
    grade = models.ForeignKey(
        Grade, on_delete=models.PROTECT, related_name="sections",
    )
    academic_year = models.ForeignKey(
        AcademicYear, on_delete=models.PROTECT, related_name="sections",
    )
    homeroom_adviser = models.ForeignKey(
        "identity.StaffProfile", on_delete=models.SET_NULL,
        related_name="advised_sections", null=True, blank=True,
    )
    capacity = models.PositiveIntegerField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = [("academic_year", "grade", "code")]
        ordering = ["academic_year__start_date", "grade__order", "code"]
```

- **Rules:**
  - A `Section` belongs to **both** a `Grade` and an `AcademicYear`. It is
    identified by `(academic_year, grade, code)` — "2026-27 / Grade 1 /
    Section A". There is no year-agnostic `Section`.
  - Because `Section` is year-scoped, `homeroom_adviser` is inherently
    year-scoped as well: each year's section row carries that year's
    adviser. Historical adviser assignments are preserved automatically as
    past-year section rows. No separate `SectionAdviser` model is needed
    for adviser history.
  - `homeroom_adviser` is a FK to `StaffProfile` (the homeroom teacher /
    adviser). This is the **only** direct StaffProfile reference in the
    academics structure. See Section 4.6 for why it lives here and not on
    enrollment.
  - `capacity` is **year-specific**. Each year's section may have a
    different capacity. It is enforced in `StudentEnrollment.clean()` as a
    warning or hard block (open decision — see Section 10, Q4).
  - Historical roster and adviser reports read past-year `Section` rows
    directly; because sections are recreated per year, those rows are never
    overwritten by next year's adviser/roster changes. This preserves
    history without an auxiliary history table.
  - `PROTECT` on `grade` and `academic_year`: a section cannot be deleted
    by deleting its grade or year while enrollments reference it; deactivation
    uses `is_active=False` rather than row deletion.

### 4.4 StudentEnrollment

- **Owner:** `apps.academics`
- **Purpose:** A student's academic registration for one `AcademicYear` —
  which grade, which section, what status, when enrolled, when (if ever)
  withdrawn.
- **Conceptual fields:**

```python
class StudentEnrollment(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        FUTURE = "future", "Future"
        TRANSFERRED = "transferred", "Transferred"
        WITHDRAWN = "withdrawn", "Withdrawn"
        GRADUATED = "graduated", "Graduated"
        REPEATED = "repeated", "Repeated"
        ARCHIVED = "archived", "Archived"

    student = models.ForeignKey(
        "identity.StudentProfile", on_delete=models.CASCADE,
        related_name="enrollments",
    )
    academic_year = models.ForeignKey(
        AcademicYear, on_delete=models.PROTECT, related_name="enrollments",
    )
    grade = models.ForeignKey(
        Grade, on_delete=models.PROTECT, related_name="enrollments",
    )
    section = models.ForeignKey(
        Section, on_delete=models.SET_NULL, related_name="enrollments",
        null=True, blank=True,
    )
    enrollment_date = models.DateField()
    withdrawal_date = models.DateField(null=True, blank=True)
    status = models.CharField(
        max_length=20, choices=Status.choices,
        default=Status.ACTIVE, db_index=True,
    )
    # provenance
    prior_enrollment = models.ForeignKey(
        "self", on_delete=models.SET_NULL, related_name="subsequent",
        null=True, blank=True,
        help_text="The enrollment this one continues from (transfer/repeat).",
    )
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-academic_year__start_date"]
        unique_together = [["student", "academic_year"]]
        indexes = [
            models.Index(fields=["academic_year", "status"]),
            models.Index(fields=["grade", "section", "status"]),
            models.Index(fields=["student", "status"]),
        ]
```

- **Key rules:**
  - `unique_together = [("student", "academic_year")]` — a student has at
    most **one** enrollment row per academic year. Transfers between
    sections update the `section` FK on the same row (with an audit trail),
    they do not create a second row for the same year.
  - `student` references `StudentProfile` (which shares its PK with
    `Person`). Academics never references the legacy `attendance.Student`.
  - `grade` and `section` are snapshot references for the year. They use
    `PROTECT` / `SET_NULL` so historical enrollments remain valid even if a
    grade/section is later deactivated.
  - **Year consistency invariant:** because `Section` is year-scoped, an
    enrollment's `section.academic_year` must equal the enrollment's
    `academic_year`. This is enforced in `StudentEnrollment.clean()` (and
    may be backed by a DB constraint at implementation time). The `grade`
    FK is also kept on the enrollment directly for fast grade-scoped
    queries and to survive the case where `section` is null.
  - `prior_enrollment` is an optional self-FK that records continuity
    across transfers and repeats, preserving auditability without duplicating
    year rows.
  - `status` is the single source of truth for the enrollment's lifecycle
    state. Booleans like `is_withdrawn` are forbidden.
  - **No finance, lunch, or attendance fields** on `StudentEnrollment`
    (see Section 7). `has_meal`/`has_bus` stay on the transitional
    `StudentProfile` (and later move to the meal/transport domains), not on
    enrollment.

### 4.5 StudentProfile relationship

`StudentProfile` (already implemented in `apps.identity`) holds:

- `person` (PK O2O → `Person`)
- `code` (student operational code)
- transitional mirrors: `grade`, `homeroom`, `has_meal`, `has_bus`
- `legacy_student` (transitional backlink)

Academics treats `StudentProfile` as the **student handle**:

- `StudentEnrollment.student` → `StudentProfile` (FK, CASCADE).
- A `StudentProfile` has many enrollments (one per year).
- "Is this person a student?" → `PersonRole(person, role_type=student)`.
- "Which grade are they in this year?" →
  `StudentEnrollment(student=profile, academic_year=current, status=active).grade`.
- **Transitional mirror maintenance:** while enrollments are introduced,
  `StudentProfile.grade` and `StudentProfile.homeroom` are backfilled from
  the active enrollment by a service method (`sync_profile_from_enrollment`).
  These mirrors are removed only after all read paths use the enrollment.
- Academics must **not** add student identity fields (name, DOB, photo,
  contact) to `Enrollment`. Those live on `Person` and are read via
  `enrollment.student.person`.

### 4.6 StaffProfile relationship (homeroom / adviser)

The only direct academics-to-StaffProfile link is
`Section.homeroom_adviser` → `StaffProfile`.

- **Why on Section, not on Enrollment:** A homeroom adviser is a property
  of the section ("who teaches Grade 1 / Section A"), not of each student.
  Putting it on enrollment would duplicate it across every student in the
  section and complicate adviser changes. Putting it on `Section` means one
  update changes the adviser for all enrolled students in that year's
  section.
- **SET_NULL** on delete: if a staff profile is removed, the section keeps
    its history but loses its adviser link.
- **Year scoping is inherent, not auxiliary:** Because `Section` is
  year-scoped (it belongs to an `AcademicYear`), `homeroom_adviser` is
  automatically year-scoped — each year's section row carries that year's
  adviser. Historical adviser assignments are preserved as past-year section
  rows. A separate `SectionAdviser(section, staff, academic_year)` model is
  **not** needed; it would only be considered if a section needed to record
  multiple concurrent advisers or mid-year adviser changes with full history,
  which is out of scope for the first iteration.

A future `StaffAssignment`/`TeachingAssignment` (out of scope here) would
link `StaffProfile` → `Subject` → `Section`/`ClassGroup` per
`AcademicYear`. That belongs to a later HR/teaching-assignment module, not
to the academics domain's first iteration.

---

## 5. Enrollment Lifecycle

The enrollment lifecycle is modeled as a **state machine**. Each transition
is a service method in `apps.academics.services`. Historical enrollments are
never deleted; closed enrollments keep their `grade`/`section` snapshot so
historical reports remain accurate.

### 5.1 Status values

| Status | Meaning |
|---|---|
| `future` | Enrolled for an upcoming academic year (not yet active). |
| `active` | Currently enrolled in the current academic year. |
| `transferred` | Moved to a different section (or, in multi-tenant, a different school) within the same year. The original row keeps this status and points to the continuing enrollment via `prior_enrollment`. |
| `withdrawn` | Left before the year ended. `withdrawal_date` is set. |
| `graduated` | Completed the final grade. |
| `repeated` | The year was repeated; a new enrollment for the same grade in the next year is the active one. |
| `archived` | Historical, no longer relevant to active operations but preserved for reporting. |

### 5.2 Valid transitions

```
future ──(year starts)──► active
active ──(section change)──► transferred*   (new row: active, prior_enrollment=this)
active ──(leaves)──► withdrawn               (set withdrawal_date)
active ──(completes final grade)──► graduated
active ──(must repeat)──► repeated           (next year's enrollment created)
active / transferred / withdrawn / repeated / graduated ──(time passes)──► archived
```

\* **Transfer between sections:** Because `unique_together = [("student",
"academic_year")]` forbids two rows for the same year, an intra-year section
transfer is handled by **updating the `section` FK on the existing active
enrollment** and recording an audit entry, rather than creating a second
year-row. The `transferred` status is reserved for transfers that cross a
boundary the model considers a new enrollment (e.g., a mid-year transfer to
a different school in multi-tenant mode), where a new row is legitimately
created for the receiving school/year. For a simple section swap within the
same school and year, the active enrollment's `section` is updated and an
audit log records the change (see Section 9). This avoids violating the
unique constraint and keeps history via the audit trail.

### 5.3 Lifecycle service methods (conceptual)

```python
# apps/academics/services.py

@transaction.atomic
def enroll_student(*, student, academic_year, grade, section=None,
                   enrollment_date=None, status="future") -> StudentEnrollment: ...

@transaction.atomic
def activate_future_enrollments(*, academic_year) -> int: ...
    # flips all `future` enrollments for the year to `active` when the year starts

@transaction.atomic
def transfer_section(*, enrollment, new_section, *, changed_by=None) -> StudentEnrollment: ...
    # updates section FK on the active enrollment; writes an audit record;
    # does NOT create a second row for the same year

@transaction.atomic
def withdraw(*, enrollment, withdrawal_date=None, *, changed_by=None) -> StudentEnrollment: ...
    # sets status=withdrawn, withdrawal_date; keeps grade/section snapshot

@transaction.atomic
def graduate(*, enrollment, *, changed_by=None) -> StudentEnrollment: ...

@transaction.atomic
def repeat_year(*, enrollment, next_academic_year, *, changed_by=None) -> StudentEnrollment: ...
    # marks current enrollment `repeated`; creates a new `future`/`active`
    # enrollment for the SAME grade in the next year, prior_enrollment=current

@transaction.atomic
def archive_year(*, academic_year) -> int: ...
    # flips non-active enrollments from a past year to `archived`
```

### 5.4 Historical preservation rules

- **Never delete** an enrollment. Use status transitions and archiving.
- **Never overwrite** `grade` on a historical enrollment. If a student
  transferred sections mid-year, update `section` (with audit) but keep
  `grade`. If grade data was wrong, use a corrective migration with a backup.
- **Never reuse** an enrollment row across academic years. Each year gets
  its own row.
- Reports for past years always read the enrollments that existed for that
  year, not a "current grade" snapshot on `StudentProfile`.
- `StudentProfile.grade`/`homeroom` mirrors reflect **only the current
  active enrollment**; they are convenience fields for the present, not a
  source of historical truth.

---

## 6. How Academics Supports Other Domains

Academics provides the year/grade/section context that other modules scope
against. Each relationship below is **one-way** (other module → academics).

### 6.1 Identity

- `StudentEnrollment.student` → `StudentProfile` → `Person`.
- "Is this person a student this year?" is answered by:
  `StudentEnrollment(student=profile, academic_year=current, status="active")`
  exists.
- `PersonRole(person, student)` still answers the timeless "is this person
  a student?" question; enrollment answers the year-specific one.
- `AcademicYear` does **not** reference `Person` directly; it is structural.

### 6.2 Attendance

- `AttendanceRecord` and `AttendanceEvent` (after FK migration) target
  `Person`. Academics adds **context**, not a FK:
  - "Was this student enrolled on the attendance date?" → check the
    `AcademicYear` whose `[start_date, end_date]` contains the date, then
    check an active enrollment for that year.
  - Section-based attendance reports join `AttendanceRecord.person` →
    `StudentEnrollment(academic_year, section)`.
- `AcademicYear` may optionally become a FK on `PeriodOccurrence`
  (scheduler app) for year-scoped timetables. That is a scheduler decision,
  not an academics-owns-attendance one.

### 6.3 Lunch

- `LunchSubscription` (future `MealSubscription` target) gains an optional
  `academic_year` FK for subscription validity windows.
- "Is this student entitled to lunch today?" combines:
  `LunchSubscription(person, academic_year, active)` **and** an active
  enrollment in that year. Academics does not store `has_meal`; the meal
  domain does (via subscription), and the transitional
  `StudentProfile.has_meal` is the interim mirror.
- Section-level lunch rosters are derived by joining subscriptions to
  enrollments by `academic_year` + `section`.

### 6.4 Wallet / Finance

- `Wallet` targets `Person` (after FK migration), not enrollment.
- `Invoice` (future) targets `Person` and is scoped by `AcademicYear` for
  per-year billing.
- **Academics does not store balance, invoice, or payment fields.** A
  student's fee category may be implied by their grade/section; finance reads
  that context via enrollment, not via a FK on enrollment.
- `AcademicYear` is the billing period anchor for finance reports.

### 6.5 Discounts

- `DiscountAssignment` (future) targets `Person` (and optionally `Family` or
  `Grade`) and is scoped by `AcademicYear`.
- Grade-level discounts reference `Grade`; the discount engine reads
  `enrollment.grade` to decide applicability. Academics does not store
  discount fields.
- **No discount field on `Enrollment`.** Whether a student receives a
  discount is resolved at charge time by the finance/meal pricing pipeline,
  not by denormalizing a discount onto enrollment.

### 6.6 Reports

- All academic, attendance, lunch, and finance reports are scoped by
  `AcademicYear` and optionally `Grade`/`Section`.
- Historical reports read past-year enrollments with their original
  `grade`/`section` snapshots. Because enrollments are append-only and
  never overwritten, a report for 2025-26 always reflects 2025-26 reality.
- Roster reports use `StudentEnrollment(academic_year, section, status=active)`
  joined to `Person` for names and `person.display_code` for rendering.

### 6.7 AI Attendance

- AI (face-recognition) attendance produces `AttendanceEvent`/`AttendanceRecord`
  for a `Person`. Academics supports it by:
  - Resolving "is this person an enrolled student on this date?" before
    writing a record (or flagging unrecognized/enrolled-vs-not).
  - Providing section context for camera-roster correlation ("camera X
    covers Grade 1 / Section A").
- `FaceEmbedding` targets `Person`; it is not scoped by enrollment. A
  staff member's embedding is independent of academic structure. Academics
  only enables staff/teacher face attendance by not tying embeddings to
  `StudentProfile`.

---

## 7. Recommended Model Boundaries

What belongs **in** the academics app:

| Belongs in `apps.academics` | Why |
|---|---|
| `AcademicYear`, `SchoolLevel`, `Grade`, `Section`, `StudentEnrollment` | Core academic structure and per-year registration. |
| `Section.homeroom_adviser` → `StaffProfile` | The only academics-StaffProfile link; a property of the section. |
| Enrollment status state machine and transitions | Lifecycle of a student's year registration. |
| Enrollment validators (overlap, capacity, unique-per-year) | Invariants of the academic structure. |

What does **not** belong in the academics app:

| Does NOT belong in `apps.academics` | Where it belongs instead |
|---|---|
| Student identity fields (name, DOB, photo, contact) | `apps.identity.Person` / `StudentProfile` |
| `has_meal`, meal plan, lunch subscription | `apps.meal` (transitional mirror on `StudentProfile`) |
| `has_bus`, bus route, transport assignment | future `apps.transport` (transitional mirror on `StudentProfile`) |
| Wallet balance, transactions, invoices | `apps.wallet` / `apps.finance` |
| Discount profiles, rules, assignments | `apps.finance` |
| Attendance records, events, face embeddings | `apps.attendance` |
| Period templates, occurrences, timetable | `apps.scheduler` / future `apps.timetable` |
| Subjects, subject offerings, class groups, teaching assignments | future academic-subject module (Phase 3+, separate from this first academics iteration) |
| Marks, exams, report cards | future `apps.assessment` |
| Staff employment, departments, HR assignments | future `apps.hr` / `StaffAssignment` |

### Boundary justification for "no finance/lunch/attendance on Enrollment"

Placing a `wallet_balance`, `has_meal`, or `attendance_status` field on
`StudentEnrollment` would:

- Couple academics to finance/meal/attendance, breaking the one-way
  dependency rule and creating circular imports.
- Force enrollment rows to change whenever a wallet/meal/attendance fact
  changes, breaking the append-only historical contract.
- Duplicate data already owned elsewhere (balance on `Wallet`, subscription
  on `LunchSubscription`, status on `AttendanceRecord`).

The only fields allowed on `Enrollment` are those that define **which
student, which year, which grade, which section, what lifecycle state**.
Everything else is read by joining through `academic_year` / `student.person`.

---

## 8. What Should NOT Be Implemented Yet

| Item | Why deferred |
|---|---|
| The `apps.academics` app itself | This document is design-only. Implementation needs a separate approved task and must follow the implementation order in Section 11. |
| `Subject`, `SubjectOffering`, `ClassGroup`, `TeachingAssignment` | Belong to a later academic-subject module; not required for year/grade/section/enrollment. |
| `Term` (sub-year periods) | Not needed until assessment or term-scoped billing exists. |
| `Exam`, `Mark`, `ReportCard` | Assessment domain, Phase 4+. |
| Multi-adviser / mid-year adviser-change history model (`SectionAdviser`) | With year-scoped `Section`, single-adviser history is already preserved per year. A `SectionAdviser` model is needed only for multiple concurrent advisers or mid-year changes with full history; defer until such a requirement exists. |
| `AcademicYear.school` becoming NOT NULL | Multi-tenant activation; defer until `Organization`/`School`/`Campus` exist. |
| Removing `StudentProfile.grade` / `homeroom` mirrors | Only after enrollments are proven and all read paths use them. |
| Capacity hard-blocking | Start with a warning in `clean()`; hard block only if a policy decision mandates it (Q4). |
| Cross-school transfers | Defer until multi-tenant; intra-school section transfers are supported from day one. |
| Archiving automation | Implement the `archive_year` service, but do not auto-run it until a retention policy is decided (Q9). |
| `AcademicYear.is_active` partial unique constraint | Use a service-level check first; a DB partial unique index is added only if contention appears. |

---

## 9. Migration Safety Notes

When the academics app is eventually implemented, these rules apply. They
are **not** actions for this document.

### General rules

- **One logical feature per migration.** Do not combine `AcademicYear`,
  `Grade`, `Section`, and `StudentEnrollment` into a single migration if
  any can be deployed independently. In practice they are introduced
  together as the app's `0001_initial`, but no unrelated schema changes
  ride along.
- **Additive first.** The academics app creates new tables only. It does
  **not** alter `StudentProfile` or `Person`. The transitional mirrors
  (`StudentProfile.grade`/`homeroom`) remain in place.
- **PROTECT/SET_NULL on academic FKs** so historical enrollments survive
  grade/section deactivation.
- **Run `python manage.py makemigrations --check --dry-run` and
  `python manage.py check`** after each step.
- **Tests run under the CREATEDB-enabled role** (`TEST_DB_USER` /
  `TEST_DB_PASSWORD`), already configured in `bisk/settings.py`. New
  migrations must pass `python manage.py test apps.academics` (and
  `apps.identity`) before merge.
- **Backup the production database** before backfilling enrollments from
  `StudentProfile.grade`/`homeroom` and `attendance.Student`.

### Backfill sequence (when implemented)

1. Create `AcademicYear` rows for past and current years from known school
   calendars (management command, not a data migration, so it is reviewable
   and re-runnable).
2. Create `SchoolLevel` / `Grade` rows from the distinct values present in
   `StudentProfile.grade` and `attendance.Student.grade`.
3. Create year-scoped `Section` rows from distinct `(academic_year, grade,
   homeroom)` triples — one section row per grade/section/year, with that
   year's homeroom adviser and capacity.
4. Create one `StudentEnrollment(student, academic_year=current,
   grade=profile.grade, section=resolved, status=active)` for each active
   student, via a management command that is **idempotent** (respects
  `unique_together`).
5. Run `sync_profile_from_enrollment` for every student to confirm the
   mirrors match the enrollments (a verification step, not a write step).
6. Once verified, flip read paths (selectors, reports, dashboards) from
   `StudentProfile.grade` to `enrollment.grade` one module at a time.
7. Only after all read paths are migrated, schedule removal of
   `StudentProfile.grade`/`homeroom` in a later phase.

### Audit trail for transfers

- Intra-year section transfers update the `section` FK on the active
  enrollment and write to an audit mechanism (a simple
  `EnrollmentHistory`/`AuditLog` model or the project's existing audit
  pattern, decided at implementation time). The enrollment row itself is
  updated, but the audit record preserves the prior `section`, the actor,
  and the timestamp.
- Do **not** implement a generic audit framework now; a per-model history
  table or service-level audit rows suffice. Decision recorded as Q8.

### Rollback

- Because the academics app is purely additive (new tables only), rolling
  back means un-applying its migrations and removing the app from
  `INSTALLED_APPS`. No existing table is altered, so rollback is safe at
  any point before the transitional mirror removal step.

---

## 10. Open Questions / Decisions Needed

1. **App name: `academics` vs `academic`.** The ERP sketch uses singular
   `academic`; Django convention favors plural `academics`. Pick one and
   enforce it in `INSTALLED_APPS` and import paths. (Recommended:
   `academics`.)

2. **`AcademicYear.code` uniqueness scope.** Globally unique now, or
   `unique_together = [("school", "code")]` from day one? Globally unique
   is simpler for single-school; per-school is more multi-tenant-ready but
   requires a nullable `school` FK that doesn't exist yet.

3. **Section rollover between academic years.** `Section` is year-scoped by
   design. The open question is operational: when a new academic year
   begins, how are last year's sections rolled forward? Options: (a) a
   management command clones last year's active sections into the new year
   (adviser/capacity left blank or copied for editing); (b) sections are
   created fresh each year by admins. Recommendation: provide a `roll_over_sections`
   service/command but do not auto-run it; an admin confirms the new year's
   sections before enrollments open.

4. **Section capacity: warn or block?** Should enrolling a student beyond
   `capacity` be a hard validation error or a soft warning? Because
   `capacity` is year-specific on each `Section` row, the check is per-year.
   Affects `StudentEnrollment.clean()` and the supervisor workflow.

5. **Mid-year adviser changes.** With year-scoped `Section`, the adviser for
   a given year is preserved on that year's section row. The remaining
   question is whether a mid-year adviser change should (a) simply update
   the section's `homeroom_adviser` (losing the prior adviser for that year
   unless audited separately), or (b) be recorded in an audit log /
   `SectionAdviser` history. Recommendation: start with (a) plus the
   transfer/audit mechanism used for section transfers (Q8); add a
   dedicated `SectionAdviser` model only if multi-adviser history is
   required.

6. **`Grade` vs `GradeLevel` naming.** Some legacy systems use
   `GradeLevel`. The ERP doc uses `SchoolLevel` + `Grade`. Confirm the
   final names before creating the app to avoid renames.

7. **Enrollment `grade` snapshot vs FK.** Should `enrollment.grade` be a
   FK to `Grade` (recommended, allows reporting) or a denormalized
   CharField snapshot? FK is recommended; a CharField would lose grade
   reorganization history.

8. **Audit mechanism for section transfers.** A dedicated
   `EnrollmentHistory` model, a generic audit-log app, or service-level
   audit rows? Decide before implementing `transfer_section`.

9. **Archiving policy.** When do `active`/`withdrawn`/`graduated`
   enrollments become `archived`? At the end of the academic year?
   Automatically or by a management command? A retention/compliance
   question.

10. **Repeated year identity.** When a student repeats, do we keep the
    `repeated` enrollment and create a new one for the same grade next
    year, or merge? Recommendation: keep both (append-only) with
    `prior_enrollment` linking them; confirm.

11. **Future enrollments and capacity.** Do `future` enrollments count
    against a section's capacity immediately, or only when they become
    `active`? Affects capacity validation logic.

12. **`AcademicYear.is_active` enforcement.** Service-level check (one
    active per school) or DB partial unique index? Service-level is
    simpler and avoids partial-index portability concerns; DB-level is
    safer under contention.

---

## 11. Recommended Implementation Order

Each step is a **separate, approved task**. This document implements
nothing.

| Step | Deliverable | Depends on | Notes |
|---|---|---|---|
| 0 | Identity foundation | — | **DONE.** `Person`, `StudentProfile`, `StaffProfile`, `PersonRole` exist. |
| 1 | Create `apps.academics` app skeleton (`apps.py`, empty `models.py`, `INSTALLED_APPS`) | identity | Additive; no models yet. |
| 2 | `AcademicYear` model + admin + seed management command | step 1 | Foundation for everything else. |
| 3 | `SchoolLevel` + `Grade` models + admin + seed from existing `StudentProfile.grade` values | step 2 | Reusable across years. |
| 4 | `Section` model (year-scoped: `academic_year` + `grade` + `code`, with `homeroom_adviser` → `StaffProfile`) + admin + seed/roll-over command | steps 2–3 | Year-scoped by design; one row per grade/section/year. |
| 5 | `StudentEnrollment` model + status enum + `unique_together` + admin | steps 2–4 | The core entity. |
| 6 | Enrollment services (`enroll_student`, `activate_future_enrollments`, `transfer_section`, `withdraw`, `graduate`, `repeat_year`, `archive_year`) + validators | step 5 | Business logic in services, not admin/views. |
| 7 | `sync_profile_from_enrollment` service + backfill management command | step 6 | Populates enrollments from transitional mirrors; verifies consistency. |
| 8 | Flip read paths: selectors/reports/dashboards read `enrollment.grade`/`section` instead of `StudentProfile.grade`/`homeroom` | step 7 | One module at a time; keep mirrors in sync. |
| 9 | Lunch / wallet / discount modules gain optional `academic_year` FK | step 2 | Can happen in parallel with steps 3–8; only needs `AcademicYear`. |
| 10 | Attendance/AI-attendance date→year resolution uses `AcademicYear` | step 2 | Parallel with steps 3–8. |
| 11 | (Later phase) Remove `StudentProfile.grade`/`homeroom` transitional mirrors | step 8 verified across all read paths | Only after all reads use enrollments. |
| 12 | (Later phase) `Subject`/`SubjectOffering`/`ClassGroup`/`TeachingAssignment` | steps 3–6 | Separate academic-subject module; not part of this first iteration. |

### Ordering rationale

- `AcademicYear` (step 2) is the unblocker for lunch/wallet/discount
  year-scoping (step 9) and attendance date resolution (step 10), so it
  comes first and the rest can proceed in parallel.
- `Grade` and `Section` (steps 3–4) must precede `StudentEnrollment` (step 5)
  because enrollment FKs to both.
- Services (step 6) and the backfill (step 7) come after the model so the
  state machine and provenance (`prior_enrollment`) are correct before any
  data exists.
- Mirror removal (step 11) is deliberately last and conditional on full
  read-path migration, preserving backward compatibility throughout.

---

## Appendix A — Enrollment State Diagram (text)

```
                 ┌─────────┐
        year     │ future  │
       starts   └────┬────┘
            ┌────────┼─────────────┐
            ▼        ▼             ▼
       ┌─────────┐  │             │
       │ active  │──┘             │
       └────┬────┘                │
            │ section change      │
            │ (update FK + audit) │
            │ (same row, same yr) │
            │                     │
   ┌────────┼─────────┬───────────┴───────────┐
   ▼        ▼         ▼                       ▼
withdrawn  graduated  repeated             transferred
(set date) (final g)  (next yr row          (cross-school,
            created)   prior=this)           new row, prior=this)

any non-active ──(time/retention)──► archived
```

---

## Appendix B — Relationship Summary

```
AcademicYear
   │
   ├──► Section ◄──── Grade
   │      │
   │      └── homeroom_adviser ──► StaffProfile (identity)
   │
   ├──► StudentEnrollment ◄─── StudentProfile (identity)
   │         │                       │
   │         ├── grade ──► Grade      │  (shares PK with Person)
   │         └── section ──► Section │  (section.academic_year == enrollment.academic_year)
   │
   ├── (referenced by) MealSubscription.academic_year   (meal domain)
   ├── (referenced by) DiscountAssignment.academic_year (finance domain)
   ├── (referenced by) Invoice.academic_year            (finance domain)
   └── (referenced by) PeriodOccurrence.academic_year   (scheduler domain, optional)

SchoolLevel ──► Grade ──► Section
```

All arrows point from the depending module to the owning module. Academics
owns the left column; identity owns `StudentProfile`/`StaffProfile`/`Person`;
the other domains own their own models and merely reference `AcademicYear`/
`Grade`/`Section`.

---

## Appendix C — Document Hierarchy

```
docs/development/PROJECT_ARCHITECTURE.md            (standards)
docs/development/AI_DEVELOPMENT_GUIDE.md            (AI workflow)
docs/architecture/person_identity_architecture.md   (identity foundation — implemented)
docs/architecture/erp_foundation_architecture.md    (top-level ERP blueprint)
docs/architecture/education_domain_architecture.md  (education domain — broad)
docs/architecture/academics_domain_architecture.md  (THIS document — academics focus)
```

This document is consistent with all of the above. Where they describe
implemented models, this document references them; where they describe
future models, this document narrows the academics subset and adds
lifecycle, boundary, safety, and open-question detail specific to
`AcademicYear`, `Grade`, `Section`, and `Enrollment`.

---

End of document.
