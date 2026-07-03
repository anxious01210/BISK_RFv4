# Education Domain Architecture — BISK_RFv4

Date: 2026-07-03
Branch: feature/person-architecture
Version: 1.0 Draft — pending review
Status: Architecture design only. No code or migrations are produced by this document.

---

## 1. Purpose

This document designs the **Education Domain Foundation** for BISK_RFv4.

It defines how the school's academic and operational entities relate to the
already-implemented identity foundation (`Person`, `StudentProfile`,
`StaffProfile`, `RoleType`, `PersonRole`) and prepares the ground for the
next subsystems:

- Guardian / parent relationships
- AcademicYear
- Grade
- Section
- Enrollment
- Wallet
- Discount
- LunchSubscription
- LunchAttendance
- AI Attendance (face-recognition attendance)
- Lunch Supervisor Dashboard

This is a **design document**, not an implementation order. No models, code,
admin, or migrations are introduced by this task. Each entity below is a
blueprint for a future, separately-approved implementation step.

### Scope boundaries

| In scope | Out of scope |
|---|---|
| Relationships between identity, academic, finance, meal, and attendance entities | Implementing the models now |
| Naming, ownership, and ownership rules | Multi-tenant activation (Organization/School/Campus) |
| Recommended implementation order | External integrations (SAP, SIS, payment gateways) |
| Migration safety notes | Portal / API / mobile app design |
| What should NOT be implemented yet | Assessment, timetable, transport, behaviour modules |

---

## 2. Design Principles

These principles are inherited from
`docs/development/PROJECT_ARCHITECTURE.md`,
`docs/development/AI_DEVELOPMENT_GUIDE.md`, and
`docs/architecture/person_identity_architecture.md`.

| # | Principle | Application here |
|---|---|---|
| 1 | **Identity before role** | Every entity anchors to `Person`, never to a legacy `Student` row. Profiles extend Person; they do not replace it. |
| 2 | **Person ≠ User** | Login identity stays on `auth.User`. Education entities reference `Person` (business identity), not `auth.User`. |
| 3 | **Roles are M2M, not a field** | Guardians, lunch supervisors, and students are expressed via `RoleType` + `PersonRole`, not via `person_type` flags. |
| 4 | **Profiles extend roles** | `StudentProfile` / `StaffProfile` hold role-specific data. New role types get their own profile, not new columns on `Person`. |
| 5 | **AcademicYear is foundational** | Enrollment, subscription, assignment, and discount validity are scoped by `AcademicYear` once it exists. Until then, date fields remain the interim source of truth. |
| 6 | **Backward compatibility always** | Existing Student-based workflows keep working during migration. New entities are additive; legacy FKs are removed only after verification. |
| 7 | **Incremental delivery** | Each entity below is independently deployable. No big-bang education migration. |
| 8 | **Service-layer business logic** | Business rules live in `services.py` / `selectors.py`, not in views, templates, or admin actions. |
| 9 | **Money is auditable** | Wallet transactions and discount applications create permanent, immutable audit records. |
| 10 | **Workflow as state machine** | Enrollment, subscription, and lunch attendance use explicit `status` enums with documented valid transitions, not scattered booleans. |
| 11 | **Domain ownership first** | Identity owns Person. Academic owns AcademicYear/Grade/Section/Enrollment. Wallet owns Wallet. Meal owns LunchSubscription/LunchAttendance. Attendance owns AI Attendance. Cross-app references only flow one way. |
| 12 | **display_code in templates** | Templates render `person.display_code`, never the legacy `h_code`. `Person.code` is the global ERP identifier; `StudentProfile.code` / `StaffProfile.code` are the operational codes. |
| 13 | **Do not recreate the old Student pattern** | No new entity may own a private `h_code` or duplicate identity fields. Identity lives on Person; everything else references it. |

---

## 3. Proposed Apps / Modules

The long-term app layout (from `erp_foundation_architecture.md`) is:

```
apps/
├── identity/        # Person, RoleType, PersonRole, StudentProfile, StaffProfile (DONE)
├── academic/        # AcademicYear, Grade, Section, Term, Subject (future)
├── enrollment/      # StudentEnrollment, StaffAssignment (future)
├── guardian/        # PersonRelationship, ParentProfile, Family (future)
├── wallet/          # Wallet, WalletTransaction (current, in attendance)
├── finance/         # Discount, DiscountRule, DiscountAssignment (future)
├── meal/            # LunchSubscription, LunchAttendance, MealProfile (future split)
├── attendance/      # AttendanceRecord, AttendanceEvent, FaceEmbedding (current)
├── scheduler/       # PeriodTemplate, PeriodOccurrence (current)
└── lunch_supervisor/# Lunch supervisor dashboard views/services (future)
```

### Ownership rules (no circular ownership)

```
identity          ← owned by no one
academic          ← depends on identity
enrollment        ← depends on identity + academic
guardian          ← depends on identity
wallet            ← depends on identity
finance/discount  ← depends on identity + academic (for year scoping)
meal              ← depends on identity + wallet (for charges)
attendance        ← depends on identity (+ scheduler)
lunch_supervisor  ← depends on meal + attendance (dashboard only, owns no models)
```

A dependency from `meal` → `wallet` is allowed (meal charges the wallet).
A dependency from `wallet` → `meal` is **not** allowed.

### Current reality

Today, `Wallet`, `MealSubscription`, `MealRecord`, `AttendanceRecord`,
`AttendanceEvent`, and `FaceEmbedding` all live inside `apps/attendance`.
Extraction into separate apps is a **future refactoring** step. The rule is:
write new code in a way that anticipates extraction, not that extraction
happens now.

---

## 4. Core Entities

Each entity below describes purpose, owner app, key fields (conceptual), and
the relationships it forms with identity.

### 4.1 Person (identity) — ALREADY IMPLEMENTED

- **Owner:** `apps.identity`
- **Key fields:** `code` (global ERP identifier), `first_name`, `middle_name`,
  `last_name`, `gender`, `date_of_birth`, `email`, `phone`, `address`,
  `photo`, `is_active`, `user` (optional O2O to `auth.User`).
- **Property:** `display_code` resolves to a profile code when a
  role-specific profile exists, otherwise returns `Person.code`.
- **Status:** Implemented, migrated, and tested.

### 4.2 StudentProfile (identity) — ALREADY IMPLEMENTED

- **Owner:** `apps.identity`
- **Key fields:** `person` (PK O2O), `code` (student operational code),
  `grade` (transitional), `homeroom` (transitional), `has_meal`
  (transitional), `has_bus` (transitional), `legacy_student` (transitional
  backlink to `attendance.Student`).
- **Transitional fields:** `grade`, `homeroom`, `has_meal`, and `has_bus`
  are Phase 1 compatibility mirrors. Their long-term owners are
  `StudentEnrollment`, the meal domain, and the transport domain
  respectively. New features must not treat them as the source of truth.

### 4.3 StaffProfile (identity) — ALREADY IMPLEMENTED

- **Owner:** `apps.identity`
- **Key fields:** `person` (PK O2O), `code` (staff operational code),
  `job_title`, `hire_date`, `is_teacher`.

### 4.4 Guardian / Parent relationships

- **Owner:** `apps.guardian` (future; until then, no model)
- **Purpose:** Represent parent/guardian links between two Person records,
  independent of role, so a Person who is a parent can also be a student or
  staff member.
- **Conceptual entities:**
  - `PersonRelationship` — links `from_person` (guardian) to `to_person`
    (dependent), with `relationship_type` (`parent`, `guardian`,
    `emergency_contact`, `sibling`, `other`), `is_primary`, `priority`.
  - `ParentProfile` — optional role-specific extension (`OneToOneField` to
    `Person`), for parent-only data (e.g., `emergency_contact_priority`).
  - `Family` — groups Person records into a household for billing and
    communication; `primary_contact` FK to `Person`.
- **Relationships:**
  - `PersonRelationship.from_person` → `Person` (guardian side)
  - `PersonRelationship.to_person` → `Person` (dependent side)
  - Relationships are Person-to-Person, never Person-to-StudentProfile.
- **Role expression:** "Is this person a guardian?" is answered by
  `PersonRole(person, role_type=parent|guardian)`, not by a boolean field.

### 4.5 AcademicYear

- **Owner:** `apps.academic` (future)
- **Purpose:** The temporal scope for enrollment, subscription, assignment,
  and discount validity.
- **Conceptual fields:** `name`, `code` (unique), `start_date`, `end_date`,
  `is_active` (only one active year per school), `school` (nullable, future
  multi-tenant).
- **Why foundational:** Without AcademicYear, enrollment and discounts
  cannot express "this applies for year X".
- **Current workaround:** `MealSubscription` and `PeriodOccurrence` use raw
  date fields. This is acceptable until AcademicYear is introduced.

### 4.6 Grade

- **Owner:** `apps.academic` (future)
- **Purpose:** A year-group level in the school (e.g., "Grade 1", "Grade 10").
- **Conceptual fields:** `name`, `code`, `level` (FK to `SchoolLevel`),
  `order` (sorting).
- **Relationships:**
  - `Grade` → `SchoolLevel` (educational stage: Primary / Secondary)
  - `StudentEnrollment.grade` → `Grade` (which grade a student is in for a
    given year)
- **Rule:** Grade is **never** a field on `Person` or `StudentProfile`. It
  lives on `StudentEnrollment`. The current `StudentProfile.grade` is a
  transitional mirror only.

### 4.7 Section

- **Owner:** `apps.academic` (future)
- **Purpose:** A class division within a grade (e.g., "Section A", "Section B").
- **Conceptual fields:** `name`, `code`, `grade` (FK), `capacity` (optional),
  `is_active`.
- **Relationships:**
  - `Section` → `Grade`
  - `StudentEnrollment.section` → `Section`
- **Rule:** Section is **never** a field on `Person`. It lives on
  `StudentEnrollment`. The current `StudentProfile.homeroom` is a
  transitional mirror only.

### 4.8 Enrollment

- **Owner:** `apps.enrollment` (future)
- **Purpose:** Track a student's academic registration per AcademicYear —
  which grade, which section, what status, when enrolled, when withdrawn.
- **Conceptual entity:** `StudentEnrollment`
  - `student` → `StudentProfile` (FK)
  - `academic_year` → `AcademicYear` (FK)
  - `grade` → `Grade` (FK)
  - `section` → `Section` (FK, nullable)
  - `enrollment_date`, `withdrawal_date` (nullable)
  - `status` enum: `enrolled`, `withdrawn`, `graduated`, `transferred`,
    `suspended`
  - `unique_together = [("student", "academic_year")]`
- **Replaces:** The current `Student.is_active` boolean and the
  transitional `StudentProfile.grade` / `homeroom` mirrors.
- **Rule:** "Is this person a student?" is answered by
  `PersonRole(person, student)`. "Which grade are they in this year?" is
  answered by `StudentEnrollment`.

### 4.9 Wallet

- **Owner:** `apps.wallet` (currently inside `apps.attendance`; split later)
- **Purpose:** Prepaid balance and transaction ledger for a Person.
- **Current state:** `Wallet.student` → `attendance.Student`.
- **Target state:** `Wallet.person` → `Person` (O2O), with
  `WalletTransaction.person` → `Person` (FK).
- **Relationships:**
  - `Wallet.person` → `Person` (O2O)
  - `WalletTransaction.wallet` → `Wallet` (FK)
  - `WalletTransaction.person` → `Person` (denormalized for fast queries;
    derived from `wallet.person`)
- **Migration:** Dual-FK pattern — add nullable `person`, backfill from
  `student.migrated_to.person`, then enforce NOT NULL, then drop `student`.
- **Audit rule:** Every charge, top-up, refund, and adjustment creates an
  immutable `WalletTransaction` row. No silent balance changes.

### 4.10 Discount

- **Owner:** `apps.finance` (future)
- **Purpose:** Apply pricing reductions to meal/lunch charges (and later to
  invoices) based on rules and validity windows.
- **Conceptual entities:**
  - `DiscountProfile` — a named discount (`name`, `code`, `type`:
    `percentage` / `fixed`, `value`, `priority`, `is_active`, `start_date`,
    `end_date`, optional `academic_year` FK).
  - `DiscountRule` — a condition (`field`, `operator`, `value`) evaluated
    against the charge context.
  - `DiscountAssignment` — links a `DiscountProfile` to a `Person` (or
    `Family`, or `Grade`) for a given `AcademicYear`.
- **Pricing pipeline:**
  ```
  base meal price
  → applicable subscription profile
  → applicable discounts/rules (ordered by priority)
  → final price
  → wallet transaction
  → audit snapshot
  ```
- **Relationships:**
  - `DiscountAssignment.person` → `Person`
  - `DiscountAssignment.academic_year` → `AcademicYear` (nullable until
    AcademicYear exists)
  - `DiscountAssignment.discount_profile` → `DiscountProfile`
- **Rule:** Discounts apply to a **Person**, not to a `StudentProfile` or a
  legacy `Student`. A discount for a younger sibling is assigned to that
  child's Person, not to the family head, unless explicitly modelled as a
  family-level discount.
- **Implementation order:** Discounts are explicitly deferred until the
  Person architecture is stable and AcademicYear exists (see Section 7).

### 4.11 LunchSubscription

- **Owner:** `apps.meal` (currently inside `apps.attendance`; split later)
- **Purpose:** A subscription that entitles a Person to lunch on specific
  days / for a period.
- **Current state:** `MealSubscription.student` → `attendance.Student`.
- **Target state:** `LunchSubscription.person` → `Person` (FK).
- **Conceptual fields:** `person`, `profile` (FK to a `MealProfile`:
  name/price/description), `start_date`, `end_date`, `status` enum
  (`active`, `paused`, `cancelled`, `expired`), optional `academic_year` FK
  (future).
- **Overlap validation:** `LunchSubscription.clean()` must check overlap
  using `person_id`, not the legacy `student_id`.
- **Rule:** A subscription belongs to a Person. Whether that Person is
  currently a student is answered by `PersonRole`, not by the subscription.

### 4.12 LunchAttendance

- **Owner:** `apps.meal`
- **Purpose:** Record that a Person actually took lunch on a given day.
- **Current state:** `MealRecord` links through `AttendanceRecord.student`,
  not directly to a Student.
- **Target state:** `LunchAttendance` references `Person` directly and the
  `AttendanceRecord`/`AttendanceEvent` that triggered it.
- **Conceptual fields:** `person`, `date`, `subscription` (FK, nullable),
  `attendance_record` (FK, nullable), `status` enum (`served`, `skipped`,
  `absent`, `manual`), `confirmed_by` (FK to `Person`, the supervisor),
  `confirmed_at`.
- **Relationships:**
  - `LunchAttendance.person` → `Person`
  - `LunchAttendance.subscription` → `LunchSubscription`
  - `LunchAttendance.confirmed_by` → `Person` (a supervisor)
- **Audit rule:** Each row records who confirmed the meal and when. Manual
  overrides are auditable.

### 4.13 AI Attendance (face-recognition attendance)

- **Owner:** `apps.attendance`
- **Purpose:** Camera-based recognition produces `AttendanceEvent` and
  `AttendanceRecord` rows for a Person.
- **Current state:** FKs target `attendance.Student`.
- **Target state:**
  - `AttendanceRecord.person` → `Person`
  - `AttendanceEvent.person` → `Person`
  - `FaceEmbedding.person` → `Person`
- **Enables:** Staff and teacher face attendance (currently embeddings are
  student-only).
- **Relationships:**
  - `FaceEmbedding.person` → `Person` (unique active embedding per person;
    the existing `uniq_active_embedding_per_student` constraint must be
  recreated for `person_id`).
  - `AttendanceEvent.person` → `Person`
  - `AttendanceRecord.person` → `Person`
- **Migration order:** See Section 9.

### 4.14 Lunch Supervisor Dashboard

- **Owner:** `apps.lunch_supervisor` (future; dashboard views + services
  only, owns no models)
- **Purpose:** A real-time dashboard for lunch supervisors to see who is
  expected, who was recognized, who has been served, who has insufficient
  wallet balance, and to confirm or override lunch attendance.
- **Composition:** It is a **view layer** composed of selectors and
  services from `meal`, `wallet`, `attendance`, and `identity`. It owns no
  models.
- **Inputs:**
  - `LunchSubscription` (who is expected today)
  - `AttendanceRecord` / `AttendanceEvent` (who was recognized today)
  - `LunchAttendance` (who has been served)
  - `Wallet` (who has balance)
  - `PersonRole(staff)` (who is the supervisor)
- **Outputs:**
  - Today's expected vs. served list
  - Wallet balance warnings
  - Manual confirm / override actions (audited via `LunchAttendance`)
- **Auth rule:** Access is granted to `PersonRole(person, staff)` (and a
  future `meal_supervisor` role metadata), combined with a Django Group for
  system access.

---

## 5. Entity Relationships

### 5.1 Relationship diagram (text)

```
                         auth.User (login identity)
                              │ optional O2O
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                            Person                                 │
│  code  = global ERP identifier                                   │
│  display_code → StudentProfile.code / StaffProfile.code / code    │
│  roles ──► RoleType  (via PersonRole)                            │
└───────────────┬───────────────────────────────────────────────────┘
                │ O2O            │ O2O
   ┌────────────┴───────┐   ┌────┴────────────┐
   │ StudentProfile    │   │ StaffProfile    │
   │ code (student op) │   │ code (staff op) │
   └────────┬───────────┘   └─────────────────┘
            │ FK (enrollment)
            ▼
   ┌────────────────────┐         ┌──────────────┐
   │ StudentEnrollment  │ ───────► │ AcademicYear │
   │ grade ──► Grade     │         └──────┬───────┘
   │ section ──► Section │                │ FK
   └────────────────────┘                used by:
                                          Discount, LunchSubscription,
                                          StaffAssignment

Person (guardian) ──PersonRelationship──► Person (dependent)
Family ──► Person (members)

Wallet ──O2O──► Person
WalletTransaction ──FK──► Wallet  (and person snapshot)

DiscountProfile ──DiscountAssignment──► Person (+ AcademicYear)
DiscountProfile ──DiscountRule (conditions)

LunchSubscription ──FK──► Person
LunchAttendance   ──FK──► Person
LunchAttendance   ──FK──► LunchSubscription
LunchAttendance   ──FK──► AttendanceRecord (optional)

AttendanceRecord ──FK──► Person
AttendanceEvent  ──FK──► Person
FaceEmbedding    ──FK──► Person

LunchSupervisorDashboard (no models) reads:
  identity + meal + wallet + attendance
```

### 5.2 Relationship matrix

| Entity ↓ references → | Person | StudentProfile | StaffProfile | AcademicYear | Grade | Section | Wallet | DiscountProfile | LunchSubscription | AttendanceRecord |
|---|---|---|---|---|---|---|---|---|---|---|
| StudentEnrollment | (via profile) | FK | — | FK | FK | FK | — | — | — | — |
| Wallet | O2O | — | — | — | — | — | — | — | — | — |
| WalletTransaction | FK | — | — | — | — | — | FK | — | — | — |
| DiscountAssignment | FK | — | — | FK | — | — | — | FK | — | — |
| LunchSubscription | FK | — | — | FK (future) | — | — | — | — | — | — |
| LunchAttendance | FK | — | — | — | — | — | — | — | FK | FK (opt) |
| AttendanceRecord | FK | — | — | — | — | — | — | — | — | — |
| AttendanceEvent | FK | — | — | — | — | — | — | — | — | — |
| FaceEmbedding | FK | — | — | — | — | — | — | — | — | — |
| PersonRelationship | FK×2 | — | — | — | — | — | — | — | — | — |

> "(via profile)" means StudentEnrollment references StudentProfile, which
> shares its PK with Person.

### 5.3 Identity-keyed lookups

Because templates must use `person.display_code`, all dashboard and report
lookups resolve a Person via:

1. `StudentProfile.code` → `StudentProfile.person`
2. `StaffProfile.code` → `StaffProfile.person`
3. `Person.code` (global ERP identifier) → `Person`

This resolution is provided by `apps.identity.selectors.get_person_by_display_code`,
already implemented. New modules must use this selector rather than
re-implementing `h_code` lookups.

---

## 6. Recommended Implementation Order

This order respects dependencies and keeps each step independently
deployable. Each step is a **separate, approved task**; nothing here is
implemented by this document.

| Step | Entity / module | Depends on | Why this order |
|---|---|---|---|
| 0 | Identity foundation (Person, StudentProfile, StaffProfile, RoleType, PersonRole) | — | **DONE.** Everything else depends on it. |
| 1 | Wallet.person (FK migration) + WalletTransaction.person | Person | Simplest, least dependent dual-FK migration. |
| 2 | LunchSubscription.person (FK migration) | Person | Depends on Person; enables meal domain split later. |
| 3 | FaceEmbedding.person + unique constraint recreation | Person | Enables staff face attendance. |
| 4 | AttendanceRecord.person + AttendanceEvent.person | Person, FaceEmbedding | Largest tables; migrate in batches. |
| 5 | AcademicYear | identity | Foundation for enrollment and discount scoping. No FK dependency beyond identity. |
| 6 | Grade + Section + SchoolLevel | AcademicYear | Academic structure; depends on year concept. |
| 7 | StudentEnrollment | StudentProfile, AcademicYear, Grade, Section | Replaces `StudentProfile.grade`/`homeroom` mirrors. |
| 8 | PersonRelationship + ParentProfile + Family | Person | Guardian links; no academic dependency. Can run in parallel with steps 5–7. |
| 9 | DiscountProfile + DiscountRule + DiscountAssignment | Person, AcademicYear | Needs year scoping for validity windows. |
| 10 | LunchAttendance (formalized) | Person, LunchSubscription, AttendanceRecord | Splits from current `MealRecord`. |
| 11 | Lunch Supervisor Dashboard | meal + wallet + attendance + identity | Pure view layer; no new models. |
| 12 | App extraction (wallet, meal, academic, enrollment, guardian, finance out of attendance) | All above | Mechanical refactor once FKs target Person. |

### Notes on ordering

- Steps 1–4 are the **Phase 1.5** FK migration sequence already approved in
  `person_identity_architecture.md`. They are listed here for completeness.
- Step 8 (guardian) is independent of academic structure and can be done
  earlier if a parent-portal need arises.
- Step 12 (app extraction) is explicitly **not** required for the education
  domain to function; it is a maintainability refactoring.

---

## 7. What Should NOT Be Implemented Yet

| Item | Why deferred |
|---|---|
| `AcademicYear` model | Not needed until enrollment and discount validity require year scoping. Current date fields suffice. |
| `Grade`, `Section`, `SchoolLevel` models | Require AcademicYear. Current `StudentProfile.grade`/`homeroom` mirrors are acceptable. |
| `StudentEnrollment` | Requires AcademicYear + Grade + Section. `StudentProfile.grade`/`homeroom` + `PersonRole` answer current needs. |
| `PersonRelationship`, `ParentProfile`, `Family` | No parent portal yet. Add when portal or family-level billing is built. |
| `DiscountProfile`, `DiscountRule`, `DiscountAssignment` | The user's standing instruction is that discounts come **after** a robust Person architecture. AcademicYear must exist first for validity windows. |
| Formal `LunchAttendance` model split | Current `MealRecord` works during FK migration. Split when the supervisor dashboard needs a first-class attendance entity. |
| Lunch Supervisor Dashboard | Depends on meal/wallet/attendance all targeting Person and on a formal `LunchAttendance` (or at least confirmed `MealRecord`) entity. |
| `AUTH_USER_MODEL = "identity.Person"` | Irreversible on existing database. Deferred to a dedicated Phase 2. |
| `Organization` / `School` / `Campus` | Multi-tenant not needed yet. |
| Removing the legacy `attendance.Student` table | Premature; keep through Phase 2. |
| Adding `Person.school` or `Person.ext_id` | Multi-tenant fields; defer until Organization/School exist. |
| Generic workflow engine | Per-domain `status` enums suffice; do not introduce Camunda/SimpleWorkflow. |
| New `h_code` fields on any new entity | Legacy pattern. Use `Person.code` / `person.display_code`. |

---

## 8. Open Questions / Decisions Needed

These are unresolved questions that must be answered before the
corresponding entity is implemented. They are recorded here so future work
does not silently make an incompatible choice.

1. **AcademicYear uniqueness scope.** Should `AcademicYear.code` be globally
   unique, or `unique_together = [("school", "code")]` from day one? Affects
   multi-tenant readiness.

2. **Grade vs. GradeLevel naming.** `erp_foundation_architecture.md` uses
   `SchoolLevel` + `Grade`; some legacy systems use `GradeLevel`. Decide the
   final name before the academic app is created.

3. **Section capacity enforcement.** Should `Section.capacity` be enforced
   at enrollment time (validate in `StudentEnrollment.clean()`) or only
   reported as a warning?

4. **Guardian relationship cardinality.** Can a `PersonRelationship` have
   more than one `relationship_type` between the same two Persons (e.g.,
   both `parent` and `emergency_contact`), or is it one row per ordered
   pair with a single type?

5. **Family vs. PersonRelationship.** Is `Family` a required grouping for
   billing, or an optional convenience? If required, who is the
   `primary_contact` when parents are divorced/separated?

6. **Discount assignment target.** Should `DiscountAssignment` target
   `Person`, `Family`, `Grade`, or all three? The user's instruction is
   that discounts apply to Person; family-level discounts need an explicit
   decision.

7. **Discount precedence.** When multiple discounts apply to the same
   charge, do they stack (additive), or does only the highest-priority one
   apply (exclusive)?

8. **LunchAttendance vs. MealRecord.** Should the existing `MealRecord` be
   renamed/extended to `LunchAttendance`, or should a new model be created
   and `MealRecord` deprecated? Affects migration safety.

9. **AI Attendance for staff.** Should `FaceEmbedding` support multiple
   active embeddings per Person (multi-angle) or keep the single-active
   constraint? The current `uniq_active_embedding_per_student` constraint
   must be recreated for `person_id` either way.

10. **Lunch Supervisor Dashboard auth.** Should supervisor access be a
    `RoleType` (`meal_supervisor`) or a Django Group? The architecture
    prefers `PersonRole` for business roles and Django Groups for system
    access; a hybrid is likely.

11. **Wallet overdraft policy.** Should a lunch charge be blocked when the
    wallet balance is insufficient, or allowed to go negative with a
    flagged `WalletTransaction`? Affects the supervisor dashboard's
    "allow override" flow.

12. **display_code fallback.** If a Person has neither `StudentProfile` nor
    `StaffProfile`, `display_code` returns `Person.code`. Confirm this is
    acceptable for guardians/guests, or whether a `GuardianProfile.code`
    should be introduced later.

---

## 9. Migration Safety Notes

These notes apply when the entities above are eventually implemented. They
are not actions for this document.

### General rules

- **Never drop a legacy FK in the same migration that adds the new FK.**
  Always: add nullable → backfill → enforce NOT NULL → (later release) drop
  legacy.
- **Every data migration must be reversible** or at least clearly
  documented as irreversible with a backup step.
- **One logical feature per migration.** Do not combine unrelated schema
  changes (per `DEVELOPMENT_STANDARDS.md`).
- **Run `python manage.py makemigrations --check --dry-run` before and
  after** any model change to detect unintended schema drift.
- **Run `python manage.py check`** after every migration step.
- **Tests run under a CREATEDB-enabled role** (`TEST_DB_USER` /
  `TEST_DB_PASSWORD`), already configured in `bisk/settings.py`. New
  migrations must pass `python manage.py test apps.identity` (and the
  relevant app's tests) before merge.
- **Backup the production database** before any FK migration on
  `Wallet`, `MealSubscription`, `AttendanceRecord`, or `FaceEmbedding`.

### FK migration order (Phase 1.5, restated for safety)

1. `Wallet` + `WalletTransaction` — smallest, least dependent.
2. `LunchSubscription` (currently `MealSubscription`).
3. `FaceEmbedding` — recreate the unique active constraint for `person_id`.
4. `AttendanceRecord` — largest table; migrate in batches.
5. `AttendanceEvent`.

### Backward-compat shims

- During transition, models expose a `student` property that returns the
  linked `StudentProfile` (or legacy `Student`) for template compatibility.
- Templates are migrated incrementally to `person.display_code`; legacy
  `h_code` references are removed app-by-app, not in a single big-bang.
- Admin: legacy `StudentAdmin` remains registered and read-only until
  Phase 2; `PersonAdmin` / `StudentProfileAdmin` / `StaffProfileAdmin` are
  already registered (see `apps/identity/admin.py`).

### Transitional fields cleanup

- `StudentProfile.grade`, `homeroom`, `has_meal`, `has_bus` are removed
  **only after** `StudentEnrollment` (for grade/homeroom) and the meal
  domain (for `has_meal`) and the transport domain (for `has_bus`) are
  proven in production. Removing them early breaks existing workflows.

---

## 10. How This Supports Future ERP Modules

This education domain foundation is designed so future ERP modules can be
added with **minimal changes to existing modules** (per
`PROJECT_ARCHITECTURE.md`: "Architecture should favor extension over
modification").

| Future module | How this foundation supports it |
|---|---|
| **Invoices / Receipts / Installments** | `Invoice.person` → `Person` reuses the identity anchor. `AcademicYear` scoping enables per-year billing. `Wallet` is the first finance-adjacent subsystem and proves the Person-FK pattern. |
| **Assessment / Marks / Report Cards** | `StudentEnrollment` tells the assessment module which grade/section a student is in for a given year. `Person` is the target of marks. No new identity model needed. |
| **Timetable** | `Section` + `Grade` + `StaffAssignment`/`TeachingAssignment` provide the class/teacher context. `PeriodTemplate`/`PeriodOccurrence` (scheduler app) already exist and gain `AcademicYear` scoping. |
| **Transport** | `PersonRelationship` + `Family` support pickup/dropoff authorization. A future `BusAssignment` targets `Person`, mirroring the enrollment pattern. `StudentProfile.has_bus` is the transitional bridge. |
| **Behaviour / Discipline** | Incident records target `Person` (offender and reporter), not `Student`. `PersonRole` distinguishes student vs. staff incidents. |
| **Health / Clinic** | Medical records target `Person`. `PersonRelationship` provides emergency-contact lookup. |
| **Portals (Student / Staff / Parent)** | Auth stays on `auth.User`; portals resolve `request.user.person` and use `PersonRole` to scope data. `display_code` is the single rendering key. |
| **Mobile API (JWT)** | JWT authenticates `auth.User`; the API exposes `Person`-keyed data. No new identity model. |
| **Multi-School / Multi-Tenant** | `AcademicYear.school`, `Grade.school`, `Section.school`, `PersonRole.school` become nullable FKs activated together. `Person.code` uniqueness moves to `unique_together = [("school", "code")]` only when multi-tenant is turned on. |
| **Notifications / Reporting** | Notifications target `Person` (or `Family` / `Section` / `Grade`). Reports are scoped by `AcademicYear` + `Grade` + `Section`. The identity anchor and academic structure are sufficient; no new identity concepts are required. |
| **External Integration (SIS / SAP)** | `Person.code` is the global ERP identifier for cross-system matching. A future `Person.ext_id` stores the external system's key. The education domain does not need to know which external system is involved. |

### Extensibility contract

A new ERP module can be added by:

1. Creating its own `apps/<domain>/` with `models.py`, `services.py`,
   `selectors.py`, `validators.py`, `admin.py` (per the identity app's
   now-established structure).
2. Referencing `Person` (via the appropriate profile where role-specific,
   via `Person` directly where role-agnostic).
3. Scoping by `AcademicYear` where the data is year-specific.
4. Using `person.display_code` in all templates and reports.
5. Reusing `apps.identity.selectors.get_person_by_display_code` for
   code-based lookups.
6. Not introducing a new `h_code` field or duplicating identity fields.

This contract keeps the architecture clean and ensures that adding a new
module does not require modifying `apps.identity`, `apps.academic`, or any
other existing module's models.

---

## Appendix A — Glossary

| Term | Definition |
|---|---|
| **Person** | The core business identity of a human in the school ecosystem. |
| **Person.code** | The global ERP identifier for a Person. |
| **StudentProfile.code** | The student operational code. |
| **StaffProfile.code** | The staff operational code. |
| **display_code** | The preferred code for rendering a Person in templates; resolves to a profile code when one exists, else `Person.code`. |
| **h_code** | Legacy identifier on the old `attendance.Student` model. Not used in new code. |
| **PersonRole** | M2M-through linking a Person to a RoleType, with optional validity window. |
| **StudentEnrollment** | A student's academic registration for one AcademicYear. |
| **AcademicYear** | The temporal scope for year-specific entities. |
| **Wallet** | Prepaid balance + transaction ledger for a Person. |
| **Discount** | A pricing reduction applied to a charge, scoped by Person and (future) AcademicYear. |
| **LunchSubscription** | Entitlement to lunch for a Person over a period. |
| **LunchAttendance** | Record that a Person actually took lunch on a day. |
| **AI Attendance** | Face-recognition-driven AttendanceEvent/AttendanceRecord for a Person. |
| **Lunch Supervisor Dashboard** | Real-time view layer for meal supervisors; owns no models. |

---

## Appendix B — Document Hierarchy

```
docs/development/PROJECT_ARCHITECTURE.md        (standards)
docs/development/AI_DEVELOPMENT_GUIDE.md        (AI workflow)
docs/architecture/person_identity_architecture.md  (identity foundation — implemented)
docs/architecture/erp_foundation_architecture.md   (top-level ERP blueprint)
docs/architecture/education_domain_architecture.md (THIS document — education domain design)
```

This document is consistent with `person_identity_architecture.md` and
`erp_foundation_architecture.md`. Where they describe implemented models,
this document references them; where they describe future models, this
document narrows the education-domain subset and adds ordering, safety,
and open-question detail.

---

End of document.
