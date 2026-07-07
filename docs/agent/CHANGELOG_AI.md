# BISK_RFv4 AI Change Log

Version: 1.0

## Initial entry — Documentation and harness preparation

Branch: `feature/person-architecture`

### Context

The wallet/pricing branch was checkpointed and pushed. A new branch was created for Person architecture.

### Completed manually before this documentation

- Committed meal dashboard cleanup:
  - Commit: `0d82202`
  - Message: `Clean up meal dashboard, remove legacy files, and sync Ubuntu 26 development environment`
- Pushed `feature/wallet-pricing-discounts`.
- Created and pushed `feature/person-architecture`.
- Verified working tree clean.

### Purpose

Prepare the project for AI-assisted development through OpenCode/OpenRouter while reducing mistakes and keeping a clear roadmap.

### Next action

Add these docs to the repository under `docs/agent/`, add root `AGENTS.md`, commit, then set up OpenCode.

---

## M1 — FaceEmbedding person FK (attendance identity adoption)

Date: 2026-07-07
Branch: `feature/person-architecture`
Phase: 1.5 (Person architecture) — Attendance Identity adoption, milestone M1.

### Context

The first attendance identity-adoption step (M1) introduces a nullable `person`
FK on `apps.attendance.FaceEmbedding`, alongside the legacy `student` FK. The
new field is dual-written by the enroll path so existing read/write flows keep
working unchanged while identity adoption proceeds in parallel.

### Schema changes

- `apps.attendance.models.FaceEmbedding.person` — nullable FK to
  `apps.identity.Person`, `on_delete=CASCADE`, `db_index=True`,
  `related_name="embeddings"`.
- New partial unique constraint `uniq_active_embedding_per_person`
  (`fields=("person",)`, `condition=Q(is_active=True)`) — coexists with the
  legacy `uniq_active_embedding_per_student` constraint. PostgreSQL allows
  multiple NULLs, so legacy embeddings with `person=NULL` are unaffected.

### Migrations

- `0032_faceembedding_person` — schema migration (AddField + AddConstraint),
  auto-generated. Dependencies: `attendance 0031`, `cameras 0009`,
  `identity 0002`.
- `0033_faceembedding_person_backfill` — data migration that backfills
  `FaceEmbedding.person_id` from `StudentProfile.legacy_student_id` →
  `Person.id`. Idempotent (`filter(person__isnull=True)`), batched
  (`chunk_size=500`), one `save(update_fields=["person_id"])` per row.
  Reverse is a no-op (column drop deferred to a future release). Logic lives
  in `_0033_helper.py` so tests can import it (the migration module name
  starts with a digit and is not importable normally).

### Service-layer changes

- `apps.attendance/services.py` — new `_resolve_person_from_student(student)`
  helper. Read-only; resolves a `Person` via
  `StudentProfile.legacy_student`. Returns `None` when the student has not
  been migrated to identity yet. Uses `django.apps.get_model` for a lazy
  import (avoids import cycles).

### Dual-write path

- `apps/attendance/utils/embeddings.py` — `enroll_student_from_folder`
  resolves `person` once after the student lookup and includes it in the
  `FaceEmbedding.objects.create(...)` payload. `person` is only added to the
  payload when non-null, so the legacy in-place update path does not stomp
  an existing `person_id` when re-enrolling an unmigrated student.
- `apps/attendance/api.py` — `EnrollView.post` resolves `person` after the
  student lookup and passes it to `FaceEmbedding.objects.create(...)`.

### Tests

- `apps/attendance/tests.py` — 26 new tests across 6 classes:
  - `FaceEmbeddingPersonFKTests` — schema, FK config, related_name.
  - `FaceEmbeddingConstraintTests` — new partial unique constraint
    (active/inactive/null/different-person cases).
  - `ResolvePersonFromStudentTests` — resolver helper (migrated, unmigrated,
    None input).
  - `EnrollViewDualWriteTests` — `EnrollView` POST sets `person` when
    migrated, leaves it NULL when unmigrated, response shape unchanged.
  - `BackfillMigrationTests` — 0033 backfill forwards/backwards (idempotent,
    skips unmigrated, noop reverse).
  - `GalleryViewRegressionTests` — `GalleryView` response shape unchanged,
    `person_id` does not leak.

### Verification

- `python manage.py check` — clean.
- `python manage.py makemigrations --check --dry-run` — no changes detected.
- `apps.attendance` — 26/26 tests OK.
- `apps.identity` — 209/209 tests OK (no regression).
- `apps.meals.tests_integrations_attendance` — 20/20 tests OK (no
  regression of meal-attendance integration).

### Files modified

- `apps/attendance/models.py`
- `apps/attendance/services.py`
- `apps/attendance/utils/embeddings.py`
- `apps/attendance/api.py`
- `apps/attendance/tests.py`
- `docs/agent/CHANGELOG_AI.md`

### Files added

- `apps/attendance/migrations/0032_faceembedding_person.py`
- `apps/attendance/migrations/0033_faceembedding_person_backfill.py`
- `apps/attendance/migrations/_0033_helper.py`

### Next milestone

M2 — `AttendanceRecord.person` FK with the same dual-write pattern, followed by
read-path migration (recognition lookups optionally prefer `person` over
`student`). See `docs/agent/ROADMAP.md`.

---

## M2 — AttendanceEvent person FK (attendance identity adoption)

Date: 2026-07-07
Branch: `feature/person-architecture`
Phase: 1.5 (Person architecture) — Attendance Identity adoption, milestone M2.

### Context

The second attendance identity-adoption step (M2) introduces a nullable
`person` FK on `apps.attendance.AttendanceEvent`, alongside the legacy
`student` FK. The new field is dual-written by the recognition write path
(`_write_from_match`) so existing read/write flows keep working unchanged while
identity adoption proceeds in parallel. M2 follows the exact dual-FK additive
pattern proven in M1 on `FaceEmbedding`.

### Schema changes

- `apps.attendance.models.AttendanceEvent.person` — nullable FK to
  `apps.identity.Person`, `on_delete=CASCADE`, `db_index=True`,
  `related_name="attendance_events"`. No new unique constraint (events are
  append-only audit log; no upsert invariant, no `is_active` flag, unlike M1's
  `uniq_active_embedding_per_person`).

### Migrations

- `0034_attendanceevent_person` — schema migration (AddField), auto-generated.
  Dependencies: `attendance 0033_faceembedding_person_backfill`,
  `identity 0002_seed_roletypes`.
- `0035_attendanceevent_person_backfill` — data migration that backfills
  `AttendanceEvent.person_id` from `StudentProfile.legacy_student_id` →
  `Person.id`. Idempotent (`filter(person__isnull=True)`), batched
  (`chunk_size=500`), grouped `update(pk__in=...)` per `person_id` to minimize
  write traffic on the (potentially large) append-only events table. Reverse
  is a no-op (column drop deferred to a future release). Logic lives in
  `_0035_helper.py` so tests can import it (the migration module name starts
  with a digit and is not importable normally).

### Service-layer changes

- `apps.attendance.services.py::_write_from_match` — resolves `person` once
  via the M1 helper `_resolve_person_from_student(student)` after the `rs`
  setup, then passes `person=person` to all four
  `AttendanceEvent.objects.create(...)` call sites:
  1. below `min_score` audit event (line ~131)
  2. no period window open (line ~142)
  3. winners loop — one event per PeriodOccurrence (line ~170)
  4. `max_periods_reached` fallback (line ~250)
  No change to the `AttendanceRecord` upsert logic (that is M3 territory).
  No change to public signatures of `record_recognition` / `ingest_match`.

### No read-path / API / admin / bridge change

- `apps.attendance.api.py` — unchanged. `IngestView` wire contract preserved
  (response does not leak `person_id` / `person`).
- `apps.attendance.admin.py` — unchanged. `AttendanceEventAdmin` keeps
  `student__*` search/display; the `person` column exists in the DB but is
  not surfaced in the admin UI (deferred polish, mirrors M1's
  `FaceEmbeddingAdmin`).
- `apps.attendance.views.py` — unchanged. `latest_event_qs` subqueries still
  filter on `student_id` (FK intact).
- `apps.meals.integrations.attendance` — unchanged. The bridge stays on the
  **fallback chain** (`attendance_event.student → StudentProfile.legacy_student
  → Person`). Switching the bridge to prefer `attendance_event.person` is
  design step S11, deliberately scheduled AFTER reconciliation confirms the
  column is populated.

### Tests

- `apps/attendance/tests.py` — 36 new tests across 7 classes (M2 sections):
  - `AttendanceEventPersonFKTests` — schema, FK config, related_name,
    db_index, no new constraint, no unique_together.
  - `ResolvePersonFromStudentReuseTests` — resolver helper reuse (migrated,
    unmigrated, None input).
  - `WriteFromMatchDualWriteTests` — all four create branches dual-write
    `person` (winners loop, below_min_score, no_period, max_periods_reached
    fallback) via `ingest_match` and `record_recognition`; multi-period tie
    case; unknown h_code; unmigrated student.
  - `EventBackfillMigrationTests` — 0035 backfill forwards/backwards
    (idempotent, skips unmigrated, noop reverse, batches large event set).
  - `ReadPathRegressionTests` — latest_event subquery still keys on
    `student_id`; admin list_display unchanged; camera health query
    unaffected; anti-tests (no person constraint, no person admin filter).
  - `BridgeFallbackRegressionTests` — bridge resolves Person via the
    backlink when `attendance_event.person` is NULL or set (M2 keeps the
    bridge on the fallback chain).
  - `IngestViewWireContractTests` — `IngestView` response shape unchanged,
    `person_id` / `person` do not leak into the response.

### Verification

- `python manage.py check` — clean (1 expected warning: GlobalResourceSettings).
- `python manage.py makemigrations --check --dry-run` — no changes detected.
- `apps.attendance.tests` — 62/62 tests OK (26 M1 + 36 M2; no regression).
- `apps.meals.tests_integrations_attendance` — 20/20 tests OK (no
  regression of meal-attendance integration bridge).

### Files modified

- `apps/attendance/models.py`
- `apps/attendance/services.py`
- `apps/attendance/tests.py`
- `docs/agent/CHANGELOG_AI.md`

### Files added

- `apps/attendance/migrations/0034_attendanceevent_person.py`
- `apps/attendance/migrations/0035_attendanceevent_person_backfill.py`
- `apps/attendance/migrations/_0035_helper.py`

### Reversibility

- `migrate attendance 0033` reverses `0034` (AddField) and `0035` (RunPython
  noop), dropping the `person_id` column. The `student` FK is untouched, so
  the legacy path is fully restored. Code revert restores single-`student`
  write path. The only irreversible step (`student` FK drop) is design step
  S13, scheduled in a separate approved release.

### Next milestone

M3 — `AttendanceRecord.person` FK with the same dual-write pattern, including
the `(person, period)` unique_together + index (mirroring the existing
`(student, period)` constraints). See `docs/agent/ROADMAP.md` and the design
document at `.tmp/reviews/2026-07-07_attendance-identity-design/attendance_
identity_adoption_design.md` §4.3 + §10 S7–S9.

---

## M3 — AttendanceRecord person FK (attendance identity adoption)

Date: 2026-07-07
Branch: `feature/person-architecture`
Phase: 1.5 (Person architecture) — Attendance Identity adoption, milestone M3.

### Context

The third and final attendance identity-adoption step (M3) introduces a
nullable `person` FK on `apps.attendance.AttendanceRecord`, alongside the
legacy `student` FK. This is the most complex milestone because
`AttendanceRecord` carries the only **upsert invariant** in the attendance
domain (`unique_together [("student", "period")]` + `get_or_create(student=...,
period=...)`). M3 follows the dual-FK additive pattern proven in M1/M2 but
takes a conservative approach to the upsert: the key stays `(student, period)`,
and `person` is added to the `defaults` dict + defensively set on found
records.

### Schema changes

- `apps.attendance.models.AttendanceRecord.person` — nullable FK to
  `apps.identity.Person`, `on_delete=CASCADE`, `db_index=True`,
  `related_name="attendance_records"`.
- New coexisting `unique_together = [("person", "period")]` alongside the
  existing `[("student", "period")]`. PostgreSQL allows multiple NULLs in a
  UNIQUE constraint, so unmigrated records (person=NULL) don't conflict.
- New coexisting `Index(fields=["person", "period"])` alongside the existing
  `Index(fields=["student", "period"])`.

### Migrations

- `0036_attendancerecord_person` — schema migration (AddField), auto-generated.
  Dependencies: `attendance 0035_attendanceevent_person_backfill`,
  `identity 0002_seed_roletypes`.
- `0037_attendancerecord_person_backfill` — data migration that backfills
  `AttendanceRecord.person_id` from `StudentProfile.legacy_student_id` →
  `Person.id`. Idempotent (`filter(person__isnull=True)`), batched
  (`chunk_size=500`), grouped `update(pk__in=...)` per `person_id` to minimize
  write traffic on the largest attendance table. Reverse is a no-op. Logic
  lives in `_0037_helper.py` so tests can import it.
- `0038_attendancerecord_person_unique` — constraint migration (auto-generated
  by `makemigrations`, renamed from Django's default name to match the
  expected scope). Adds `AlterUniqueTogether` (both `(student, period)` and
  `(person, period)`) + `AddIndex` for `(person, period)`. Dependencies:
  `attendance 0037`, `cameras 0009`, `identity 0002`.

### Service-layer changes

- `apps.attendance.services.py::_write_from_match` — **conservative dual-write**:
  - The `get_or_create` upsert key **remains `(student, period)`**. It is NOT
    switched to `(person, period)` — that is deferred to design step S9 after
    reconciliation confirms all records have `person_id` set.
  - `person=person` is added to the `defaults` dict so new records get
    `person_id` set on creation.
  - In the `if not created` block, `rec.person_id` is defensively set to
    `person.pk` when `rec.person_id is None` and `person is not None` (edge
    case: student migrated after the 0037 backfill ran).
  - A `person_id_set` flag tracks whether `person_id` was changed; the flag is
    used to include `"person_id"` in the `save(update_fields=...)` call of the
    re-register window short-circuit path.
  - The default update path uses `rec.save()` (no `update_fields`), so
    `person_id` is always saved there.

### Critical design decision: upsert key NOT switched

M3 deliberately does NOT switch the upsert key from `(student, period)` to
`(person, period)`. The hazard (design R3): if a record exists with
`person=NULL` (created before backfill, or the student was migrated after
backfill), a person-keyed `get_or_create(person=P, period=O)` would NOT find
it, try to CREATE a new one, and violate the existing `(student, period)`
unique constraint → `IntegrityError`. The safe approach keeps the
`(student, period)` key and uses `defaults` + defensive set. The key switch
is deferred to design step S9.

### No read-path / API / admin / serializer / resource / bridge change

- `apps/attendance.api.py` — unchanged.
- `apps/attendance.admin.py` — unchanged. `AttendanceRecordAdmin` keeps
  `student__*` search/display.
- `apps/attendance.views.py` — unchanged. All subqueries filter on
  `student_id` (FK intact).
- `apps/attendance.serializers.py` — unchanged. Sources from `obj.student.*`.
- `apps/attendance.resources.py` — unchanged.
- `apps.meals.integrations.attendance` — unchanged. Bridge stays fallback-only.

### Tests

- `apps/attendance/tests.py` — 39 new tests across 8 classes (M3 sections):
  - `AttendanceRecordPersonFKTests` — schema, FK config, related_name,
    db_index, unique_together + index coexistence.
  - `AttendanceRecordConstraintTests` — `(person, period)` unique constraint
    behavior (duplicate raises, NULL allowed, different persons allowed,
    different periods allowed).
  - `RecordDualWriteDefaultsTests` — new records get person via defaults;
    existing records found by `(student, period)` get person_id defensively
    set; person_id not overwritten when already set; unmigrated stays NULL;
    no duplicate records created.
  - `RecordReregisterWindowTests` — defensive `person_id` set persisted via
    `save(update_fields=...)` in the re-register short-circuit path.
  - `UpsertInvariantRegressionTests` — multiple events same student+period =
    1 record; mixed migrated/unmigrated = 2 records; multi-period tie.
  - `RecordBackfillMigrationTests` — 0037 backfill forwards/backwards
    (idempotent, skips unmigrated, noop reverse, batches).
  - `RecordReadPathRegressionTests` — admin list unchanged, serializer sources
    student.
  - `RecordBridgeFallbackRegressionTests` — bridge resolves Person via the
    backlink when record.person is NULL or set.

### Verification

- `python manage.py check` — clean (1 expected warning: GlobalResourceSettings).
- `python manage.py makemigrations --check --dry-run` — no changes detected.
- `apps.attendance.tests` — 101/101 tests OK (26 M1 + 36 M2 + 39 M3; no
  regression).
- `apps.meals.tests_integrations_attendance` — 20/20 tests OK (no regression).

### Files modified

- `apps/attendance/models.py`
- `apps/attendance/services.py`
- `apps/attendance/tests.py`
- `docs/agent/CHANGELOG_AI.md`

### Files added

- `apps/attendance/migrations/0036_attendancerecord_person.py`
- `apps/attendance/migrations/0037_attendancerecord_person_backfill.py`
- `apps/attendance/migrations/_0037_helper.py`
- `apps/attendance/migrations/0038_attendancerecord_person_unique.py`

### Reversibility

- `migrate attendance 0035` reverses `0036` (AddField → drops column),
  `0037` (RunPython noop), and `0038` (AlterUniqueTogether + AddIndex →
  drops constraint + index). The `student` FK and the existing
  `(student, period)` invariant are untouched.
- Code revert restores the original `get_or_create` defaults (without
  `person`) and removes the defensive `person_id` set.
- The only irreversible step (`student` FK drop) is design step S13, a
  separate approved release.

### Pre-deployment check

Before applying the constraint migration (0038) on production, run:
```sql
SELECT person_id, period_id, COUNT(*)
FROM attendance_attendancerecord
WHERE person_id IS NOT NULL
GROUP BY person_id, period_id
HAVING COUNT(*) > 1;
```
If this returns rows, resolve the duplicates before applying 0038.

### Next milestone

S9 — switch the `AttendanceRecord` upsert key from `(student, period)` to
`(person, period)` when `person` is non-null. This is deferred until
reconciliation confirms every record whose Student is migrated has
`person_id` set. See the design document §4.3 + §10 S9 and the M3 readiness
review's special warning section.

---

## S12 — Attendance identity reconciliation (read-only)

Date: 2026-07-07
Branch: `feature/person-architecture`
Phase: 1.5 (Person architecture) — Attendance Identity adoption, step S12.

### Context

After M1 (`FaceEmbedding.person`), M2 (`AttendanceEvent.person`), and M3
(`AttendanceRecord.person`), a read-only reconciliation layer is needed to
verify that all `person_id` values are correctly populated. This layer is the
**S9 gate** — the `AttendanceRecord` upsert key switch from `(student, period)`
to `(person, period)` cannot proceed until this reconciliation reports zero
issues.

### What was implemented

- `apps/attendance/identity_reconciliation.py` — read-only service module
  with dataclasses (`AttendanceIdentityIssue`,
  `ModelReconciliationReport`, `AttendanceIdentityReconciliationReport`) and
  four reconcile functions:
  - `reconcile_face_embeddings()`
  - `reconcile_attendance_events()`
  - `reconcile_attendance_records()`
  - `reconcile_all_attendance_identity()`
- `apps/attendance/management/commands/reconcile_attendance_identity.py` —
  thin CLI wrapper with `--json`, `--only-issues`, `--model` flags. Exit
  codes: 0 (clean), 1 (issues found), 2 (unexpected error).

### Checks implemented

| Check | Models | Description |
|---|---|---|
| `missing_person_id` | All 3 | `person_id IS NULL` but `student_id` points to a migrated student |
| `mismatched_person_id` | All 3 | `person_id` doesn't match the StudentProfile's person |
| `orphan_person_no_student` | All 3 | `person_id` set but `student_id` is NULL (defensive) |
| `dangling_person_id` | All 3 | `person_id` points to non-existent Person (defensive) |
| `duplicate_person_periods` | AttendanceRecord | Duplicate `(person_id, period_id)` where person_id IS NOT NULL |
| `duplicate_active_embeddings` | FaceEmbedding | Duplicate active embedding per `person_id` |

### Read-only

Both the service module and the command are **read-only**. No functions write
to the database. There is no `--fix` flag. Auto-fix is explicitly forbidden in
this milestone — `missing_person_id` is fixed by re-running the backfill
migration, `mismatched_person_id` is a human decision, and duplicate removal
is destructive.

### S9 gate criteria

Before switching the upsert key to `(person, period)`:
1. `python manage.py reconcile_identity_migration` → exit 0 (identity migration complete).
2. `python manage.py reconcile_attendance_identity` → exit 0 (all three models clean).
3. Specifically: zero `missing_person_id`, zero `mismatched_person_id`, zero
   duplicates.

### No migrations

This milestone creates **no migrations**. No models were modified. No
service-layer code was changed. It is a pure additive module + command + tests.

### Tests

- `apps/attendance/tests.py` — 23 new tests across 5 classes:
  - `FaceEmbeddingReconciliationTests` (5 tests) — clean, missing, mismatched,
    skips unmigrated, duplicate active.
  - `AttendanceEventReconciliationTests` (4 tests) — clean, missing,
    mismatched, skips unmigrated.
  - `AttendanceRecordReconciliationTests` (5 tests) — clean, missing,
    mismatched, skips unmigrated, duplicate person_period.
  - `AggregateReconciliationTests` (3 tests) — clean, has issues, report shape.
  - `ReconciliationCommandTests` (6 tests) — clean exit 0, issues exit 1,
    JSON output, only-issues, model filter, read-only verification.
- Also fixed a pre-existing time-sensitivity bug in M2/M3 test base classes
  (`cls.now = timezone.localtime()` → `cls.now.replace(hour=12, ...)`) that
  caused `test_max_periods_reached_branch_dual_writes_person` to fail when
  tests ran after 22:00 local time.

### Verification

- `python manage.py check` — clean.
- `python manage.py makemigrations --check --dry-run` — no changes detected.
- `apps.attendance.tests` — 124/124 tests OK (26 M1 + 36 M2 + 39 M3 + 23
  reconciliation; no regression).
- `apps.meals.tests_integrations_attendance` — 20/20 tests OK.
- `python manage.py reconcile_attendance_identity --json` — exit 0, clean.
- `python manage.py reconcile_attendance_identity --only-issues` — exit 0,
  "No issues found. S9 gate passes."

### Files modified

- `apps/attendance/tests.py`
- `docs/agent/CHANGELOG_AI.md`

### Files added

- `apps/attendance/identity_reconciliation.py`
- `apps/attendance/management/commands/reconcile_attendance_identity.py`

### Next milestone

S9 — switch the `AttendanceRecord` upsert key from `(student, period)` to
`(person, period)` when `person` is non-null. Must wait until
`reconcile_attendance_identity` reports exit 0 on production.

