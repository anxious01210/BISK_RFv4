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

