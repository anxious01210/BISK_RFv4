---
date: 2026-07-07
branch: feature/person-architecture
model: z-ai/glm-5.2
mode: implementation
status: completed
topic: M2 — AttendanceEvent person FK implementation
tags:
  - attendance
  - identity
  - person-architecture
  - migration
  - dual-fk
  - m2
related:
  - .tmp/reviews/2026-07-07_m2-attendanceevent-readiness/m2_readiness_review.md
  - .tmp/reviews/2026-07-07_m2-attendanceevent-readiness/m2_exact_scope.md
  - .tmp/reviews/2026-07-07_m2-attendanceevent-readiness/m2_test_plan.md
  - .tmp/reviews/2026-07-07_m2-attendanceevent-person-fk/implementation_summary.md
  - .tmp/reviews/2026-07-07_m2-attendanceevent-person-fk/architecture_review.md
---

# Objective

Implement M2 — `AttendanceEvent.person` FK — following the exact scope from the
M2 readiness review. This is the second attendance identity-adoption milestone
(after M1 `FaceEmbedding.person`), using the same dual-FK additive pattern.

# Findings

- The M1 pattern (schema migration + batched backfill + dual-write in services)
  transferred cleanly to `AttendanceEvent`.
- `AttendanceEvent` is simpler than `FaceEmbedding`: no unique constraint, no
  upsert, no `is_active` flag, no `Meta` constraints. M2 adds only a nullable FK
  column + a backfill.
- All 4 `AttendanceEvent.objects.create()` call sites in `_write_from_match`
  were updated to pass `person=person`.
- A pre-existing latent bug was discovered: `_to_media_rel("")` returns `None`,
  but `AttendanceEvent.crop_path` is NOT NULL in the DB. This is NOT introduced
  by M2 and is out of scope. Tests work around it by passing a non-empty
  `crop_path`.

# Risks

- **Pre-existing `_to_media_rel` bug** (N1): `ingest_match` with empty
  `crop_path` raises `IntegrityError` on a fresh DB. Out of scope for M2; should
  be tracked as a separate S0-style fix. Production runners always send a
  non-empty path, and production DB drift may mask the bug.
- **One extra query per recognition event** (N2): `_resolve_person_from_student`
  runs a `StudentProfile` lookup on every `_write_from_match` call. Accepted —
  the write path runs at recognition rate, not page-render rate.
- **`RecognitionSettings` singleton sharing** (N3): tests that mutate the
  singleton must reset it in `setUp`. Mitigated via `_reset_settings()` helper.

# Decisions

1. Used `crop_path="captures/test.jpg"` in all test calls to `ingest_match` /
   `record_recognition` to avoid triggering the pre-existing `_to_media_rel`
   bug. The `_make_event` helper (which bypasses `_to_media_rel`) keeps
   `crop_path=""` as default.
2. Split M2 into two migrations (0034 schema + 0035 backfill), mirroring M1's
   0032/0033 split. This keeps each migration independently revertible.
3. Used grouped `update(pk__in=ev_ids, person_id=person_id)` per `person_id`
   in the backfill helper instead of per-row `save(update_fields=...)` to
   minimize write traffic on the append-only events table.
4. Did NOT modify the meals bridge (kept fallback-only). Switching the bridge
   to prefer `attendance_event.person` is design step S11, scheduled after
   reconciliation confirms the column is populated.
5. Did NOT add `person` to `AttendanceEventAdmin` (deferred polish, mirrors
   M1's `FaceEmbeddingAdmin`).

# Implementation

- Added nullable `person` FK to `AttendanceEvent` in `models.py`.
- Generated `0034_attendanceevent_person.py` (auto-generated `AddField`).
- Wrote `_0035_helper.py` (idempotent batched backfill, noop reverse).
- Wrote `0035_attendanceevent_person_backfill.py` (RunPython calling helper).
- Modified `_write_from_match` in `services.py` to resolve `person` once and
  pass it to all 4 `AttendanceEvent.objects.create()` call sites.
- Added 36 M2 tests across 7 classes in `tests.py`.
- Appended M2 entry to `docs/agent/CHANGELOG_AI.md`.

# Remaining Work

1. **S0** (separate fix): Fix `_to_media_rel` to return `""` instead of `None`
   for empty input, OR add `null=True` to `AttendanceEvent.crop_path`. Track
   separately.
2. **M3**: `AttendanceRecord.person` FK with `(person, period)` unique_together
   + index + dual-write upsert keyed on `(person, period)`.
3. **S11**: Update `apps.meals.integrations.attendance` bridge to prefer
   `attendance_event.person` with fallback to the backlink.
4. **S12**: Reconciliation pass confirming every event/record whose Student is
   migrated has `person_id` set.
5. **S13**: (Separate release, with backup) Drop `student` FK + old constraints.

# Files Modified

- `apps/attendance/models.py`
- `apps/attendance/services.py`
- `apps/attendance/tests.py`
- `docs/agent/CHANGELOG_AI.md`

# Files Added

- `apps/attendance/migrations/0034_attendanceevent_person.py`
- `apps/attendance/migrations/0035_attendanceevent_person_backfill.py`
- `apps/attendance/migrations/_0035_helper.py`

# Commands Executed

```bash
git status --short
git log --oneline -5
python manage.py check
python manage.py makemigrations --check --dry-run
python manage.py makemigrations attendance
TEST_DB_USER=bisk_test TEST_DB_PASSWORD='BiskTestDB1!' python manage.py test apps.attendance.tests --noinput
TEST_DB_USER=bisk_test TEST_DB_PASSWORD='BiskTestDB1!' python manage.py test apps.meals.tests_integrations_attendance --noinput
git diff --stat
```

# Recommendations

- **Do not commit yet** — the user explicitly said "Do NOT commit. Do NOT push."
- **Review the diff** in
  `.tmp/reviews/2026-07-07_m2-attendanceevent-person-fk/git_diff_tracked.patch`
  before committing.
- **Track the pre-existing `_to_media_rel` bug** as a separate S0-style fix.
  It is not blocking M2 but should be fixed before M3 to avoid the same test
  workaround.
- **For M3**: follow the same M1/M2 pattern. The `AttendanceRecord` upsert is
  more complex (dual `unique_together` + `get_or_create` keyed on `(person,
  period)` when person is non-null), so allocate more test coverage for the
  race-condition scenarios (design R3).

Report generated:
.ai/sessions/opencode/2026-07-07_m2-attendanceevent-person-fk.md
