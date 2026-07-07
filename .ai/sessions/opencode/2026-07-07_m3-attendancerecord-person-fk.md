---
date: 2026-07-07
branch: feature/person-architecture
model: z-ai/glm-5.2
mode: implementation
status: completed
topic: M3 — AttendanceRecord person FK implementation
tags:
  - attendance
  - identity
  - person-architecture
  - migration
  - dual-fk
  - m3
  - upsert
related:
  - .tmp/reviews/2026-07-07_m3-attendancerecord-readiness/m3_readiness_review.md
  - .tmp/reviews/2026-07-07_m3-attendancerecord-readiness/m3_exact_scope.md
  - .tmp/reviews/2026-07-07_m3-attendancerecord-readiness/m3_test_plan.md
  - .tmp/reviews/2026-07-07_m3-attendancerecord-person-fk/implementation_summary.md
  - .tmp/reviews/2026-07-07_m3-attendancerecord-person-fk/architecture_review.md
---

# Objective

Implement M3 — `AttendanceRecord.person` FK — following the exact scope from the
M3 readiness review. This is the third and final attendance identity-adoption
milestone, using the dual-FK additive pattern with a **conservative upsert
approach** (key stays `(student, period)`).

# Findings

- `AttendanceRecord` is the most complex attendance model because it carries
  the only **upsert invariant** (`unique_together [("student", "period")]` +
  `get_or_create(student=..., period=...)`).
- The conservative approach — keep `(student, period)` key, add `person` to
  `defaults`, defensively set `person_id` on found records — eliminates the
  race condition (design R3).
- A `person_id_set` flag is needed to ensure `person_id` is included in the
  `save(update_fields=...)` call of the re-register window short-circuit path.
  Without this flag, the defensive set would be lost on that path.
- Django auto-generated 0038 as `0038_alter_attendancerecord_unique_together_
  and_more.py`; it was renamed to `0038_attendancerecord_person_unique.py`.

# Risks

- **Upsert key switch race (R3)**: Mitigated by NOT switching the key.
- **`save(update_fields=...)` missing `person_id` (R12)**: Mitigated by the
  `person_id_set` flag.
- **Constraint migration fails on duplicates (R5)**: Mitigated by pre-deployment
  duplicate-detection query documented in the changelog.
- **Pre-existing `_to_media_rel("")` bug (N1)**: Out of scope; tests use
  non-empty `crop_path`.

# Decisions

1. **Keep the upsert key as `(student, period)`** — do NOT switch to
   `(person, period)`. The key switch is deferred to design step S9.
2. **Use a `person_id_set` flag** to track whether `person_id` was defensively
   set, ensuring it's included in `save(update_fields=...)`.
3. **Split into three migrations** (schema → backfill → constraint) following
   the M1/M2 pattern.
4. **Rename 0038** from Django's auto-generated name to match the expected
   scope file name.
5. **Did NOT modify any read paths** (views, API, admin, serializers,
   resources, bridge) — same conservative posture as M1/M2.

# Implementation

- Added nullable `person` FK to `AttendanceRecord` in `models.py`.
- Added `("person", "period")` to `unique_together` and
  `Index(fields=["person", "period"])` to `indexes`.
- Generated `0036_attendancerecord_person.py` (auto-generated AddField).
- Wrote `_0037_helper.py` (idempotent batched backfill, noop reverse).
- Wrote `0037_attendancerecord_person_backfill.py` (RunPython).
- Generated `0038_attendancerecord_person_unique.py` (auto-generated
  AlterUniqueTogether + AddIndex, renamed).
- Modified `_write_from_match` in `services.py`:
  - Added `person=person` to `get_or_create` defaults.
  - Added defensive `person_id` set with `person_id_set` flag.
  - `person_id` included in `save(update_fields=...)` when `person_id_set`.
- Added 39 M3 tests across 8 classes in `tests.py`.
- Appended M3 entry to `docs/agent/CHANGELOG_AI.md`.

# Remaining Work

1. **S0** (separate fix): Fix `_to_media_rel` to return `""` instead of `None`
   for empty input. Track separately.
2. **S9**: Switch the upsert key from `(student, period)` to `(person, period)`
   when person is non-null — AFTER reconciliation confirms all records have
   `person_id` set.
3. **S10**: Additive API/serializer/admin `person` fields and filters.
4. **S11**: Update bridge to prefer `attendance_event.person` with fallback.
5. **S12**: Reconciliation pass confirming every record has `person_id` set.
6. **S13**: (Separate release, with backup) Drop `student` FK + old constraints.

# Files Modified

- `apps/attendance/models.py`
- `apps/attendance/services.py`
- `apps/attendance/tests.py`
- `docs/agent/CHANGELOG_AI.md`

# Files Added

- `apps/attendance/migrations/0036_attendancerecord_person.py`
- `apps/attendance/migrations/0037_attendancerecord_person_backfill.py`
- `apps/attendance/migrations/_0037_helper.py`
- `apps/attendance/migrations/0038_attendancerecord_person_unique.py`

# Commands Executed

```bash
git status --short
git log --oneline -3
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
  `.tmp/reviews/2026-07-07_m3-attendancerecord-person-fk/git_diff_tracked.patch`
  before committing.
- **Run the pre-deployment duplicate check** before applying 0038 on production:
  ```sql
  SELECT person_id, period_id, COUNT(*)
  FROM attendance_attendancerecord
  WHERE person_id IS NOT NULL
  GROUP BY person_id, period_id
  HAVING COUNT(*) > 1;
  ```
- **For S9**: before switching the upsert key, verify that zero records exist
  for migrated students with `person_id=NULL`.

Report generated:
.ai/sessions/opencode/2026-07-07_m3-attendancerecord-person-fk.md
