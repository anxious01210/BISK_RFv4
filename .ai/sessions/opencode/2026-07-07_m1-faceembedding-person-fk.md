---
date: 2026-07-07
branch: feature/person-architecture
model: openrouter/z-ai/glm-5.2
mode: implementation
status: completed
topic: M1 — FaceEmbedding person FK (attendance identity adoption)
tags:
  - attendance
  - identity
  - person-architecture
  - migration
  - dual-write
related:
  - docs/agent/ARCHITECTURE.md
  - docs/agent/ROADMAP.md
  - docs/agent/DECISIONS.md
  - docs/agent/CHANGELOG_AI.md
  - .tmp/reviews/2026-07-07_attendance-identity-design/
  - .tmp/reviews/2026-07-07_attendance-m1-readiness/
  - .tmp/reviews/2026-07-07_m1-faceembedding-person-fk/
---

# Objective

Implement the smallest first attendance identity-adoption step (M1) per the
readiness review's exact scope: add a nullable `person` FK to
`apps.attendance.FaceEmbedding`, dual-write it on enroll, add a partial unique
constraint, write a backfill migration, add 26 tests, append a CHANGELOG
entry, run all required verification commands, and produce deliverables for
ChatGPT review.

# Findings

- The readiness review (`.tmp/reviews/2026-07-07_attendance-m1-readiness/`)
  had already produced an exact scope and a 28-case test plan; the
  implementation followed it precisely.
- `apps.attendance.tests.py` was an empty 3-line placeholder before this
  session; it is now the M1 test suite.
- `EnrollView` and `GalleryView` are gated by `X-BISK-KEY` against
  `settings.RUNNER_HEARTBEAT_KEY`. Tests use
  `@override_settings(RUNNER_HEARTBEAT_KEY="")` to disable the gate.
- `RoleType(code="student")` is seeded by an identity data migration; tests
  use `get_or_create` instead of `create`.
- Django's migration loader ignores underscore-prefixed modules, so
  `_0033_helper.py` is a safe home for the backfill forward/reverse logic
  that tests need to import (the `0033_…` migration module name starts with
  a digit and is not importable normally).
- `enroll_student_from_folder` does an in-place update when an active
  embedding already exists. Adding `person` to the payload only when
  non-null preserves the in-place update path's idempotency (re-enrolling an
  unmigrated student does not stomp an existing `person_id`).

# Risks

- **Dual-FK divergence.** Any future enroll path that does not call
  `_resolve_person_from_student` will leave `person` NULL. M1 only updates
  the two known paths; M2/M3 should add a regression test that asserts
  every enroll path dual-writes.
- **Other write paths.** If a future code path creates `FaceEmbedding`
  without going through the resolver, `person` will stay NULL.
- **`AttendanceRecord` backfill.** `AttendanceRecord` has far more rows
  than `FaceEmbedding`; the per-row `save(update_fields=…)` approach used
  here will need `bulk_update` or raw SQL at that scale.
- **Read path.** Recognition lookups still go through `student`. Switching
  them to `person` is M3 and will need a feature flag.

# Decisions

- `person` FK is nullable + `db_index=True` + `on_delete=CASCADE` +
  `related_name="embeddings"`. No unique-together with `student`.
- New constraint `uniq_active_embedding_per_person` is partial
  (`condition=Q(is_active=True)`), matching the legacy
  `uniq_active_embedding_per_student` shape. PostgreSQL allows multiple
  NULLs, so legacy rows are unaffected.
- Backfill migration 0033 reverse is a noop; column drop deferred to a
  future release (keeps rollback-then-re-apply idempotent).
- `_0033_helper.py` holds the backfill forward/reverse logic so tests can
  import it; the migration module imports from it.
- `_resolve_person_from_student` uses `django.apps.get_model` for a lazy
  import (avoids import cycle; keeps the `attendance → identity`
  dependency direction explicit).
- `person` is added to the create payload **only when non-null** so the
  in-place update path does not stomp an existing `person_id`.

# Implementation

All M1 deliverables are implemented:

- Schema: `FaceEmbedding.person` FK + `uniq_active_embedding_per_person`
  constraint in `apps/attendance/models.py`.
- Migrations: `0032_faceembedding_person` (schema, auto-generated),
  `0033_faceembedding_person_backfill` (data, hand-written), and
  `_0033_helper.py` (importable helper for tests).
- Resolver: `_resolve_person_from_student` in
  `apps/attendance/services.py`.
- Dual-write: `enroll_student_from_folder` in
  `apps/attendance/utils/embeddings.py`, `EnrollView.post` in
  `apps/attendance/api.py`.
- Tests: 26 tests across 6 classes in `apps/attendance/tests.py`.
- Docs: CHANGELOG_AI entry appended in `docs/agent/CHANGELOG_AI.md`.

# Remaining Work

1. M2 — `AttendanceRecord.person` FK with the same dual-write pattern.
2. M3 — read-path migration (recognition lookups optionally prefer `person`
   over `student`) behind a feature flag.
3. Add a regression test that asserts every `FaceEmbedding` write path
   dual-writes `person`.
4. Plan `AttendanceRecord` backfill using `bulk_update` or raw SQL (scale).

# Files Modified

- `apps/attendance/models.py`
- `apps/attendance/services.py`
- `apps/attendance/utils/embeddings.py`
- `apps/attendance/api.py`
- `apps/attendance/tests.py`
- `docs/agent/CHANGELOG_AI.md`

# Files Added

- `apps/attendance/migrations/0032_faceembedding_person.py`
- `apps/attendance/migrations/0033_faceembedding_person_backfill.py`
- `apps/attendance/migrations/_0033_helper.py`
- `.tmp/reviews/2026-07-07_m1-faceembedding-person-fk/implementation_summary.md`
- `.tmp/reviews/2026-07-07_m1-faceembedding-person-fk/architecture_review.md`
- `.tmp/reviews/2026-07-07_m1-faceembedding-person-fk/test_output.txt`
- `.tmp/reviews/2026-07-07_m1-faceembedding-person-fk/git_diff_tracked.patch`
- `.tmp/reviews/2026-07-07_m1-faceembedding-person-fk/git_status.txt`

# Commands Executed

```
git status --short
git branch --show-current
python manage.py check
python manage.py makemigrations --check --dry-run
python manage.py makemigrations            # to generate 0032
TEST_DB_USER=bisk_test TEST_DB_PASSWORD='BiskTestDB1!' \
  python manage.py test apps.attendance --noinput
TEST_DB_USER=bisk_test TEST_DB_PASSWORD='BiskTestDB1!' \
  python manage.py test apps.identity --noinput
TEST_DB_USER=bisk_test TEST_DB_PASSWORD='BiskTestDB1!' \
  python manage.py test apps.meals.tests_integrations_attendance --noinput
git diff --stat
```

# Recommendations

- Treat M1 as the canonical pattern for the next identity-adoption steps
  (M2 AttendanceRecord, M3 read path). Same shape: nullable FK + partial
  unique constraint + dual-write + idempotent backfill + regression tests.
- Before M2, audit all `AttendanceRecord` write paths (not just the obvious
  one) so the dual-write coverage is complete from day one.
- Consider adding a `manage.py inspect_person_dual_write` check command that
  reports rows where `student` is set but `person` is NULL after the
  backfill, to track migration completion in production.

# Deliverables for ChatGPT review

- `.tmp/reviews/2026-07-07_m1-faceembedding-person-fk/implementation_summary.md`
- `.tmp/reviews/2026-07-07_m1-faceembedding-person-fk/architecture_review.md`
- `.tmp/reviews/2026-07-07_m1-faceembedding-person-fk/test_output.txt`
- `.tmp/reviews/2026-07-07_m1-faceembedding-person-fk/git_diff_tracked.patch`
- `.tmp/reviews/2026-07-07_m1-faceembedding-person-fk/git_status.txt`

Report generated:
.ai/sessions/opencode/2026-07-07_m1-faceembedding-person-fk.md
