---
date: 2026-07-07
branch: feature/person-architecture
model: z-ai/glm-5.2
mode: implementation
status: completed
topic: Attendance identity reconciliation (S12) — read-only verification layer
tags:
  - attendance
  - identity
  - reconciliation
  - s12
  - s9-gate
  - read-only
related:
  - .tmp/reviews/2026-07-07_attendance-identity-reconciliation-readiness/attendance_identity_reconciliation_readiness.md
  - .tmp/reviews/2026-07-07_attendance-identity-reconciliation-readiness/reconciliation_exact_scope.md
  - .tmp/reviews/2026-07-07_attendance-identity-reconciliation/implementation_summary.md
---

# Objective

Implement the read-only attendance identity reconciliation layer (S12) that
verifies M1/M2/M3 dual-FK data integrity and serves as the S9 gate.

# Findings

- The reconciliation pattern mirrors `apps/identity/migration_reconciliation.py`
  exactly — dataclasses + read-only functions + thin CLI command.
- The `student_id → person_id` mapping is built once and reused across all
  three model checks, minimizing queries.
- A pre-existing time-sensitivity bug was found in M2/M3 test base classes:
  `cls.now = timezone.localtime()` caused `self.now + 2h` to cross midnight
  when tests ran after 22:00 local time, making
  `test_max_periods_reached_branch_dual_writes_person` fail. Fixed by using
  `cls.now.replace(hour=12, ...)`.

# Risks

- None. The module is purely read-only with no writes, no migrations, no
  model changes.

# Decisions

1. Read-only only — no `--fix` flag. Auto-fix is explicitly forbidden.
2. Exit codes: 0 (clean), 1 (issues found), 2 (unexpected error).
3. Use `transaction.atomic()` savepoint for the duplicate-active-embedding
   test to safely catch IntegrityError from raw SQL inside Django's TestCase
   atomic block.

# Implementation

- Added `apps/attendance/identity_reconciliation.py` with 3 dataclasses and
  4 reconcile functions + 2 duplicate detection helpers.
- Added `apps/attendance/management/commands/reconcile_attendance_identity.py`
  with --json, --only-issues, --model flags.
- Added 23 reconciliation tests across 5 classes.
- Fixed M2/M3 test time-sensitivity bug.
- Appended S12 entry to CHANGELOG_AI.md.

# Remaining Work

1. S9: Switch upsert key to (person, period) — AFTER `reconcile_attendance_identity`
   exits 0 on production.
2. S11: Update Meals bridge to prefer `attendance_event.person` with fallback.
3. S13: Drop `student` FK (separate release, with backup).

# Files Modified

- `apps/attendance/tests.py`
- `docs/agent/CHANGELOG_AI.md`

# Files Added

- `apps/attendance/identity_reconciliation.py`
- `apps/attendance/management/commands/reconcile_attendance_identity.py`

# Commands Executed

```bash
python manage.py check
python manage.py makemigrations --check --dry-run
TEST_DB_USER=bisk_test TEST_DB_PASSWORD='BiskTestDB1!' python manage.py test apps.attendance.tests --noinput
TEST_DB_USER=bisk_test TEST_DB_PASSWORD='BiskTestDB1!' python manage.py test apps.meals.tests_integrations_attendance --noinput
python manage.py reconcile_attendance_identity --json
python manage.py reconcile_attendance_identity --only-issues
```

Report generated:
.ai/sessions/opencode/2026-07-07_attendance-identity-reconciliation.md
