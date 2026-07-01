---
date: 2026-07-01
branch: feature/person-architecture
model: opencode/big-pickle
mode: read-only
status: completed
topic: Attendance Migration Squash Planning
tags:
  - migrations
  - squash
  - planning
  - Phase-0
related:
  - apps/attendance/migrations/
  - apps/cameras/migrations/
  - apps/scheduler/migrations/
  - docs/architecture/erp_foundation_architecture.md
  - docs/architecture/person_identity_architecture.md
  - .ai/sessions/opencode/2026-07-01_15-30-00_erp-foundation-architecture-doc.md
---

# Session Report — Attendance Migration Squash Planning

## Objective

Inspect the attendance migration history and produce a safe, executable plan for squashing migrations before Phase 1 Person architecture implementation. Read-only analysis — no files modified, no migrations created, no database changes.

## Findings

### 1. Migration Graph — Attendance App

```
0001_initial                                    (initial schema)
0002_recognitionsettings_max_periods_per_day
0003_alter_periodtemplate_early_grace_minutes_and_more
0004_remove_student_full_name_student_first_name_and_more
0005_faceembedding_arcface_model_and_more
0006_faceembedding_uniq_active_embedding_per_student
0007_faceembedding_attendance__embeddi_68477a_idx_and_more
0008_alter_faceembedding_arcface_model_and_more
0009_faceembedding_used_images_detail
0010_faceembedding_last_used_min_score
0011_faceembedding_avg_used_score
0012_alter_attendanceevent_ts
0013_recognitionsettings_min_face_px
0014_alter_recognitionsettings_min_face_px
0015_recognitionsettings_changed_at
0016_alter_attendanceevent_crop_path_and_more
0017_attendancerecord_last_pass_at_and_more
0018_recognitionsettings_crops_apply_all_students_and_more
0019_faceembedding_crops_opt_in
0020_attendancerecord_confirmed_and_more
0021_dashboardtag_periodtemplate_usage_tags
├── 0022_alter_student_grade                     (same number!)
└── 0022_lunchsubscription                        (same number!)
    └── 0023_merge_0022_alter_student_grade_0022_lunchsubscription  (merge)
        └── 0024_alter_student_grade
            └── 0025_student_gender_alter_student_grade
                └── 0026_attendancerecord_lunch_eligible_at_time_and_more
                    └── 0027_alter_lunchsubscription_end_date_and_more
                        └── 0028_discountprofile_and_more
                            └── 0029_mealsubscription_source
                                └── 0030_alter_mealsubscription_options_and_more
                                    └── 0031_alter_mealsubscription_plan_type  [UNAPPLIED]
```

### 2. Merge Migration and Duplicate Numbering

**Duplicate numbering at 0022:**
- `0022_alter_student_grade.py` — generated 2025-11-17
- `0022_lunchsubscription.py` — generated 2025-11-18

Both have the same parent (`0021_dashboardtag_periodtemplate_usage_tags`), same number, but different filenames and different creation dates. This was a branch/merge scenario.

**Merge migration at 0023:**
- `0023_merge_0022_alter_student_grade_0022_lunchsubscription.py` — an empty operations merge that reconciles the two branches.

This is technically valid Django (both `0022_*` files share the same number because they diverged from `0021`), but it is untidy and makes the migration history harder to read. A squash would clean this up.

### 3. Cross-App Dependencies (Safety Analysis)

| App | Latest migration | Depends on attendance? |
|-----|-----------------|----------------------|
| **cameras** | `0009_camera_period_templates` | Yes — `cameras/0008` depends on `attendance/0021`; `cameras/0009` depends on `attendance/0025` |
| **scheduler** | `0020_globalresourcesettings_save_debug_unmatched` | **No** — zero attendance dependencies |
| **auth** | `0012_alter_user_first_name_max_length` | No |
| **contenttypes** | `0002_remove_content_type_name` | No |
| **sessions** | `0001_initial` | No |
| **admin** | `0003_logentry_add_action_flag_choices` | No |

### 4. State of the Tag

- Tag `architecture-approved-v1` exists and is at **HEAD** commit `c2d3aa0`.
- No uncommitted changes exist on the branch (only `.ai/sessions/opencode/` untracked files).

### 5. Django Health

| Check | Result |
|-------|--------|
| `python manage.py check` | No issues (0 silenced) |
| `makemigrations --check --dry-run` | No changes detected |
| Unapplied migrations | Only `attendance.0031_alter_mealsubscription_plan_type` |
| Database | PostgreSQL |

### 6. Backup Status

| File | Size | Date |
|------|------|------|
| `backups/manual/pre_migration_squash_20260701_193012.json` | 23.6 MB | 2026-07-01 |
| `backups/manual/pre_person_architecture_20260630_201157.json` | 23.6 MB | 2026-06-30 |

Two verified JSON backups exist. The second (`pre_migration_squash_*`) is explicitly named for pre-squash safety. `backups/` is gitignored.

### 7. Migration 0031 Details

The single unapplied migration (`0031_alter_mealsubscription_plan_type`) is a trivial `AlterField` on `MealSubscription.plan_type` — adding `help_text` and reordering choice labels. It is safe to apply or to absorb into the squash.

---

## Decisions

1. **Squash only the attendance app.** Cameras and scheduler have clean linear histories with no duplicate numbers or merge migrations. They do not need squashing.

2. **The squash target should be `0021`** (the last migration before the duplicate numbering). All migrations from `0001_initial` through `0021_dashboardtag_periodtemplate_usage_tags` should be squashed into a single migration. This preserves `0021` and `0025` as stable labels that `cameras/0008` and `cameras/0009` depend on.

3. **Do NOT squash through `0025`** because `cameras/0009` depends on `attendance/0025`. If we squash past `0025`, the cameras dependency must be updated to point to the squash. We CAN squash through `0025` if we also update cameras dependencies — this is a valid option but adds risk.

4. **Recommended approach**: Squash `0001` through `0021` into a single squashed migration. The resulting graph will be:

    ```
    attendance/0001_squashed_0001_0021  (replaces 0001–0021)
    attendance/0022_alter_student_grade  (unchanged)
    attendance/0022_lunchsubscription    (unchanged)
    attendance/0023_merge  (unchanged)
    attendance/0024_alter_student_grade  (unchanged)
    attendance/0025_student_gender...    (unchanged)
    attendance/0026...
    ...
    attendance/0031  (unchanged)
    ```

    This eliminates the duplicate/messy history while keeping all existing cross-app dependency labels intact.

5. **Optional expanded approach**: Squash `0001` through `0030` into a single migration, update cameras dependencies from `0021` and `0025` to the new squash. This is cleaner but requires cameras migration edits. Recommended only if cameras migrations are also being edited.

6. **Apply 0031 before squashing** to keep the squash as the union of all applied state. This ensures the squash captures the current database schema.

---

## Implementation Plan

### Step 1: Apply the pending migration

```bash
.venv/bin/python manage.py migrate attendance 0031
```

This unblocks the squash by ensuring the database is at the latest schema before the squash captures it.

### Step 2: Verify Django health after applying

```bash
.venv/bin/python manage.py check
.venv/bin/python manage.py makemigrations --check --dry-run
```

### Step 3: Create the squash

```bash
.venv/bin/python manage.py makemigrations attendance --squash-name squashed_0001_0021
```

This creates a single squashed migration that replaces `0001_initial` through `0021_dashboardtag_periodtemplate_usage_tags`.

### Step 4: Verify the squash file

The squashed migration will be created at:
```
apps/attendance/migrations/0032_squashed_0001_0021.py
```

It will have `replaces = [...]` listing all replaced migrations. It will depend on `attendance.0031` (the last before it) or on the previous node in the dependency chain of `0021`.

The `replaces` list must include every migration from `0001` through `0021`:
```
replaces = [
    ('attendance', '0001_initial'),
    ('attendance', '0002_recognitionsettings_max_periods_per_day'),
    ('attendance', '0003_alter_periodtemplate_early_grace_minutes_and_more'),
    ...
    ('attendance', '0021_dashboardtag_periodtemplate_usage_tags'),
]
```

Verify this in the generated file.

### Step 5: Fake-apply the squash

```bash
.venv/bin/python manage.py migrate attendance 0032 --fake
```

This tells Django that the new squash represents the already-applied migrations. The actual database schema does not change.

### Step 6: Verify the graph

```bash
.venv/bin/python manage.py showmigrations
```

Expected output at this stage:

```
attendance
 [X] 0032_squashed_0001_0021    (fake-applied, replaces 0001–0021)
 [X] 0022_alter_student_grade   (still listed as applied)
 [X] 0022_lunchsubscription     (still listed as applied)
 [X] 0023_merge_0022_alter_student_grade_0022_lunchsubscription
 [X] 0024_alter_student_grade
 [X] 0025_student_gender_alter_student_grade
 ...
 [X] 0031_alter_mealsubscription_plan_type
```

Note: `showmigrations` may show the replaced migrations as `(migrated)` or omit them once the squash is fake-applied. Django's behavior:
- With `--fake`, the squash is recorded in the `django_migrations` table.
- The replaced migrations are no longer tracked individually.
- If they are also in the `django_migrations` table, they remain listed but ignored.

### Step 7: Run full health checks

```bash
.venv/bin/python manage.py check
.venv/bin/python manage.py makemigrations --check --dry-run
```

### Step 8: Verify cameras dependency still resolves

The cameras app depends on `attendance/0021`. After the squash, `0021` is no longer an individual migration — it is part of the squash. Django resolves this through the `replaces` list. Verify:

```bash
.venv/bin/python manage.py migrate cameras --plan
```

This should show no pending migrations for cameras.

---

## Expected Files Created

| File | Purpose |
|------|---------|
| `apps/attendance/migrations/0032_squashed_0001_0021.py` | Squashed migration replacing 0001–0021 |

No files are deleted automatically. The original migration files (0001–0021) remain in the directory. They can be manually removed later once the squash is stable, but they should **not** be deleted before Phase 2 (deprecate Student). Keeping them allows rollback.

---

## What Files Should Remain Temporarily

All original migration files should remain in the directory until Phase 2 or later:

- `apps/attendance/migrations/0001_initial.py` through `0021_dashboardtag_periodtemplate_usage_tags.py`
- `apps/attendance/migrations/0022_alter_student_grade.py`, `0022_lunchsubscription.py`
- `apps/attendance/migrations/0023_merge_*.py` through `0031_alter_mealsubscription_plan_type.py`

**Do not delete any original migrations.** Django will ignore files listed in `replaces = [...]` once the squash is applied, but keeping them is a safety net for rollback.

---

## Post-Squash Checks

| # | Check | Command | Expected |
|---|-------|---------|----------|
| 1 | Django system check | `.venv/bin/python manage.py check` | No issues |
| 2 | No pending migrations | `.venv/bin/python manage.py makemigrations --check --dry-run` | No changes |
| 3 | Cameras plan | `.venv/bin/python manage.py migrate cameras --plan` | No pending for cameras |
| 4 | Scheduler plan | `.venv/bin/python manage.py migrate scheduler --plan` | No pending for scheduler |
| 5 | Show migrations | `.venv/bin/python manage.py showmigrations` | All [X] except removed internal ones |
| 6 | Squash replaces list | Inspect `0032_squashed_0001_0021.py` | Covers 0001–0021 |

---

## Rollback Plan

If the squash causes issues:

### Option A: Reverse the fake-apply (simplest)

```bash
.venv/bin/python manage.py migrate attendance 0031 --fake
```

This removes the squash from the `django_migrations` table and re-records migrations 0001–0021. No SQL is executed because the schema is the same.

**Prerequisite:** All original migration files (0001–0021) are still present in the migrations directory.

### Option B: Full database restore (nuclear option)

```bash
# Drop and recreate the database
createdb bisk_dev
# Restore from backup
.venv/bin/python manage.py migrate
```

With the verified JSON backup at `backups/manual/pre_migration_squash_20260701_193012.json`, the data can also be restored via:

```bash
.venv/bin/python manage.py flush
.venv/bin/python manage.py loaddata backups/manual/pre_migration_squash_20260701_193012.json
```

**Only needed if migrations are corrupted or data is lost.**

---

## Whether to Squash Only Attendance or Other Apps Too

| App | Migrations | Need squash? | Reason |
|-----|------------|-------------|--------|
| **attendance** | 31 (1 unapplied), 2 duplicate numbers, 1 merge | **Yes** | Messy history; merge + duplicates will complicate Phase 1 Person migrations |
| **cameras** | 9 (linear, clean) | **No** | Clean history; no merge or duplicate issues |
| **scheduler** | 20 (linear, clean) | **No** | Clean history; no merge or duplicate issues |

**Squash only attendance.** Cameras and scheduler have clean linear histories. Squashing them would add risk with no benefit.

However, note the recommended squash target is **0001–0021 only** (not through 0025 or 0030). This minimizes the change surface:

- 21 migrations replaced (down to 1).
- 10 migrations preserved (0022–0031).
- 2 cross-app camera dependencies remain valid without changes.

---

## Summary of Commands (Copy-Paste Ready)

```bash
# 1. Ensure we are on the correct branch
git branch --show-current    # must be feature/person-architecture

# 2. Apply pending migration
.venv/bin/python manage.py migrate attendance 0031

# 3. Verify health
.venv/bin/python manage.py check
.venv/bin/python manage.py makemigrations --check --dry-run

# 4. Create the squash
.venv/bin/python manage.py makemigrations attendance --squash-name squashed_0001_0021

# 5. Inspect generated squash file
ls -la apps/attendance/migrations/0032_squashed_0001_0021.py
# Verify `replaces` list covers 0001 through 0021

# 6. Fake-apply the squash
.venv/bin/python manage.py migrate attendance 0032 --fake

# 7. Post-squash verification
.venv/bin/python manage.py check
.venv/bin/python manage.py makemigrations --check --dry-run
.venv/bin/python manage.py showmigrations
.venv/bin/python manage.py migrate cameras --plan
.venv/bin/python manage.py migrate scheduler --plan

# 8. List the squash file in git
git status
git diff --stat
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Squash `replaces` list incorrect | Low | Medium | Inspect generated file; verify list covers exactly 0001–0021 |
| Cameras dependency broken | Low | High | Cameras depends on 0021 and 0025 — both preserved outside the squash target range |
| Database state mismatch | Low | High | Apply 0031 before squashing so the squash captures the exact DB state |
| Cannot rollback | Low | Medium | Keep all original migration files; Option A rollback is safe |
| Person migrations conflict with squash | Low | Medium | Create Person migrations AFTER the squash is applied; never before |

---

## Recommendations

1. **Squash now** — before any Person architecture migrations are created. This ensures the Person migration is migration `0033` (or whatever follows the squash), not competing with existing numbered migrations.

2. **Target 0001–0021 only** — this preserves `0025` for the cameras dependency and avoids the need to edit any cross-app migration files.

3. **Keep all original files** — do not delete the original 0001–0021 .py files. They are harmless (Django ignores them once the `replaces` list is active) and provide a safety net.

4. **Do NOT squash cameras or scheduler** — their histories are clean and linear.

5. **Run all post-squash checks** before proceeding to Phase 1.

6. **Delete original migration files only after Phase 2** (Student deprecation), when the squash is fully proven in production with the Person models.

---

Report generated:
`.ai/sessions/opencode/2026-07-01_16-30-00_migration-squash-plan.md`
