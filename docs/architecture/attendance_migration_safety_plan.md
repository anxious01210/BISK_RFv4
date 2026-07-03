# Attendance Migration Safety Plan — BISK_RFv4

Date: 2026-07-03
Branch: feature/person-architecture
Version: 1.0 Draft — pending review
Status: Planning document only. No code, models, or migrations changed.

---

## 1. Purpose

This document plans how to safely handle the legacy `apps.attendance`
migration drift **before** building the new Meals and Finance domains.

It is a direct response to the findings in
`docs/architecture/legacy_lunch_wallet_attendance_investigation.md`, which
discovered that the attendance app's migration state does not match its
actual database schema. The existing lunch/wallet/dashboard behavior works
and **must not be broken**; this plan ensures new development can proceed
without forcing a risky reconciliation of the legacy migration graph.

This is a **planning document**. It produces no code, models, or migrations.

---

## 2. Summary of the Drift

The investigation found the following inconsistencies in
`apps/attendance/migrations/`:

| Finding | Detail |
|---|---|
| **Both `0001_initial` and `0001_squashed_0021_dashboardtag_periodtemplate_usage_tags` are recorded as applied.** | The `django_migrations` table has both rows. The squash was **fake-applied** — recorded without actually running its operations against the DB. |
| **The `confirmed` field (from squash migration 0020) was never applied to the DB.** | The squash includes `AddField('attendancerecord', 'confirmed')`, but the actual `attendance_attendancerecord` table does not have the column. |
| **The `lunch_*` fields (from migration 0026) were never applied to the DB.** | `lunch_eligible_at_time`, `lunch_reason_code`, `lunch_reason_notes`, and `lunch_subscription` were added by migration 0026 but the DB columns do not exist. |
| **`models.py` was edited to remove those fields**, aligning with the DB state but not with the migration state. | `makemigrations --check` reports "No changes detected" because it compares models against migration state (not the actual DB). The migration graph is internally consistent but dishonest about the DB. |
| **`LunchSubscription` → `MealSubscription` was a destructive delete+recreate.** | Migration 0028 `DeleteModel('LunchSubscription')` + `CreateModel('MealSubscription')` rather than a `RenameModel`. |
| **`resources.py` references non-existent fields.** | `AttendanceRecordResource` exports `meal_eligible_at_time`, `meal_reason_code`, `meal_reason_notes`, `meal_subscription_id` — none exist on the model or DB. Export would fail at runtime. |

### Current state

- The **legacy lunch/wallet/dashboard behavior works** because the code paths
  that run in production use the columns that *do* exist in the DB.
- The **migration graph lies** about which columns exist — it claims
  `confirmed` and `lunch_*` are present when they are not.
- `makemigrations --check` passes because `models.py` was edited to match
  the DB (not the migration graph), so Django sees no diff.

---

## 3. Why This Is Risky

| Risk | Explanation |
|---|---|
| **Any new migration that assumes `confirmed`/`lunch_*` columns exist will fail at runtime.** | Django's migration state says they exist; the DB says they don't. A migration that does `AlterField` on `confirmed` will crash. |
| **`migrate --fake` or `--fake-initial` on a fresh DB would create the wrong schema.** | A fresh `migrate` would apply the squash honestly (creating `confirmed`/`lunch_*`), producing a DB that differs from production. |
| **Rollback is unsafe.** | If you un-apply a migration that touches `confirmed`/`lunch_*`, it will fail because the columns don't exist. |
| **CI/test DBs may diverge from production.** | If CI creates a fresh DB via `migrate`, it gets the squash-honest schema (with `confirmed`/`lunch_*`). Tests may pass in CI but fail in production (or vice versa). |
| **New Meals/Finance apps that FK into attendance models may depend on columns that don't exist in production.** | If the new apps' migrations assume the attendance schema matches the migration graph, they will break. |
| **`resources.py` is silently broken.** | An admin export of `AttendanceRecord` will crash when it tries to read `meal_eligible_at_time`. |

---

## 4. What Must Not Be Touched Yet

| Do not touch | Why |
|---|---|
| **The legacy `0001_squashed_0021...` migration file.** | Editing it would change the recorded state but not the DB; it could corrupt the migration graph further. |
| **Any migration between `0001` and `0031`.** | These are recorded as applied; altering them risks breaking `migrate` on any environment. |
| **The `django_migrations` table rows for attendance.** | Manually deleting the squash row would make Django try to re-apply it, which would fail (columns already exist or don't exist depending on the table). |
| **`models.py` fields that production code relies on.** | The current model matches the production DB; do not re-add `confirmed`/`lunch_*` without a real migration. |
| **`views.py` lunch/wallet/dashboard behavior.** | It works; wrapping with adapters is safe but rewriting is not. |
| **`signals.py` meal-flag recalculation.** | Working; preserve until enrollment-based eligibility replaces it. |
| **The `face_gallery/<h_code>/` filesystem layout.** | Keyed by `h_code`; remap only after Person migration is verified. |

---

## 5. Safe Inspection Commands

These commands are **read-only** and safe to run on any environment:

```bash
# 1. Show which migrations Django thinks are applied:
python manage.py showmigrations attendance

# 2. Check if models match migration state (does NOT check the DB):
python manage.py makemigrations --check --dry-run

# 3. System check (model/config sanity):
python manage.py check

# 4. Inspect actual DB columns for a table (PostgreSQL):
python manage.py shell -c "
from django.db import connection
table_desc = connection.introspection.get_table_description(
    connection.cursor(), 'attendance_attendancerecord'
)
print([c.name for c in table_desc])
"

# 5. List all tables Django knows about:
python manage.py shell -c "
from django.db import connection
print(connection.introspection.table_names())
"

# 6. Check which migrations are recorded as applied in the DB:
python manage.py shell -c "
from django.db import connection
with connection.cursor() as c:
    c.execute(\"SELECT name, applied FROM django_migrations WHERE app='attendance' ORDER BY applied\")
    for r in c.fetchall():
        print(r)
"

# 7. Inspect a model's declared fields vs DB columns:
python manage.py shell -c "
from apps.attendance.models import AttendanceRecord
declared = {f.name for f in AttendanceRecord._meta.get_fields()}
print('Declared:', sorted(declared))
from django.db import connection
db_cols = {c.name for c in connection.introspection.get_table_description(
    connection.cursor(), AttendanceRecord._meta.db_table
)}
print('DB:', sorted(db_cols))
print('In model but not DB:', sorted(declared - db_cols))
print('In DB but not model:', sorted(db_cols - declared))
"
```

> **Do NOT run** `python manage.py migrate attendance --fake`,
> `--fake-initial`, `migrate attendance zero`, or any `--run-syncdb` on a
> production environment without a verified backup and a tested rollback
> plan.

---

## 6. How to Compare Django Migration State vs Actual DB Schema

The drift exists because Django's migration recorder (`django_migrations`
table) says operations were applied, but the actual DB was never modified.
To detect this:

### 6.1 Per-model introspection

For each attendance model, compare:
- **Declared fields** (`Model._meta.get_fields()`) — what `models.py` says.
- **DB columns** (`connection.introspection.get_table_description(...)`) — what Postgres actually has.
- **Migration-state fields** — what the migration graph says the model
  should have (reconstructed from the squash + subsequent migrations).

A field is "drifted" if the migration graph says it exists but the DB
doesn't (or vice versa).

### 6.2 Full audit script (conceptual, not to be run as a command)

```python
# Pseudocode — do not run blindly; review output manually.
from django.apps import apps
from django.db import connection

for model in apps.get_app_config('attendance').get_models():
    table = model._meta.db_table
    declared = {f.column for f in model._meta.local_fields}
    try:
        db_cols = {c.name for c in connection.introspection.get_table_description(
            connection.cursor(), table
        )}
    except Exception as e:
        print(f"{model.__name__}: table {table} not found ({e})")
        continue
    missing_in_db = declared - db_cols
    extra_in_db = db_cols - declared
    if missing_in_db or extra_in_db:
        print(f"DRIFT {model.__name__} ({table}):")
        if missing_in_db:
            print(f"  In model but not DB: {missing_in_db}")
        if extra_in_db:
            print(f"  In DB but not model: {extra_in_db}")
```

### 6.3 What the audit would reveal (based on the investigation)

| Model | Drift |
|---|---|
| `AttendanceRecord` | Migration graph says `confirmed` + `lunch_*` columns exist; DB does not have them. Model (`models.py`) also omits them, so `makemigrations --check` passes. |
| `MealSubscription` | Migration 0028 deleted `LunchSubscription` and created `MealSubscription`; the DB has `attendance_mealsubscription` but not `attendance_lunchsubscription`. |

---

## 7. Recommended Immediate Strategy

**Do not reconcile the legacy attendance migration graph.** Leave it as-is.

| Step | Action | Why |
|---|---|---|
| 1 | **Freeze the attendance migration graph.** No new migrations on `apps.attendance` unless they are purely additive and don't reference drifted fields. | Editing the graph risks breaking production. |
| 2 | **Fix `resources.py` only.** Remove references to non-existent `meal_*` fields (or guard them with `getattr`). This is a bug fix, not a migration. | The export is silently broken; this is safe to fix without touching migrations. |
| 3 | **Delete `views_.py`** (after confirming `views.py` is the live module). | Stale duplicate; a maintenance hazard. |
| 4 | **Build new Meals/Finance apps in separate apps** (`apps.meals`, `apps.finance`) with their own clean migration histories. | New apps don't inherit attendance's drift. |
| 5 | **New apps FK to `identity.Person` / `identity.StudentProfile`**, never to `attendance.Student`. | Avoids depending on the drifted attendance schema. |
| 6 | **New apps reference `attendance` models only through service boundaries** (function calls, not FKs or migration dependencies). | No migration-level coupling to the broken graph. |
| 7 | **Defer full attendance migration reconciliation to a dedicated future task** (Phase 0 of the identity-architecture migration plan). | It's a separate, high-risk effort that should not block Meals/Finance. |

---

## 8. Options

### Option A: Leave legacy attendance migrations alone (RECOMMENDED)

- **What:** Do not touch `apps.attendance/migrations/`. Do not re-run
  `migrate`. Do not `--fake` anything. Accept that the migration graph is
  dishonest about the DB but matches `models.py`.
- **Pros:** Zero risk to production; no migration changes; existing behavior
  continues.
- **Cons:** The drift remains; fresh-DB CI may differ from production; any
  future migration that touches drifted fields will fail.
- **Mitigation:** New apps don't depend on attendance migrations; CI uses
  the existing production DB (or a dump) for attendance-related tests.

### Option B: Create future compatibility migrations

- **What:** Add a new migration `0032_reconcile_drift` on `apps.attendance`
  that explicitly `RemoveField`s the `confirmed`/`lunch_*` columns from the
  migration state (bringing the graph in line with the DB), or `AddField`s
  them to the DB (bringing the DB in line with the graph).
- **Pros:** Makes the graph honest; `makemigrations --check` and fresh-DB CI
  become trustworthy.
- **Cons:** Risky — `RemoveField` on a non-existent column fails;
  `AddField` adds columns the code doesn't use. Requires careful
  `state_operations` vs `db_operations` separation. Must be tested on a
  staging DB first.
- **When to choose:** Only if a concrete need arises (e.g. CI divergence
  blocking merges, or a new attendance migration is required).

### Option C: Rebuild attendance app migrations later

- **What:** Squash all attendance migrations (`0001`–`0031`) into a single
  new `0001_initial` that honestly reflects the current `models.py` and DB,
  then `--fake`-apply it on production.
- **Pros:** Clean slate; no drift; future attendance migrations are safe.
- **Cons:** High risk; requires a production backup; requires verifying
  every table/index/constraint matches the new `0001`; `--fake` must be
  done carefully. This is the identity-architecture Phase 0 squash.
- **When to choose:** As a dedicated, separately-approved task with a
  tested rollback plan. **Not now.**

### Option D: Extract new Meals/Finance apps without touching attendance migrations (RECOMMENDED alongside Option A)

- **What:** Build `apps.meals` and `apps.finance` with their own
  `0001_initial` migrations, FK-ing to `identity.Person` /
  `identity.StudentProfile`. Do not add migrations to `apps.attendance`.
  The new apps coexist with the legacy lunch/wallet code.
- **Pros:** New development is unblocked; no risk to production; clean
  migration histories for new domains.
- **Cons:** Two parallel systems (legacy `MealSubscription`/`Wallet` in
  attendance + new meal-period subscription/`Wallet` in meals/finance) until
  migration is complete.
- **Migration path:** Dual-FK pattern when ready — the new models gain a
  nullable FK to the legacy attendance models for backfill, then the legacy
  FKs are dropped after verification.

---

## 9. Recommended Path for BISK_RFv4

**Combine Option A + Option D:**

1. **Freeze** the attendance migration graph (Option A). Do not reconcile
   drift now.
2. **Build** `apps.meals` and `apps.finance` as new apps with clean
   migration histories (Option D), FK-ing to `identity.Person` /
   `identity.StudentProfile`.
3. **Fix** `resources.py` to remove references to non-existent fields (a
   code fix, not a migration).
4. **Delete** `views_.py` (after confirming it's stale).
5. **Wrap** legacy lunch/wallet/dashboard logic with service-boundary
   adapters so new code calls `finance.charge()` / `meals.resolve_eligibility()`
   instead of touching `Wallet.balance_iqd` directly — but the legacy views
   keep working unchanged.
6. **Defer** the full attendance squash/reconciliation to a dedicated
   Phase 0 task (Option C) with a tested rollback plan, to be scheduled
   separately.

### Why this path

- It **unblocks** Meals/Finance development immediately.
- It **preserves** all working lunch/wallet/dashboard behavior.
- It **avoids** any risk to the production attendance schema.
- It **isolates** the drift to `apps.attendance` — new apps don't inherit it.
- It **prepares** for the eventual squash by establishing clean new
  migration histories that don't depend on the broken graph.

---

## 10. How New Apps Should Avoid Depending on Broken Legacy Migration History

| Rule | Enforcement |
|---|---|
| **New apps (`apps.meals`, `apps.finance`) must not include `apps.attendance` in their migration `dependencies`.** | New migrations depend only on `apps.identity` (and `apps.academics` when it exists). |
| **New apps must not FK to `attendance.Student`, `attendance.Wallet`, or `attendance.MealSubscription` in their models.** | New models FK to `identity.Person` / `identity.StudentProfile`. If a link to a legacy model is needed during migration, use a nullable FK with `on_delete=SET_NULL` and `db_constraint=False`, or use a generic reference (`source_module` + `reference_id`). |
| **New apps may *call* attendance services** (e.g. reading an `AttendanceEvent`) **but must not *depend on* attendance migrations.** | Service-boundary calls (function imports) are allowed; migration-level dependencies are not. |
| **New apps' tests must not require the attendance migration graph to be honest.** | Test with the actual DB state (or a fixture) rather than assuming `migrate` produces the production schema. |
| **New apps' CI should verify against a production-DB dump or a fixture**, not a fresh `migrate`, if attendance schema matters. | Prevents CI-vs-production divergence. |

---

## 11. Rollback / Safety Notes

| Scenario | Rollback |
|---|---|
| **A new `apps.meals`/`apps.finance` migration breaks something.** | Un-apply only the new app's migrations (`migrate apps.meals zero`). The attendance DB is untouched. |
| **A service-boundary adapter breaks the dashboard.** | Revert the adapter; the legacy `views.py` code path still works (it was never deleted). |
| **`resources.py` fix breaks export.** | Revert the file; the old (broken) state is no worse than before. |
| **`views_.py` deletion breaks an import.** | Re-add the file; it was a stale duplicate, so restoring it is safe. |
| **Production DB is corrupted during a future squash.** | Restore from the pre-squash backup (taken before any `--fake` operation). This is why the squash is deferred. |

### Golden rules

1. **Never `--fake` a migration on production without a backup.**
2. **Never delete a row from `django_migrations` on production.**
3. **Never run `migrate attendance zero` on production.**
4. **Always test a new migration on a staging DB (a dump of production) first.**
5. **If in doubt, leave it alone and build new apps instead.**

---

## 12. Open Questions

1. **Should `resources.py` be fixed now or deferred?** The export is
   broken (references non-existent fields), but it may not be actively
   used. Recommendation: fix it now (remove the fields or guard with
   `getattr`); it's a safe code change.

2. **Should `views_.py` be deleted?** It's a stale duplicate of `views.py`
   missing only the `meal_bucket` filter. Confirm it's not imported
   anywhere, then delete. Check `urls.py` imports.

3. **Does CI use a fresh `migrate` or a production dump?** If fresh, CI
   has the squash-honest schema (with `confirmed`/`lunch_*`), which differs
   from production. This affects whether attendance-related tests are
   trustworthy.

4. **Are there other drifted tables beyond `AttendanceRecord`?** The
   investigation focused on AR; a full audit (Section 6.2) should check
   every attendance model before the eventual squash.

5. **When should the full attendance squash (Option C) be scheduled?**
   Recommendation: after Identity + Academics + Meals + Finance foundations
   are in place and the dual-FK migration is verified. The squash is a
   standalone, high-risk task.

6. **Should the `LunchSubscription` → `MealSubscription` rename be
   reconciled?** Migration 0028 used `DeleteModel` + `CreateModel` rather
   than `RenameModel`. If any `LunchSubscription` data existed, it was
   lost. Confirm whether the table ever had data before the eventual
   squash.

7. **Should the `django_migrations` table have both `0001_initial` and
   `0001_squashed_...` rows?** The squash `replaces` the originals, so
   having both is unusual. The squash row may have been `--fake`-applied
   while the original rows were left in place. This should be cleaned up
   during the eventual squash, not now.

---

## 13. Final Recommendation Before Meals Coding

> **Do not reconcile the attendance migration drift now. Build `apps.meals`
> and `apps.finance` as new apps with clean migration histories that FK to
> `identity.Person` / `identity.StudentProfile`. Freeze the attendance
> migration graph. Fix `resources.py` and delete `views_.py` as safe code
> changes. Defer the full attendance squash to a dedicated future task.**

### Pre-Meals checklist

- [ ] `resources.py` fixed (remove non-existent field references).
- [ ] `views_.py` deleted (after confirming no imports).
- [ ] `apps.meals` and `apps.finance` app skeletons created with clean
      `0001_initial` migrations.
- [ ] New app migrations depend only on `apps.identity` (and
      `apps.academics`), never on `apps.attendance`.
- [ ] New app models FK to `Person` / `StudentProfile`, never to
      `attendance.Student`.
- [ ] Legacy `views.py` lunch/wallet/dashboard behavior left unchanged.
- [ ] Service-boundary adapters planned but not yet required for the
      initial `apps.meals`/`apps.finance` `0001_initial` (they're needed
      when live charges begin, not when models are created).
- [ ] Attendance migration graph frozen — no new attendance migrations
      unless purely additive and not referencing drifted fields.

### What this enables

- Meals and Finance domain development can proceed immediately.
- The legacy lunch/wallet/dashboard continues to work in production.
- The migration drift is contained to `apps.attendance` and does not
  contaminate new apps.
- The eventual squash (Option C) can be scheduled as a dedicated task
  with a tested rollback plan, unblocked by Meals/Finance pressure.

---

End of document.
