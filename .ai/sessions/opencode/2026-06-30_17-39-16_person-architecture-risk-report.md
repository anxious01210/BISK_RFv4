# OpenCode Session — Person Architecture Risk Report

Date: 2026-06-30
Model: big-pickle
Branch: feature/person-architecture
Mode: read-only

## Objective

Produce a comprehensive migration risk report for introducing a Person architecture (Person, PersonRole, StudentProfile, StaffProfile) into BISK_RFv4. The report covers: current auth model usage, Student model fields/relationships, Wallet/Lunch model relationships, migration risks, phased implementation plan, affected files, affected database tables, and backward compatibility strategy.

## Findings

- **AUTH_USER_MODEL is not set.** The project uses Django's default `auth.User`. No custom user model exists. Three FK fields reference `settings.AUTH_USER_MODEL`: `WalletTransaction.created_by`, `MealRecord.confirmed_by`, `MealRecord.reversed_by`.
- **Student has NO FK/O2O to `auth.User`.** Student is completely independent of the user model.
- **6 models have direct FK/O2O to Student:** `AttendanceRecord`, `AttendanceEvent`, `FaceEmbedding`, `MealSubscription`, `Wallet`, `WalletTransaction`.
- **Authorization is group-based** with three hardcoded groups: `supervisor`, `meal_supervisor`, `api_user`. No custom permissions, no `has_perm()` calls, no template-level perms.
- **30+ existing migrations** all reference the `attendance_student` table.
- **Student fields:** `h_code` (unique), `is_active`, `first_name`, `middle_name`, `last_name`, `gender`, `grade`, `has_meal`, `has_bus`.

## Risks

### HIGH
- Setting `AUTH_USER_MODEL` after migrations exist is irreversible without complex db rename.
- 6 models with direct Student FKs require dual-write migration with data migration per model.
- 30+ existing migrations create migration chain complexity when introducing new models.
- `MealSubscription.clean()` uses `self.student_id` directly and `Wallet.Meta.ordering` uses `student__h_code`.
- Template `_meal_rows.html` accesses `r.student.*` in multiple places.
- Views use `rec.student` and `student=rec.student` throughout.

### MEDIUM
- `has_meal` is a denormalized field on Student synced by signal.
- `h_code` uniqueness currently on Student — must be re-evaluated for Person.
- Admin inlines, import/export resources tied to Student model.
- 5 external scripts in `extras/` use `Student.objects.get(h_code=...)`.

### LOW
- Management commands, serializers, services, scheduler — surface-level Student references.

## Decisions

1. **Do NOT introduce Person as AUTH_USER_MODEL.** Keep `auth.User` as the canonical auth model. Person gets an optional `OneToOneField('auth.User')`. This avoids the irreversible AUTH_USER_MODEL migration trap.
2. **Use dual-FK pattern during migration.** Add `person_id` FK alongside existing `student_id` in each model. Populate both, then switch views to Person, then drop `student_id` in a later release.
3. **Squash migrations first.** Combine 30+ existing migrations into a single squashed migration before adding Person models.
4. **Incremental per-model migration.** Migrate FK targets one model at a time: Wallet → WalletTransaction → MealSubscription → AttendanceRecord → AttendanceEvent → FaceEmbedding.
5. **Keep Student table through Phase 4.** Do not drop Student until the parallel Person-based code has been running for at least one release cycle.

## Implementation

No implementation was performed. This was a read-only analysis session.

## Remaining Work

1. Squash existing migrations into a clean base.
2. Create Person model with shared fields and optional user FK.
3. Create StudentProfile, StaffProfile, PersonRole models.
4. Add data migration to create Person+StudentProfile for every existing Student.
5. Migrate FK targets one model at a time with dual-write.
6. Update all code paths (views, admin, templates, services, API, serializers, signals, utils, management commands, extras scripts).
7. Deprecate and drop Student table.
8. Update all documentation.

## Files Modified

None — read-only analysis.

## Commands Executed

- `git status`
- `git branch --show-current`
- Various `grep`, `glob`, `read`, and `task` calls for codebase exploration.

## Recommendations

1. **Do not change AUTH_USER_MODEL.** Keep auth.User forever.
2. **Squash migrations before introducing Person.**
3. **Use dual-FK pattern** to allow zero-downtime migration of each FK model.
4. **Add backward-compat properties** (`Student.person`, `AttendanceRecord.student` as cached_property) so existing code continues working.
5. **Migrate one FK model at a time** — do not attempt a "big bang" migration.
6. **Run with both schemas for at least one release cycle** before dropping the Student table.
