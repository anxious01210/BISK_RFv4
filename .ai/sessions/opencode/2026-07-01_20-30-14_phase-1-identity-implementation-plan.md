# Phase 1 Implementation Plan — Person Identity Layer

**Date:** 2026-07-01
**Project:** BISK_RFv4
**Branch:** `feature/person-architecture`
**Tags:** `architecture-approved-v1` `pre-squash-clean-db`
**Status:** Plan — awaiting approval

---

## 1. Current Model Inventory (Phase-1-Relevant)

### Models in `apps/attendance/models.py` that reference `Student`

| Model | FK to Student | Cardinality | Lines | Notes |
|-------|--------------|-------------|-------|-------|
| `Student` | — (self) | — | 36–93 | 721 rows, standalone model |
| `AttendanceRecord` | `student` | FK | 173 | unique_together=(student, period) |
| `AttendanceEvent` | `student` | FK | 203 | no unique constraint |
| `FaceEmbedding` | `student` | FK | 357 | unique constraint `uniq_active_embedding_per_student` |
| `MealSubscription` | `student` | FK | 506–7 | clean() does overlap check on `student_id` |
| `Wallet` | `student` | O2O | 635–6 | ordering=`student__h_code` |
| `WalletTransaction` | `student` | FK | 673–7 | also has `wallet` FK |
| `MealRecord` | none (direct) | — | 874+ | indirect via `attendance_record.student` |

### auth.User
- **4 users:** `mehdi` (superuser), `api`, `lunch`, `sibtain` (all is_staff)
- No `AUTH_USER_MODEL` override — Django default `auth.User`
- No relationship to `Student` exists

### Other data
- **Students:** 721 rows
- **No Person, RoleType, PersonRole, StudentProfile, StaffProfile** tables exist

### Key constraint requiring migration-time care
- `FaceEmbedding`: `UniqueConstraint(fields=("student",), condition=Q(is_active=True), name="uniq_active_embedding_per_student")`

---

## 2. Exact Files Likely Modified (Phase 1 only — additive)

| File | Change | Phase 1 Scope |
|------|--------|---------------|
| `apps/attendance/models.py` | Append new models (Person, RoleType, PersonRole, StudentProfile, StaffProfile) at end of file | Required |
| `apps/attendance/admin.py` | Append new ModelAdmins (PersonAdmin, StudentProfileAdmin, StaffProfileAdmin, RoleTypeAdmin, PersonRoleAdmin) | Required |
| `apps/attendance/signals.py` | Add `post_save` signals: auth.User ↔ Person sync | Strongly recommended |
| `apps/attendance/resources.py` | Add `PersonResource`, `StudentProfileResource` (optional — not needed in Phase 1) | Optional |
| `.ai/sessions/opencode/` | This plan report | Required |
| *(migrations)* | 4 new migrations (0032–0035) | Auto-generated |

### Files NOT modified in Phase 1
- `apps/attendance/services.py` — preserves Student imports
- `apps/attendance/utils/meal.py` — preserves Student imports
- `apps/attendance/api.py` — no change
- `apps/cameras/models.py` — no change
- `apps/scheduler/` — no change
- `bisk/settings.py` — no change (AUTH_USER_MODEL stays default)
- `extras/*.py` — no change

---

## 3. Exact Migration Sequence

| # | Migration Name | Type | Content |
|---|---------------|------|---------|
| `0032` | `add_person_role_profile_models` | Auto-generated `CreateModel` | Person, RoleType, PersonRole, StudentProfile, StaffProfile |
| `0033` | `seed_role_types` | `RunPython` | Insert 6 RoleType rows (student, staff, teacher, parent, guest, vendor) |
| `0034` | `migrate_student_to_person` | `RunPython` | For each Student → 1 Person + 1 StudentProfile + 1 PersonRole(student) |
| `0035` | `link_users_to_person` | `RunPython` | For each auth.User → lookup or create Person, set `person.user` |

### Dependency chain
```
0031 (current head)
  └── 0032 (schema: new tables)
        └── 0033 (seed data)
              └── 0034 (student data migration)
                    └── 0035 (user link migration)
```

### Why 4 migrations instead of 1
- **0032** is pure schema — zero data risk, fast to apply/revert
- **0033** is independent seed data — can be rerun safely
- **0034** is the bulk data migration — largest, most review-worthy, most likely to need debugging
- **0035** is a separate concern (staff users are a different population from students)

---

## 4. Recommended Model Fields

### Person

```python
class Person(models.Model):
    # --- Identity ---
    h_code = models.CharField(max_length=32, unique=True, db_index=True,
                              help_text="Unique human-readable identifier. Replaces Student.h_code.")
    first_name = models.CharField(max_length=100, blank=True, default="")
    middle_name = models.CharField(max_length=100, blank=True, default="")
    last_name = models.CharField(max_length=100, blank=True, default="")
    gender = models.CharField(max_length=6,
                              choices=[("MALE", "male"), ("FEMALE", "female")],
                              blank=True, null=True, db_index=True)
    date_of_birth = models.DateField(blank=True, null=True)

    # --- Contact ---
    email = models.EmailField(blank=True, default="")
    phone = models.CharField(max_length=20, blank=True, default="")
    address = models.TextField(blank=True, default="")

    # --- Photo ---
    photo = models.ImageField(upload_to="person_photos/", blank=True)

    # --- Status ---
    is_active = models.BooleanField(default=True, db_index=True)

    # --- Auth bridge (Phase 1 only — removed in Phase 2) ---
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="person",
    )

    # --- Multi-school / ERP future ---
    ext_id = models.CharField(max_length=64, blank=True, default="")
    school = models.ForeignKey(
        "attendance.School",  # future model
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="persons",
    )

    # --- Metadata ---
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

### RoleType

```python
class RoleType(models.Model):
    ROLE_STUDENT = "student"
    ROLE_STAFF = "staff"
    ROLE_TEACHER = "teacher"
    ROLE_PARENT = "parent"
    ROLE_GUEST = "guest"
    ROLE_VENDOR = "vendor"

    ROLE_CHOICES = [ ... ]

    code = models.CharField(max_length=32, unique=True, choices=ROLE_CHOICES)
    name = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)
```

### PersonRole

```python
class PersonRole(models.Model):
    person = models.ForeignKey(Person, CASCADE, related_name="roles")
    role_type = models.ForeignKey(RoleType, PROTECT, related_name="person_roles")
    academic_year = models.ForeignKey("AcademicYear", CASCADE, null=True, blank=True)  # future
    school = models.ForeignKey("School", CASCADE, null=True, blank=True)  # future
    is_active = models.BooleanField(default=True, db_index=True)
    notes = models.CharField(max_length=200, blank=True, default="")
    assigned_at = models.DateTimeField(auto_now_add=True)
    assigned_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL, null=True, blank=True,
    )
```

### StudentProfile

```python
class StudentProfile(models.Model):
    person = models.OneToOneField(
        Person, CASCADE, primary_key=True, related_name="student_profile",
    )
    grade = models.CharField(max_length=32, blank=True, null=True, db_index=True)
    homeroom = models.CharField(max_length=64, blank=True, default="")
    has_meal = models.BooleanField(default=False, db_index=True)
    has_bus = models.BooleanField(default=False, db_index=True)
    legacy_student = models.OneToOneField(
        "attendance.Student", on_delete=SET_NULL,
        null=True, blank=True, related_name="migrated_to",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

### StaffProfile

```python
class StaffProfile(models.Model):
    person = models.OneToOneField(
        Person, CASCADE, primary_key=True, related_name="staff_profile",
    )
    employee_id = models.CharField(max_length=32, blank=True, default="")
    job_title = models.CharField(max_length=100, blank=True, default="")
    department = models.CharField(max_length=100, blank=True, default="")
    hire_date = models.DateField(null=True, blank=True)
    is_teacher = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

---

## 5. Recommended Constraints and Indexes

### Person
| Type | Definition | Rationale |
|------|-----------|-----------|
| UNIQUE | `h_code` | Natural key, replaces Student.h_code uniqueness |
| INDEX | `is_active` | Soft-delete filtering |
| INDEX | `(last_name, first_name)` | Common admin search pattern |

### RoleType
| Type | Definition | Rationale |
|------|-----------|-----------|
| UNIQUE | `code` | Machine key for lookups |

### PersonRole
| Type | Definition | Rationale |
|------|-----------|-----------|
| UNIQUE | `(person, role_type, academic_year, school)` | Prevent duplicate role assignment |
| INDEX | `(person, is_active)` | "Find all active roles for this person" |
| INDEX | `(role_type, is_active)` | "Find all people with this role" |

### StudentProfile
| Type | Definition | Rationale |
|------|-----------|-----------|
| PK | `person_id` | Shared PK with Person (OneToOneField primary_key=True) |
| INDEX | `grade` | Filtering |
| INDEX | `has_meal` | Meal eligibility filtering |
| INDEX | `has_bus` | Bus filtering |

### StaffProfile
| Type | Definition | Rationale |
|------|-----------|-----------|
| PK | `person_id` | Shared PK with Person |

### Meta for all models
- `ordering = ["h_code"]` on Person
- `ordering = ["code"]` on RoleType
- `ordering = ["person", "role_type"]` on PersonRole

---

## 6. Data Migration Algorithm

### Migration 0033: Seed RoleTypes

```
for each (code, name) in:
    student → Student
    staff → Staff
    teacher → Teacher
    parent → Parent
    guest → Guest
    vendor → Vendor
    RoleType.objects.get_or_create(code=code, defaults={'name': name})
```

Idempotent — safe to run multiple times.

### Migration 0034: Student → Person + StudentProfile + PersonRole

```
student_role = RoleType.objects.get(code='student')

for each student in Student.objects.iterator():
    person = Person.objects.create(
        h_code=student.h_code,
        first_name=student.first_name,
        middle_name=student.middle_name,
        last_name=student.last_name,
        gender=student.gender,
        is_active=student.is_active,
    )
    StudentProfile.objects.create(
        person=person,
        grade=student.grade,
        has_meal=student.has_meal,
        has_bus=student.has_bus,
        legacy_student=student,
    )
    PersonRole.objects.create(
        person=person,
        role_type=student_role,
        is_active=student.is_active,
    )
```

**Edge cases handled:**
- 0 students → nothing created, no error
- Duplicate h_code → impossible (Student.h_code has UNIQUE constraint)
- Null grade/gender → passed through cleanly

**Performance:** 721 rows. Single transaction is fine. Add `--atomic` to migration.

### Migration 0035: Link auth.User → Person

```
student_role = RoleType.objects.get(code='student')
staff_role = RoleType.objects.get(code='staff')

for each user in User.objects.all():
    # Strategy 1: match by username as h_code
    person = Person.objects.filter(h_code__iexact=user.username).first()

    if not person:
        # Strategy 2: create a new Person for staff users
        person = Person.objects.create(
            h_code=f"USER_{user.id}",      # temporary — can be renamed later
            first_name=user.first_name or "",
            last_name=user.last_name or "",
            email=user.email or "",
            is_active=user.is_active,
        )
        # Auto-create StaffProfile + staff role
        StaffProfile.objects.get_or_create(person=person)
        PersonRole.objects.get_or_create(
            person=person,
            role_type=staff_role,
            defaults={'is_active': True}
        )

    # Link
    person.user = user
    person.save(update_fields=['user'])
```

---

## 7. Admin Strategy

### New registrations (all in `apps/attendance/admin.py`)

```python
@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ("h_code", "full_name_display", "gender", "is_active", "has_user")
    search_fields = ("h_code", "first_name", "middle_name", "last_name")
    list_filter = ("is_active", "gender")
    readonly_fields = ("created_at", "updated_at")
    # No inlines in Phase 1 — keep it simple

    def full_name_display(self, obj):
        return obj.full_name()
    full_name_display.short_description = "Full name"

    def has_user(self, obj):
        return obj.user_id is not None
    has_user.boolean = True
    has_user.short_description = "Has login"

@admin.register(StudentProfile)
class StudentProfileAdmin(admin.ModelAdmin):
    list_display = ("person", "grade", "has_meal", "has_bus", "legacy_student")
    search_fields = ("person__h_code", "person__first_name", "person__last_name")
    list_filter = ("has_meal", "has_bus", "grade")
    # Read-only since data is owned by Student model during Phase 1
    readonly_fields = ("person", "legacy_student")

@admin.register(StaffProfile)
class StaffProfileAdmin(admin.ModelAdmin):
    list_display = ("person", "employee_id", "job_title", "department", "is_teacher")
    search_fields = ("person__h_code", "person__first_name", "person__last_name", "employee_id")
    list_filter = ("is_teacher",)

@admin.register(RoleType)
class RoleTypeAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "is_active")
    list_editable = ("is_active",)

@admin.register(PersonRole)
class PersonRoleAdmin(admin.ModelAdmin):
    list_display = ("person", "role_type", "is_active", "assigned_at")
    list_filter = ("role_type", "is_active")
    search_fields = ("person__h_code",)
    readonly_fields = ("assigned_at",)
```

### What stays unchanged
- `StudentAdmin` — fully active, all actions, inlines, resources
- `MealSubscriptionAdmin`, `WalletAdmin`, `WalletTransactionAdmin` — all student references untouched
- `AttendanceRecordAdmin`, `AttendanceEventAdmin`, `FaceEmbeddingAdmin` — all student references untouched

### Rationale
Phase 1 is **observability-only** for Person. Operators continue to use `StudentAdmin` for all workflows. Person/Profile data is visible for verification and debugging only.

---

## 8. Backward Compatibility Strategy

### Principle
**All existing Student-based code continues working identically. Zero regressions.**

| Area | Strategy |
|------|----------|
| **Models** | `Student` model is untouched. No FK fields added to it. |
| **Queries** | All `Student.objects.*`, `attendance_record.student.*` work as before. |
| **Admin** | `StudentAdmin` unchanged. Phase 1 Person admins are additive. |
| **Signals** | `meal_subscription_saved` still uses `instance.student_id`. Works as before. |
| **Utils** | `recalc_meal_flags_for_students(Student.objects.all())` works unchanged. |
| **Auth** | `request.user` is still `auth.User`. No middleware changes. |
| **API** | No API changes in Phase 1. |
| **Resources** | `StudentResource`, `AttendanceRecordResource` unchanged. |
| **Management commands** | All `Student` imports continue working. |
| **Extras scripts** | No changes. |

### The `legacy_student` backlink
`StudentProfile.legacy_student` points back to the original `Student` row. This enables:
- Reverse lookup: `student.migrated_to.person` (if Student model has `related_name="migrated_to"`)
- Forward lookup: `profile.legacy_student` → original Student

### Sync signals (recommended)
Add these to `signals.py` to prevent drift between Student↔Person during Phase 1:

```python
# When Student is saved, sync to linked Person
@receiver(post_save, sender=Student)
def sync_student_to_person(sender, instance, **kwargs):
    try:
        profile = instance.migrated_to  # StudentProfile via legacy_student backlink
        person = profile.person
        changed = False
        for field in ('h_code', 'first_name', 'middle_name', 'last_name', 'gender', 'is_active'):
            if getattr(person, field) != getattr(instance, field):
                setattr(person, field, getattr(instance, field))
                changed = True
        if changed:
            person.save(update_fields=['h_code', 'first_name', 'middle_name', 'last_name', 'gender', 'is_active', 'updated_at'])
    except StudentProfile.DoesNotExist:
        pass  # No Person link yet (pre-migration)
```

---

## 9. Risk List

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | **Student.h_code is not unique** | Very low | High | Impossible — Student has DB-level UNIQUE constraint |
| 2 | **Name drift** between Student↔Person | Medium | Low | Add sync signal on Student post_save |
| 3 | **Admin confusion** (two systems) | Medium | Low | Clear documentation; Phase 1 is additive |
| 4 | **Signal recursion** (User↔Person) | Medium | Medium | Guard conditions, `update_fields`, exclude from recursion |
| 5 | **Large migration performance** | Very low | Low | 721 students — completes in <1s |
| 6 | **Missing `date_of_birth`/`email`/`phone`** on Student | Certain | None | Person fields are blank/default — no data loss |
| 7 | **Migration ordering conflict** with future FK migrations | Low | Medium | Phase 1.5 migrations must depend on 0034 |
| 8 | **Accidental import of new models** breaks existing code | Low | Low | New models are at bottom of models.py; existing imports unchanged |

---

## 10. Rollback Plan

### Full rollback (safe at any time)
```bash
python manage.py migrate attendance 0031
python manage.py check
git checkout -- apps/attendance/models.py apps/attendance/admin.py
```

This drops all 5 new tables (Person, RoleType, PersonRole, StudentProfile, StaffProfile) and any data in them. Existing Student data is untouched.

### Partial rollback during development
```bash
# Revert schema only (0032)
python manage.py migrate attendance 0031

# Revert seed data (0033)
python manage.py migrate attendance 0032

# Revert migration 0034 (data migration) — will error if 0035 has been applied
python manage.py migrate attendance 0033    # drops Person/Profile/Role rows
```

### Data integrity on rollback
- Student table: **untouched** in all Phase 1 migrations
- auth.User: **untouched** (we only set `person.user`, which is a nullable FK — dropping Person table cascades to null)
- Wallet, meals, attendance: **untouched**

---

## 11. Verification Commands

### Pre-migration sanity
```bash
python manage.py check
python manage.py makemigrations --check --dry-run
git status
git diff --stat
```

### Apply migrations
```bash
python manage.py migrate attendance 0032
python manage.py migrate attendance 0033
python manage.py migrate attendance 0034
python manage.py migrate attendance 0035
python manage.py check
```

### Data integrity verification
```bash
python manage.py shell -c "
from apps.attendance.models import Person, StudentProfile, PersonRole, Student
from django.contrib.auth.models import User

p_count = Person.objects.count()
sp_count = StudentProfile.objects.count()
s_count = Student.objects.count()
r_count = PersonRole.objects.count()

print(f'Persons: {p_count}')
print(f'StudentProfiles: {sp_count}')
print(f'Students (unchanged): {s_count}')
print(f'PersonRoles: {r_count}')

# All students got a Person
assert p_count == s_count, f'Person count {p_count} != Student count {s_count}'
# All students got a StudentProfile
assert sp_count == s_count, f'StudentProfile count {sp_count} != Student count {s_count}'
# All StudentProfiles have a student role
student_roles = PersonRole.objects.filter(role_type__code='student').count()
assert student_roles == s_count, f'Student roles {student_roles} != Student count {s_count}'

# Verify legacy_student backlinks
orphans = StudentProfile.objects.filter(legacy_student__isnull=True)
assert orphans.count() == 0, f'{orphans.count()} StudentProfiles missing legacy_student'

# Verify auth.User links
linked = User.objects.filter(person__isnull=False).count()
total_users = User.objects.count()
print(f'Users linked to Person: {linked}/{total_users}')

print('ALL CHECKS PASSED')
"
```

### Rollback verification
```bash
python manage.py migrate attendance 0031
python manage.py check
python manage.py migrate
python manage.py check
```

---

## 12. Recommended Implementation Split (Small Commits)

Each commit is independently reviewable and individually revertible.

### Commit 1 — `feat: add Person, RoleType, PersonRole, StudentProfile, StaffProfile models`
- Append model classes to `apps/attendance/models.py`
- `__str__`, `full_name()`, `Meta` (ordering, indexes)
- Auto-generated migration `0032`

### Commit 2 — `feat: seed RoleType values`
- Migration `0033` with `RunPython(seed_role_types)`
- 6 roles: student, staff, teacher, parent, guest, vendor

### Commit 3 — `feat: data migrate existing Student records to Person + StudentProfile + PersonRole`
- Migration `0034` with `RunPython(migrate_students_to_persons)`
- Each Student → 1 Person + 1 StudentProfile + 1 PersonRole(student)
- Verify counts match

### Commit 4 — `feat: link existing auth.User accounts to Person`
- Migration `0035` with `RunPython(link_users_to_persons)`
- Match by username-as-h_code strategy, fallback to create Person

### Commit 5 — `feat: register Person, StudentProfile, StaffProfile, RoleType, PersonRole in admin`
- Append ModelAdmin classes to `apps/attendance/admin.py`
- All read-only where appropriate
- StudentAdmin left unchanged

### Commit 6 (optional) — `feat: add auth.User ↔ Person sync signals`
- Add `post_save` signals to `apps/attendance/signals.py`
- Sync Student → Person name fields
- Guard against recursion

### Merge strategy
All commits on `feature/person-architecture`. After review, the branch can be merged to main with a merge commit summarizing the 5–6 changes.

---

## Appendix A: Field Mapping (Student → Person + StudentProfile)

| Student field | Target | Notes |
|--------------|--------|-------|
| `h_code` | `Person.h_code` | Direct copy |
| `first_name` | `Person.first_name` | Direct copy |
| `middle_name` | `Person.middle_name` | Direct copy |
| `last_name` | `Person.last_name` | Direct copy |
| `gender` | `Person.gender` | Direct copy |
| `is_active` | `Person.is_active` | Direct copy |
| `grade` | `StudentProfile.grade` | Direct copy (temporary — will move to Enrollment) |
| `has_meal` | `StudentProfile.has_meal` | Direct copy |
| `has_bus` | `StudentProfile.has_bus` | Direct copy |
| *no equivalent* | `Person.date_of_birth` | Blank — collected later |
| *no equivalent* | `Person.email` | Blank — collected later |
| *no equivalent* | `Person.phone` | Blank — collected later |
| *no equivalent* | `Person.photo` | Blank — future portrait |
| *no equivalent* | `Person.user` | Set in migration 0035 for staff users |
| *no equivalent* | `Person.ext_id` | Blank — future ERP integration |
| *no equivalent* | `Person.school` | Null — future multi-tenant |
| *no equivalent* | `StudentProfile.homeroom` | Blank — future use |
| *no equivalent* | `StudentProfile.legacy_student` | Set to original Student instance |

## Appendix B: Dependency Order for Future Phases

```
Phase 1 (this plan)
  └── Phase 1.5: Wallet.person → starts here
  └── Phase 1.5: MealSubscription.person → starts here
  └── Phase 1.5: FaceEmbedding.person → starts here
  └── Phase 1.5: AttendanceRecord.person → starts here
  └── Phase 1.5: AttendanceEvent.person → starts here
        └── Phase 2: Deprecate Student → requires all Phase 1.5 FK migrations done
              └── Phase 3: AcademicYear + Enrollment → requires Phase 2 done
```

## Appendix C: Prerequisites Before Starting Commit 1

- [x] Architecture docs approved and tagged
- [x] Migrations squashed (0001_squashed_0021 exists)
- [x] Database backed up and verified
- [x] Django checks pass (`python manage.py check` — 0 issues)
- [x] No pending migrations (`makemigrations --check --dry-run` — no changes)
- [x] This plan reviewed and approved
