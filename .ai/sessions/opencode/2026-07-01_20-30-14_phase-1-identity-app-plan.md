# Phase 1 Implementation Plan (Revised) — Identity App

**Date:** 2026-07-01
**Project:** BISK_RFv4
**Branch:** `feature/person-architecture`
**Tags:** `architecture-approved-v1` `pre-squash-clean-db`
**Status:** Plan revision — identity lives in `apps/identity/`

---

## 1. Current Model Inventory (Phase-1-Relevant)

### Models in `apps/attendance/models.py` that reference `Student` — unchanged

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

### Existing app structure
```
apps/
├── __init__.py
├── attendance/     (models: Student, AttendanceRecord, FaceEmbedding, Wallet, MealSubscription, …)
├── cameras/         (models: Camera)
└── scheduler/       (models: PeriodTemplate, PeriodOccurrence, …)
```

No `apps/identity/` directory exists yet.

---

## 2. Creating `apps/identity`

### Directory structure to create

```
apps/identity/
├── __init__.py
├── apps.py              # IdentityConfig
├── models.py            # Person, RoleType, PersonRole, StudentProfile, StaffProfile
├── admin.py             # PersonAdmin, RoleTypeAdmin, PersonRoleAdmin, StudentProfileAdmin, StaffProfileAdmin
├── migrations/
│   ├── __init__.py
│   ├── 0001_initial.py          # auto-generated (CreateModel for all 5)
│   ├── 0002_seed_role_types.py   # RunPython
│   ├── 0003_migrate_students_to_persons.py  # RunPython
│   └── 0004_link_users_to_persons.py       # RunPython
├── tests.py              # empty for now
```

### `apps/identity/apps.py`

```python
from django.apps import AppConfig

class IdentityConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.identity"
    verbose_name = "Identity"
```

No `ready()` method in Phase 1 — signals are postponed (see Section 10).

---

## 3. Adding Identity to `INSTALLED_APPS`

### Edit `bisk/settings.py`

Add `"apps.identity"` to the `INSTALLED_APPS` block. Recommended placement — insert before `apps.attendance` because identity is a foundation dependency:

```python
INSTALLED_APPS += [
    "import_export",
    "rest_framework",
    "django_filters",
    "apps.cameras",
    "apps.scheduler",
    "apps.identity",                           # ← ADD (foundation app, before attendance)
    "apps.attendance.apps.AttendanceConfig",
    "django_completion",
]
```

`AUTH_USER_MODEL` remains unset. No other settings changes.

---

## 4. Model Fields — Revised

### Person — in `apps/identity/models.py`

```python
from django.db import models
from django.conf import settings

class Person(models.Model):
    # --- Identity ---
    h_code = models.CharField(max_length=32, unique=True, db_index=True,
                              help_text="Unique human-readable identifier.")
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

    # --- Multi-school / ERP future — NOT IMPLEMENTED IN PHASE 1 ---
    # School FK, ext_id — documented but not created.
    # ext_id will be added in a future phase when ERP integration is needed.
    # school FK will be added in Phase 5+ when Organization/School models exist.

    # --- Metadata ---
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["is_active"]),
            models.Index(fields=["last_name", "first_name"]),
        ]
        ordering = ["h_code"]

    def __str__(self):
        return f"{self.h_code} — {self.full_name()}"

    def full_name(self) -> str:
        parts = [self.first_name.strip(), self.middle_name.strip(), self.last_name.strip()]
        return " ".join(p for p in parts if p).strip() or self.h_code
```

**Changes from original plan:**
- `school` FK: **REMOVED** — will be added in Phase 5+
- `ext_id` field: **REMOVED** — will be added in a later phase when ERP integration is required

### RoleType — in `apps/identity/models.py`

```python
class RoleType(models.Model):
    ROLE_STUDENT = "student"
    ROLE_STAFF = "staff"
    ROLE_TEACHER = "teacher"
    ROLE_PARENT = "parent"
    ROLE_GUARDIAN = "guardian"
    ROLE_GUEST = "guest"
    ROLE_VENDOR = "vendor"
    ROLE_ADMINISTRATOR = "administrator"
    ROLE_FINANCE = "finance"
    ROLE_HR = "hr"
    ROLE_PRINCIPAL = "principal"
    ROLE_VICE_PRINCIPAL = "vice_principal"
    ROLE_LIBRARIAN = "librarian"
    ROLE_NURSE = "nurse"

    ROLE_CHOICES = [
        (ROLE_STUDENT, "Student"),
        (ROLE_STAFF, "Staff"),
        (ROLE_TEACHER, "Teacher"),
        (ROLE_PARENT, "Parent"),
        (ROLE_GUARDIAN, "Guardian"),
        (ROLE_GUEST, "Guest"),
        (ROLE_VENDOR, "Vendor"),
        (ROLE_ADMINISTRATOR, "Administrator"),
        (ROLE_FINANCE, "Finance"),
        (ROLE_HR, "HR"),
        (ROLE_PRINCIPAL, "Principal"),
        (ROLE_VICE_PRINCIPAL, "Vice Principal"),
        (ROLE_LIBRARIAN, "Librarian"),
        (ROLE_NURSE, "Nurse"),
    ]

    code = models.CharField(max_length=32, unique=True, choices=ROLE_CHOICES)
    name = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["code"]

    def __str__(self):
        return self.name
```

**Changes from original plan:**
- Added 8 roles: `guardian`, `administrator`, `finance`, `hr`, `principal`, `vice_principal`, `librarian`, `nurse`
- 14 seeded roles total (was 6)

### PersonRole — in `apps/identity/models.py`

```python
class PersonRole(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name="roles")
    role_type = models.ForeignKey(RoleType, on_delete=models.PROTECT, related_name="person_roles")
    is_active = models.BooleanField(default=True, db_index=True)
    notes = models.CharField(max_length=200, blank=True, default="")
    assigned_at = models.DateTimeField(auto_now_add=True)
    assigned_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL, null=True, blank=True,
    )

    # Future scoping fields — NOT IMPLEMENTED IN PHASE 1:
    # academic_year = ForeignKey(AcademicYear, null=True)  — Phase 3+
    # school = ForeignKey(School, null=True)               — Phase 5+

    class Meta:
        unique_together = [("person", "role_type")]
        indexes = [
            models.Index(fields=["person", "is_active"]),
            models.Index(fields=["role_type", "is_active"]),
        ]
        ordering = ["person", "role_type"]

    def __str__(self):
        return f"{self.person.h_code} → {self.role_type.code}"
```

**Changes from original plan:**
- `academic_year` FK: **REMOVED** — deferred to Phase 3+
- `school` FK: **REMOVED** — deferred to Phase 5+
- `unique_together` simplified to `(person, role_type)` only (without year/school)
- Documentation comment left for clarity

### StudentProfile — in `apps/identity/models.py`

```python
class StudentProfile(models.Model):
    person = models.OneToOneField(
        Person, on_delete=models.CASCADE,
        primary_key=True, related_name="student_profile",
    )
    grade = models.CharField(max_length=32, blank=True, null=True, db_index=True)
    homeroom = models.CharField(max_length=64, blank=True, default="")
    has_meal = models.BooleanField(default=False, db_index=True)
    has_bus = models.BooleanField(default=False, db_index=True)
    legacy_student = models.OneToOneField(
        "attendance.Student", on_delete=models.SET_NULL,
        null=True, blank=True, related_name="migrated_to",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["person__h_code"]

    def __str__(self):
        return f"Student: {self.person.h_code}"
```

**Note:** `legacy_student` uses the string `"attendance.Student"` — Django resolves this cross-app FK automatically as long as `apps.identity` appears after `apps.attendance` in `INSTALLED_APPS` (or Django can resolve it regardless). This is safe.

### StaffProfile — in `apps/identity/models.py`

```python
class StaffProfile(models.Model):
    person = models.OneToOneField(
        Person, on_delete=models.CASCADE,
        primary_key=True, related_name="staff_profile",
    )
    employee_id = models.CharField(max_length=32, blank=True, default="")
    job_title = models.CharField(max_length=100, blank=True, default="")
    hire_date = models.DateField(null=True, blank=True)
    is_teacher = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["person__h_code"]

    def __str__(self):
        return f"Staff: {self.person.h_code} ({self.job_title or 'No title'})"
```

**Changes from original plan:**
- `department` CharField: **REMOVED** — deferred to Phase 3+ when StaffAssignment exists
- StaffProfile is minimal: employee_id, job_title, hire_date, is_teacher only

---

## 5. Constraints and Indexes — Revised

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
| UNIQUE | `(person, role_type)` | Prevent duplicate role assignment per person |
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

---

## 6. Migration Sequence — Revised

| # | Migration Name | App | Type | Content |
|---|---------------|-----|------|---------|
| `identity.0001` | `initial` | identity | Auto-generated `CreateModel` | Person, RoleType, PersonRole, StudentProfile, StaffProfile |
| `identity.0002` | `seed_role_types` | identity | `RunPython` | Insert 14 RoleType rows |
| `identity.0003` | `migrate_students_to_persons` | identity | `RunPython` | For each Student → 1 Person + 1 StudentProfile + 1 PersonRole(student) |
| `identity.0004` | `link_users_to_persons` | identity | `RunPython` | For each auth.User → lookup or create Person, set `person.user` |

No migrations are added to the `attendance` app. All Phase 1 migrations live in `apps/identity/migrations/`.

### Dependency chain
```
identity.0001 (schema: new tables)
  └── identity.0002 (seed data)
        └── identity.0003 (student data migration)
              └── identity.0004 (user link migration)
```

No dependency on `attendance` migrations — the data migrations reference `attendance.Student` and `auth.User` via `apps.get_model`.

---

## 7. Data Migration Algorithm — Revised

### Migration `identity.0002`: Seed RoleTypes

```python
def seed_role_types(apps, schema_editor):
    RoleType = apps.get_model("identity", "RoleType")
    roles = [
        ("student", "Student"),
        ("staff", "Staff"),
        ("teacher", "Teacher"),
        ("parent", "Parent"),
        ("guardian", "Guardian"),
        ("guest", "Guest"),
        ("vendor", "Vendor"),
        ("administrator", "Administrator"),
        ("finance", "Finance"),
        ("hr", "HR"),
        ("principal", "Principal"),
        ("vice_principal", "Vice Principal"),
        ("librarian", "Librarian"),
        ("nurse", "Nurse"),
    ]
    for code, name in roles:
        RoleType.objects.get_or_create(code=code, defaults={"name": name})
```

Idempotent — safe to run multiple times.

### Migration `identity.0003`: Student → Person + StudentProfile + PersonRole

```python
def migrate_students_to_persons(apps, schema_editor):
    Student = apps.get_model("attendance", "Student")
    Person = apps.get_model("identity", "Person")
    StudentProfile = apps.get_model("identity", "StudentProfile")
    PersonRole = apps.get_model("identity", "PersonRole")
    RoleType = apps.get_model("identity", "RoleType")

    student_role = RoleType.objects.get(code="student")

    for student in Student.objects.iterator():
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

**Cross-app import:** `apps.get_model("attendance", "Student")` works because `attendance` is a registered app.

**Edge cases:**
- 0 students → nothing created, no error
- Duplicate h_code → impossible (Student.h_code has UNIQUE constraint)
- Null grade/gender → passed through cleanly
- 721 students → single transaction, expected time < 1s

### Migration `identity.0004`: Link auth.User → Person

```python
def link_users_to_persons(apps, schema_editor):
    User = apps.get_model("auth", "User")
    Person = apps.get_model("identity", "Person")
    RoleType = apps.get_model("identity", "RoleType")
    StaffProfile = apps.get_model("identity", "StaffProfile")
    PersonRole = apps.get_model("identity", "PersonRole")

    staff_role = RoleType.objects.get(code="staff")

    for user in User.objects.all():
        # Strategy 1: match by username as h_code
        person = Person.objects.filter(h_code__iexact=user.username).first()

        if not person:
            # Strategy 2: create a new Person for staff users
            person = Person.objects.create(
                h_code=f"USER_{user.id}",
                first_name=user.first_name or "",
                last_name=user.last_name or "",
                email=user.email or "",
                is_active=user.is_active,
            )
            StaffProfile.objects.get_or_create(person=person)
            PersonRole.objects.get_or_create(
                person=person,
                role_type=staff_role,
                defaults={"is_active": True},
            )

        person.user = user
        person.save(update_fields=["user"])
```

---

## 8. Admin Strategy — Revised

### New file: `apps/identity/admin.py`

```python
from django.contrib import admin
from .models import Person, RoleType, PersonRole, StudentProfile, StaffProfile

@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ("h_code", "full_name_display", "gender", "is_active", "has_user")
    search_fields = ("h_code", "first_name", "middle_name", "last_name")
    list_filter = ("is_active", "gender")
    readonly_fields = ("created_at", "updated_at")

    def full_name_display(self, obj):
        return obj.full_name()
    full_name_display.short_description = "Full name"

    def has_user(self, obj):
        return obj.user_id is not None
    has_user.boolean = True
    has_user.short_description = "Has login"

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

@admin.register(StudentProfile)
class StudentProfileAdmin(admin.ModelAdmin):
    list_display = ("person", "grade", "has_meal", "has_bus", "legacy_student")
    search_fields = ("person__h_code", "person__first_name", "person__last_name")
    list_filter = ("has_meal", "has_bus", "grade")
    readonly_fields = ("person", "legacy_student")

@admin.register(StaffProfile)
class StaffProfileAdmin(admin.ModelAdmin):
    list_display = ("person", "employee_id", "job_title", "is_teacher")
    search_fields = ("person__h_code", "person__first_name", "person__last_name", "employee_id")
    list_filter = ("is_teacher",)
```

### `apps/attendance/admin.py` — UNCHANGED in Phase 1

- `StudentAdmin` remains fully active with all actions, inlines, resources
- `MealSubscriptionAdmin`, `WalletAdmin`, `WalletTransactionAdmin` — all student references untouched
- `AttendanceRecordAdmin`, `AttendanceEventAdmin`, `FaceEmbeddingAdmin` — all student references untouched
- No imports from `apps.identity`

### Admin navigation note
Django admin will show two sections: **Identity** (Person, RoleType, PersonRole, StudentProfile, StaffProfile) and **Attendance** (Student, MealSubscription, Wallet, etc.). Operators continue to use StudentAdmin for daily workflows and use Identity admins for verification and debugging.

---

## 9. Backward Compatibility Strategy — Unchanged

| Area | Strategy |
|------|----------|
| **Models** | `Student` model is untouched. No FK fields added to it. |
| **Queries** | All `Student.objects.*`, `attendance_record.student.*` work as before. |
| **Admin** | `StudentAdmin` unchanged. Identity admins are additive in separate app. |
| **Signals** | `meal_subscription_saved` still uses `instance.student_id`. Works as before. |
| **Utils** | `recalc_meal_flags_for_students(Student.objects.all())` works unchanged. |
| **Auth** | `request.user` is still `auth.User`. No middleware changes. |
| **API** | No API changes in Phase 1. |
| **Resources** | `StudentResource`, `AttendanceRecordResource` unchanged. |
| **Management commands** | All `Student` imports continue working. |
| **Extras scripts** | No changes. |
| **`apps/identity`** | New app — no existing code references it yet |

---

## 10. Signals Strategy — Revised (Postpone)

**Recommendation: Do NOT add sync signals in Phase 1.**

### Rationale
- Student ↔ Person sync signals would require the `identity` app to import from `attendance` and vice versa, creating a cross-app coupling that defeats the purpose of separating identity.
- During Phase 1, `Student` data is the source of truth. Person data is a read-only mirror created by the data migration.
- Operators continue to edit Student records through `StudentAdmin`. Person data may become stale, but this is acceptable until Phase 1.5 when FK migration begins and Person becomes the source of truth.
- 4 auth.User accounts exist. Manual sync via admin or a one-off management command is simpler than a signal.

### If minimal one-way sync is desired
A single `apps/identity/signals.py` file can be deferred to Commit 6 but should only sync **Student → Person** (not bidirectional):

```python
# apps/identity/signals.py — DEFERRED, not in Phase 1
from django.db.models.signals import post_save
from django.dispatch import receiver
from apps.attendance.models import Student

@receiver(post_save, sender=Student)
def sync_student_to_person(sender, instance, **kwargs):
    try:
        profile = instance.migrated_to  # StudentProfile via related_name
        person = profile.person
        changed = False
        for field in ("h_code", "first_name", "middle_name", "last_name", "gender", "is_active"):
            if getattr(person, field) != getattr(instance, field):
                setattr(person, field, getattr(instance, field))
                changed = True
        if changed:
            person.save(update_fields=["h_code", "first_name", "middle_name", "last_name", "gender", "is_active", "updated_at"])
    except Person.DoesNotExist:
        pass  # StudentProfile.legacy_student exists but person may not
```

This would be added in a later commit or Phase 1.5, not in the initial Phase 1 migration set.

---

## 11. ADR-016 Recommendation

### Exact wording for `docs/architecture/ADR.md`

Add the following ADR entry (insert after ADR-015, before the ADR Index):

```
---

## ADR-016: Identity models live in dedicated `apps.identity` app

**Status:** Accepted

**Context:** The initial Phase 1 plan placed Person, RoleType, PersonRole, StudentProfile, and StaffProfile inside `apps/attendance/models.py`. However, BISK_RFv4 is evolving into a full ERP/LMS/portal platform where identity is a foundation domain. Attendance, finance, academics, portals, HR, and future modules all depend on Person. Placing identity models in the attendance app creates an ownership problem — attendance should not own the core identity layer, and every other app would need to import from attendance to reference Person.

**Decision:** A new `apps.identity` Django app owns the Person identity layer. All identity models — Person, RoleType, PersonRole, StudentProfile, StaffProfile — live in `apps.identity/models.py`. The `attendance` app's `Student` model remains untouched during Phase 1. Future profile types (ParentProfile, GuestProfile, etc.) will be added to `apps.identity`. Future FK migrations (Wallet → Person, AttendanceRecord → Person, etc.) will reference `identity.Person`.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Models in `apps/attendance/models.py` | Creates ownership ambiguity; every other app imports from attendance; violates domain boundaries |
| Models in a shared `apps/core` module | Too vague; identity has clear ownership and domain scope |
| Separate `apps/person` app | Acceptable, but `apps/identity` better signals the broader scope (roles, profiles, future auth) |

**Consequences:**

- Positive: Clean domain boundary; identity is a foundation app that other apps import from; aligns with the long-term architecture vision where identity is separate from attendance, finance, and academics.
- Negative: Cross-app FK (`StudentProfile.legacy_student → attendance.Student`) introduces a dependency from identity to attendance; this is temporary and will be removed in Phase 2 when Student is deprecated.
- Mitigation: The cross-app FK uses a string reference (`"attendance.Student"`) which Django resolves automatically. The dependency is one-directional (identity → attendance) and intentional during the transition.

**Related documents:**

- `docs/architecture/erp_foundation_architecture.md` Section 4 (Domain Boundaries)
- `docs/architecture/person_identity_architecture.md` Sections 5, 9, 10
- `.ai/sessions/opencode/2026-07-01_20-30-14_phase-1-identity-app-plan.md`
```

### ADR Index update

Add to the ADR Index table:

```
| 016 | Identity models live in dedicated identity app | Phase 1 |
```

---

## 12. Risks of Introducing a New Django App Now

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | **`INSTALLED_APPS` ordering bug** — identity registered after attendance, causing FK resolution failure | Very low | Medium | Place `apps.identity` before `apps.attendance` in INSTALLED_APPS; Django resolves string FKs at runtime regardless of order |
| 2 | **Cross-app FK** `StudentProfile.legacy_student → attendance.Student` fails if attendance app is unregistered | Very low | High | Attendance is a core app; won't be unregistered |
| 3 | **Migration discovery** — Django may not auto-discover `apps/identity/migrations/` on first run | Low | Medium | Run `python manage.py makemigrations identity` explicitly, or verify `python manage.py showmigrations identity` lists the app |
| 4 | **Import confusion** — developer imports `Person` from wrong module | Medium | Low | Clear convention: `from apps.identity.models import Person` |
| 5 | **Two admin sections** — operators may find Identity vs. Attendance separation confusing | Medium | Low | Document in release notes; StudentAdmin remains the primary workflow interface |
| 6 | **`makemigrations` creates attendance migration** if `StudentProfile.legacy_student` FK triggers a dependency | Low | Low | Run `makemigrations identity` specifically; verify no attendance migrations are generated |

---

## 13. Benefits of Introducing a New Django App Now

| # | Benefit | Impact |
|---|---------|--------|
| 1 | **Clean domain boundary** — identity is independent, not nested inside attendance | Future apps import from identity, not attendance |
| 2 | **Ownership clarity** — identity team/changes are separate from attendance team/changes | Reduces merge conflicts and cognitive load |
| 3 | **Migration isolation** — identity migrations are versioned independently from attendance migrations | Simpler history, easier rollback |
| 4 | **Future-proof** — when `AUTH_USER_MODEL` changes to `identity.Person` in Phase 2, the model is already in the right app | Avoids moving models between apps mid-stream |
| 5 | **Aligns with architecture docs** — `erp_foundation_architecture.md` Section 4 defines `apps/identity/` as the target | Consistent with documented long-term structure |
| 6 | **No big-bang refactor** — introducing the app now (when it has zero dependents) is the cheapest time to do it | Adding it later after FK migrations start would be disruptive |

---

## 14. Verification Commands — Revised

### Pre-migration sanity
```bash
python manage.py check
python manage.py makemigrations --check --dry-run
git status
git diff --stat
```

### Verify identity app is discovered
```bash
python manage.py showmigrations identity
# Expected output:
# identity
#  [ ] 0001_initial
#  [ ] 0002_seed_role_types
#  [ ] 0003_migrate_students_to_persons
#  [ ] 0004_link_users_to_persons
```

### Apply migrations
```bash
python manage.py migrate identity 0001
python manage.py migrate identity 0002
python manage.py migrate identity 0003
python manage.py migrate identity 0004
python manage.py check
```

### Data integrity verification
```bash
python manage.py shell -c "
from apps.identity.models import Person, StudentProfile, PersonRole
from apps.attendance.models import Student
from django.contrib.auth.models import User

p_count = Person.objects.count()
sp_count = StudentProfile.objects.count()
s_count = Student.objects.count()
r_count = PersonRole.objects.count()

print(f'Persons: {p_count}')
print(f'StudentProfiles: {sp_count}')
print(f'Students (unchanged): {s_count}')
print(f'PersonRoles: {r_count}')

assert p_count == s_count, f'Person count {p_count} != Student count {s_count}'
assert sp_count == s_count, f'StudentProfile count {sp_count} != Student count {s_count}'

student_roles = PersonRole.objects.filter(role_type__code='student').count()
assert student_roles == s_count, f'Student roles {student_roles} != Student count {s_count}'

orphans = StudentProfile.objects.filter(legacy_student__isnull=True)
assert orphans.count() == 0, f'{orphans.count()} StudentProfiles missing legacy_student'

linked = User.objects.filter(person__isnull=False).count()
total_users = User.objects.count()
print(f'Users linked to Person: {linked}/{total_users}')

print('ALL CHECKS PASSED')
"
```

### Verify attendance app is unaffected
```bash
python manage.py shell -c "
from apps.attendance.models import Student
print(f'Students still work: {Student.objects.count()} records')
print(f'Sample: {Student.objects.first()}')
"
```

---

## 15. Rollback Plan — Revised

### Full rollback (safe at any time)
```bash
python manage.py migrate identity zero    # drops all 5 identity tables
python manage.py check
git checkout -- apps/identity/ bisk/settings.py
```

This drops all 5 new tables (identity_person, identity_roletype, identity_personrole, identity_studentprofile, identity_staffprofile) and all migrated data. Existing Student and attendance data is untouched.

### Partial rollback during development
```bash
python manage.py migrate identity 0003    # keep schema, drop migrated data
python manage.py migrate identity 0002    # keep schema, drop seeded roles
python manage.py migrate identity 0001    # drop schema only
python manage.py migrate identity zero    # drop all
```

### Revert settings.py
```bash
git checkout -- bisk/settings.py
```

### Data integrity on rollback
- `attendance_student` table: **untouched**
- `auth_user` table: **untouched** (Person.user FK is nullable — dropping Person table auto-nulls the FK)
- Wallet, meals, attendance: **untouched**
- No cascade effects on attendance data

---

## 16. Small Commit Sequence — Revised

### Commit 1 — `feat: create apps/identity app structure`
- Create `apps/identity/__init__.py`, `apps/identity/apps.py` (IdentityConfig)
- Create `apps/identity/migrations/__init__.py`
- Create `apps/identity/models.py` with Person, RoleType, PersonRole, StudentProfile, StaffProfile
- Create `apps/identity/admin.py` with all ModelAdmins
- Create `apps/identity/tests.py` (empty)
- Add `"apps.identity"` to `INSTALLED_APPS` in `bisk/settings.py`
- Auto-generated migration `identity.0001`

### Commit 2 — `feat: seed RoleType values`
- Migration `identity.0002` with `RunPython(seed_role_types)`
- 14 roles: student, staff, teacher, parent, guardian, guest, vendor, administrator, finance, hr, principal, vice_principal, librarian, nurse

### Commit 3 — `feat: data migrate existing Student records to Person + StudentProfile + PersonRole`
- Migration `identity.0003` with `RunPython(migrate_students_to_persons)`
- Each Student → 1 Person + 1 StudentProfile + 1 PersonRole(student)
- Verify counts match

### Commit 4 — `feat: link existing auth.User accounts to Person`
- Migration `identity.0004` with `RunPython(link_users_to_persons)`
- Match by username-as-h_code, fallback to create Person + StaffProfile

### Commit 5 (optional) — `feat: add one-way Student→Person sync signal (deferred)`
- Create `apps/identity/signals.py`
- Wire in `apps/identity/apps.py:ready()`
- Student post_save syncs identity fields to Person
- **This can be postponed to Phase 1.5 without impact**

### Merge strategy
All commits on `feature/person-architecture`. Review commits 1–4 as a group; commit 5 is optional.

---

## Appendix A: File Manifest (New and Modified)

### New files created
```
apps/identity/
├── __init__.py
├── apps.py
├── models.py
├── admin.py
├── tests.py
├── migrations/
│   ├── __init__.py
│   ├── 0001_initial.py
│   ├── 0002_seed_role_types.py
│   ├── 0003_migrate_students_to_persons.py
│   └── 0004_link_users_to_persons.py
```

### Existing files modified
```
bisk/settings.py        # add "apps.identity" to INSTALLED_APPS
```

### Existing files untouched
```
apps/attendance/models.py   # no change
apps/attendance/admin.py    # no change
apps/attendance/signals.py  # no change
apps/attendance/services.py # no change
apps/attendance/resources.py# no change
apps/attendance/utils/meal.py # no change
apps/scheduler/             # no change
apps/cameras/               # no change
extras/                     # no change
```

---

## Appendix B: Field Mapping (Student → Person + StudentProfile)

| Student field | Target in identity app | Notes |
|--------------|------------------------|-------|
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
| *no equivalent* | `Person.user` | Set in migration 0004 for staff users |
| *no equivalent* | `StudentProfile.homeroom` | Blank — future use |
| *no equivalent* | `StudentProfile.legacy_student` | Set to original Student instance (cross-app FK) |

---

## Appendix C: Dependency Order for Future Phases

```
Phase 1 (this plan — identity app)
  └── Phase 1.5: Wallet.person → identity.Person
  └── Phase 1.5: MealSubscription.person → identity.Person
  └── Phase 1.5: FaceEmbedding.person → identity.Person
  └── Phase 1.5: AttendanceRecord.person → identity.Person
  └── Phase 1.5: AttendanceEvent.person → identity.Person
        └── Phase 2: Deprecate Student → requires all Phase 1.5 done
              └── Phase 3: AcademicYear + Enrollment → requires Phase 2 done
```

All Phase 1.5 FK migrations target `identity.Person`, not `attendance.Person`.

---

## Appendix D: Prerequisites Before Starting Commit 1

- [x] Architecture docs approved and tagged
- [x] Migrations squashed (0001_squashed_0021 exists)
- [x] Database backed up and verified
- [x] Django checks pass (`python manage.py check` — 0 issues)
- [x] No pending migrations (`makemigrations --check --dry-run` — no changes)
- [ ] This revised plan reviewed and approved
