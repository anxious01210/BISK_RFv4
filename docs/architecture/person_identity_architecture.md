# Person Identity Architecture — BISK_RFv4

Date: 2026-06-30
Branch: feature/person-architecture
Status: Draft — architecture blueprint

---

## 1. Vision

BISK_RFv4 will evolve from a school attendance and meal-management system into a full-school ERP / LMS / portal platform. The identity system is the foundation.

The vision is a **unified person identity** that underlies every subsystem:

- Attendance (face recognition → Person)
- Meals and wallet (subscription → Person, wallet → Person, charges → Person)
- Academics (enrollment → Person, grading → Person, timetable → Person)
- Finance (payments → Person, invoices → Person)
- Portals (student portal, staff portal, parent portal — all keyed to Person)
- Mobile API (JWT-authenticated access to Person data)
- Multi-school (Person can exist across schools via Organization/tenant)

The core principle: **one identity, many roles, optional login.**

---

## 2. Design Principles

| Principle | Description |
|---|---|
| **Identity before role** | A Person exists independently of their role. The same person can be a student one year and staff the next. |
| **Person ≠ User** | Business identity (Person) is separate from login identity (auth.User). A person may have zero, one, or multiple login accounts. |
| **Roles are M2M, not a field** | Do not use a single `person_type` CharField. Use RoleType + PersonRole so a person can hold multiple simultaneous roles (e.g., staff + parent). |
| **Profiles extend roles** | StudentProfile, StaffProfile, etc. are role-specific data extensions. They are optional — a Person with role "parent" may have no profile. |
| **Backward compatibility always** | Never break existing Student-based code during migration. Dual-write, proxy models, and compat properties. |
| **Incremental delivery** | Each phase is independently deployable. No big-bang migrations. |
| **AcademicYear awareness** | Enrollments, subscriptions, and assignments are scoped to academic years. The model is designed now but implemented later. |
| **Multi-tenant ready** | Organization/School/Campus hierarchy is designed into the identity model but activated when needed. |

---

## 3. Organization / School / Campus Concept

The system envisions a future legal/physical hierarchy:

```
Organization (e.g., "BISK Educational Group")
    └── School (e.g., "BISK K-12 Baghdad")
            └── Campus (e.g., "Main Campus", "Girls Campus")
```

**Department is not part of this physical hierarchy.**

Departments are domain-owned organizational units or cost centers. Depending on the future feature, a department may belong to the academic, HR, finance, or operations domain. This aligns with `docs/architecture/erp_foundation_architecture.md` and `docs/architecture/ADR.md` ADR-011.

### Department placement

| Domain | Example Department | Future ownership |
|---|---|---|
| Academic | Mathematics, English, Science | Academic structure / subject grouping |
| HR/Admin | Administration, HR, Payroll | HR / staff assignment |
| Operations | Maintenance, Transport, Security | Operations domain |
| Finance | Accounting, Billing, Procurement | Finance / cost center |

A single future `Department` model with a `domain` or `category` field may be enough, or separate domain-specific models may be introduced later. That decision is deferred until a real department-driven feature is implemented.

### Design (future implementation)

```python
class Organization(models.Model):
    name = CharField(max_length=200)
    code = CharField(max_length=32, unique=True)        # short code
    is_active = BooleanField(default=True)
    # ... contact, address, branding

class School(models.Model):
    organization = ForeignKey(Organization, CASCADE, related_name="schools")
    name = CharField(max_length=200)
    code = CharField(max_length=32, unique=True)
    is_active = BooleanField(default=True)

class Campus(models.Model):
    school = ForeignKey(School, CASCADE, related_name="campuses")
    name = CharField(max_length=200)
    is_active = BooleanField(default=True)

# Department is intentionally omitted from this hierarchy.
# It will be modeled later in the relevant domain: academic, HR, finance, or operations.
```

### Current implementation

Not implemented. Person will have an optional `school` FK and `ext_id` field prepared for future multi-tenant activation.

Person models (StudentProfile, StaffProfile) will NOT require a School FK initially. They become required when multi-tenant is activated.

---

## 4. AcademicYear Concept

Many school-domain models are scoped to an academic year:

- StudentEnrollment (which year a student is enrolled)
- StaffAssignment (which year a staff member is assigned)
- MealSubscription (which year a subscription covers)
- DiscountProfile (which year a discount applies)
- PeriodTemplate / PeriodOccurrence (which year's timetable)

### Design (future implementation)

```python
class AcademicYear(models.Model):
    name = CharField(max_length=100)                    # e.g., "2026-2027"
    code = CharField(max_length=32, unique=True)        # e.g., "2026-27"
    start_date = DateField()
    end_date = DateField()
    is_active = BooleanField(default=False)             # current/active year
    school = ForeignKey(School, CASCADE, related_name="academic_years")  # future

    class Meta:
        ordering = ["-start_date"]
```

### Current implementation

Not implemented. AcademicYear is referenced in design diagrams and model comments but is not a required FK in the first Person migration. Existing models (MealSubscription, PeriodOccurrence) use date fields directly.

---

## 5. Person Identity Model

Person is the **core business identity**. It represents a real human being in the school ecosystem.

### Design

```python
class Person(models.Model):
    # --- Identity ---
    h_code = models.CharField(
        max_length=32, unique=True,
        help_text="Unique human-readable identifier. Replaces Student.h_code."
    )
    first_name = models.CharField(max_length=100, blank=True, default="")
    middle_name = models.CharField(max_length=100, blank=True, default="")
    last_name = models.CharField(max_length=100, blank=True, default="")
    gender = models.CharField(
        max_length=6, choices=[("MALE", "male"), ("FEMALE", "female")],
        blank=True, null=True, db_index=True
    )
    date_of_birth = models.DateField(blank=True, null=True)

    # --- Contact ---
    email = models.EmailField(blank=True, default="")
    phone = models.CharField(max_length=20, blank=True, default="")
    address = models.TextField(blank=True, default="")

    # --- Photo ---
    photo = models.ImageField(
        upload_to="person_photos/", blank=True,
        help_text="Official portrait photo."
    )

    # --- Status ---
    is_active = models.BooleanField(
        default=True, db_index=True,
        help_text="Deactivate to soft-delete a person."
    )

    # --- Auth bridge (Phase 1 only) ---
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="person",
        help_text="Optional link to a login account.",
    )

    # --- Multi-school / ERP future ---
    ext_id = models.CharField(
        max_length=64, blank=True, default="",
        help_text="External ERP or cross-system identifier.",
    )
    school = models.ForeignKey(
        "attendance.School",                            # future model
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="persons",
        help_text="Primary school/campus affiliation (future multi-tenant).",
    )

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

### Key design decisions

| Decision | Rationale |
|---|---|
| `h_code` as unique identifier | Replaces `Student.h_code`. Already the primary lookup key. Avoids exposing internal PKs. |
| No `username` field | `auth.User` handles usernames. Person uses `h_code` as the natural key. |
| No `password` field | Person is NOT the auth model (Option D hybrid). Passwords stay on `auth.User`. |
| `user` FK is nullable | A child in kindergarten does not need login access. |
| `school` FK is nullable | Multi-tenant is future. A Person can exist without school affiliation initially. |
| `ext_id` for ERP | Prepares for integration with external SIS/ERP systems. |

---

## 6. CustomUser / Authentication Strategy

### Current state

- `AUTH_USER_MODEL` is **not set**. Django defaults to `auth.User`.
- 4 login users exist (staff/supervisors only).
- Students have zero login accounts.

### Recommended strategy: Option D (Progressive Hybrid)

**Do NOT change `AUTH_USER_MODEL` during the first Person migration.** Keep `auth.User` as the authentication model. Person has an optional `OneToOneField("auth.User")` bridge.

#### Phase 1 (immediate)

```
auth.User ──(optional O2O)──→ Person
```

- All authentication flows remain unchanged.
- `request.user` is still `auth.User`.
- Login accounts are created manually or via management commands.
- A `post_save` signal on `auth.User` syncs key fields to the linked `Person` (if one exists).
- A `post_save` signal on `Person` creates an `auth.User` if one is needed (e.g., for staff).

#### Phase 2 (future — after squash + preparation)

```
Person(AbstractUser) with AUTH_USER_MODEL = "attendance.Person"
```

- Set `AUTH_USER_MODEL` in settings.
- Copy existing `auth_user` data into `attendance_person`.
- Update the 3 FK fields that reference `settings.AUTH_USER_MODEL`.
- Drop the `user` O2O bridge from Person.
- Drop the `auth_user` table.
- `request.user` is now `Person`.

#### Why not do Phase 2 now?

Django explicitly warns that changing `AUTH_USER_MODEL` after migrations have been run is not supported on a production database. It requires either:
- A new database (unrealistic for production data)
- Manual SQL rename operations (high risk)
- Squashed migrations + careful state migration (doable, but blocks all other work)

Phase 2 should be scheduled as a separate, dedicated migration effort after the Person model is proven in production.

---

## 7. RoleType Model

RoleType is a **controlled vocabulary** of roles a Person can hold. It replaces a simple `person_type` CharField.

### Why not `person_type`?

A single `person_type` field (`student`, `staff`, `parent`, etc.) cannot represent:
- A staff member who is also a parent of a student
- A student who is also a teacher assistant (dual role)
- A vendor who is also a parent
- A person changing roles across academic years

`RoleType` + `PersonRole` handles all of these.

### Design

```python
class RoleType(models.Model):
    ROLE_STUDENT = "student"
    ROLE_STAFF = "staff"
    ROLE_TEACHER = "teacher"
    ROLE_PARENT = "parent"
    ROLE_GUEST = "guest"
    ROLE_VENDOR = "vendor"

    ROLE_CHOICES = [
        (ROLE_STUDENT, "Student"),
        (ROLE_STAFF, "Staff"),
        (ROLE_TEACHER, "Teacher"),
        (ROLE_PARENT, "Parent"),
        (ROLE_GUEST, "Guest"),
        (ROLE_VENDOR, "Vendor"),
    ]

    code = models.CharField(
        max_length=32, unique=True, choices=ROLE_CHOICES,
        help_text="Machine-readable role code.",
    )
    name = models.CharField(
        max_length=100,
        help_text="Human-readable role name (e.g., 'Student', 'Teacher').",
    )
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["code"]

    def __str__(self):
        return self.name
```

---

## 8. PersonRole Model

PersonRole links a Person to a RoleType, optionally scoped to an AcademicYear and/or School.

### Design

```python
class PersonRole(models.Model):
    person = models.ForeignKey(
        Person, on_delete=models.CASCADE,
        related_name="roles",
    )
    role_type = models.ForeignKey(
        RoleType, on_delete=models.PROTECT,
        related_name="person_roles",
    )
    academic_year = models.ForeignKey(
        AcademicYear, on_delete=models.CASCADE,
        null=True, blank=True,                 # null = ongoing/permanent role
        related_name="person_roles",
        help_text="Academic year this role is active for. Null = not time-bound.",
    )
    school = models.ForeignKey(
        School, on_delete=models.CASCADE,
        null=True, blank=True,
        related_name="person_roles",
        help_text="School/campus this role applies to (future multi-tenant).",
    )

    # --- Status ---
    is_active = models.BooleanField(default=True, db_index=True)
    notes = models.CharField(max_length=200, blank=True, default="")

    # --- Metadata ---
    assigned_at = models.DateTimeField(auto_now_add=True)
    assigned_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL, null=True, blank=True,
    )

    class Meta:
        unique_together = [
            ("person", "role_type", "academic_year", "school"),
        ]
        indexes = [
            models.Index(fields=["person", "is_active"]),
            models.Index(fields=["role_type", "is_active"]),
        ]
        ordering = ["person", "role_type", "-academic_year__start_date"]

    def __str__(self):
        return f"{self.person.h_code} → {self.role_type.code}"
```

### Usage patterns

| Pattern | Example |
|---|---|
| Active student this year | `PersonRole(person=X, role_type=student, academic_year=2026)` |
| Staff member (ongoing) | `PersonRole(person=Y, role_type=staff, academic_year=None)` |
| Staff + Parent | Two PersonRole rows for Z: one staff, one parent |
| Student becomes staff next year | Old student role closed, new staff role opened |

---

## 9. StudentProfile

StudentProfile extends Person with student-specific data. It is the eventual replacement for the current `Student` model.

### Design

```python
class StudentProfile(models.Model):
    person = models.OneToOneField(
        Person, on_delete=models.CASCADE,
        primary_key=True,                       # shares PK with Person
        related_name="student_profile",
    )

    # --- Academic ---
    grade = models.CharField(max_length=32, blank=True, null=True, db_index=True)
    homeroom = models.CharField(max_length=64, blank=True, default="")

    # --- Flags (denormalized, synced by signals) ---
    has_meal = models.BooleanField(default=False, db_index=True)
    has_bus = models.BooleanField(default=False, db_index=True)

    # --- Legacy ---
    legacy_student = models.OneToOneField(
        "attendance.Student",                   # current Student model
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="migrated_to",
        help_text="Temporary backlink to original Student record during migration.",
    )

    # --- Metadata ---
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["person__h_code"]

    def __str__(self):
        return f"Student: {self.person.h_code}"
```

### Migration from Student

Each current `Student` row produces:
- One `Person` row (h_code, first_name, middle_name, last_name, gender, grade copied)
- One `StudentProfile` row (grade, has_meal, has_bus copied)
- One `PersonRole` row (role_type=student)

The `legacy_student` FK provides a backlink for backward compatibility during the transition.

---

## 10. StaffProfile

StaffProfile extends Person with staff-specific data.

### Design

```python
class StaffProfile(models.Model):
    person = models.OneToOneField(
        Person, on_delete=models.CASCADE,
        primary_key=True,
        related_name="staff_profile",
    )

    # --- Employment ---
    employee_id = models.CharField(
        max_length=32, blank=True, default="",
        help_text="HR/ERP employee identifier.",
    )
    job_title = models.CharField(max_length=100, blank=True, default="")
    department = models.CharField(max_length=100, blank=True, default="")
    hire_date = models.DateField(null=True, blank=True)

    # --- Flags ---
    is_teacher = models.BooleanField(
        default=False,
        help_text="If true, this staff member can be assigned teaching periods.",
    )

    # --- Metadata ---
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["person__h_code"]

    def __str__(self):
        return f"Staff: {self.person.h_code} ({self.job_title or 'No title'})"
```

---

## 11. Future Profile Types

These profiles are **designed conceptually but not implemented** in the first Person migration.

### ParentProfile

```python
class ParentProfile(models.Model):
    person = OneToOneField(Person, CASCADE, primary_key=True, related_name="parent_profile")
    children = ManyToManyField(StudentProfile, related_name="parents", blank=True)
    emergency_contact_priority = PositiveSmallIntegerField(default=1)
    # ...
```

### GuestProfile

```python
class GuestProfile(models.Model):
    person = OneToOneField(Person, CASCADE, primary_key=True, related_name="guest_profile")
    organization = CharField(max_length=200, blank=True)
    purpose = CharField(max_length=200, blank=True)
    access_expires = DateTimeField(null=True, blank=True)
    # ...
```

### VendorProfile

```python
class VendorProfile(models.Model):
    person = OneToOneField(Person, CASCADE, primary_key=True, related_name="vendor_profile")
    company = CharField(max_length=200)
    contract_ref = CharField(max_length=64, blank=True)
    # ...
```

### Implementation order

```
Phase 1: StudentProfile, StaffProfile, PersonRole
Phase 3+: ParentProfile, GuestProfile, VendorProfile (as needed)
```

---

## 12. StudentEnrollment by AcademicYear

StudentEnrollment tracks a student's enrollment status **per academic year**. This replaces the simplistic `Student.is_active` flag.

### Design (future implementation)

```python
class StudentEnrollment(models.Model):
    student = models.ForeignKey(
        StudentProfile, on_delete=models.CASCADE,
        related_name="enrollments",
    )
    academic_year = models.ForeignKey(
        AcademicYear, on_delete=models.CASCADE,
        related_name="enrollments",
    )
    grade = models.CharField(max_length=32, blank=True, null=True)
    homeroom = models.CharField(max_length=64, blank=True, default="")
    enrollment_date = models.DateField()
    withdrawal_date = models.DateField(null=True, blank=True)

    STATUS_ENROLLED = "enrolled"
    STATUS_WITHDRAWN = "withdrawn"
    STATUS_GRADUATED = "graduated"
    STATUS_TRANSFERRED = "transferred"

    status = models.CharField(
        max_length=20, choices=[...], default=STATUS_ENROLLED, db_index=True,
    )

    class Meta:
        unique_together = [("student", "academic_year")]
```

### Implementation order

Not in Phase 1. The current `Student.is_active` boolean suffices until AcademicYear is modeled.

---

## 13. StaffAssignment by AcademicYear

StaffAssignment tracks staff roles and departments per academic year.

### Design (future implementation)

```python
class StaffAssignment(models.Model):
    staff = models.ForeignKey(
        StaffProfile, on_delete=models.CASCADE,
        related_name="assignments",
    )
    academic_year = models.ForeignKey(AcademicYear, on_delete=models.CASCADE)
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True)
    job_title = models.CharField(max_length=100)
    is_primary = models.BooleanField(default=True)
    # ...
```

### Implementation order

Not in Phase 1.

---

## 14. Wallet Linked to Person

### Current

```python
Wallet.student = OneToOneField(Student, CASCADE, related_name="wallet")
```

### Target

```python
Wallet.person = OneToOneField(Person, CASCADE, related_name="wallet")
```

### Migration strategy

Phase 1.5 (after Person + StudentProfile are created):

1. Add `Wallet.person` as a nullable `OneToOneField(Person)`.
2. Data migration: for each `Wallet`, set `person = wallet.student.migrated_to` (the linked Person).
3. Once verified, remove `Wallet.student` in a subsequent migration.
4. Update `Wallet.Meta.ordering` from `student__h_code` to `person__h_code`.
5. Update `Wallet.__str__` to use `self.person.h_code`.

Same pattern for `WalletTransaction`:
- `WalletTransaction.student` → `WalletTransaction.person`
- `WalletTransaction` already has `wallet` FK, so `student` is denormalized — after migration it can be derived from `wallet.person`.

---

## 15. Meal / Lunch Linked to Person

### Current

```python
MealSubscription.student = ForeignKey(Student, CASCADE, related_name="meal_subscriptions")
```

### Target

```python
MealSubscription.person = ForeignKey(Person, CASCADE, related_name="meal_subscriptions")
```

### Migration strategy

Same dual-FK pattern as Wallet.

`MealRecord` does not have a direct FK to Student — it links through `AttendanceRecord.student`. When `AttendanceRecord.student` is migrated to `AttendanceRecord.person`, `MealRecord` inherits the correct Person association.

`MealSubscription.clean()` uses `self.student_id` for overlap checking. Must be updated to `self.person_id` (with backward compat for existing callers).

---

## 16. FaceEmbedding Linked to Person

### Current

```python
FaceEmbedding.student = ForeignKey(Student, CASCADE, related_name="embeddings")
```

### Target

```python
FaceEmbedding.person = ForeignKey(Person, CASCADE, related_name="embeddings")
```

### Migration strategy

Same dual-FK pattern. This also enables face recognition for staff (currently face embeddings are student-only).

Note: The unique constraint `uniq_active_embedding_per_student` (migration 0006) must be recreated for `person_id`.

---

## 17. Portal and API Access

### Authentication flow

```
Request → [Middleware] → request.user (auth.User)
                               ↓
                          user.person (Person, if linked)
                               ↓
                     Person.roles → RoleType list
                               ↓
                     [Permission check: group-based + role-based]
```

### Portal types

| Portal | Auth | Person→Role | Data scope |
|---|---|---|---|
| Student portal | Student logs in via `auth.User` (created for them) | `PersonRole(person, student)` | Own attendance, meals, grades, schedule |
| Staff portal | Staff logs in via existing `auth.User` | `PersonRole(person, staff)` | Their students, classes, reports |
| Parent portal | Parent logs in via `auth.User` | `PersonRole(person, parent)` | Their children's data |
| Admin | Existing staff superuser | `PersonRole(person, staff)` + `is_superuser` | Full system |

### API design

```python
# Current user endpoint — returns Person data, not User data
GET /api/v1/me/
→ { person: { h_code, name, roles: [...], profile: { ... } } }
```

DRF permission classes:

```python
class HasPersonRole(BasePermission):
    """Check if the authenticated user's Person has a specific RoleType."""
    def has_permission(self, request, view):
        person = getattr(request.user, "person", None)
        if not person:
            return False
        required_roles = getattr(view, "required_roles", [])
        return person.roles.filter(role_type__code__in=required_roles).exists()
```

### Mobile API

- JWT authentication via `rest_framework_simplejwt` or similar.
- Token-based, no session cookies.
- Scoped to the authenticated Person's roles and data.

---

## 18. Permissions/Groups vs Business Roles

### Two separate systems

| System | Purpose | Technology |
|---|---|---|
| **Django Groups** | System access control. Who can access admin, API, dashboards. | `auth.Group` + `auth.User.groups` |
| **PersonRole** | Business role. What a person IS in the school context. | `RoleType` + `PersonRole` |

### Rule of thumb

- If a permission controls **access to a Django view/endpoint/admin page**, use **Django Groups** (unchanged from current system).
- If a permission controls **business logic** (e.g., "can this staff member confirm meals for this student?"), use **PersonRole**.
- The two systems can be combined: a PersonRole can imply automatic membership in a Django Group (via signal).

### Mapping currently hardcoded groups (Phase 1 compatibility)

| Current Group | Equivalent RoleType | Notes |
|---|---|---|
| `supervisor` | `staff` (with `is_supervisor` flag) | Used for admin-gate access |
| `meal_supervisor` | `staff` (with `role_metadata`) | Used for meal dashboard |
| `api_user` | `staff` or `guest` | Used for API-only access |

Long-term: Django groups remain for auth, but the hardcoded group-name checks are replaced by PersonRole-based checks.

---

## 19. Microsoft/Google Login (Future)

### Vision

```
                        ┌──────────────────┐
                        │   Social Auth    │
                        │ (django-allauth) │
                        └────────┬─────────┘
                                 │
                    ┌────────────▼──────────┐
                    │     auth.User         │
                    │  (login identity)     │
                    └────────────┬──────────┘
                                 │ optional O2O
                    ┌────────────▼──────────┐
                    │       Person          │
                    │  (business identity)  │
                    └───────────────────────┘
```

### Approach

- Use `django-allauth` or `python-social-auth` for Microsoft/Google OAuth.
- Social accounts link to `auth.User` (the login identity).
- `auth.User` links to `Person` via the existing O2O bridge.
- Staff can "Login with Microsoft" → auto-creates/link to their existing Person.
- Parents can "Login with Google" → creates `auth.User` with `person_type=parent`, auto-creates Person + PersonRole.

### Implementation order

Phase 3+, not in the first Person migration.

---

## 20. Multi-School Readiness

### Design for multi-tenant

The identity model is designed to support multi-school without schema changes:

| Element | Multi-school support |
|---|---|
| `Person.school` | Optional FK. Null = single-school mode. |
| `PersonRole.school` | Scopes a role to a specific school. |
| `StudentProfile` | Inherits school from Person. |
| `StaffProfile` | Inherits school from Person. |
| `h_code` uniqueness | Change from `unique=True` to `unique_together=[("school", "h_code")]` when multi-tenant is activated. |
| `ext_id` | Cross-system identifier for syncing with external school SIS/ERP. |

### What changes when multi-school is activated

1. `AUTH_USER_MODEL` may need a `school` FK (or each school has its own auth realm).
2. All business queries become scoped by `school_id`.
3. Django `django-tenants` or similar library for schema-per-tenant.

### Implementation order

Not implemented in the first Person migration. The `school` FK and `ext_id` field are prepared but nullable.

---

## 21. Migration Strategy from Current Student Model

### Overview

```
Phase 0                      Phase 1                    Phase 1.5                  Phase 2
[Squash migrations]     [Add Person + Profiles]    [Migrate FKs to Person]    [Deprecate Student]
                              │                           │                          │
  30+ migrations             Person (new)              Wallet.person              Student
  ──→ squashed base          StudentProfile (new)      Meals.person              ──→ dropped
                              StaffProfile (new)        FaceEmbedding.person
                              PersonRole (new)          Attendance.person
                              │                           │
                              Student (unchanged)         Student (unchanged)
                              Data: mirror Student        Data: dual FK write
                              → Person+Profile            → verify consistency
```

### Step-by-step

#### Phase 0: Squash

- Combine all 30+ current attendance migrations into one `0001_squashed`.
- This resets the migration baseline before introducing Person.
- Run: `python manage.py makemigrations --squash-name squashed` (or manual squash).

#### Phase 1: Add Person (additive)

- Create `Person`, `StudentProfile`, `StaffProfile`, `RoleType`, `PersonRole` models.
- Create data migration: for each `Student`, create one `Person` + one `StudentProfile` + one `PersonRole(student)`.
- `StudentProfile.legacy_student` links back to the original `Student` row.
- No existing tables are altered. Full rollback by dropping new tables.

#### Phase 1.5: Migrate FKs (dual-write)

For each FK model in order:

1. Add `person` FK (nullable).
2. Data migration: populate `person_id` from `student_id` → `StudentProfile` → `Person`.
3. Add database constraint (not null).
4. Update all query code to use `person` instead of `student`.
5. In a separate release: drop `student` FK.

Order:
1. `Wallet` + `WalletTransaction` — simplest, least dependent
2. `MealSubscription` — depends on Student gone
3. `FaceEmbedding` — depends on Person existing
4. `AttendanceRecord` — largest table, migration in batches
5. `AttendanceEvent` — large table, similar approach

#### Phase 2: Deprecate Student

- Remove `legacy_student` backlink from `StudentProfile`.
- Mark `Student` model as `managed = False`.
- Once confirmed no code references Student, drop the table in a final migration.

---

## 22. Backward Compatibility Strategy

### Principle

**All existing Student-based code continues working until Phase 2.**

### Compatibility mechanisms

#### 1. Student → Person read-through proxy

```python
# On AttendanceRecord (and all FK models during transition)
@property
def student(self):
    """Backward-compat accessor. Returns StudentProfile or None."""
    if hasattr(self, "person") and self.person_id:
        return getattr(self.person, "student_profile", None)
    return None
```

Template code like `r.student.full_name` continues working because `r.student` returns a StudentProfile-like object (or the actual Student model during early phases).

#### 2. Dual FK fields

During Phase 1.5, models have both `student_id` and `person_id`. Writes go to both. Reads toggle from `student` to `person` via a feature flag or setting.

#### 3. Import/Export compatibility

`StudentResource` remains for admin export. A new `PersonResource` is added. The old resource delegates to the new one internally.

#### 4. API backward compatibility

API endpoints that accept `h_code` continue to work — the lookup function first checks `Person`, then falls back to `Student`:

```python
def resolve_person(h_code: str) -> Person | None:
    person = Person.objects.filter(h_code=h_code).first()
    if person:
        return person
    student = Student.objects.filter(h_code=h_code).first()
    if student:
        return student.migrated_to.person  # via legacy_student backlink
    return None
```

#### 5. Admin backward compatibility

- `StudentAdmin` continues to be registered during Phase 1.
- `PersonAdmin` + `StudentProfileAdmin` are registered alongside it.
- `StudentAdmin` becomes read-only or is hidden when migration is complete.

#### 6. Signal compatibility

```python
# On MealSubscription post_save (Phase 1)
# Old signal path:
#   subscription.student.has_meal = recalc(subscription.student)
# New signal path:
#   subscription.person.student_profile.has_meal = recalc(subscription.person)

# Both paths active during Phase 1. Old path removed in Phase 2.
```

---

## 23. What NOT to Implement Yet

These are explicitly **excluded from the first Person migration** (Phase 1):

| Feature | Why deferred |
|---|---|
| `AUTH_USER_MODEL = "attendance.Person"` | Irreversible on existing database. Deferred to Phase 2. |
| `Organization` / `School` / `Campus` models | Multi-tenant not needed yet. `Person.school` FK is nullable. |
| `AcademicYear` model | Not needed until enrollment and subscription scoping require it. |
| `StudentEnrollment` | Replaces `is_active` flag. Deferred until AcademicYear exists. |
| `StaffAssignment` | Deferred until AcademicYear exists. |
| `ParentProfile` | No parent portal yet. Add when portal is built. |
| `GuestProfile` | Not needed. |
| `VendorProfile` | Not needed. |
| Social auth (Microsoft/Google) | No requirement yet. Add when portals launch. |
| JWT/OAuth2 for mobile | No mobile app yet. Add when API is built. |
| `Department` model | Not needed until staff assignment scoping requires it. |
| Removing `Student` table | Premature. Keep through Phase 2. |
| `Student.user` FK (adding FK to auth.User on Student) | Student is being replaced by Person+StudentProfile. Adding a FK to a deprecated model is wasted effort. |

---

## 24. Phased Implementation Roadmap

### Phase 0: Preparation

| Step | Deliverable |
|---|---|
| Squash existing migrations | Single `0001_squashed` migration |
| Dump production database | Rollback safety |
| Freeze current schema | Document current table structure |
| Update `AGENTS.md` | Record architecture decisions |
| **Risk** | None — no production impact |

### Phase 1: Add Person (additive — no existing table changes)

| Step | Deliverable |
|---|---|
| Create `Person` model | Identity fields, optional `user` O2O, `school` FK, `ext_id` |
| Create `RoleType` model | Seeded with student, staff, teacher, parent |
| Create `PersonRole` model | M2M through with year/school scoping |
| Create `StudentProfile` model | Student-specific fields, `legacy_student` backlink |
| Create `StaffProfile` model | Staff-specific fields |
| Add auth.User ↔ Person sync signal | Bidirectional sync |
| Data migration: `Student` → `Person` + `Profile` | For every existing Student |
| Data migration: link existing `auth.User` → `Person` | Link 4 existing users |
| Add `PersonAdmin` + `StudentProfileAdmin` | Admin interface |
| Update `docs/` | Architecture documentation |
| **Risk** | Low — additive only, full rollback possible |

### Phase 1.5: Migrate FK Targets (one model at a time)

| Step | Deliverable |
|---|---|
| Migrate `Wallet` FK | `student` → `person` |
| Migrate `WalletTransaction` FK | `student` → `person` |
| Migrate `MealSubscription` FK | `student` → `person` |
| Migrate `FaceEmbedding` FK | `student` → `person` |
| Migrate `AttendanceRecord` FK | `student` → `person` |
| Migrate `AttendanceEvent` FK | `student` → `person` |
| Update `MealSubscription.clean()` | Use `person_id` instead of `student_id` |
| Update `Wallet.Meta.ordering` | Use `person__h_code` |
| Update all views, templates, admin, services | Remove `student` FK references |
| Add backward-compat properties | `AttendanceRecord.student`, etc. |
| Update `extras/` scripts | Use Person lookups |
| **Risk** | Medium — each FK migration is independent and reversible |

### Phase 2: Deprecate Student

| Step | Deliverable |
|---|---|
| Remove `legacy_student` backlink | Clean up StudentProfile |
| Mark `Student` as `managed = False` | Django ignores the model |
| Archive `Student` data (if needed) | Before dropping |
| Drop `attendance_student` table | Final migration |
| Remove backward-compat shims | Properties, dual-FK code |
| **Risk** | Medium — irreversible once table is dropped |

### Phase 3: AcademicYear + Enrollment

| Step | Deliverable |
|---|---|
| Create `AcademicYear` model | Year definition |
| Create `StudentEnrollment` model | Per-year enrollment tracking |
| Data migration: populate enrollments from current data | One enrollment per active student |
| Update `MealSubscription` to reference AcademicYear | Scoping |
| **Risk** | Medium — new models, but additive |

### Phase 4: Portals + Social Auth + Mobile

| Step | Deliverable |
|---|---|
| Create `ParentProfile` | Parent-specific fields |
| Create student/Staff/parent portals | Separate Django apps or SPA |
| Add `django-allauth` | Microsoft/Google login |
| Add JWT authentication | Mobile API |
| **Risk** | Low-Medium — new features |

### Phase 5: Multi-School

| Step | Deliverable |
|---|---|
| Create `Organization`, `School`, `Campus` | Full legal/physical hierarchy |
| Create domain-owned Department model if needed | Academic/HR/finance/operations department concept |
| Activate `Person.school` FK | Required |
| Change `h_code` uniqueness to `unique_together` | Multi-school scope |
| Add tenant middleware | Request scoping |
| **Risk** | High — significant architectural change |

---

## Appendix: Model Relationship Diagram (Text)

```
auth.User (unchanged)
    │
    │ optional O2O
    ▼
Person ──────► RoleType (via PersonRole)
  │                │
  │ O2O            │ M2M through
  ├── StudentProfile
  ├── StaffProfile
  ├── ParentProfile (future)
  ├── GuestProfile (future)
  └── VendorProfile (future)

Wallet ───────────► Person (was Student)
WalletTransaction ──► Person (was Student)
MealSubscription ──► Person (was Student)
FaceEmbedding ─────► Person (was Student)
AttendanceRecord ──► Person (was Student)
AttendanceEvent ───► Person (was Student)

MealRecord ────────► AttendanceRecord (no direct Person FK)
```

---

## Appendix: Current → Target Field Mapping

| Current `Student` | Target `Person` | Target `StudentProfile` |
|---|---|---|
| `h_code` | `h_code` | — |
| `first_name` | `first_name` | — |
| `middle_name` | `middle_name` | — |
| `last_name` | `last_name` | — |
| `gender` | `gender` | — |
| `grade` | — | `grade` |
| `has_meal` | — | `has_meal` |
| `has_bus` | — | `has_bus` |
| `is_active` | `is_active` | — |
| — | `email` | — |
| — | `phone` | — |
| — | `date_of_birth` | — |
| — | `photo` | — |
| — | `user` (O2O to auth.User) | — |
| — | `ext_id` | — |
| — | — | `homeroom` |
| — | — | `legacy_student` |
| — | — | `person` (PK) |
