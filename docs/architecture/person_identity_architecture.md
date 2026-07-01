# Person Identity Architecture — BISK_RFv4

Date: 2026-06-30
Branch: feature/person-architecture
Status: Version 1.0 Candidate — aligned with dedicated identity app plan

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
| **Multi-tenant ready** | Organization/School/Campus hierarchy is designed conceptually but activated only when needed. |
| **Domain ownership first** | Identity models live in `apps.identity`; attendance, finance, academics, portal, HR, and future modules depend on identity, not the other way around. |
| **Transitional fields are temporary** | `StudentProfile.grade`, `has_meal`, and `has_bus` are Phase 1 compatibility mirrors. New features should not treat them as long-term source-of-truth fields. |

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

Not implemented. Phase 1 does **not** add `Organization`, `School`, `Campus`, `Department`, `Person.school`, or `Person.ext_id`.

Identity models will live in `apps.identity`. Multi-school fields are intentionally deferred until the Organization/School/Campus models exist and a real multi-school requirement is implemented.

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

In Phase 1, Person is owned by the dedicated Django app:

```text
apps.identity
```

Attendance, finance, academics, portal, HR, and future modules should depend on `apps.identity`, not on `apps.attendance`.

### Design

```python
# apps/identity/models.py

class Person(models.Model):
    # --- Identity ---
    h_code = models.CharField(
        max_length=32,
        unique=True,
        db_index=True,
        help_text="Unique human-readable identifier. Replaces Student.h_code.",
    )
    first_name = models.CharField(max_length=100, blank=True, default="")
    middle_name = models.CharField(max_length=100, blank=True, default="")
    last_name = models.CharField(max_length=100, blank=True, default="")
    gender = models.CharField(
        max_length=6,
        choices=[("MALE", "male"), ("FEMALE", "female")],
        blank=True,
        null=True,
        db_index=True,
    )
    date_of_birth = models.DateField(blank=True, null=True)

    # --- Contact ---
    email = models.EmailField(blank=True, default="")
    phone = models.CharField(max_length=20, blank=True, default="")
    address = models.TextField(blank=True, default="")

    # --- Photo ---
    photo = models.ImageField(
        upload_to="person_photos/",
        blank=True,
        help_text="Official portrait photo.",
    )

    # --- Status ---
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        help_text="Deactivate to soft-delete a person.",
    )

    # --- Auth bridge (Phase 1 only) ---
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="person",
        help_text="Optional link to a login account.",
    )

    # --- Multi-school / ERP future ---
    # Do NOT add school/ext_id in Phase 1.
    # They are deferred until Organization/School/Campus and external ERP/SIS
    # integration are implemented.

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
| No `school` FK in Phase 1 | A Person may eventually have roles in multiple schools; school scoping belongs later on roles/assignments when Organization/School/Campus exists. |
| No `ext_id` in Phase 1 | External ERP/SIS integration is deferred until a real integration is implemented. |

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
- Sync signals are postponed in Phase 1 to avoid unnecessary bidirectional complexity.
- Login accounts can be linked through migration/admin/management commands.

#### Phase 2 (future — after squash + preparation)

```
Person(AbstractUser) with AUTH_USER_MODEL = "identity.Person"
```

- Set `AUTH_USER_MODEL` in settings.
- Copy existing `auth_user` data into `identity_person`.
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
# apps/identity/models.py

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

    code = models.CharField(
        max_length=32,
        unique=True,
        choices=ROLE_CHOICES,
        help_text="Machine-readable role code.",
    )
    name = models.CharField(
        max_length=100,
        help_text="Human-readable role name.",
    )
    is_active = models.BooleanField(default=True)
    is_system = models.BooleanField(
        default=False,
        help_text="Seeded/system roles should not be renamed or deleted casually.",
    )

    class Meta:
        ordering = ["code"]

    def __str__(self):
        return self.name
```

---

## 8. PersonRole Model

PersonRole links a Person to a RoleType. In Phase 1 it is not scoped to AcademicYear or School because those models are not implemented yet. Future versions can add AcademicYear/School scoping when the academic and multi-school domains are introduced.

### Design

```python
# apps/identity/models.py

class PersonRole(models.Model):
    person = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name="roles",
    )
    role_type = models.ForeignKey(
        RoleType,
        on_delete=models.PROTECT,
        related_name="person_roles",
    )

    # --- Validity window ---
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)

    # Future scoping fields — NOT IMPLEMENTED IN PHASE 1:
    # academic_year = ForeignKey(AcademicYear, null=True)  # Phase 3+
    # school = ForeignKey(School, null=True)               # Phase 5+

    # --- Status ---
    is_active = models.BooleanField(default=True, db_index=True)
    notes = models.CharField(max_length=200, blank=True, default="")

    # --- Metadata ---
    assigned_at = models.DateTimeField(auto_now_add=True)
    assigned_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )

    class Meta:
        unique_together = [("person", "role_type")]
        indexes = [
            models.Index(fields=["person", "is_active"]),
            models.Index(fields=["role_type", "is_active"]),
            models.Index(fields=["start_date", "end_date"]),
        ]
        ordering = ["person", "role_type"]

    def __str__(self):
        return f"{self.person.h_code} → {self.role_type.code}"
```

### Usage patterns

| Pattern | Example |
|---|---|
| Active student | `PersonRole(person=X, role_type=student, is_active=True)` |
| Staff member | `PersonRole(person=Y, role_type=staff, is_active=True)` |
| Staff + Parent | Two PersonRole rows for Z: one staff, one parent |
| Student becomes staff next year | Old student role closed, new staff role opened |

---

## 9. StudentProfile

StudentProfile extends Person with student-specific data. It is the eventual replacement for the current `Student` model.

Important: `grade`, `homeroom`, `has_meal`, and `has_bus` are **transitional compatibility mirrors** from the current `Student` model. New features should not treat them as long-term source-of-truth fields. Grade/homeroom move to `StudentEnrollment`; meal status moves to meal subscriptions; bus status moves to the future transport domain.

### Design

```python
# apps/identity/models.py

class StudentProfile(models.Model):
    person = models.OneToOneField(
        Person,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="student_profile",
    )

    # --- Transitional compatibility fields ---
    # These mirror current Student fields during Phase 1 only.
    # Long-term source of truth:
    # - grade/homeroom → StudentEnrollment + AcademicYear
    # - has_meal → MealSubscription / meal domain
    # - has_bus → Transport domain
    grade = models.CharField(max_length=32, blank=True, null=True, db_index=True)
    homeroom = models.CharField(max_length=64, blank=True, default="")
    has_meal = models.BooleanField(default=False, db_index=True)
    has_bus = models.BooleanField(default=False, db_index=True)

    # --- Legacy ---
    legacy_student = models.OneToOneField(
        "attendance.Student",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
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
# apps/identity/models.py

class StaffProfile(models.Model):
    person = models.OneToOneField(
        Person,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="staff_profile",
    )

    # --- Employment ---
    employee_id = models.CharField(
        max_length=32,
        blank=True,
        default="",
        help_text="HR/ERP employee identifier.",
    )
    job_title = models.CharField(max_length=100, blank=True, default="")
    hire_date = models.DateField(null=True, blank=True)

    # --- Flags ---
    is_teacher = models.BooleanField(
        default=False,
        help_text="If true, this staff member can later be assigned teaching periods.",
    )

    # Department is intentionally omitted in Phase 1.
    # It belongs later to StaffAssignment / HR / Academic domain models.

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

The identity model is designed so multi-school support can be added later without forcing a redesign, but Phase 1 does **not** add multi-school fields.

| Element | Phase 1 | Future multi-school support |
|---|---|---|
| `Person.school` | Not implemented | Add only when `Organization`, `School`, and `Campus` exist and the ownership decision is confirmed |
| `PersonRole.school` | Not implemented | Add when roles must be scoped per school/campus |
| `StudentProfile` | No school FK | Inherits school context through future enrollment/role scope |
| `StaffProfile` | No school FK | Inherits school context through future assignment/role scope |
| `h_code` uniqueness | Globally unique | Revisit when multi-school mode is activated |
| `ext_id` | Not implemented | Add when external SIS/ERP integration requires it |

### Why no `Person.school` in Phase 1?

A Person may eventually hold roles across multiple schools, campuses, or organizations. Therefore, school scoping should usually belong to `PersonRole`, `StudentEnrollment`, `StaffAssignment`, or other domain-specific assignment models rather than being forced onto the Person identity itself too early.

### What changes when multi-school is activated

1. Create `Organization`, `School`, and `Campus`.
2. Decide whether school scoping belongs to Person, PersonRole, Enrollment, Assignment, or all of them.
3. Add required FKs only where real business queries need them.
4. Decide whether `h_code` remains globally unique or becomes unique per school.
5. Consider tenant middleware or schema-per-tenant only when the deployment model requires it.

### Implementation order

Not implemented in the first Person migration.

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
| `AUTH_USER_MODEL = "identity.Person"` | Irreversible on existing database. Deferred to Phase 2. |
| `Organization` / `School` / `Campus` models | Multi-tenant not needed yet. No `Person.school` FK in Phase 1. |
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
| Create `Person` model | Identity fields and optional `user` O2O only; no `school` FK or `ext_id` in Phase 1 |
| Create `RoleType` model | Seeded system roles including student, staff, teacher, parent, guardian, guest, vendor, administrator, finance, HR, principal, vice principal, librarian, nurse |
| Create `PersonRole` model | M2M through model with status and optional validity dates; year/school scoping deferred |
| Create `StudentProfile` model | Student-specific fields, `legacy_student` backlink |
| Create `StaffProfile` model | Staff-specific fields |
| Auth.User ↔ Person sync signal | Deferred; avoid bidirectional sync in Phase 1 |
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
| Decide and implement school scoping | Possibly PersonRole/Enrollment/Assignment, not automatically Person |
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
| `grade` | — | `grade` (transitional; future StudentEnrollment) |
| `has_meal` | — | `has_meal` (transitional; future meal domain) |
| `has_bus` | — | `has_bus` (transitional; future transport domain) |
| `is_active` | `is_active` | — |
| — | `email` | — |
| — | `phone` | — |
| — | `date_of_birth` | — |
| — | `photo` | — |
| — | `user` (O2O to auth.User) | — |
| — | — | `homeroom` |
| — | — | `legacy_student` |
| — | — | `person` (PK) |
