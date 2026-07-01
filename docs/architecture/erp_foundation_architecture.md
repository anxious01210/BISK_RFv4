# ERP Foundation Architecture — BISK_RFv4

Date: 2026-07-01
Branch: feature/person-architecture
Version: 1.0 Candidate
Status: Draft — pending ChatGPT review

---

## 1. Vision

BISK_RFv4 will evolve from a school attendance and meal-management system into a **full-school ERP / LMS / portal / mobile / API platform** capable of serving a single school today and multiple schools/tenants in the future.

The long-term vision:

```
BISK_RFv4 Platform
├── Identity & Access (Person, User, Role, Auth)
├── Academic Core (AcademicYear, Enrollment, Assignment, Grades, Classes, Subjects)
├── Timetable & Scheduling (Periods, Lectures, Rooms)
├── Attendance & Recognition (Face, Period, Daily, Staff)
├── Finance & Payments (Wallet, Invoices, Receipts, Installments)
├── Meals & Services (Subscriptions, Meal Records, Service Categories)
├── Assessment & Marks (Exams, Continuous Assessment, Report Cards)
├── Admissions (Applications, Assessments, Enrollment Pipeline)
├── Communication (Notifications, Chat, News, Events)
├── Behaviour & Discipline (Incidents, House Points)
├── Health & Clinic (Referrals, Records)
├── Portals (Student, Staff, Parent — Web + Mobile)
├── API Layer (REST, JWT, Mobile-First)
├── Reporting & Analytics (Attendance, Finance, Academic, Behavior)
├── Multi-Tenant / Multi-School (Organization, School, Campus)
└── External Integration (SAP, SIS, Payment Gateways, SSO)
```

The core principle: **one unified identity, many roles, many subsystems, optional login.**

---

## 2. Scope

### What BISK_RFv4 IS

- A Django-based school ERP/LMS platform.
- A face-recognition attendance system.
- A meal/wallet management system.
- A future academic/assessment/timetable system.
- A future portal/mobile/API platform.
- A future multi-tenant/multi-school platform.

### What BISK_RFv4 is NOT

- NOT a replacement for external accounting/ERP (SAP integration is a future connector).
- NOT a real-time communication platform (Chat is a future module, not the core product).
- NOT a Learning Management System in the Moodle/Canvas sense (LMS features are academic management, not course-authoring/content-delivery).
- NOT a payment gateway (it records payments, but does not process card/bank transactions directly).

### Boundary with legacy BISK LMS

The legacy Node.js/Next.js LMS is a separate system with overlapping domains. BISK_RFv4 will **not** reimplement all legacy features immediately. It will build a cleaner foundation that can eventually absorb the legacy system's workflows. See Section 30 (Relationship to Legacy BISK LMS).

---

## 3. Architecture Principles

| # | Principle | Description |
|---|-----------|-------------|
| 1 | **Identity before role** | A Person exists independently of their role. The same person can be a student one year and staff the next. |
| 2 | **Person ≠ User** | Business identity (Person) is separate from login identity (auth.User). A person may have zero, one, or multiple login accounts. |
| 3 | **Roles are M2M, not a field** | Do not use a single `person_type` CharField. Use RoleType + PersonRole so a person can hold multiple simultaneous roles. |
| 4 | **Profiles extend roles** | StudentProfile, StaffProfile, etc. are optional role-specific data extensions. A Person with role "parent" may have no profile. |
| 5 | **AcademicYear is foundational** | Many domain models are scoped to an academic year: enrollments, assignments, subscriptions, timetables, assessments. |
| 6 | **Backward compatibility always** | Never break existing Student-based code during migration. Dual-write, proxy models, and compat properties. |
| 7 | **Incremental delivery** | Each phase is independently deployable. No big-bang migrations. |
| 8 | **Multi-tenant ready, not active** | Organization/School/Campus hierarchy is designed into the identity model but not activated until needed. |
| 9 | **Service-layer business logic** | Business logic lives in services/helpers, not in views, templates, or models. |
| 10 | **Money is auditable** | All financial transactions create permanent records. No silent balance changes. |
| 11 | **Configuration over hard-coding** | Where possible, domain concepts are configurable from Django admin. Core domain structure remains coded. |
| 12 | **Domain boundaries respected** | Apps have clear ownership. The attendance app does not own finance or academic domains long-term. |
| 13 | **Workflow as state machine** | Business processes should be modeled as state transitions, not scattered boolean flags. Use status enums, valid transition maps, and explicit workflow steps. |

### Workflow/state-machine principle — detail

The system must resist the temptation to model business processes with boolean flags:

| Anti-pattern (scattered booleans) | Pattern (state machine) |
|----------------------------------|------------------------|
| `is_approved`, `is_paid`, `is_completed` on the same model | Single `status` field with valid transitions |
| Two booleans creating 4 states, 2 of which are invalid | Enum with exactly the valid states |
| Checking `if is_approved and not is_cancelled` everywhere | One `status` switch |

Examples of domain workflows that should use state machines:

| Domain | States | Transition triggers |
|--------|--------|-------------------|
| Admissions | `applied → assessed → accepted → enrolled → rejected` | Assessment score, admin decision |
| Invoice | `draft → issued → partial_paid → paid → overdue → cancelled → written_off` | Payment received, date passed |
| Purchase/Procurement | `requested → approved → ordered → received → invoiced → paid` | Approver action, delivery |
| Refund | `requested → reviewed → approved → processed → completed → rejected` | Admin review, payment processor |
| Leave request | `submitted → approved_by_supervisor → hr_approved → taken → cancelled` | Approver actions |
| Behaviour/Discipline | `reported → investigated → resolved → appealed → closed` | Investigator action, appeal |

**This does NOT mean implementing a general-purpose workflow engine (Camunda, SimpleWorkflow, django-workflows).** Instead:

- Each model carries its own `status` CharField with `choices`.
- Each model documents valid transitions (as a `VALID_TRANSITIONS` dict or `clean()` method).
- Each transition may trigger side effects via signals or service methods.
- Status changes are audited (who changed, when, from what state, to what state).

A general-purpose workflow engine is explicitly **not** implemented in any planned phase. The recommendation is to model workflows per-domain with simple enums and service methods. If the system later has 10+ workflows with complex branching, a workflow library can be introduced as a refactoring rather than an up-front architecture decision.

---

## 4. Domain Boundaries

Future long-term app structure (not all apps exist yet):

```
apps/
├── identity/          # Person, RoleType, PersonRole, profiles (moved from attendance)
├── attendance/        # AttendanceRecord, AttendanceEvent, FaceEmbedding, RecognitionSettings, Camera (current)
├── scheduler/         # PeriodTemplate, PeriodOccurrence, ResourceSettings (current)
├── academic/          # AcademicYear, Grade, Class, Section, Subject, Term, Exam (future)
├── enrollment/        # StudentEnrollment, StaffAssignment, TeachingAssignment (future)
├── timetable/         # Day, Period, Lecture, Classroom, Location (future)
├── meal/              # MealProfile, MealSubscription, MealRecord (current, may split)
├── wallet/            # Wallet, WalletTransaction (current)
├── finance/           # Invoice, Receipt, Installment, Discount (future)
├── assessment/        # Mark, Exam, ReportCard, ContinuousAssessment (future)
├── admissions/        # Application, AdmissionAssessment (future)
├── guardian/          # ParentProfile, GuardianRelationship, Family (future)
├── communication/     # Notification, Chat, News, Event (future)
├── behaviour/         # BehaviourIncident, HousePoint (future)
├── health/            # ClinicReferral, MedicalRecord (future)
├── portal/            # Student/Staff/Parent portal views (future)
├── api/               # API versioning, JWT, throttling (future)
├── reporting/         # Report generation, exports, analytics (future)
└── licensing/         # Remote software licensing (future)
```

### Current reality

Currently, everything lives in `apps/attendance/` or `apps/cameras/` or `apps/scheduler/`. Extraction into separate apps is a future refactoring step. The important constraint is that **new code should be written in a way that anticipates extraction**, not that extraction happens now.

---

## 5. Organization / School / Campus

### Long-term hierarchy

```
Organization (e.g., "BISK Educational Group")
    └── School (e.g., "BISK K-12 Baghdad")
            └── Campus (e.g., "Main Campus", "Girls Campus")
```

**Department is NOT part of this physical/legal hierarchy.** Departments belong to academic, HR, or finance domains depending on use. They are cost centers and organizational units, not physical locations.

### Department placement

| Domain | Example Department | Belongs to |
|--------|-------------------|------------|
| Academic | Mathematics, English, Science | Academic structure (linked to Campus optionally) |
| HR/Admin | Administration, HR, Payroll | Organization/HR domain |
| Operations | Maintenance, Transport, Security | Operations domain |
| Finance | Accounting, Billing, Procurement | Finance domain |

Departments may later act as academic departments (subject-grouping for teachers), HR departments (cost centers for staff assignment), or finance departments (budget centers). A single Department model with a type/category field may serve all domains, or separate models per domain may be created when needed. This decision is deferred.

### Current implementation

Not implemented. Person has an optional `school` FK (nullable) and `ext_id` field prepared for future multi-tenant activation.

### Key rules

- A Person belongs to exactly one **primary** School/Campus.
- A Person can hold roles at multiple Schools via PersonRole (future).
- Organization/School/Campus models are NOT required for single-school mode.
- The `h_code` uniqueness constraint must be changed to `unique_together = [("school", "h_code")]` when multi-tenant is activated.
- `ext_id` enables cross-system identity matching (ERP, SIS, legacy LMS).

---

## 6. AcademicYear

### Status

Foundation concept. Not yet a database model.

### Design intent

```python
class AcademicYear(models.Model):
    name = CharField(max_length=100)             # e.g., "2026-2027"
    code = CharField(max_length=32, unique=True) # e.g., "2026-27"
    start_date = DateField()
    end_date = DateField()
    is_active = BooleanField(default=False)      # current/active year
    school = ForeignKey(School, null=True)       # multi-tenant future
```

### Why AcademicYear is foundational

These future models are scoped by AcademicYear:

| Model | Why year-scoped |
|-------|----------------|
| StudentEnrollment | Student grade/class/status per year |
| StaffAssignment | Staff role/department per year |
| TeachingAssignment | Teacher-subject-class per year |
| MealSubscription | Subscription coverage period |
| DiscountAssignment | Discount validity period |
| PeriodTemplate/Occurrence | Timetable per year |
| AssessmentScheme | Mark/grade scheme per year |
| Invoice | Billing per academic term/year |

### Current workaround

Existing models (MealSubscription, PeriodOccurrence) use date fields directly. This is acceptable until the AcademicYear model is introduced.

---

## 7. Person / Identity

### Core model

Person is the **central business identity** in the entire ERP system. Every human actor — student, staff, teacher, parent, guardian, guest, vendor — is represented as a Person.

### Design (from person_identity_architecture.md)

```python
class Person(models.Model):
    h_code = CharField(max_length=32, unique=True)   # natural key
    first_name, middle_name, last_name               # name fields
    gender, date_of_birth                            # demographic
    email, phone, address                            # contact
    photo                                            # portrait
    is_active                                        # soft delete
    user = OneToOneField(auth.User, null=True)       # optional auth bridge
    ext_id                                           # external ERP/SIS ID
    school = ForeignKey("School", null=True)          # future multi-tenant
```

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| `h_code` is the natural key | Already used as the primary lookup. Avoids exposing internal PKs. |
| No `username` field | `auth.User` handles usernames. Person uses `h_code`. |
| No `password` field | Person is NOT the auth model. Passwords stay on `auth.User`. |
| `user` FK is nullable | A child in kindergarten does not need login access. |
| `school` FK is nullable | Multi-tenant is future. |
| `ext_id` for ERP | Prepares for external SIS/ERP integration. |

### Person is NOT a Django User

- Person represents a real human being in the school ecosystem.
- `auth.User` represents a login account.
- A Person may exist without any login account.
- A Person may have multiple login accounts in the future (multi-system).
- `auth.User` may exist for system access without a corresponding Person (service accounts, API-only users).

---

## 8. Authentication

### Current state

- `AUTH_USER_MODEL` is not set. Django defaults to `auth.User`.
- 4 login users exist (staff/supervisors only).
- Students have zero login accounts.

### Strategy: Option D (Progressive Hybrid)

**Do NOT change `AUTH_USER_MODEL` during Phase 1.** Keep `auth.User` as the authentication model. Person has an optional `OneToOneField("auth.User")` bridge.

#### Phase 1 (current phase)

```
auth.User ──(optional O2O)──→ Person
```

- All authentication flows remain unchanged.
- `request.user` is still `auth.User`.
- A `post_save` signal on `auth.User` syncs key fields to linked Person.
- A `post_save` signal on Person creates `auth.User` if needed (e.g., for staff).

#### Phase 2 (future — after squash + preparation)

```
Person(AbstractUser) with AUTH_USER_MODEL = "attendance.Person"
```

- Set `AUTH_USER_MODEL` in settings.
- Copy existing `auth_user` data into `attendance_person`.
- Update FK fields referencing `settings.AUTH_USER_MODEL`.
- Drop the `user` O2O bridge from Person.
- `request.user` is now `Person`.

#### Why not now

Django explicitly warns against changing `AUTH_USER_MODEL` after migrations have been applied. Phase 2 should be a separate, dedicated migration effort after Person is proven in production.

### Future auth methods

- JWT for mobile API (via `rest_framework_simplejwt`).
- Microsoft/Google OAuth for portals (via `django-allauth`).
- Social accounts link to `auth.User`, which links to Person.

### IdentityProvider concept (future extension point)

The authentication architecture should anticipate multiple identity providers beyond local `auth.User`. An IdentityProvider model is a future extension point:

```python
class IdentityProvider(models.Model):
    code = CharField(max_length=32, unique=True)   # "local", "google", "microsoft", "ldap", "azure_ad", "oidc"
    name = CharField(max_length=100)
    is_active = BooleanField(default=True)
    config = JSONField(default=dict, blank=True)   # provider-specific settings
```

**This model is NOT implemented now.** It is documented to ensure that the Person-User bridge does not assume only local authentication will ever exist.

| Provider | Use case | Phase |
|----------|----------|-------|
| `local` | Username/password via Django auth | Phase 1 (current) |
| `google` | Google Workspace SSO for staff/parent portals | Phase 4+ |
| `microsoft` | Microsoft 365 / Azure AD SSO | Phase 4+ |
| `ldap` | Active Directory / LDAP for on-prem deployments | Phase 6+ |
| `azure_ad` | Azure AD with OIDC | Phase 6+ |
| `oidc` | Generic OpenID Connect provider | Phase 6+ |

The key design constraint: **auth.User remains the single Phase 1 login identity.** IdentityProvider is purely a future concept to keep the architecture extensible. When implemented, `auth.User` would gain an optional `identity_provider` FK or a separate `ExternalIdentity` model would store provider-specific identifiers linked to Person.

---

## 9. Authorization

### Two separate systems

| System | Purpose | Technology |
|--------|---------|------------|
| **Django Groups/Permissions** | System access control. Who can access admin, API, dashboards. | `auth.Group` + `auth.User.groups` + Django permissions framework |
| **PersonRole** | Business role. What a person IS in the school context. | `RoleType` + `PersonRole` |

### Rule of thumb

- If a permission controls **access to a Django view/endpoint/admin page**, use **Django Groups**.
- If a permission controls **business logic** (e.g., "can this staff member confirm meals for this student?"), use **PersonRole**.
- The two systems can be combined: a PersonRole can imply automatic membership in a Django Group via signal.

### Long-term permission model

DRF permission classes should check PersonRole:

```python
class HasPersonRole(BasePermission):
    def has_permission(self, request, view):
        person = getattr(request.user, "person", None)
        if not person:
            return False
        required_roles = getattr(view, "required_roles", [])
        return person.roles.filter(role_type__code__in=required_roles).exists()
```

### Mapping from current hardcoded groups

| Current Group | Equivalent RoleType | Notes |
|---------------|-------------------|-------|
| `supervisor` | `staff` (with supervisor flag) | Admin-gate access |
| `meal_supervisor` | `staff` (with meal role metadata) | Meal dashboard access |
| `api_user` | `staff` or `guest` | API-only access |

---

## 10. RoleType / PersonRole

### RoleType

A controlled vocabulary of roles a person can hold.

```python
class RoleType(models.Model):
    code = CharField(max_length=32, unique=True)   # "student", "staff", "teacher", "parent", "guest", "vendor"
    name = CharField(max_length=100)                # "Student", "Staff", etc.
    is_active = BooleanField(default=True)
```

Seeded roles:

| Code | Description |
|------|-------------|
| `student` | Enrolled student |
| `staff` | Non-teaching staff (admin, operations, finance) |
| `teacher` | Teaching staff |
| `parent` | Parent/guardian of a student |
| `guest` | Visitor, contractor, temporary access |
| `vendor` | External vendor/supplier |

### PersonRole

Links a Person to a RoleType, optionally scoped to AcademicYear and/or School.

```python
class PersonRole(models.Model):
    person = ForeignKey(Person, CASCADE)
    role_type = ForeignKey(RoleType, PROTECT)
    academic_year = ForeignKey(AcademicYear, null=True)  # null = ongoing
    school = ForeignKey(School, null=True)                # multi-tenant future
    is_active = BooleanField(default=True)
    assigned_at = DateTimeField(auto_now_add=True)
```

### Why not `person_type` CharField

A single `person_type` field cannot represent:
- A staff member who is also a parent of a student.
- A student who is also a teacher assistant.
- A person changing roles across academic years.
- A vendor who is also a parent.

---

## 11. Status Strategy

### Person-level status

`Person.is_active` is the primary status. When deactivated, the person is soft-deleted from the system. This replaces `Student.is_active`.

### Status per role

`PersonRole.is_active` allows deactivating a specific role while keeping the person active in other roles. For example, a student graduates (student role deactivated) but remains in the system as an alumni contact.

### Status per enrollment

`StudentEnrollment.status` (future) tracks enrollment-specific states:

| Status | Meaning |
|--------|---------|
| `enrolled` | Actively enrolled |
| `withdrawn` | Withdrawn mid-year |
| `graduated` | Completed final year |
| `transferred` | Transferred to another school |
| `suspended` | Temporarily suspended |

### Status per assignment

`StaffAssignment.status` (future) tracks employment states:

| Status | Meaning |
|--------|---------|
| `active` | Currently employed |
| `resigned` | Resigned |
| `terminated` | Terminated |
| `on_leave` | On leave of absence |
| `retired` | Retired |

### Guiding principle

- `is_active` on Person = can this person be found in the system?
- `PersonRole.is_active` = does this person currently hold this role?
- `StudentEnrollment.status` = what is the academic status of this enrollment?
- These are separate concerns and should not be conflated into a single field.

---

## 12. Profile Strategy

### Profile = role-specific extension

Profiles extend Person with role-specific data. They are optional — a Person with role "parent" may have no profile.

```
Person
├── StudentProfile   (grade, homeroom, legacy_student, has_meal, has_bus)
├── StaffProfile     (employee_id, job_title, department, hire_date, is_teacher)
├── ParentProfile    (future: children, emergency_contact_priority)
├── GuestProfile     (future: organization, purpose, access_expires)
└── VendorProfile    (future: company, contract_ref)
```

### Implementation order

| Phase | Profiles |
|-------|----------|
| Phase 1 | StudentProfile, StaffProfile |
| Phase 3+ | ParentProfile |
| Future | GuestProfile, VendorProfile |

### Profile design rules

- Each profile uses `OneToOneField(Person, primary_key=True)` — shares PK with Person.
- Profiles should be lean. Do not move every possible field into a profile.
- Fields that belong to a year/term should live on Enrollment or Assignment, not on the profile.
- Medical, financial, and other domain-specific fields should live in their own domain models, not on profiles.

---

## 13. Assignment Concept

### General pattern

Many school operations involve **assigning a Person to something for a period of time**. Rather than building one generic assignment table (which would become a dumping ground), the architecture treats "assignment" as a shared pattern that each domain specializes.

Specialized assignment models (future):

| Assignment model | Binds | Scoped by | Purpose |
|-----------------|-------|-----------|---------|
| **StudentEnrollment** | StudentProfile → AcademicYear | Grade, Section, status | Academic registration |
| **StaffAssignment** | StaffProfile → AcademicYear | Department, job title, is_primary | Employment/role |
| **TeachingAssignment** | StaffProfile → Subject → ClassGroup → AcademicYear | Teaching load, sections | Who teaches what to whom |
| **ClubAssignment** | Person → Club → AcademicYear | Role in club | Extracurricular |
| **BusAssignment** | Person → BusRoute → AcademicYear | Stop, pickup/dropoff | Transport |
| **ServiceAssignment** | Person → ServiceCategory → AcademicYear | Allocation | Student services |

### Design rules

- Each assignment model has its own table, fields, and constraints.
- Every assignment is scoped by **AcademicYear** (for year-specific) or uses date fields (for date-range assignments).
- Every assignment has a **status** field (enum, not boolean) to support state transitions.
- The assignment targets a Person (via a Profile FK), not the Person directly, unless the assignment applies to any person regardless of role.
- Assignments are **additive** — a person can have multiple assignments in the same domain per year (e.g., teach two subjects, ride two bus routes).
- Overlap validation is the responsibility of each assignment model's `clean()` method.

### What is NOT an assignment

Temporary or transactional records are not assignments:

| Not an assignment | Why |
|------------------|-----|
| `AttendanceRecord` | Transactional event, not a period assignment |
| `MealSubscription` | Product subscription, not a role/location assignment |
| `PersonRole` | Identity role, not a domain-specific assignment |

### Why not one generic assignment table

A single `Assignment` model with a type discriminator (e.g., `assignment_type = "enrollment"|"staff"|"teaching"`) would:

- Lose per-type constraints and validation.
- Require nullable columns for type-specific fields.
- Make FK targets ambiguous (generic FK or JSON).
- Be harder to query, index, and migrate.

The specialized approach keeps each assignment model clean and independently evolvable.

---

## 14. StudentEnrollment

### Purpose

StudentEnrollment tracks a student's enrollment **per academic year**. This replaces the simplistic `Student.is_active` flag and allows the system to know which grade/class a student was in during a given year.

### Future design

```python
class StudentEnrollment(models.Model):
    student = ForeignKey(StudentProfile, CASCADE)
    academic_year = ForeignKey(AcademicYear, CASCADE)
    grade = CharField(max_length=32)
    homeroom = CharField(max_length=64, blank=True)
    enrollment_date = DateField()
    withdrawal_date = DateField(null=True)
    status = CharField(max_length=20, choices=ENROLLMENT_STATUSES, default="enrolled")
```

### Implementation order

Implemented **after** AcademicYear model exists. Not in Phase 1. Current `Student.is_active` suffices temporarily.

### Key rules

- A StudentProfile can have multiple enrollments (one per AcademicYear).
- `PersonRole(person, student)` answers "is this person a student?"
- `StudentEnrollment` answers "what grade/class were they in for year X?"
- Grade and homeroom on StudentProfile are **current** denormalized values, updated by the most recent active enrollment.

---

## 15. StaffAssignment (Assignment subtype)

### Purpose

StaffAssignment tracks a staff member's role, department, and job title per academic year.

### Future design

```python
class StaffAssignment(models.Model):
    staff = ForeignKey(StaffProfile, CASCADE)
    academic_year = ForeignKey(AcademicYear, CASCADE)
    department = ForeignKey(Department, null=True)
    job_title = CharField(max_length=100)
    is_primary = BooleanField(default=True)

class TeachingAssignment(models.Model):
    staff_assignment = ForeignKey(StaffAssignment, CASCADE)
    subject = ForeignKey(Subject)
    class_group = ForeignKey(Class)
    section = ForeignKey(Section, null=True)
```

### Implementation order

Not in Phase 1. Requires AcademicYear, Department, Subject, Class, and Section models.

### Key rules

- A staff member can have multiple assignments in the same year (e.g., teach Math and Science).
- `PersonRole(person, staff)` answers "is this person staff?"
- `PersonRole(person, teacher)` answers "is this person a teacher?"
- `StaffAssignment` answers "what is their role/department for year X?"
- `TeachingAssignment` answers "what subjects and classes do they teach?"

---

## 16. Guardian/Parent Relationship Strategy

### Purpose

Support parent/guardian relationships for parent portal, emergency contacts, and family-level billing.

### Long-term design

```python
class PersonRelationship(models.Model):
    from_person = ForeignKey(Person, CASCADE, related_name="guardianships")
    to_person = ForeignKey(Person, CASCADE, related_name="dependents")
    RELATION_PARENT = "parent"
    RELATION_GUARDIAN = "guardian"
    RELATION_EMERGENCY_CONTACT = "emergency_contact"
    RELATION_SIBLING = "sibling"
    RELATION_OTHER = "other"
    relationship_type = CharField(max_length=20, choices=RELATION_TYPES)
    is_primary = BooleanField(default=False)
    priority = PositiveSmallIntegerField(default=1)

class Family(models.Model):
    name = CharField(max_length=200)
    primary_contact = ForeignKey(Person, null=True)
    is_active = BooleanField(default=True)
```

### Strategy

- Relationships are between two Person records (not Person-to-Student).
- This allows a Person who is a "parent" to also be a "student" (older student who is a parent).
- `PersonRole(person, parent)` indicates the person has a parental role.
- `PersonRelationship` links parent to child.
- `Family` groups people into households for billing and communication.
- Guardian/Parent profiles (`ParentProfile`) are for parent-specific data (e.g., emergency contact priority).

### Implementation order

`PersonRelationship` can be added as a simple model early (Phase 2 or 3). `Family` and `ParentProfile` come later when portals are built.

---

## 17. Academic Structure

### Long-term design

```
SchoolLevel (e.g., "Primary", "Secondary", "High School")
    └── Grade (e.g., "Grade 1", "Grade 10")
            └── Section (e.g., "Section A", "Section B")
                    └── Students (via StudentEnrollment)

Subject (e.g., "Mathematics", "English")
    └── Offered at Grade level via SubjectOffering

Term (e.g., "Term 1", "Term 2", "Term 3")
    └── Belongs to AcademicYear

Exam (e.g., "Midterm", "Final")
    └── Belongs to Term

Mark / Assessment
    └── Belongs to Exam + Student + Subject + Term
```

### Key concepts

| Concept | Description |
|---------|-------------|
| **SchoolLevel** | Educational stage (Primary, Secondary) |
| **Grade** | Year group (Grade 1...Grade 12) |
| **Section** | Class division within a grade (Section A, B) |
| **Subject** | Academic subject (Math, English, Science) |
| **SubjectOffering** | Subject offered at a specific grade in a specific year |
| **Term** | Academic period within a year (Term 1, Term 2, Term 3) |
| **ClassGroup** | A specific group of students taking a subject with a teacher |

### Design rules

- Grade is NOT a field on Person. It belongs to StudentEnrollment for a given AcademicYear.
- Section is NOT on Person. It belongs to StudentEnrollment.
- Subject is a reusable entity, not tied to a specific grade.
- SubjectOffering links Subject + Grade + AcademicYear.
- ClassGroup links Teacher + Subject + Section + AcademicYear for timetable and assessment purposes.

### Implementation order

| Component | Phase |
|-----------|-------|
| Grade | Phase 3 (with AcademicYear) |
| Section | Phase 3 |
| Subject | Phase 3 |
| SubjectOffering | Phase 3 |
| Term | Phase 3 |
| ClassGroup | Phase 3 |
| Exam | Phase 4+ |
| Mark/Assessment | Phase 4+ |
| ReportCard | Phase 5+ |

---

## 18. Finance/Payment Foundation

### Vision

Beyond the meal wallet, BISK_RFv4 should support a general finance module for invoices, receipts, installments, payments, discounts, and financial reporting.

### Long-term design

```python
class Invoice(models.Model):
    person = ForeignKey(Person, CASCADE)
    academic_year = ForeignKey(AcademicYear, null=True)
    invoice_number = CharField(max_length=64, unique=True)
    issue_date = DateField()
    due_date = DateField()
    total_amount = DecimalField(max_digits=12, decimal_places=3)
    paid_amount = DecimalField(max_digits=12, decimal_places=3, default=0)
    status = CharField(max_length=20, choices=INVOICE_STATUSES)
    notes = TextField(blank=True)

class InvoiceItem(models.Model):
    invoice = ForeignKey(Invoice, CASCADE)
    description = CharField(max_length=200)
    quantity = PositiveIntegerField(default=1)
    unit_price = DecimalField(max_digits=10, decimal_places=3)
    total_price = DecimalField(max_digits=12, decimal_places=3)

class Receipt(models.Model):
    invoice = ForeignKey(Invoice, CASCADE)
    person = ForeignKey(Person, CASCADE)
    amount = DecimalField(max_digits=12, decimal_places=3)
    payment_method = CharField(max_length=32)   # "cash", "card", "bank_transfer", "wallet"
    reference = CharField(max_length=128, blank=True)
    received_at = DateTimeField(auto_now_add=True)

class Installment(models.Model):
    invoice = ForeignKey(Invoice, CASCADE)
    due_date = DateField()
    amount = DecimalField(max_digits=12, decimal_places=3)
    paid = BooleanField(default=False)
```

### Design rules

- All financial records target Person, not Student.
- Invoice is at Person level, but items can be scoped to a specific student/child.
- AcademicYear scoping enables per-year billing.
- The meal wallet is the first implemented finance-adjacent subsystem.
- Later finance features should reuse the same Person+AcademicYear anchoring.

### Implementation order

| Component | Phase |
|-----------|-------|
| Wallet (current) | Phase 0 (exists) |
| Wallet → Person (FK migration) | Phase 1.5 |
| Discount engine | Phase 4 |
| Invoice/Receipt/Installment | Phase 5+ |

---

## 19. Wallet/Lunch Foundation

### Current state

- `Wallet` has FK to `Student`.
- `MealSubscription` has FK to `Student`.
- `MealRecord` links through `AttendanceRecord.student`.

### Target state

- `Wallet.person` replaces `Wallet.student`.
- `MealSubscription.person` replaces `MealSubscription.student`.
- `FaceEmbedding.person` replaces `FaceEmbedding.student`.
- `AttendanceRecord.person` replaces `AttendanceRecord.student`.

### Migration strategy

Dual-FK pattern for each model:

1. Add `person` FK (nullable).
2. Data migration: populate `person_id` from `student_id` → `StudentProfile` → `Person`.
3. Add database constraint (not null).
4. Update query code to use `person`.
5. In a separate release: drop `student` FK.

Order of migration:

1. `Wallet` + `WalletTransaction`
2. `MealSubscription`
3. `FaceEmbedding`
4. `AttendanceRecord`
5. `AttendanceEvent`

### Discount engine (future)

Pricing pipeline:

```
base meal price
→ applicable subscription profile
→ applicable discounts/rules
→ final price
→ wallet transaction
→ audit snapshot
```

---

## 20. Attendance/Recognition Foundation

### Current scope

- Student face-recognition attendance.
- Camera/runner infrastructure.
- Attendance events and records.

### Future scope (Person-targeted)

```
AttendanceRecord.person          (replaces Student FK)
AttendanceEvent.person           (replaces Student FK)
FaceEmbedding.person             (replaces Student FK)
```

### Expanded attendance types

| Type | Description | Phase |
|------|-------------|-------|
| Face recognition | Camera-based student attendance | Current |
| Period attendance | Per-class/period attendance | Future |
| Staff attendance | Staff check-in/check-out | Future |
| Daily/Morning attendance | Roll-call style | Future |

### Design rules

- `AttendanceRecord` and `FaceEmbedding` target Person, enabling staff/teacher attendance and face recognition.
- Attendance can be scoped by AcademicYear for reporting.
- Attendance records should be auditable (who confirmed, when, via which camera/method).
- The camera/runner infrastructure is already Person-agnostic — only the FK target changes.

---

## 21. Timetable Foundation

### Concepts from legacy LMS

The legacy system includes: Day, Period, Lecture, Classroom, Location.

### Long-term design

```python
class Day(models.Model):
    name = CharField(max_length=20)    # "Sunday", "Monday", ...
    code = CharField(max_length=10)    # "SUN", "MON", ...
    order = PositiveSmallIntegerField()

class PeriodTemplate(models.Model):
    # Already exists in scheduler app
    academic_year = ForeignKey(AcademicYear, null=True)  # future scoping
    day = ForeignKey(Day)
    start_time = TimeField()
    end_time = TimeField()
    name = CharField(max_length=100)

class Lecture(models.Model):
    class_group = ForeignKey(ClassGroup)
    period = ForeignKey(PeriodTemplate)
    teacher = ForeignKey(StaffProfile)
    subject = ForeignKey(Subject)
    classroom = ForeignKey(Classroom, null=True)
    academic_year = ForeignKey(AcademicYear)
```

### Implementation order

Not in early phases. The existing scheduler/PeriodTemplate infrastructure already supports period definitions. Full timetable with teacher-subject-classroom assignment is Phase 4+.

---

## 22. LMS/Online Learning Foundation

### BISK_RFv4 is NOT a full LMS

BISK_RFv4 will not implement:
- Course authoring tools.
- Content delivery/SCORM.
- Student assignment submission and grading workflow.
- Discussion forums.
- Quiz/assessment engines.

### What BISK_RFv4 WILL support

- Academic structure (grades, subjects, terms, exams, marks).
- Teacher-student-class assignment.
- Mark/grade entry and report cards.
- Attendance tracking per class/period.
- Student/parent portal for viewing academic data.
- Mobile API for grade/attendance lookup.

### Boundary

LMS features are **academic management features** — they manage the structure and records of learning, not the content delivery of learning.

If the school later needs full online learning (Moodle, Canvas, Google Classroom), BISK_RFv4 should be able to sync identity and enrollment data to those platforms via API, not replace them.

---

## 23. Portal/Mobile/API Foundation

### Portal types

| Portal | Auth | Role | Data scope |
|--------|------|------|------------|
| Student portal | Student logs in via `auth.User` | PersonRole(student) | Own attendance, meals, grades, schedule |
| Staff portal | Staff logs in via `auth.User` | PersonRole(staff/teacher) | Their students, classes, reports |
| Parent portal | Parent logs in via `auth.User` | PersonRole(parent) | Their children's data |
| Admin | Existing staff superuser | PersonRole(staff) + is_superuser | Full system |

### API design

```
GET /api/v1/me/
→ { person: { h_code, name, roles: [...], profile: { ... } } }

GET /api/v1/students/
→ PersonRole-scoped student list

GET /api/v1/attendance/
→ Person-scoped attendance records

GET /api/v1/wallet/
→ Person-scoped wallet and transactions
```

### Authentication for mobile

- JWT via `rest_framework_simplejwt`.
- Token-based, no session cookies.
- Scoped to the authenticated Person's roles and data.

### Implementation order

| Component | Phase |
|-----------|-------|
| Basic REST API (DRF) | Exists partially |
| JWT authentication | Phase 4 |
| Student portal | Phase 4 |
| Parent portal | Phase 5 |
| Staff portal | Phase 5 |
| Mobile API hardening | Phase 5+ |

---

## 24. Notifications/Reporting Foundation

### Notification concepts

- Targetable by Person, Role, Class, Family, or Group.
- Multiple channels: in-app, email, SMS, push (FCM).
- Notification templates configurable from admin.
- Automated triggers: attendance events, low wallet balance, payment due, behavior incidents.

### Reporting concepts

- Academic reports: attendance, marks, behavior, finance.
- Scoped by AcademicYear, Grade, Section, Subject, Person.
- Export formats: PDF, Excel, CSV.
- Scheduled/automated report generation.
- Report templates configurable from admin.

### Implementation order

| Component | Phase |
|-----------|-------|
| Notification model + FCM | From legacy, Phase 4+ |
| Email notifications | Phase 4+ |
| Report generation infrastructure | Phase 5+ |
| Automated reporting tasks | Phase 5+ |

---

## 25. Multi-School / Multi-Tenant Readiness

### Design approach

The architecture is designed for multi-tenant but not activated until needed. This means:

- `Person.school` FK exists but is nullable.
- `PersonRole.school` FK exists but is nullable.
- `h_code` is unique now. When multi-tenant is activated, becomes `unique_together = [("school", "h_code")]`.
- `ext_id` enables cross-system identity matching.

### What changes when multi-tenant is activated

1. Organization/School/Campus/Department models become active.
2. `Person.school` becomes required.
3. All business queries become scoped by `school_id`.
4. `h_code` uniqueness changes to per-school scope.
5. Django `django-tenants` or schema-per-tenant approach.
6. `AUTH_USER_MODEL` may need a `school` FK.

### What does NOT change

- The Person model structure.
- The RoleType/PersonRole pattern.
- The Profile strategy.
- The AcademicYear/Enrollment/Assignment patterns.
- The Wallet/Finance patterns.

### Implementation order

Phase 5+. Not now.

---

## 26. Data Ownership Rules

### Core principle

**Person owns their identity data. Domain models own their domain data.**

| Data | Owner | Notes |
|------|-------|-------|
| Person identity fields | Person model | h_code, name, gender, DOB, contact |
| Student-specific fields | StudentProfile | Grade, homeroom (current). Enrollment (future). |
| Staff-specific fields | StaffProfile | Employee ID, job title, department |
| Financial data | Wallet / Finance app | Wallet balance, transactions, invoices |
| Academic data | Academic / Assessment app | Enrollments, marks, grades |
| Attendance data | Attendance app | Records, events, embeddings |
| Authentication data | auth.User | Passwords, groups, permissions |
| Relationship data | PersonRelationship | Guardian/parent links |
| Communication data | Communication app | Notifications, chat, messages |

### Access rules

- A Person can always view their own identity data.
- A Person can view their own wallet, attendance, meals, and assessment data.
- A parent can view their children's data (via PersonRelationship).
- A teacher can view their students' data (via TeachingAssignment + ClassGroup).
- Staff with appropriate Roles can view/export data within their permission scope.
- System admins (Django superusers) can access everything.

### Audit rules

- All financial transactions (wallet deductions, refunds, payments) create permanent records.
- Attendance events log who confirmed and via which method.
- Role assignments log who assigned and when.
- No silent data modification.

---

## 27. What Must Be Configurable from Django Admin

This is a living list. These concepts should be manageable through Django admin rather than requiring code changes:

| Concept | Configurable fields |
|---------|--------------------|
| RoleType | code, name, is_active |
| Academic Years | name, code, start_date, end_date, is_active |
| School Levels | name, code, order |
| Grades | name, code, level, order |
| Sections | name, code, grade |
| Subjects | name, code |
| Terms | name, code, start_date, end_date, academic_year |
| Meal Profiles | name, price, description, active |
| Discount Profiles | name, type, value, priority, active, date range |
| Discount Rules | condition type, operator, value |
| Notification Templates | subject, body, channel, event type |
| Behavior Categories | name, code |
| Service Categories | name, code, description |
| Camera/Device config | name, IP, location, active |
| Recognition Settings | thresholds, modes |
| Global Resource Settings | Period templates, overrides |
| Automated Tasks | schedule, enabled, parameters |
| Report Templates | format, filters, schedule |

---

## 28. What Should Remain Domain-Modeled/Hard-Coded

These are structural domain concepts that should remain coded and migration-controlled:

| Concept | Why not configurable |
|---------|---------------------|
| Person model fields | Core identity structure. Changes require migration. |
| RoleType code values | "student", "staff", "teacher", etc. are domain constants. |
| PersonRole structure | The M2M-through pattern is architectural, not configuration. |
| Profile models (Student, Staff) | One-to-one profile extension is a pattern decision. |
| Wallet/finance structure | Decimal fields, transaction records, audit trail — structural. |
| Attendance/FaceEmbedding pattern | The FK-to-Person design is architectural. |
| AcademicYear-as-FK pattern | Every year-scoped model uses this pattern. |
| Enrollment/Assignment pattern | Year-scoped enrollment and assignment is a design decision. |
| Person-User separation | The identity architecture principle. |
| Django Groups/Permissions separation from PersonRole | Authorization architecture. |
| Relationship pattern (Person-to-Person) | Relationship model structure. |
| Multi-tenant readiness (nullable FKs, ext_id) | Architectural preparation. |

### Flexibility within structure

Within these hard-coded structures, administration and configuration are possible:
- Adding new RoleType codes (via admin: "librarian", "bus_driver").
- Adding new Profile types (requires coded model + migration).
- Adding new PersonRelationship types (via choices, requires coded update).
- Adding new Enrollment statuses (requires coded update to choices).

---

## 29. Migration Strategy from Current BISK_RFv4

### Overview

```
Phase 0                    Phase 1                    Phase 1.5                  Phase 2
[Squash migrations]  [Add Person + Profiles]    [Migrate FKs to Person]    [Deprecate Student]
                            │                           │                          │
  30+ migrations           Person (new)              Wallet.person              Student
  ──→ squashed base         StudentProfile (new)      Meals.person              ──→ dropped
                            StaffProfile (new)        FaceEmbedding.person
                            PersonRole (new)          Attendance.person
                            │                           │
                            Student (unchanged)         Student (unchanged)
                            Data: mirror Student        Data: dual FK write
                            → Person+Profile            → verify consistency
```

### Phase 0: Squash Migrations

- Combine all 30+ current attendance migrations into one `0001_squashed`.
- Run `python manage.py makemigrations --squash-name squashed`.
- Dump production database before proceeding.

### Phase 1: Add Person (additive — no existing table changes)

- Create `Person`, `StudentProfile`, `StaffProfile`, `RoleType`, `PersonRole`.
- Data migration: for each `Student`, create one `Person` + one `StudentProfile` + one `PersonRole(student)`.
- `StudentProfile.legacy_student` links back to original Student row.
- Data migration: link existing 4 `auth.User` rows to their Person.
- Add `PersonAdmin` + `StudentProfileAdmin`.
- Full rollback possible by dropping new tables.

### Phase 1.5: Migrate FK Targets (one model at a time)

1. Wallet + WalletTransaction — simplest, least dependent.
2. MealSubscription.
3. FaceEmbedding.
4. AttendanceRecord — largest table, migrate in batches.
5. AttendanceEvent.

Each follows the same pattern:
1. Add `person` FK (nullable).
2. Data migration: populate `person_id`.
3. Add not-null constraint.
4. Update query code to use `person`.
5. Later release: drop `student` FK.

### Phase 2: Deprecate Student

- Remove `legacy_student` backlink.
- Mark Student as `managed = False`.
- Archive data if needed.
- Drop `attendance_student` table.
- Remove backward-compat shims.

### Phase 3+: AcademicYear, Enrollment, Academic Features

See Implementation Phases (Section 32).

---

## 30. Relationship to Legacy BISK LMS

### Nature of relationship

The legacy BISK LMS (Node.js/Next.js/React) is a separate system. BISK_RFv4 does not depend on it. However, the legacy system contains valuable business knowledge about the school's real workflows.

### What we take from legacy

- Business domain knowledge (what modules exist, what entities are needed).
- Workflow understanding (how enrollment, billing, attendance, assessment work in practice).
- Feature scope reference (what a complete school ERP should include).
- Entity relationships (how students, guardians, families, teachers, classes, subjects relate).

### What we do NOT take

- Database schema (Sequelize models are not copied to Django).
- Code architecture (Express routes are not copied to DRF).
- Implementation decisions (no one-to-one port).
- Technical debt (the legacy system's accumulated workarounds).

### Coexistence strategy

- The systems can run in parallel during migration.
- `ext_id` on Person stores the legacy system's person/student ID for cross-reference.
- Data can be imported from legacy via management commands or API sync.
- Eventually, BISK_RFv4 may fully replace the legacy system module by module.

### Migration order from legacy

```
Identity (Person, roles) ──→ Academic structure (years, grades, subjects)
    ──→ Enrollments ──→ Timetable ──→ Attendance ──→ Marks/Assessment
    ──→ Finance ──→ Admissions ──→ Communication ──→ Reports
```

---

## 31. Relationship to person_identity_architecture.md

### Hierarchy

```
erp_foundation_architecture.md          (this document — top-level ERP blueprint)
    └── person_identity_architecture.md  (detailed identity design — phase 1 implementation guide)
            └── legacy_lms_feature_inventory.md  (business feature source)
```

### What this document adds

- Broader ERP scope beyond identity.
- Domain boundaries for all future subsystems.
- Multi-school / multi-tenant readiness.
- Academic structure (grades, classes, subjects, terms, exams, marks).
- Finance/payment foundation.
- Timetable foundation.
- LMS/online learning foundation.
- Portal/mobile/API foundation.
- Notifications/reporting foundation.
- Data ownership rules.
- Configurable vs hard-coded guidance.
- Open questions.

### What this document inherits from person_identity_architecture.md

- Person model design.
- Person-User separation.
- RoleType + PersonRole pattern.
- Profile strategy.
- AcademicYear concept.
- Organization/School/Campus hierarchy.
- Authentication strategy (Option D progressive hybrid).
- Authorization separation (Django Groups vs PersonRole).
- Wallet/lunch/attendance FK migration strategy.
- Phased implementation approach.

### Consistency requirement

- All decisions in this document are compatible with person_identity_architecture.md.
- Implementation phases align between both documents.
- Migration strategy is shared.
- Model designs are consistent (Person fields, RoleType, PersonRole etc.).

---

## 32. Implementation Phases

### Phase 0: Preparation

| Step | Deliverable |
|------|-------------|
| Squash existing migrations | Single `0001_squashed` migration |
| Dump production database | Rollback safety |
| Freeze current schema | Document current table structure |
| Update AGENTS.md | Record architecture decisions |

### Phase 1: Person Architecture (additive)

| Step | Deliverable |
|------|-------------|
| Create Person model | Identity fields, optional user O2O, school FK, ext_id |
| Create RoleType model | Seeded with student, staff, teacher, parent |
| Create PersonRole model | M2M through with year/school scoping |
| Create StudentProfile model | Student-specific fields, legacy_student backlink |
| Create StaffProfile model | Staff-specific fields |
| Add auth.User ↔ Person sync signal | Bidirectional sync |
| Data migration: Student → Person + Profile | For every existing Student |
| Data migration: link existing auth.User → Person | Link 4 existing users |
| Add PersonAdmin + StudentProfileAdmin | Admin interface |
| Update docs/ | Architecture documentation |

### Phase 1.5: Migrate FK Targets (one model at a time)

| Step | Deliverable |
|------|-------------|
| Migrate Wallet FK | student → person |
| Migrate WalletTransaction FK | student → person |
| Migrate MealSubscription FK | student → person |
| Migrate FaceEmbedding FK | student → person |
| Migrate AttendanceRecord FK | student → person |
| Migrate AttendanceEvent FK | student → person |
| Update MealSubscription.clean() | Use person_id instead of student_id |
| Update all views, templates, admin, services | Remove student FK references |
| Add backward-compat properties | AttendanceRecord.student, etc. |

### Phase 2: Deprecate Student

| Step | Deliverable |
|------|-------------|
| Remove legacy_student backlink | Clean up StudentProfile |
| Mark Student as managed = False | Django ignores the model |
| Archive Student data (if needed) | Before dropping |
| Drop attendance_student table | Final migration |
| Remove backward-compat shims | Properties, dual-FK code |

### Phase 3: Academic Year + Enrollment

| Step | Deliverable |
|------|-------------|
| Create AcademicYear model | Year definition |
| Create Grade, Section, Subject models | Academic structure |
| Create SubjectOffering model | Subject-Grade-Year linking |
| Create StudentEnrollment model | Per-year enrollment tracking |
| Data migration: populate enrollments | One enrollment per active student |
| Create StaffAssignment model | Per-year staff role/department |
| Create TeachingAssignment model | Teacher-Subject-Class linking |
| Update MealSubscription to reference AcademicYear | Scoping |
| Create Term model | Academic periods within a year |

### Phase 3.5: Guardian/Family Relationships

| Step | Deliverable |
|------|-------------|
| Create PersonRelationship model | Parent/guardian linking |
| Create Family model | Household grouping |
| Create ParentProfile model | Parent-specific fields |

### Phase 4: Portals + Assessment + Timetable

| Step | Deliverable |
|------|-------------|
| Create Day, Lecture, Classroom models | Timetable structure |
| Create Exam, Mark, Assessment models | Assessment structure |
| Create Student portal | Student-facing views |
| Add JWT authentication | Mobile API |
| Create Notification infrastructure | FCM, email, templates |
| Create Discount engine | Discount models + pricing service |

### Phase 5: Finance + Reporting

| Step | Deliverable |
|------|-------------|
| Create Invoice, Receipt, Installment models | Finance structure |
| Create Staff portal | Staff-facing views |
| Create Parent portal | Parent-facing views |
| Create Report generation infrastructure | PDF, Excel, automated |
| Add djang-allauth | Microsoft/Google login |

### Phase 6: Multi-School + Admissions + Advanced Features

| Step | Deliverable |
|------|-------------|
| Create Organization, School, Campus, Department | Full hierarchy |
| Activate Person.school FK | Required |
| Change h_code uniqueness to unique_together | Multi-school scope |
| Add tenant middleware | Request scoping |
| Create Admissions module | Application, assessment |
| Create Behaviour/Discipline module | Incidents, house points |
| Create Health/Clinic module | Referrals, records |
| Create External integration connectors | SAP, SIS, payment gateways |

### Phase 7: Remote Software Licensing

| Step | Deliverable |
|------|-------------|
| Create licensing app | License model, validation |
| Add feature flags | Module enablement |
| Add remote check endpoint | Central licensing server |
| Add admin license UI | Status and management |

---

## 33. What NOT to Implement Yet

These are explicitly deferred:

| Feature | Why deferred |
|---------|--------------|
| `AUTH_USER_MODEL = "attendance.Person"` | Irreversible on existing database. Deferred to Phase 2. |
| Organization / School / Campus models | Multi-tenant not needed yet. Nullable FKs suffice. |
| AcademicYear as a model | Not needed until enrollment/subscription scoping requires it. |
| StudentEnrollment (full model) | Requires AcademicYear first. |
| StaffAssignment (full model) | Requires AcademicYear + Department first. |
| TeachingAssignment | Requires Subject + ClassGroup first. |
| Department model | Not needed until staff assignment requires it. |
| Family / PersonRelationship | Not needed until parent portal is built. |
| ParentProfile | Not needed until parent portal is built. |
| GuestProfile / VendorProfile | Not needed. |
| Grade / Section / Subject models | Not needed until AcademicYear + Enrollment. |
| Exam / Mark / Assessment models | Not needed until academic features are implemented. |
| Invoice / Receipt / Installment | Not needed until finance module is built. |
| Timetable (full) | Not needed until academic structure exists. |
| LMS features (content, quizzes) | Not in scope — BISK_RFv4 does not replace Moodle. |
| Social auth (Microsoft/Google) | No requirement yet. Add when portals launch. |
| JWT for mobile | No mobile app yet. Add when API is built. |
| Notification infrastructure | Not needed until portals launch. |
| Reporting engine | Not needed until academic/finance data exists. |
| Removing Student table | Premature. Keep through Phase 2. |
| Extracting apps (identity, academic, etc.) | Premature. Keep in attendance app until patterns stabilize. |
| Student.user FK on Student | Student is being replaced by Person+StudentProfile. |

---

## 34. Open Questions

These questions must be answered before or during implementation:

1. **h_code uniqueness**: In multi-school mode, should `h_code` be globally unique or per-school? Per-school is more flexible but requires changing the unique constraint.

2. **AcademicYear overlap**: Can a person have roles in two academic years simultaneously (e.g., teaching in one year while enrolled in another)? Yes — PersonRole allows it. But enrollment and assignment models need to handle edge cases.

3. **Grade promotion**: When a student is promoted to the next grade, does this create a new StudentEnrollment or update the existing one? Answer: new enrollment per year.

4. **Mid-year transfers**: Can a student transfer sections mid-year? Should StudentEnrollment support this via status changes or new records?

5. **Family billing**: Can an invoice be issued to a family (multiple students) rather than an individual? The Family model should support this, but the invoice structure needs design.

6. **Parent accounts**: Can one parent account be linked to multiple children in different grades/classes? Yes — via PersonRole(parent) + multiple PersonRelationships.

7. **Staff who are also parents**: How does the parent portal distinguish between "my data" and "my child's data"? Role-based data scoping.

8. **Legacy data**: How much historical attendance/meal data needs to be migrated to Person-based FKs? Entire current dataset, or only active records?

9. **Face embeddings**: Are staff face embeddings needed immediately, or can they wait? Wait — Phase 1.5 or later.

10. **Photo migration**: Should existing Student photos be migrated to Person.photo? Yes — in the data migration.

11. **ext_id format**: What is the format for ext_id? Should it be a URI, a UUID, or a school-specific code?

12. **School code format**: For multi-school, what is the format for School.code? Short abbreviation ("BAGHDAD", "ERBIL") or numeric?

13. **Department model design**: Should there be one Department model with a type/category discriminator, or separate models per domain (AcademicDepartment, HRDepartment, FinanceDepartment)? Recommendation: start with one Department model with a `domain` field when implementation begins. Deferred.

14. **AcademicYear vs financial year**: Does the school use academic year or calendar year for billing? If different, both must be supported.

15. **Discount priority**: When multiple discounts apply (student discount + staff discount + family discount), what is the resolution order?

16. **Wallet per person or per role**: Does a person have one wallet (shared across roles) or one per role? Recommendation: one wallet per person for simplicity.

17. **Negative wallets**: Should wallets be allowed to go negative? Current design allows it for postpaid meals.

18. **Parent guardian hierarchy**: When both parents are in the system, who is the primary contact for billing, emergencies, and communications?

19. **Notification preferences**: Should each person be able to configure their notification channel preferences (email, SMS, push)?

20. **Multi-school Person**: When multi-school is active, can a Person have records at two schools simultaneously (e.g., a teacher who teaches at two campuses)? PersonRole.school handles this, but wallet and enrollment scoping need design.

---

## Appendix A: Model Relationship Diagram (High-Level)

```
Organization (future)
    └── School (future)
            └── Campus (future)

AcademicYear
    ├── StudentEnrollment
    ├── StaffAssignment
    ├── TeachingAssignment
    └── Term

Person
    ├── RoleType (via PersonRole)
    ├── StudentProfile
    ├── StaffProfile
    ├── ParentProfile (future)
    ├── PersonRelationship (future)
    ├── Wallet
    ├── MealSubscription
    ├── FaceEmbedding
    ├── AttendanceRecord
    ├── AttendanceEvent
    └── Invoice (future)

auth.User
    └── Person (optional O2O)
```

---

## Appendix B: Current → Target FK Migration Table

| Current FK | Target FK | Model |
|------------|-----------|-------|
| `Wallet.student` | `Wallet.person` | Wallet |
| `WalletTransaction.student` | `WalletTransaction.person` | WalletTransaction |
| `MealSubscription.student` | `MealSubscription.person` | MealSubscription |
| `FaceEmbedding.student` | `FaceEmbedding.person` | FaceEmbedding |
| `AttendanceRecord.student` | `AttendanceRecord.person` | AttendanceRecord |
| `AttendanceEvent.student` | `AttendanceEvent.person` | AttendanceEvent |

---

## Appendix C: Document Relationships

```
docs/architecture/
├── erp_foundation_architecture.md          ← THIS DOCUMENT
│   └── person_identity_architecture.md     ← Detailed identity design
│       └── legacy_lms_feature_inventory.md ← Business feature source

docs/agent/
├── PROJECT_MASTER.md     ← Project overview and state
├── ARCHITECTURE.md       ← High-level architecture notes
├── DECISIONS.md          ← Architecture decisions
├── ROADMAP.md            ← Implementation roadmap
├── CODING_RULES.md       ← Coding rules for AI agents
└── CHANGELOG_AI.md       ← AI-session changelog
```
