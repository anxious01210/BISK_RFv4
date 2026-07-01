# Architecture Decision Records — BISK_RFv4

Date: 2026-07-01
Branch: feature/person-architecture
Status: Living document

---

## ADR-001: BISK_RFv4 will evolve into ERP/LMS/Portal platform

**Status:** Accepted

**Context:** BISK_RFv4 currently serves as a school attendance and meal-management system centered on students. The school's existing workflows (visible in the legacy Node.js LMS backend) include academic management, finance, admissions, timetables, assessments, communication, behaviour, and reporting. The project must define its long-term scope to ensure incremental work does not paint the architecture into a corner.

**Decision:** BISK_RFv4 will evolve from a single-purpose attendance/meal system into a full-school ERP / LMS / portal / mobile / API platform. The long-term platform includes identity, academic core, timetable, attendance, finance, meals, assessment, admissions, communication, behaviour, health, portals, API, reporting, multi-tenant, and external integration domains. The system is explicitly NOT a full LMS (content delivery, authoring, SCORM) nor a payment gateway.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Stay attendance/meal only | Would require a future rewrite for academic/finance features |
| Rebuild as microservices from scratch | Too disruptive; incremental evolution preserves existing value |
| Port legacy LMS code directly to Django | Would inherit legacy technical debt and Sequelize schema |

**Consequences:**

- Positive: Single platform serves all school needs; unified identity across all subsystems; incremental delivery allows prioritization.
- Negative: Broader scope creates risk of scope creep; deferred features must be explicitly documented as "not yet."
- Mitigation: Clear "What NOT to implement yet" section in foundation document; phased roadmap with explicit boundaries.

**Related documents:**

- `docs/architecture/erp_foundation_architecture.md` Sections 1, 2
- `docs/architecture/legacy_lms_feature_inventory.md` Section 3

---

## ADR-002: Person is business identity, User is login identity

**Status:** Accepted

**Context:** The current system uses `Student` as the business identity and `auth.User` for the 4 existing staff login accounts. There is no unified concept of a "person" that can be a student, staff, teacher, parent, guest, or vendor. As the system expands to support staff meals, parent portals, and guest access, a shared identity model is required.

**Decision:** Person is the central business identity in the entire ERP system. Every human actor — student, staff, teacher, parent, guardian, guest, vendor — is represented as a Person. `auth.User` represents a login/authentication identity only. Person and User are separate models linked by an optional `OneToOneField`. A Person may exist without any login account; a User may exist without a corresponding Person (service accounts).

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Person extends AbstractUser (Option A) | Changes AUTH_USER_MODEL, irreversible on existing DB |
| Single model for identity + auth (Option B) | Cannot represent persons without login |
| Keep Student as sole business identity (Option C) | Cannot represent staff, parents, guests |

**Consequences:**

- Positive: Clean separation of concerns; backward compatible with existing Student model; enables optional login; enables multi-system identity.
- Negative: Temporary complexity of linking Person ↔ User; requires migration from Student → Person.
- Mitigation: Backward-compat properties on models; dual-write during transition.

**Related documents:**

- `docs/architecture/person_identity_architecture.md` Sections 1, 5
- `docs/architecture/erp_foundation_architecture.md` Section 7
- `docs/agent/ARCHITECTURE.md` Target Person architecture

---

## ADR-003: Keep auth.User during Phase 1

**Status:** Accepted

**Context:** Django explicitly warns that changing `AUTH_USER_MODEL` after migrations have been run on a production database is not supported. The current system uses Django's default `auth.User` with 4 existing staff login accounts. Changing it would require manual SQL rename operations or squashed migrations with state migration, blocking all other work.

**Decision:** Phase 1 keeps `auth.User` as the authentication model. Person has an optional `OneToOneField("auth.User")` bridge. All authentication flows remain unchanged; `request.user` remains `auth.User`. A future Phase 2 will set `AUTH_USER_MODEL = "attendance.Person"` after squashing migrations and copying data.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Change AUTH_USER_MODEL immediately | High risk; Django warns against it on existing databases |
| Person(AbstractUser) from start | Would require new database; incompatible with production data |
| No auth change ever | auth.User cannot be extended for Person; keeps unnecessary separation |

**Consequences:**

- Positive: Zero risk to existing authentication; all auth flows work unchanged; backward compatible.
- Negative: Phase 2 will require careful migration; temporary O2O bridge must be maintained.
- Mitigation: Signal-based sync between auth.User and Person; documented Phase 2 plan.

**Related documents:**

- `docs/architecture/person_identity_architecture.md` Section 6
- `docs/architecture/erp_foundation_architecture.md` Section 8

---

## ADR-004: Use RoleType + PersonRole instead of person_type

**Status:** Accepted

**Context:** Many school systems use a single `person_type` CharField (`student`, `staff`, `parent`). This cannot represent: a staff member who is also a parent, a student who is also a teacher assistant, a person changing roles across academic years, or a vendor who is also a parent.

**Decision:** RoleType is a controlled vocabulary of roles (student, staff, teacher, parent, guest, vendor). PersonRole is a many-to-many through model linking Person to RoleType, optionally scoped by AcademicYear and School. A person can hold multiple simultaneous roles.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Single `person_type` CharField | Cannot represent multiple simultaneous roles |
| Django Group-based roles | Groups control system access, not business roles |
| JSON field for roles | Loses referential integrity, queryability |

**Consequences:**

- Positive: Flexible M2M roles; supports staff+parent, student+assistant; role changes across years; scoped by school for multi-tenant.
- Negative: Slightly more complex than a single field; requires RoleType seed data migration.
- Mitigation: RoleType is configurable from Django admin; PersonRole has a clean admin interface.

**Related documents:**

- `docs/architecture/person_identity_architecture.md` Sections 7, 8
- `docs/architecture/erp_foundation_architecture.md` Section 10

---

## ADR-005: Profiles extend Person roles

**Status:** Accepted

**Context:** Different roles need different data fields. Students need grade, homeroom, has_meal flags. Staff need employee_id, job title, department, hire_date. Overloading Person with all possible role-specific fields would create a wide, sparse table with many nullable columns.

**Decision:** Person stores only shared identity fields (h_code, name, gender, DOB, contact, photo, status, optional auth bridge). Role-specific data lives in OneToOne profile models (StudentProfile, StaffProfile, future ParentProfile, GuestProfile, VendorProfile) that share their primary key with Person. Profiles are optional — a Person with role "parent" may have no profile.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Single Person model with all role fields | Wide sparse table; maintenance burden |
| Separate identity models per role (Student, Staff, etc.) | No common identity; cross-role features impossible |
| All data on Person, profiles for extra only | Same as rejected alternative |

**Consequences:**

- Positive: Clean separation; profiles are optional and share PK with Person; backward compatible via StudentProfile.legacy_student; easy to add new profile types.
- Negative: Requires join to read role-specific data; extra models to maintain.

**Related documents:**

- `docs/architecture/person_identity_architecture.md` Sections 9, 10, 11
- `docs/architecture/erp_foundation_architecture.md` Section 12

---

## ADR-006: AcademicYear is a foundation concept

**Status:** Accepted

**Context:** Many school subsystems are scoped to an academic year: enrollment (which grade a student is in), staff assignments (which department a teacher belongs to), subscriptions (which year a meal plan covers), timetables (which periods apply when), assessments (which marking scheme), and billing (which year's fees). The legacy LMS confirms this pattern with Year and Term entities.

**Decision:** AcademicYear is a core foundational model. Future models (StudentEnrollment, StaffAssignment, TeachingAssignment, MealSubscription, DiscountAssignment, PeriodTemplate, AssessmentScheme, Invoice) will reference AcademicYear. Current models using date fields are acceptable until the AcademicYear model is introduced in Phase 3.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| No year scoping (use date fields only) | Complex queries for "current year" data; no clear year boundaries |
| Year as a simple string/char field | Loses date-based queries; no start/end validation |
| Year as part of each model (denormalized) | Duplication; inconsistent year definitions across models |

**Consequences:**

- Positive: Consistent year-scoping across all domains; enables easy reporting per year; clean "current year" queries.
- Negative: Not implemented in Phase 1; Phase 3 is a significant migration.
- Mitigation: Existing date-field approach works; documented as deferred.

**Related documents:**

- `docs/architecture/person_identity_architecture.md` Section 4
- `docs/architecture/erp_foundation_architecture.md` Section 6
- `docs/architecture/legacy_lms_feature_inventory.md` Section 5

---

## ADR-007: Grade/Class belong to StudentEnrollment, not Person

**Status:** Accepted

**Context:** A student's grade and class change every academic year. Storing grade on Person (or StudentProfile) permanently would require overwriting each year, losing historical data. The legacy LMS has Year, Grade, and Class entities separate from student identity.

**Decision:** Grade and section/homeroom live on StudentEnrollment, which is scoped by AcademicYear. StudentProfile may carry the current grade as a denormalized convenience value for fast lookup, synced from the most recent active enrollment.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Grade on Person permanently | Overwrites each year; no historical tracking |
| Grade on StudentProfile only | No historical tracking; unclear which year's grade |
| Grade on separate GradeHistory model | Same as StudentEnrollment pattern but less integrated |

**Consequences:**

- Positive: Accurate historical tracking; one enrollment per year; clean academic record.
- Negative: Requires StudentEnrollment model (Phase 3); temporary denormalization on StudentProfile.
- Mitigation: Current `Student.grade` suffices until Phase 3; documented migration path.

**Related documents:**

- `docs/architecture/person_identity_architecture.md` Section 12
- `docs/architecture/erp_foundation_architecture.md` Sections 13, 14, 17
- `docs/architecture/legacy_lms_feature_inventory.md` Section 5

---

## ADR-008: Staff duties belong to StaffAssignment/TeachingAssignment

**Status:** Accepted

**Context:** Staff roles, departments, and teaching responsibilities change by academic year. Storing job_title and department on StaffProfile would lose history and make multi-year reporting impossible. The legacy LMS has teacher and department entities separate from staff identity.

**Decision:** StaffAssignment tracks a staff member's role, department, and job title per AcademicYear. TeachingAssignment links teacher, subject, and class/group per AcademicYear. StaffProfile may carry current values as denormalized convenience.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Job title on StaffProfile only | No historical tracking; cannot report past assignments |
| Single Assignment model with type discriminator | Generic table loses per-type constraints and validation |
| No assignment tracking | Cannot support multi-year staff reporting |

**Consequences:**

- Positive: Historical tracking; multi-assignment support (teach two subjects); clean separation from identity.
- Negative: Requires AcademicYear, Department, Subject, Class, Section models (Phase 3).
- Mitigation: Not needed until academic features are implemented; current StaffProfile suffices temporarily.

**Related documents:**

- `docs/architecture/person_identity_architecture.md` Section 13
- `docs/architecture/erp_foundation_architecture.md` Sections 13, 15
- `docs/architecture/legacy_lms_feature_inventory.md` Section 8

---

## ADR-009: Wallet, lunch, attendance, and face recognition target Person

**Status:** Accepted

**Context:** Currently Wallet, MealSubscription, FaceEmbedding, AttendanceRecord, and AttendanceEvent all have foreign keys to Student. This means staff, teachers, guests, and vendors cannot use these features without duplicating the entire model set.

**Decision:** All FK targets migrate from Student to Person. Wallet, MealSubscription, FaceEmbedding, AttendanceRecord, and AttendanceEvent will each gain a `person` FK using the dual-FK pattern: (1) add nullable person FK, (2) data-migrate existing student links, (3) add not-null constraint, (4) update query code, (5) later drop student FK.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Keep Student FKs, add parallel Staff FKs | Duplicates every FK relationship; doubles maintenance |
| Convert Student to proxy of Person immediately | Breaks existing code; high risk |
| Polymorphic FK (GenericForeignKey) | Loses referential integrity; poor query performance |

**Consequences:**

- Positive: Universal FK target enables staff meals, staff attendance, staff face recognition; cleaner architecture.
- Negative: Temporary dual-FK period; multi-step migration; must update all query code.
- Mitigation: Documented migration order (Wallet → MealSubscription → FaceEmbedding → AttendanceRecord → AttendanceEvent); backward-compat properties.

**Related documents:**

- `docs/architecture/person_identity_architecture.md` Sections 14, 15, 16
- `docs/architecture/erp_foundation_architecture.md` Sections 18, 19, 20
- `docs/architecture/legacy_lms_feature_inventory.md` Section 9

---

## ADR-010: Organization/School/Campus designed now, implemented later

**Status:** Accepted

**Context:** BISK_RFv4 serves a single school today but may need to serve multiple schools or campuses in the future. Adding a full multi-tenant hierarchy now would add complexity with no immediate benefit. Ignoring multi-school entirely would require a major schema migration later.

**Decision:** The Person model includes a nullable `school` FK and `ext_id` field for future multi-tenant activation. The Organization→School→Campus hierarchy is designed conceptually but not implemented as database models. The `h_code` uniqueness constraint is prepared to change to `unique_together = [("school", "h_code")]` when multi-tenant is activated. Organization/School/Campus models will be added in Phase 5+.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Implement full multi-tenant now | Premature complexity; no current requirement |
| Ignore multi-school entirely | Would require major schema migration later |
| Use django-tenants schema-per-tenant from start | Overhead for single-school deployment |

**Consequences:**

- Positive: Future-ready without current overhead; nullable FKs are safe; ext_id enables cross-system identity.
- Negative: Nullable FKs need migration to become required later; h_code uniqueness change requires data migration.
- Mitigation: Deferred to Phase 5+; documented in architecture.

**Related documents:**

- `docs/architecture/person_identity_architecture.md` Sections 3, 20
- `docs/architecture/erp_foundation_architecture.md` Sections 5, 25

---

## ADR-011: Department is domain-owned, not physical hierarchy

**Status:** Accepted

**Context:** The initial design in `person_identity_architecture.md` placed Department under Campus as a physical hierarchy. The legacy LMS feature inventory recommends that departments belong to academic/HR domains, not physical locations. Departments are organizational units — they can be academic departments (Mathematics, English), HR departments (Administration, Payroll), finance departments (Accounting, Procurement), or operations departments (Maintenance, Transport).

**Decision:** Department is NOT part of the Organization→School→Campus physical/legal hierarchy. Departments are cost centers and organizational units belonging to academic, HR, finance, or operations domains. Whether a single Department model with a domain/category discriminator or separate models per domain is used is deferred until implementation.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Department under Campus (physical hierarchy) | Incorrect domain model; departments are not physical locations |
| No Department model | Need some grouping for staff assignment, subject grouping |
| Separate models per domain from the start | Premature; may create more complexity than needed |

**Consequences:**

- Positive: More accurate domain modeling; flexible placement per domain; avoids physical/non-physical confusion.
- Negative: Inconsistent with `person_identity_architecture.md` which still shows Department under Campus (needs update).
- Mitigation: Decision deferred on single vs multiple Department models; documented in foundation document.

**Related documents:**

- `docs/architecture/erp_foundation_architecture.md` Section 5
- `docs/architecture/legacy_lms_feature_inventory.md` Section 14 (item "Treating departments as part of the physical campus hierarchy")
- Note: `person_identity_architecture.md` Section 3 still shows Department under Campus — pending alignment update.

---

## ADR-012: Business workflows use state machines/status fields

**Status:** Accepted

**Context:** Business processes like admissions applications, invoice lifecycle, leave requests, refund processing, procurement approvals, and behaviour discipline involve multiple states and transitions. Without explicit state modeling, these workflows tend to accumulate scattered boolean flags (is_approved, is_paid, is_completed, is_cancelled) that create invalid states and complex conditional logic.

**Decision:** Business workflows use status enums with documented valid state transitions. Each workflow model carries a single `status` CharField with choices, a documented transition map, and audit logging of status changes. A general-purpose workflow engine (Camunda, SimpleWorkflow, django-workflows) is explicitly NOT implemented in any planned phase.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Boolean flags for each state | Creates invalid state combinations; complex conditional logic |
| General-purpose workflow engine | Premature over-engineering for current scale; added deployment complexity |
| No explicit workflow modeling | Each domain invents its own inconsistent approach |

**Consequences:**

- Positive: Consistent workflow modeling; auditable transitions; no invalid states; no premature workflow engine complexity.
- Negative: Each model must define its own transitions; per-domain service methods for state changes.
- Mitigation: Do per-domain with simple enums and service methods; a workflow library can be introduced later if needed.

**Related documents:**

- `docs/architecture/erp_foundation_architecture.md` Section 3 (Principle 13)

---

## ADR-013: Legacy LMS is business reference, not code to copy

**Status:** Accepted

**Context:** A legacy Node.js/Next.js/React LMS backend exists with ~20 business domains including students, guardians, families, teachers, academic structure, timetables, attendance, finance, admissions, assessment, behaviour, chat, notifications, shop, clinic, and reporting. The code is in Sequelize/Express and is not directly reusable.

**Decision:** The legacy LMS is treated exclusively as a business-domain and workflow source. Its database schema, code architecture, and implementation decisions are NOT copied into BISK_RFv4. The `ext_id` field on Person enables cross-reference with the legacy system during coexistence.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Port legacy code directly to Django | Would inherit legacy schema issues and technical debt |
| Ignore legacy system entirely | Would lose valuable business workflow knowledge |
| Rebuild legacy features in BISK_RFv4 immediately | Premature; foundation must be stable first |

**Consequences:**

- Positive: Clean architecture; no legacy technical debt; business workflows inform design; ext_id enables coexistence.
- Negative: Requires separate feature inventory document; team must learn legacy workflows without reusing code.
- Mitigation: `legacy_lms_feature_inventory.md` captures business knowledge; incremental migration order planned.

**Related documents:**

- `docs/architecture/legacy_lms_feature_inventory.md` Sections 1, 14, 16, 17
- `docs/architecture/erp_foundation_architecture.md` Section 30

---

## ADR-014: Implementation must be incremental and backward compatible

**Status:** Accepted

**Context:** BISK_RFv4 is a production system with real student, meal, wallet, and attendance data. The Person architecture migration touches many models, views, templates, services, and admin interfaces. A big-bang migration would risk data loss, extended downtime, or deployment failure.

**Decision:** Each phase is additive and independently deployable. Phase 1 adds Person/Profiles/Roles without changing existing tables. Phase 1.5 migrates FK targets one model at a time using dual-write. Phase 2 deprecates Student only after all FKs are migrated and verified. Backward-compatibility mechanisms include: read-through proxy properties, dual FK fields during transition, fallback lookup in API endpoints, continued StudentAdmin registration, and parallel signal paths.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Big-bang migration in one release | High risk; extended downtime; difficult rollback |
| Freeze all features during migration | Blocks business value delivery |
| Rewrite Student as proxy of Person immediately | Too many code paths depend on Student; high breakage risk |

**Consequences:**

- Positive: Lower risk; easier rollback; smaller commits; production remains functional throughout.
- Negative: Longer transition period; temporary code duplication (dual FKs, compat properties, parallel signals).
- Mitigation: Each FK migration is independent and reversible; documented order from simplest (Wallet) to most complex (AttendanceRecord).

**Related documents:**

- `docs/architecture/person_identity_architecture.md` Sections 21, 22
- `docs/architecture/erp_foundation_architecture.md` Sections 29, 32
- `docs/agent/CODING_RULES.md` Person architecture rules

---

## ADR-015: Do not delete Student model early

**Status:** Accepted

**Context:** The current `Student` model is referenced directly or indirectly by Wallet, MealSubscription, FaceEmbedding, AttendanceRecord, AttendanceEvent, MealRecord, MealProfile, templates, views, admin, services, signals, imports/exports, and external scripts. Removing or disabling it prematurely would break the production system.

**Decision:** The Student model is preserved through Phase 1 (add Person + Profiles) and Phase 1.5 (migrate FKs to Person). It is only deprecated in Phase 2 after all FK migrations are completed and verified. Deprecation steps: (1) remove StudentProfile.legacy_student backlink, (2) mark Student as `managed = False`, (3) archive data if needed, (4) drop attendancestudent table, (5) remove backward-compat shims.

**Alternatives considered:**

| Alternative | Reason rejected |
|-------------|----------------|
| Delete Student in Phase 1 | Breaks every FK, view, template, signal, export immediately |
| Make Student a proxy of Person in Phase 1 | Too many code paths depend on Student model directly; high risk of subtle breakage |
| Keep Student forever | Perpetuates dual models; increases maintenance burden |

**Consequences:**

- Positive: Safe, reversible migration path; no disruption to existing code; time to verify Person-based code works.
- Negative: Temporary dual-maintenance; cleanup required in Phase 2.
- Mitigation: Phase 1.5 verification period; backward-compat properties; documented Phase 2 steps.

**Related documents:**

- `docs/architecture/person_identity_architecture.md` Sections 21, 22
- `docs/architecture/erp_foundation_architecture.md` Sections 29, 32, 33
- `docs/agent/ARCHITECTURE.md` Compatibility strategy

---



---

## ADR-016: Identity models live in a dedicated `apps.identity` application

**Status:** Accepted

**Context:** As BISK_RFv4 grows into an ERP platform, identity will be shared across attendance, meals, finance, academics, HR, portals, and future modules. Keeping `Person` and related identity models inside the `attendance` app would create unnecessary coupling.

**Decision:** The long-term home for `Person`, `RoleType`, `PersonRole`, `StudentProfile`, `StaffProfile`, and future identity models is a dedicated `apps.identity` Django application. Existing attendance models will reference these models via foreign keys. The initial implementation may remain in `attendance` temporarily for incremental migration, but all new architectural work should target `apps.identity`.

**Consequences:**

- Identity becomes reusable across all domains.
- Attendance becomes a consumer of identity instead of owning it.
- Future ERP modules can depend on `apps.identity` without importing attendance.

**Related documents:**

- `docs/architecture/person_identity_architecture.md`
- `docs/agent/ARCHITECTURE.md`

## ADR Index

| ADR | Title | Phase |
|-----|-------|-------|
| 001 | BISK_RFv4 will evolve into ERP/LMS/Portal platform | All |
| 002 | Person is business identity, User is login identity | Phase 1 |
| 003 | Keep auth.User during Phase 1 | Phase 1 |
| 004 | Use RoleType + PersonRole instead of person_type | Phase 1 |
| 005 | Profiles extend Person roles | Phase 1 |
| 006 | AcademicYear is a foundation concept | Phase 3 |
| 007 | Grade/Class belong to StudentEnrollment, not Person | Phase 3 |
| 008 | Staff duties belong to StaffAssignment/TeachingAssignment | Phase 3 |
| 009 | Wallet, lunch, attendance, and face recognition target Person | Phase 1.5 |
| 010 | Organization/School/Campus designed now, implemented later | Phase 5+ |
| 011 | Department is domain-owned, not physical hierarchy | Phase 5+ |
| 012 | Business workflows use state machines/status fields | Phase 4+ |
| 013 | Legacy LMS is business reference, not code to copy | All |
| 014 | Implementation must be incremental and backward compatible | All |
| 015 | Do not delete Student model early | Phase 1–2 |
| 016 | Identity models live in `apps.identity` | Phase 1 |
