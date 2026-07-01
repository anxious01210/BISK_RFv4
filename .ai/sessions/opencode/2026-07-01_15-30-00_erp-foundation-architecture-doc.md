---
date: 2026-07-01
branch: feature/person-architecture
model: opencode/big-pickle
mode: planning
status: completed
topic: ERP Foundation Architecture Document
tags:
  - architecture
  - foundation
  - erp
  - identity
  - multi-school
related:
  - docs/architecture/person_identity_architecture.md
  - docs/architecture/legacy_lms_feature_inventory.md
  - docs/architecture/erp_foundation_architecture.md
  - AGENTS.md
  - docs/agent/ARCHITECTURE.md
  - docs/agent/PROJECT_MASTER.md
  - docs/agent/ROADMAP.md
  - docs/agent/DECISIONS.md
  - docs/agent/CODING_RULES.md
---

# Session Report — ERP Foundation Architecture Document

## Objective

Create the top-level ERP foundation blueprint document at `docs/architecture/erp_foundation_architecture.md` for BISK_RFv4. The document must define the long-term foundation for BISK_RFv4 as a future school ERP/LMS/portal/mobile/API/multi-school platform.

No models, migrations, database schema changes, or app code modifications were to be made.

## Findings

1. **Existing architecture documents**: `person_identity_architecture.md` (1080 lines) already contains detailed identity design. The new document needed to be compatible and sit above it as a top-level blueprint.

2. **Legacy LMS inventory**: `legacy_lms_feature_inventory.md` identified 20 business domains from the legacy Node.js backend. The foundation document needed to be broad enough to accommodate all of them while not committing to implement everything immediately.

3. **Current project state**: The project currently has ~30+ attendance migrations, a `Student`-centric model set, and 4 login `auth.User` accounts. The `feature/person-architecture` branch is the active branch.

4. **Architecture principles were well-defined** but scattered across multiple agent docs. The foundation document consolidates them with 12 explicit principles.

5. **The legacy LMS contains significant academic/finance/guardian features** that BISK_RFv4 will need to support eventually but should not reimplement immediately.

## Risks

- The foundation document is aspirational. Over-committing to future features could create pressure to implement prematurely.
- The 33-section structure is comprehensive, but some sections (especially around academic structure, timetable, and finance) speculate on designs that will need refinement when implementation begins.
- Multi-school readiness via nullable FKs is a reasonable compromise, but actual multi-tenant activation will require significant architectural work.
- The document defers AUTH_USER_MODEL changes, which is correct for now but may become a bottleneck for portal/mobile authentication.

## Decisions

- The ERP foundation document sits above `person_identity_architecture.md` as the top-level blueprint.
- All 33 requested sections were addressed in the document.
- Domain boundaries were defined with a future target app structure, but the document acknowledges that everything currently lives in `apps/attendance/`.
- AcademicYear is treated as a core foundation concept even though it won't be implemented until Phase 3.
- Person-User separation is reinforced as a hard architectural principle.
- Django Groups/Permissions and PersonRole are explicitly separated.
- 12 architecture principles were codified.
- 20 open questions were documented for future resolution.
- A clear "What NOT to implement yet" section (32 items) was included to constrain scope.
- BISK_RFv4 is explicitly defined as NOT a full LMS (no course authoring, content delivery, SCORM, etc.).

## Implementation

No implementation was performed. This was a documentation-only task.

### Created file

- `docs/architecture/erp_foundation_architecture.md` — ~750 lines across 33 sections + 3 appendices

### Sections covered

1. Vision
2. Scope (including boundary with legacy LMS)
3. Architecture principles (12 principles)
4. Domain boundaries (target future app structure)
5. Organization / School / Campus
6. AcademicYear
7. Person / identity
8. Authentication (Option D progressive hybrid)
9. Authorization (Django Groups vs PersonRole)
10. RoleType / PersonRole
11. Status strategy (Person-level + Role-level + Enrollment-level + Assignment-level)
12. Profile strategy (StudentProfile, StaffProfile, ParentProfile, GuestProfile, VendorProfile)
13. StudentEnrollment strategy
14. StaffAssignment strategy
15. Guardian/parent relationship strategy (PersonRelationship + Family)
16. Academic structure (SchoolLevel, Grade, Section, Subject, Term, Exam, Mark)
17. Finance/payment foundation (Invoice, Receipt, Installment)
18. Wallet/lunch foundation (migration strategy)
19. Attendance/recognition foundation (Person-targeted, expanded types)
20. Timetable foundation (Day, Period, Lecture, Classroom)
21. LMS/online learning foundation (BISK_RFv4 is NOT a full LMS)
22. Portal/mobile/API foundation (portal types, API design, JWT)
23. Notifications/reporting foundation
24. Multi-school / multi-tenant readiness
25. Data ownership rules (10 domains, access rules, audit rules)
26. What must be configurable from Django admin (17 concept categories)
27. What should remain domain-modeled/hard-coded (12 concept categories)
28. Migration strategy from current BISK_RFv4 (Phases 0, 1, 1.5, 2)
29. Relationship to legacy BISK LMS
30. Relationship to person_identity_architecture.md
31. Implementation phases (Phases 0–7, 17 implementation steps)
32. What NOT to implement yet (32 deferred features)
33. Open questions (20 questions)

## Files Modified

Created:
- `docs/architecture/erp_foundation_architecture.md`

Created:
- `.ai/sessions/opencode/2026-07-01_15-30-00_erp-foundation-architecture-doc.md`

## Commands Executed

- `mkdir -p .ai/sessions/opencode`

## Remaining Work

1. Review and approve `erp_foundation_architecture.md` with the project owner.
2. Ensure alignment with `person_identity_architecture.md` before Phase 1 implementation begins.
3. Begin Phase 1: Add Person / RoleType / PersonRole / StudentProfile / StaffProfile models.
4. Update `docs/agent/` documentation after meaningful implementation work.
5. Address the 20 open questions as implementation progresses.

## Recommendations

1. **Approve the foundation document before beginning Phase 1 implementation.** The document serves as the architectural contract for all future work.

2. **Keep the foundation document alive.** Update it as implementation reveals new patterns or constraints. It should not be a static document.

3. **Use the open questions section as a backlog** for design discussions with the project owner.

4. **Stay incremental.** The document's breadth could create pressure to implement too much. The "What NOT to implement yet" section (Section 32) and the phased roadmap (Section 31) provide guardrails.

5. **The next task should be Phase 1 implementation** — creating Person, RoleType, PersonRole, StudentProfile, and StaffProfile models with migrations, data migration from Student, and admin interfaces.

6. **Do not change AUTH_USER_MODEL.** The progressive hybrid approach (Option D) remains the correct strategy for now.

7. **Document multi-school decisions early** even though implementation is deferred, because they affect the Person model (nullable school FK, h_code uniqueness approach).

---

Report generated:
`.ai/sessions/opencode/2026-07-01_15-30-00_erp-foundation-architecture-doc.md`
