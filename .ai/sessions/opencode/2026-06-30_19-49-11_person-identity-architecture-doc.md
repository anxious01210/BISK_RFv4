# OpenCode Session — Person Identity Architecture Document

Date: 2026-06-30
Branch: feature/person-architecture
Mode: read-only architecture documentation

## Objective

Create the comprehensive architecture blueprint for the BISK_RFv4 identity system at `docs/architecture/person_identity_architecture.md`. The document defines the vision for Person, RoleType, PersonRole, StudentProfile, StaffProfile, and the migration strategy from the current Student model — covering 24 sections from design principles to phased implementation roadmap.

## Findings

- The `docs/architecture/` directory existed but was empty — no prior architecture documentation.
- The AGENTS.md and previous session reports defined the target (Person → StudentProfile/StaffProfile) but lacked detailed model specifications or migration strategy.
- All identity architecture decisions from the previous analysis session (Option D: Progressive Hybrid) were confirmed and incorporated into the document.
- No existing models (Organization, School, AcademicYear, RoleType, PersonRole) existed in the codebase — everything in the document is forward-looking design.

## Decisions

1. **Person is business identity, User is login identity.** Auth.User is kept as-is (no AUTH_USER_MODEL change). Person has optional O2O bridge to auth.User.
2. **Roles use RoleType + PersonRole (M2M through), not a single person_type field.** This allows a person to hold multiple simultaneous roles.
3. **Profiles extend roles.** StudentProfile and StaffProfile are optional OneToOne extensions of Person, triggered by the presence of the corresponding PersonRole.
4. **AcademicYear and Organization hierarchies are designed but deferred.** Including `school` FK and `ext_id` as nullable pre-provisioned fields.
5. **6 FK models must migrate Student → Person.** Wallet, WalletTransaction, MealSubscription, FaceEmbedding, AttendanceRecord, AttendanceEvent — in that order.
6. **Backward compatibility via dual-FK pattern, compat properties, and legacy_student backlink.** No existing code breaks during migration.
7. **What NOT to implement in Phase 1:** AUTH_USER_MODEL switch, AcademicYear, Enrollment, Assignment, ParentProfile, social auth, JWT, multi-tenant activation.

## Implementation

No code was modified. A single architecture document was created:

- `docs/architecture/person_identity_architecture.md` (293 lines, 24 sections + appendixes)

## Files Modified

None (new file created):
- `docs/architecture/person_identity_architecture.md` — created

## Commands Executed

- `date -u '+%Y-%m-%d_%H-%M-%S'` — timestamp for filenames
- Directory checks on `docs/architecture/`

## Remaining Work

1. Review and approve the architecture document with stakeholders.
2. Update `AGENTS.md` to reference the new architecture document.
3. Begin Phase 0: squash existing migrations.
4. Upload the session report to ChatGPT for architectural review as requested.
5. Implement Phase 1 models (Person, RoleType, PersonRole, StudentProfile, StaffProfile).

## Recommendations

1. **Use this document as the single source of truth** for identity architecture decisions. All future identity work should be checked against it.
2. **Review the RoleType ↔ Django Group mapping** before implementation. Confirm whether PersonRole should automatically assign corresponding Django Groups.
3. **Decide on the `h_code` generation strategy** early — sequential, hash-based, or ERP-sourced — as it affects Person creation workflows.
4. **Plan the migration squash** as the very next engineering step. It blocks all Phase 1 work.
5. **Upload this report to ChatGPT** for external review as requested by the user.
