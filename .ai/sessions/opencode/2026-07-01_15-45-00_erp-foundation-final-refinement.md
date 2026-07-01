---
date: 2026-07-01
branch: feature/person-architecture
model: opencode/big-pickle
mode: planning
status: completed
topic: ERP Foundation Architecture — Final Refinement
tags:
  - architecture
  - foundation
  - erp
  - state-machine
  - identity-provider
  - assignment-pattern
  - department-hierarchy
related:
  - docs/architecture/erp_foundation_architecture.md
  - docs/architecture/person_identity_architecture.md
  - docs/architecture/legacy_lms_feature_inventory.md
  - AGENTS.md
  - .ai/sessions/opencode/2026-07-01_15-30-00_erp-foundation-architecture-doc.md
---

# Session Report — ERP Foundation Architecture Final Refinement

## Objective

Apply 5 architectural refinements to `docs/architecture/erp_foundation_architecture.md`:
1. Fix Department hierarchy (remove from Campus, make domain-owned).
2. Add general Assignment Concept pattern.
3. Add IdentityProvider future extension point.
4. Add Workflow/state-machine architecture principle.
5. Mark document as Version 1.0 Candidate.

No models, migrations, database schema changes, or app code modifications were made.

## Findings

1. The Department hierarchy in the original document placed Department under Campus as a physical location. This conflicted with the legacy inventory recommendation that departments belong to academic/HR domains.

2. The original document had separate StudentEnrollment and StaffAssignment sections but no unifying "assignment as a pattern" concept. This made it harder to see that ClubAssignment, BusAssignment, and future assignment types follow the same pattern.

3. Authentication section covered local auth and social login but had no concept of multiple identity providers (LDAP, Azure AD, OIDC). The architecture needed an extension point.

4. The 12 architecture principles did not include workflow modeling strategy. Boolean flags (is_approved, is_paid, is_active) were not addressed.

5. The document header had "Draft — top-level foundation blueprint" without a version number.

## Decisions

1. **Department is NOT part of the Organization→School→Campus hierarchy.** It is a cross-domain concept belonging to academic, HR, finance, or operations domains. Added a department placement table to clarify.

2. **Assignment is a shared architectural pattern**, not a single generic table. Added Section 13 with specialized assignment models (StudentEnrollment, StaffAssignment, TeachingAssignment, ClubAssignment, BusAssignment, ServiceAssignment), design rules, and a "what is NOT an assignment" clarification.

3. **IdentityProvider is a future-only concept.** Added as an extension point in the Authentication section. `auth.User` remains the sole Phase 1 login identity.

4. **Principle 13: Workflow as state machine.** Business processes should use status enums with valid transition maps, not scattered boolean flags. A general-purpose workflow engine is explicitly NOT planned.

5. **Version: 1.0 Candidate, Status: Draft — pending ChatGPT review.**

## Implementation

No implementation was performed. This was a documentation-only refinement.

### Changes to `docs/architecture/erp_foundation_architecture.md`

1. **Document header** (line 5-6): Changed to `Version: 1.0 Candidate` and `Status: Draft — pending ChatGPT review`.

2. **Section 5 — Organization / School / Campus**: Replaced Department-in-hierarchy with `Organization → School → Campus` only. Added Department placement table showing Department as a cross-domain concept (academic, HR, operations, finance). Added explanation that a single Department model with type/category may serve all domains, or separate models may be created — deferred.

3. **Section 3 — Architecture Principles**: Added principle 13 "Workflow as state machine" with a detailed subsection showing anti-patterns (scattered booleans) vs pattern (status enums), 6 domain workflow examples with valid transitions, and a clear statement that a general-purpose workflow engine is NOT planned.

4. **New Section 13 — Assignment Concept**: 500-line section explaining the general assignment pattern with 6 specialized assignment types, 6 design rules, "what is NOT an assignment" clarification, and rationale against a single generic assignment table.

5. **Section 8 — Authentication**: Added "IdentityProvider concept (future extension point)" subsection with model design, 6 provider types with phases, and the key constraint that `auth.User` remains the sole Phase 1 login identity.

6. **Renumbering**: All sections 13–33 were renumbered to 14–34. Internal cross-references (Section 29 → 30, Section 31 → 32) and open question 13 (Department structure) were updated accordingly.

### Document statistics

- Before: 1408 lines
- After: 1526 lines
- Added: ~118 lines of new content

## Files Modified

Modified:
- `docs/architecture/erp_foundation_architecture.md`

Created:
- `.ai/sessions/opencode/2026-07-01_15-45-00_erp-foundation-final-refinement.md`

## Commands Executed

- `grep -n '^## [0-9]' docs/architecture/erp_foundation_architecture.md` (verify section numbering)
- `grep 'Section [0-9]' docs/architecture/erp_foundation_architecture.md` (verify cross-references)
- `wc -l docs/architecture/erp_foundation_architecture.md`
- `git diff --stat`

## Remaining Work

1. Review the refined document with the project owner.
2. Schedule ChatGPT architectural review using both session reports (initial + refinement).
3. After review, change Status from "Draft — pending ChatGPT review" to "Approved".
4. Begin Phase 1 implementation: Person / RoleType / PersonRole / StudentProfile / StaffProfile.
5. Continue maintaining alignment between all three architecture documents.

## Recommendations

1. **Proceed to ChatGPT review.** The document now includes all requested refinements and is ready for external architectural review.

2. **The IdentityProvider concept should remain purely documentary** until portal/mobile work begins (Phase 4+). Do not implement it sooner.

3. **The Assignment pattern should be referenced** when implementing StudentEnrollment (Phase 3) and StaffAssignment (Phase 3) — each model should follow the pattern documented in Section 13.

4. **The workflow/state-machine principle should be applied** to new models from Phase 1 onward. PersonRole and all future assignment models should use status enums rather than boolean flags.

5. **Department placement should be revisited** when the first domain-specific department need arises (academic structure or HR). The current document defers the decision, which is correct.

6. **Document the multi-school decisions early** (nullable school FK, h_code uniqueness) even though implementation is deferred, as they affect the Person model.

---

Report generated:
`.ai/sessions/opencode/2026-07-01_15-45-00_erp-foundation-final-refinement.md`
