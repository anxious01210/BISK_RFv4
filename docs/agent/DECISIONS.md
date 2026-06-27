# BISK_RFv4 Architecture Decisions

Version: 1.0

## Decision 001 — Use Person architecture before discount engine

Status: Accepted  
Branch: `feature/person-architecture`

### Context

The lunch/meal system originally centered around students. The next academic year requires meal usage by students, staff, teachers, and potentially future person types.

### Decision

Introduce a shared `Person` architecture before implementing discount rules.

### Reason

If discounts are implemented directly against `Student`, the system will likely need a second major redesign when staff meal support is added.

### Consequences

Positive:

- Discounts can apply to students and staff.
- Wallets can belong to any person type.
- Meal subscriptions can apply to any person type.
- Future expansion is easier.

Tradeoff:

- More work now.
- Careful migration/backfill needed.
- Existing student workflows must be preserved.

## Decision 002 — Keep refactor incremental

Status: Accepted

### Decision

Do not rewrite the whole system in one pass. Add compatibility layers and migrate gradually.

### Consequences

Positive:

- Lower risk.
- Easier testing.
- Easier rollback.
- Smaller commits.

Tradeoff:

- Some temporary duplication may exist.
- Later cleanup phase will be needed.

## Decision 003 — Use docs/agent as living AI knowledge base

Status: Accepted

### Decision

Create and maintain `docs/agent/` as the living knowledge base, and use root `AGENTS.md` for agent instructions.

## Decision 004 — Licensing system is last roadmap phase

Status: Accepted

### Context

The owner wants remote subscription/license control per customer.

### Decision

Add remote software licensing as the final long-term phase, after the core product stabilizes.

### Reason

Licensing depends on knowing the final product modules, customer model, feature boundaries, and deployment pattern.
