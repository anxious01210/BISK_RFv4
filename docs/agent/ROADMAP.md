# BISK_RFv4 Roadmap

Version: 1.0

## Phase 0 — Current foundation

Status: mostly completed.

Includes:

- Django project foundation.
- Admin customization.
- Attendance models.
- Face embedding workflows.
- Recognition runner integration.
- Camera management.
- Scheduler/resource management.
- Dashboards.
- Meal/lunch system.
- Wallet system.
- Meal dashboard cleanup.
- Git branch cleanup.

Current working branch:

```bash
feature/person-architecture
```

## Phase 1 — Documentation and AI harness

Status: next immediate task.

Tasks:

- Add `docs/agent/` documentation set.
- Add root `AGENTS.md`.
- Commit docs.
- Install/set up OpenCode.
- Connect OpenCode to OpenRouter.
- Select economical coding model.
- Test with read-only prompt.
- Confirm agent respects repository rules.

## Phase 2 — Person architecture

Goal: support multiple person types before discounts.

Tasks:

1. Inspect current `Student`, `Wallet`, `MealSubscription`, `MealRecord`, `AttendanceRecord`.
2. Design migration plan.
3. Add `Person` model.
4. Backfill existing students into Person.
5. Add `StaffProfile`.
6. Add admin support.
7. Add helper methods for display name, person type, student grade, staff department/title.
8. Keep existing student workflows working.

Acceptance:

- Existing student attendance still works.
- Existing meal dashboard still works.
- Existing wallet deductions still work.
- Admin can view persons.
- Staff records can be created.
- No data loss.

## Phase 3 — Meal/wallet owner refactor

Goal: make wallet and meal subscriptions support Person, not only Student.

Tasks:

1. Add nullable `person` relation to wallet/subscription models if needed.
2. Backfill from existing student relation.
3. Update services to resolve owner.
4. Update admin filters/search.
5. Update exports.
6. Update dashboard display.
7. Keep compatibility during transition.
8. Later remove direct student dependency only after stable.

## Phase 4 — Discount engine

Goal: flexible discounts for students, staff, and future person types.

Tasks:

1. Add discount models.
2. Add pricing service.
3. Add priority logic.
4. Add date-range support.
5. Add fixed, percentage, and free-meal discounts.
6. Add audit snapshots.
7. Integrate with wallet deductions.
8. Add admin UI.
9. Add exports/reports.

## Phase 5 — Payments and finance reporting

Future:

- Wallet top-ups.
- Receipts.
- Payment method tracking.
- Daily/monthly reports.
- Student/staff statements.
- Excel exports.
- Admin reconciliation.

## Phase 6 — Parent/staff portal

Future:

- Parent view of student wallet.
- Staff profile access.
- Meal history.
- Wallet balance.
- Notifications.
- Optional online payment integration.

## Phase 7 — API and mobile support

Future:

- REST API hardening.
- Mobile-friendly endpoints.
- Token/auth strategy.
- API documentation.
- Rate limiting.
- Audit logs.

## Phase 8 — Multi-school / multi-campus support

Future optional:

- Campus model.
- School/customer model.
- Per-campus cameras.
- Per-campus pricing.
- Per-campus reports.
- Permission scoping.

## Phase 9 — Remote software licensing

Final long-term commercial phase.

Goal: allow flexible remote subscription/license control per customer.

Tasks:

1. Add licensing app.
2. Add local license model.
3. Add signed license validation.
4. Add remote check endpoint/client.
5. Add offline grace period.
6. Add feature flags tied to license.
7. Add customer usage limits.
8. Add admin license status page.
9. Add audit logs.
10. Add emergency override.
11. Build separate licensing server/admin later.
