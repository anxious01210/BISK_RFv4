# Decision Record

ID: DECISION-2026-06-30-001
Date: 2026-06-30
Status: Accepted
Topic: Person Architecture — Auth Model Strategy

## Context

The project currently uses Django's default `auth.User` model with no `AUTH_USER_MODEL` setting in any settings file. Introducing a Person architecture raised the question: should Person become the custom user model (`AUTH_USER_MODEL = "attendance.Person"`)?

The codebase has:
- 30+ existing migrations referencing `auth.User` via `settings.AUTH_USER_MODEL`
- 3 FK fields to `auth.User` in production models (WalletTransaction.created_by, MealRecord.confirmed_by, MealRecord.reversed_by)
- 6 models with direct FK/O2O to Student
- No existing Student-to-User relationship

## Decision

**Do NOT introduce Person as AUTH_USER_MODEL.** Keep `auth.User` as the canonical authentication model. Person will have an optional `OneToOneField('auth.User', null=True, blank=True)` to link login accounts to Person records.

## Reasoning

1. **Irreversible constraint.** Django's documentation explicitly warns that changing `AUTH_USER_MODEL` after migrations have been created and applied is not supported without a new database or manual SQL renaming.
2. **Existing migrations locked in.** Migration 0028 uses `migrations.swappable_dependency(settings.AUTH_USER_MODEL)`. Changing AUTH_USER_MODEL would break the migration chain.
3. **No benefit for current use case.** The system authenticates staff/supervisors, not students. Students are recognized by face, not by login. There is no requirement for Student-as-User.
4. **Simpler migration path.** Adding Person as a regular model with an optional user FK is purely additive — no existing schema changes needed, no migration chain breakage.
5. **Future flexibility.** Staff can get login accounts via Person.user FK; students can be linked later if needed (e.g., parent portals).

## Consequences

- Person model has an `Optional[OneToOneField[auth.User]]` rather than inheriting from AbstractUser.
- Auth logic remains unchanged — group-based authorization with `supervisor`, `meal_supervisor`, `api_user` groups.
- Student remains a standalone model until fully migrated to Person+StudentProfile.
- No need to update any of the 3 existing `settings.AUTH_USER_MODEL` FK references.
- All existing auth-related code (login views, decorators, permission checks) continues working unchanged.

## Related Files

- `bisk/settings.py` (no AUTH_USER_MODEL — intentionally not adding one)
- `apps/attendance/models.py` (future: Person model with user FK)
- `apps/attendance/migrations/0028_discountprofile_and_more.py` (swappable dependency on settings.AUTH_USER_MODEL)

## Related Sessions

- `2026-06-27-person-architecture-inspection.md`
- `2026-06-30_17-39-16_person-architecture-risk-report.md`
