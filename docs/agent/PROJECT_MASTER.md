# BISK_RFv4 Agent Knowledge File

Version: 1.0  
Recommended location: `docs/agent/PROJECT_MASTER.md`  
Current development branch: `feature/person-architecture`

## Purpose

This is the living knowledge file for BISK_RFv4. It is designed for the project owner, AI coding agents such as OpenCode/Claude Code/Codex, and future developers.

Any AI agent must read this file first, then read:

1. `docs/agent/ARCHITECTURE.md`
2. `docs/agent/CODING_RULES.md`
3. `docs/agent/ROADMAP.md`
4. `docs/agent/DECISIONS.md`
5. `docs/agent/CHANGELOG_AI.md`

## Project summary

BISK_RFv4 is a Django-based school attendance, face-recognition, meal, wallet, dashboard, and operations system.

The project currently includes:

- Student records.
- Attendance events and attendance records.
- Face embedding and recognition workflows.
- Camera and runner management.
- Scheduler/resource management.
- Meal/lunch dashboard.
- Meal subscriptions.
- Wallets and wallet transactions.
- Meal confirmation/refund behavior.
- Admin customization.
- Export/reporting improvements.

The project is now entering a major architecture phase: supporting multiple person types such as Students, Staff, Teachers, Guests, and future types before implementing discount rules.

## Current Git state

Repository: `BISK_RFv4`  
Example local path: `~/PycharmProjects/BISK_RFv4-1`

Important recent state:

- `feature/wallet-pricing-discounts` contains the meal/wallet/pricing work.
- Latest checkpoint on that branch:
  - Commit: `0d82202`
  - Message: `Clean up meal dashboard, remove legacy files, and sync Ubuntu 26 development environment`
- New branch created from that checkpoint:
  - `feature/person-architecture`
- New branch pushed to origin and tracking `origin/feature/person-architecture`.

Before editing, always run:

```bash
git status
git branch --show-current
```

Expected branch:

```text
feature/person-architecture
```

## Recent completed work

### Meal naming direction

The project moved from lunch-specific concepts toward meal naming so the system can later support lunch, breakfast, snack, staff meals, event meals, or other meal types.

Examples of current concepts:

- `MealProfile`
- `MealRecord`
- `MealSubscription`
- Meal dashboard/template naming

Do not reintroduce hard-coded lunch-only architecture except where preserving legacy compatibility.

### Wallet behavior

Wallet-based meal confirmation/refund behavior has been developed. Conceptual wallet behaviors include:

- Enough balance: confirm meal and deduct.
- Insufficient balance and deny: do not confirm.
- Insufficient balance but allow unpaid/postpaid behavior: confirm with unpaid handling.
- Insufficient balance and allow negative: confirm and allow negative wallet balance.

There was later cleanup around postpaid behavior and multiple `MealProfile` behavior per student/person, including date-range and wallet-priority behavior.

### Meal dashboard cleanup

The latest cleanup removed old backup/temporary files and kept the active dashboard cleaner.

Deleted legacy examples included:

- `admin_before_best_latest_fix.py`
- `meal_G1.html`, `meal_G2.html`, `meal_G3.html`
- `meal__.html`
- `meal_before_live_status_strip.html`
- `meal_before_phase3_cards_refresh.html`
- `views_C1.py`, `views_G1.py`
- `views_before_best_latest_fix.py`
- `views_before_meal_idempotent_confirm_and_denied_reconfirm.py`
- `views_before_phase3_cards_refresh.py`

Important active files include:

```text
apps/attendance/templates/attendance/dash/meal.html
apps/attendance/templates/attendance/dash/_meal_rows.html
apps/attendance/views.py
apps/attendance/views_.py
apps/attendance/services.py
apps/attendance/utils/meal.py
```

### Git hygiene

The `.gitignore` was updated to avoid committing local/dev files such as:

```text
.django-completion-cache.json
tree.txt
tree_n.txt
migration_backup/
*.bak
__pycache__/
*.pyc
media/
db.sqlite3
```

AI agents must not add virtualenvs, pycache, media, SQLite DBs, API keys, `.env` files, service-account JSON, dumps, or local backups.

## Known Django layout

Important apps:

```text
apps/
├── attendance
├── cameras
└── scheduler
```

Important project package:

```text
bisk/
├── settings.py
├── settings_home.py
├── settings_prod.py
├── urls.py
└── views.py
```

Important attendance files:

```text
apps/attendance/
├── admin.py
├── api.py
├── models.py
├── resources.py
├── serializers.py
├── services.py
├── signals.py
├── urls.py
├── views.py
├── views_.py
├── utils/meal.py
└── templates/attendance/dash/
    ├── meal.html
    └── _meal_rows.html
```

## Current strategic goal

The next major goal is:

> Introduce a robust Person architecture before adding discount rules.

Reason: meal, wallet, subscriptions, reports, and discounts must support Students, Staff, Teachers, Guests, and future person categories. If discounts are implemented first against only `Student`, the project may need another redesign later.

Recommended order:

1. Person architecture.
2. Wallet/meal subscription migration to Person.
3. Staff support.
4. Discount engine.
5. Payment/reporting improvements.
6. Remote software licensing as a final long-term commercial feature.

## Desired future Person architecture

Preferred design:

```text
Person
├── StudentProfile
├── StaffProfile
└── Future profile types
```

Conceptually:

- `Person` stores shared identity and status.
- `StudentProfile` stores student-only fields such as grade, homeroom, code, guardian-related fields.
- `StaffProfile` stores staff-only fields such as department, role/title, employee ID.
- Future profiles may support guests, contractors, parents, or other categories.

Target direction:

```text
Wallet → Person
MealSubscription → Person
MealRecord → Person
Discounts → Person
```

Do this incrementally. Existing student features must keep working.

## AI workflow rule

When an AI coding agent works on this project:

1. Inspect relevant files before editing.
2. Produce a short implementation plan.
3. Make one small change set.
4. Run checks.
5. Update docs when meaningful.
6. Show `git diff --stat`.
7. Wait for user approval before committing unless explicitly told to commit.

Do not implement large architecture changes in one pass.

## Required checks before success

At minimum:

```bash
python manage.py check
python manage.py makemigrations --check --dry-run
git status
git diff --stat
```

If model/migration changes are intentional:

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py check
```

If tests exist and are safe:

```bash
python manage.py test
```

## OpenCode/OpenRouter harness goal

Recommended harness:

- OpenCode in the terminal.
- OpenRouter provider.
- Primary economical model: `z-ai/glm-5.2`.
- Root `AGENTS.md` for concise agent instructions.
- Long knowledge files under `docs/agent/`.

Use bounded tasks. Do not let the agent freely rewrite the system.

## Documentation update rule

After meaningful work, update:

- `docs/agent/CHANGELOG_AI.md`
- `docs/agent/DECISIONS.md` if a new architectural decision was made
- `docs/agent/ROADMAP.md` if the roadmap changes
- `docs/agent/PROJECT_MASTER.md` if major project state changes

Append entries instead of rewriting the whole files unless explicitly instructed.

## Long-term commercial feature: remote software licensing

Final long-term roadmap item: flexible software license/subscription control per customer.

Planned capabilities:

- Customer license records.
- License keys or signed license tokens.
- Remote license verification.
- Offline grace period.
- Feature/module enablement: attendance, recognition, meal, wallet, reports, API/mobile, multi-campus.
- Usage limits: persons/students, cameras, campuses, users.
- Trial/monthly/yearly/lifetime/internal license types.
- License states: active, expiring, expired, suspended, grace period.
- Emergency local override for support.
- Audit logging of license checks.
- Admin UI for current license state.
- Future central licensing server.

Implement this last, after the core product stabilizes.
