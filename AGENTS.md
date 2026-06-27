# AGENTS.md — BISK_RFv4 Agent Instructions

This repository is BISK_RFv4, a Django school attendance, recognition, meal, wallet, and operations system.

Before making changes, read:

1. `docs/agent/PROJECT_MASTER.md`
2. `docs/agent/ARCHITECTURE.md`
3. `docs/agent/CODING_RULES.md`
4. `docs/agent/ROADMAP.md`
5. `docs/agent/DECISIONS.md`
6. `docs/agent/CHANGELOG_AI.md`

## Current branch goal

Work should happen on:

```bash
feature/person-architecture
```

The current major goal is to introduce a robust Person architecture before implementing discounts.

## Critical rules

- Do not rewrite the project broadly.
- Do not touch unrelated files.
- Do not commit unless explicitly instructed.
- Do not add secrets, `.env`, database files, media, pycache, virtualenvs, or local backups.
- Prefer small, reviewable changes.
- Keep existing student/meal/wallet workflows working.
- Use service-layer business logic.
- Update docs after meaningful work.

## Required checks

Before editing:

```bash
git status
git branch --show-current
```

Before declaring success:

```bash
python manage.py check
python manage.py makemigrations --check --dry-run
git status
git diff --stat
```

If migrations are intentionally created:

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py check
```

## Current architecture direction

Target:

```text
Person
├── StudentProfile
├── StaffProfile
└── Future profile types
```

Eventually:

```text
Wallet → Person
MealSubscription → Person
MealRecord → Person
Discounts → Person
```

Do this incrementally. Do not delete the current Student model early.

## AI workflow

For large tasks:

1. Inspect files.
2. Propose plan.
3. Wait for approval.
4. Implement small step.
5. Run checks.
6. Summarize diff.
7. Update docs.
