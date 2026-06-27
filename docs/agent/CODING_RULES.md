# BISK_RFv4 Coding Rules for AI Agents

Version: 1.0

## Required behavior

Before changes:

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

## Branch discipline

Current branch for Person work:

```bash
feature/person-architecture
```

Do not switch branches unless explicitly asked. Do not commit unless explicitly asked.

## Never add to Git

```text
.venv/
venv/
env/
migration_backup/
__pycache__/
*.pyc
media/
db.sqlite3
tree.txt
tree_n.txt
.django-completion-cache.json
*.bak
.env
*.sqlite3
```

Never add secrets, API keys, DB dumps, or service account JSON files.

## Editing rules

Prefer small edits and reversible migrations. Do not perform broad rewrites.

Avoid:

- Renaming many models at once.
- Deleting old fields before migration/backfill.
- Large generated diffs.
- Touching unrelated apps.
- Editing production settings without need.
- Changing dashboard UI while doing model migrations unless necessary.

## Django rules

- Use migrations for model changes.
- Do not manually edit existing applied migrations unless explicitly instructed.
- Use `PROTECT` or careful `SET_NULL` for important data relations.
- Avoid destructive cascades for financial data.
- Use `DecimalField` for money.
- Avoid floats for wallet/pricing calculations.
- Prefer explicit `related_name`.
- Use `TextChoices` for stable choices.

## Money and wallet rules

Money must be auditable.

- Use `Decimal`.
- Store transaction records.
- Never silently modify balances without a transaction.
- Refunds should create transaction history.
- Meal confirmations should be idempotent.
- Store price and discount snapshots when discounts are implemented.

## Person architecture rules

The Person refactor must be incremental. Do not delete `Student` early.

Temporary compatibility may be acceptable:

```text
Student → Person
Wallet → Student and/or Person temporarily
MealSubscription → Student and/or Person temporarily
```

Later target:

```text
Wallet → Person
MealSubscription → Person
```

## Documentation update rules

After meaningful changes:

- Append to `docs/agent/CHANGELOG_AI.md`.
- Add a decision entry to `docs/agent/DECISIONS.md` for architectural choices.
- Update `docs/agent/ROADMAP.md` if milestones changed.
- Update `docs/agent/PROJECT_MASTER.md` for major state changes.

Do not rewrite the entire docs without permission.

## Prompt discipline

Good prompt:

```text
Inspect apps/attendance/models.py and propose the minimal Person model migration plan. Do not edit files yet.
```

Bad prompt:

```text
Refactor the whole system to support staff.
```
