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

## AI session persistence

Every meaningful OpenCode session must produce permanent project knowledge.

Unless explicitly instructed otherwise:

- Never finish an important session without generating a session report.
- Never overwrite previous reports.
- Reports are part of the repository knowledge base.

All OpenCode session reports must be saved under:

```text
.ai/sessions/opencode/
```

Filename format:

```text
YYYY-MM-DD_HH-MM-SS_<short-topic>.md
```

Example:

```text
.ai/sessions/opencode/2026-06-30_17-39-16_person-architecture-risk-report.md
```

Raw terminal logging with `opencode-log` is allowed as a backup, but it is not a replacement for a clean Markdown session report.

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

## Session report requirements

At the end of every architecture discussion, implementation task, investigation, debugging session, migration planning session, or significant code review, generate a clean GitHub-flavored Markdown report.

The report must begin with metadata:

```yaml
---
date: YYYY-MM-DD
branch: <current-branch>
model: <model-used>
mode: <read-only | planning | implementation | debugging | review>
status: <completed | partial | blocked>
topic: <short topic>
tags:
  - <tag>
related:
  - <related file or document>
---
```

The report must contain these sections:

```md
# Objective

What was requested.

# Findings

Important discoveries.

# Risks

Technical risks or architectural concerns.

# Decisions

Decisions made during the session.

# Implementation

What was implemented.

If nothing was implemented, explicitly state:

No implementation was performed.

# Remaining Work

Numbered list.

# Files Modified

List every modified file.

If none:

None

# Commands Executed

Important commands executed.

# Recommendations

Recommendations for the next session.
```

The report must be human-readable and suitable to upload into ChatGPT for architectural review.

## Deliverables for ChatGPT review

Whenever a session contains architecture analysis, migration planning, database changes, major implementation, debugging, or design decisions, always generate a Markdown report before finishing.

The final OpenCode response should include the generated report path:

```text
Report generated:
.ai/sessions/opencode/<filename>.md
```

If no report was generated, explain why.

## AI workflow

For large tasks:

1. Inspect the repository.
2. Propose a technical plan.
3. Wait for approval.
4. Implement one logical step.
5. Run verification checks.
6. Summarize the code changes.
7. Update project documentation if required.
8. Generate the session report.
9. Save the report under `.ai/sessions/opencode/`.
10. Mention the generated report path in the final response.
