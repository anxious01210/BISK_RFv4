# BISK_RFv4 AI Development Platform

## Purpose

Define how AI tools are used safely and economically in BISK_RFv4.

## Roles

| Role | Tool | Responsibility |
|---|---|---|
| Architect | ChatGPT | Architecture, planning, review |
| Implementer | OpenCode + OpenRouter GLM 5.2 | Small code changes |
| Knowledge Graph | Graphifyy | Code/document relationship discovery |
| Reviewer | Ponytail or review agent | Review diffs, migrations, risks |
| Lead Engineer | User | Final approval, commits, releases |

## Rules

- AI implementers must read `AGENTS.md`.
- AI implementers must follow `docs/checklists/AI_Change_Checklist.md`.
- OpenCode should start with read-only inspection.
- Graphifyy is for understanding, not rewriting.
- Ponytail/review agents are for review, not architecture.
- No AI tool commits unless explicitly approved.
- All meaningful AI work updates `docs/agent/CHANGELOG_AI.md`.

## Standard workflow

1. ChatGPT designs the task.
2. OpenCode inspects files read-only.
3. User approves plan.
4. OpenCode implements one small step.
5. OpenCode runs checks.
6. Ponytail/reviewer checks the diff.
7. User commits.
8. Docs are updated.

## Cost control

- Use small prompts.
- Avoid full-repo repeated scans.
- Prefer targeted file inspection.
- Use docs and checklists instead of long repeated prompts.
- Commit stable checkpoints.

## First tool setup order

1. OpenCode + OpenRouter
2. Read-only test
3. Graphifyy
4. Ponytail/review tool
5. Person architecture implementation
