# BISK_RFv4 Documentation Map

## Purpose

This document explains where every type of project knowledge belongs.

When humans or AI assistants need information, this should be one of the first documents they read after `AGENTS.md`.

---

# Root

## AGENTS.md

Entry point for every AI coding assistant.

Contains:

* AI rules
* Engineering workflow
* Required reading
* Repository behavior

---

# docs/

## ENGINEERING_HANDBOOK.md

Master index of all engineering documentation.

---

## DOCUMENTATION_MAP.md

Explains where project information is stored.

---

# docs/agent/

Purpose:

Fast onboarding for AI assistants.

Contains:

* PROJECT_MASTER.md
* ARCHITECTURE.md
* CHANGELOG_AI.md
* ROADMAP.md
* DECISIONS.md
* CODING_RULES.md
* OPENCODE_SETUP.md

These are concise documents intended for quick context.

---

# docs/architecture/

Long-term architecture documentation.

Examples:

* Identity Architecture
* Attendance Architecture
* Wallet Architecture
* Meal Architecture

---

# docs/database/

Database documentation.

Examples:

* ER diagrams
* Schema decisions
* Migration strategy

---

# docs/workflows/

Business workflows.

Examples:

* Meal confirmation
* Attendance recognition
* Wallet charging
* Refund workflow

---

# docs/modules/

One document per Django app/module.

Examples:

* attendance.md
* students.md
* recognition.md

---

# docs/development/

Development process.

Contains:

* AI tooling
* AI memory pipeline
* AI knowledge pipeline
* Development workflow
* Tool evaluations

---

# docs/prompts/

Reusable prompts and personas.

Examples:

* Architect
* Engineer
* Reviewer
* Read-only inspection

---

# docs/checklists/

Reusable engineering checklists.

---

# docs/templates/

Templates for future documentation.

---

# .ai/

AI working knowledge.

Not business documentation.

---

## .ai/sessions/

Raw AI conversations worth preserving.

---

## .ai/summaries/

Condensed reusable summaries.

---

## .ai/graph/

Structured JSON describing:

* entities
* workflows
* relationships

Future Graphifyy integration will primarily use this folder.

---

## .ai/decisions/

AI-generated architectural decision drafts.

Accepted decisions should eventually be reflected in `docs/agent/DECISIONS.md`.

---

# Ownership Rules

Permanent project knowledge belongs in `docs/`.

Temporary AI working knowledge belongs in `.ai/`.

Raw conversations should never replace documentation.

When information becomes stable, promote it from `.ai/` into `docs/`.

---

# Long-Term Vision

The repository should become self-describing.

A new developer—or a new AI assistant—should understand the project by reading:

1. AGENTS.md
2. ENGINEERING_HANDBOOK.md
3. DOCUMENTATION_MAP.md
4. PROJECT_MASTER.md

before reading source code.
