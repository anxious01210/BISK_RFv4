# BISK_RFv4 AI Development Guide

## Purpose

This document defines how AI assistants should contribute to the BISK_RFv4 project.

It applies to ChatGPT, OpenCode, Claude Code, Gemini, and any future AI coding assistant.

The goal is to ensure that AI accelerates development while maintaining a consistent architecture and high code quality.

---

# Core Principles

AI should assist development, not control the architecture.

Architectural decisions are made before implementation.

Implementation follows the approved architecture.

---

# Development Workflow

Every feature follows this order:

1. Requirements
2. Architecture
3. Implementation
4. Review
5. Commit
6. Push

Never skip the review step.

---

# Responsibilities

## ChatGPT

Responsible for:

- Architecture
- Domain modeling
- Design reviews
- Code reviews
- Migration reviews
- Naming consistency
- Long-term maintainability

ChatGPT reviews work before commits whenever practical.

---

## OpenCode

Responsible for:

- Implementing approved tasks
- Refactoring repetitive code
- Generating boilerplate
- Producing implementation reports

OpenCode should not introduce architectural changes unless explicitly requested.

---

## Other AI Assistants

Claude Code, Gemini, and similar assistants may assist implementation, but should follow the same project standards and architecture.

---

# Coding Rules

AI should:

- Follow DEVELOPMENT_STANDARDS.md
- Follow PROJECT_ARCHITECTURE.md
- Prefer readability over cleverness
- Keep commits focused
- Avoid unrelated formatting changes
- Preserve backward compatibility when possible

---

# Migrations

One logical feature should normally produce one migration.

Do not mix unrelated schema changes.

Review migrations before applying them.

---

# Commits

One feature per commit.

Avoid combining:

- formatting
- tooling
- refactoring
- business features

into a single commit.

---

# Reviews

Every implementation should be reviewed for:

- Architecture
- Naming
- Database design
- Relationships
- Extensibility
- Django best practices

---

# Temporary Files and Generated Artifacts

AI assistants must keep the repository clean.

Unless explicitly requested otherwise, all temporary files and generated review artifacts must be created under:

```text
.tmp/
```

The directory should be created automatically if it does not already exist.

Recommended structure:

```text
.tmp/
├── archives/
├── exports/
├── opencode/
├── reports/
├── reviews/
└── work/
```

Examples:

```text
.tmp/reviews/person_review_2026-07-03.tar.gz
.tmp/reports/identity_summary.md
.tmp/opencode/session_2026-07-03.log
.tmp/exports/person_export.csv
```

Rules:

- Never generate archives in the project root.
- Never generate zip or tar files beside source code.
- Never leave temporary files inside `apps/`.
- Never write scratch files into `docs/` unless they are intended to become permanent documentation.
- Remove obsolete temporary files when they are no longer needed.
- All AI-generated temporary artifacts are disposable unless explicitly promoted into the repository.
- Review archives should use clear names such as `<feature>_review_<YYYY-MM-DD>.tar.gz`.

---

# Creating New Files

AI assistants should avoid creating new files unless there is a clear architectural reason.

Before creating a new file, prefer:

- extending an existing module
- extending an existing document
- extending an existing test

When a new file is justified:

- use the project's naming conventions
- place it in the correct architectural location
- avoid duplicate functionality
- explain why a new file is necessary

---

# Repository Hygiene

The project root should remain minimal.

Only long-term project files belong in the repository root.

Temporary artifacts, generated reports, review packages, exported data, AI scratch files, and intermediate outputs must not clutter the project root.

AI assistants should not create files in tracked project directories unless those files are part of the intended source code, documentation, tests, or configuration change.

---

# Review Artifact Standard

When an AI assistant completes an implementation task, it should generate review artifacts under:

```text
.tmp/reviews/
```

A normal review package should include, when practical:

- `git_status.txt`
- `changed_files.txt`
- `diff.patch`
- `check_output.txt`
- `makemigrations_check_output.txt`
- relevant test output, such as `test_output.txt`

The review archive should be created under `.tmp/reviews/`, not in the repository root.

Example:

```bash
mkdir -p .tmp/reviews/identity_review_2026-07-03

git status > .tmp/reviews/identity_review_2026-07-03/git_status.txt
git diff --name-only > .tmp/reviews/identity_review_2026-07-03/changed_files.txt
git diff > .tmp/reviews/identity_review_2026-07-03/diff.patch
python manage.py check > .tmp/reviews/identity_review_2026-07-03/check_output.txt 2>&1
python manage.py makemigrations --check --dry-run > .tmp/reviews/identity_review_2026-07-03/makemigrations_check_output.txt 2>&1
python manage.py test apps.identity > .tmp/reviews/identity_review_2026-07-03/test_output.txt 2>&1

tar -czf .tmp/reviews/identity_review_2026-07-03.tar.gz -C .tmp/reviews identity_review_2026-07-03
```

---

# OpenCode Session Logs

OpenCode produces raw session logs (terminal captures of the full AI
interaction). These logs are useful for debugging but must not bloat the
Git repository.

## Rules

1. **Raw OpenCode session logs should not be committed if they are large.**
   Raw logs are verbose and grow quickly.

2. **Raw logs should be stored locally under `.tmp/opencode/`** because
   `.tmp/` is ignored by Git. This keeps them available locally without
   polluting the repository.

3. **Committed session records under `.ai/sessions/opencode/` should be
   concise Markdown summaries, not huge raw logs.** Use the
   `SESSION_SUMMARY_TEMPLATE.md` template located in that directory.

4. **A session summary must include:**
   - Task name
   - Date/time
   - Tools/model used
   - Files changed
   - Decisions made
   - Tests/checks run
   - Review archive path
   - Final outcome

5. **If a raw session log is important, keep it locally or archive it
   outside Git** (e.g., in `.tmp/opencode/` or an external archive). Do not
   commit the raw `.txt` capture.

6. **Avoid committing session files larger than 5 MB.** If a session file
   exceeds 5 MB, replace it with a concise Markdown summary.

7. **Never commit session files larger than 50 MB.** This is a hard limit.
   A 50 MB+ session file must not enter version control under any
   circumstance.

## Process

When an OpenCode session completes:

1. Save the raw log to `.tmp/opencode/` (it is gitignored).
2. Create a concise Markdown summary under `.ai/sessions/opencode/` using
   the template.
3. Verify the summary file size is well under 5 MB before committing.
4. Never `git add` raw `.txt` session captures.

## Template

Use `.ai/sessions/opencode/SESSION_SUMMARY_TEMPLATE.md` as the starting
point for every committed session summary.

---

# Long-Term Vision

BISK_RFv4 is intended to become a modular ERP platform.

AI should optimize for long-term maintainability rather than short-term convenience.

When in doubt, choose the solution that keeps the architecture clean and extensible.
