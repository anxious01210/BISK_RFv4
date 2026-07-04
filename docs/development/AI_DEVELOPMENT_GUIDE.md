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

# OpenCode Implementation Review Packages

Every OpenCode implementation task — whether or not the prompt repeats
the instruction — must produce a **complete review package** under:

```text
.tmp/reviews/YYYY-MM-DD_<feature-name>/
```

`<feature-name>` is a short, kebab-case slug for the work performed
(e.g. `finance-backend-foundation`, `person-architecture`,
`attendance-migration-plan`). The date is the day the package is
assembled (ISO `YYYY-MM-DD`).

The package is the durable record of what was done, why, and how it was
verified. It must be **complete enough for another engineer or AI
assistant to review the implementation without rerunning the work**.

## Required files

Every package must contain **all** of the following files. Files must
not be placeholders, stubs, or empty. If a file is genuinely not
applicable (e.g. no tests for a docs-only change), say so explicitly
inside the file with a one-line reason — do not omit it.

| File | Contents |
|---|---|
| `README.md` | Overview of the package: task name, date, branch, scope, files changed, and a map of the other files in the package. The first thing a reviewer reads. |
| `architecture_review.md` | How the change fits the architecture, design decisions, assumptions, deviations from prior architecture docs, and concerns. Reference the relevant architecture docs by path. |
| `implementation_notes.md` | What was implemented, model/service/selector/validator/admin breakdown, migration status, and any conscious implementation choices. |
| `test_results.md` | Test commands run and their full output (success or failure). Include counts and any warnings. |
| `commands_executed.txt` | Every shell command executed during the task, one per line, in order. Reproducible from this file alone. |
| `git_status.txt` | Output of `git status` at the end of the task. |
| `git_diff.patch` | Output of `git diff` (and `git diff --cached` if anything was staged). The full reviewable patch. |
| `git_diff_stat.txt` | Output of `git diff --stat` for a quick file/line summary. |
| `generated_files.txt` | List of files created or modified by the task, with a one-line description of each. Include migration files. |
| `tree.txt` | Output of `tree -L 2 apps/<app>` (or the relevant directory) so the reviewer can see the resulting structure without checking out the branch. If `tree` is unavailable, use `find` with a depth limit. |

## Rules

1. **Always create the package.** This is not optional and does not need
   to be repeated in each prompt. Completing an implementation task
   without a package is an incomplete task.
2. **Files must not be placeholders.** Every file must carry real
   content captured from the actual session. A `README.md` that says
   "TODO" or a `test_results.md` that says "tests passed" without the
   output is a failed package.
3. **Completeness over conciseness.** Prefer full command output over
   summaries; reviewers must not need to rerun the work.
4. **Captured at the end of the task.** `git_status.txt`, `git_diff.patch`
   and `git_diff_stat.txt` reflect the state after all work is done and
   before any commit.
5. **Local only.** `.tmp/reviews/` is local review output and is
   gitignored. It **must not be committed** unless the user explicitly
   requests it. The committed record of a session is the concise
   Markdown summary under `.ai/sessions/opencode/`, not the package
   itself.
6. **One package per task.** Do not overwrite an earlier package from a
   different task; create a new dated directory. Re-runs of the same
   task may suffix the directory with `_v2`, `_v3`, etc.
7. **No archives in the repository root.** If a `.tar.gz` snapshot is
   useful, keep it under `.tmp/reviews/` (also gitignored).
8. **Documentation-only changes.** A docs-only change that does not run
   code still produces the package; `test_results.md` and
   `commands_executed.txt` then document that no tests/commands apply
   (with a one-line reason) rather than being omitted.

## Example

```bash
PKG=.tmp/reviews/2026-07-04_finance-backend-foundation
mkdir -p "$PKG"

git status > "$PKG/git_status.txt" 2>&1
git diff > "$PKG/git_diff.patch" 2>&1
git diff --stat > "$PKG/git_diff_stat.txt" 2>&1
git diff --name-only > "$PKG/generated_files.txt" 2>&1
# Append untracked new files so generated_files.txt is complete:
git status --short >> "$PKG/generated_files.txt" 2>&1

python manage.py check > "$PKG/check_output.txt" 2>&1
python manage.py makemigrations --check --dry-run > "$PKG/makemigrations_check_output.txt" 2>&1
TEST_DB_USER=bisk_test TEST_DB_PASSWORD='BiskTestDB1!' \
    python manage.py test apps.finance > "$PKG/test_output.txt" 2>&1

tree -L 2 apps/finance > "$PKG/tree.txt" 2>/dev/null || find apps/finance -maxdepth 2 > "$PKG/tree.txt"
```

`README.md`, `architecture_review.md`, `implementation_notes.md`, and
`test_results.md` are then written by hand from the session; the raw
captures above are embedded or referenced as appropriate.

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
