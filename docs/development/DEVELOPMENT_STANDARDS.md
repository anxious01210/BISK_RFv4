# BISK_RFv4 Development Standards

## Purpose

This document defines the development standards for the BISK_RFv4 project.

These standards apply to:

- Human developers
- ChatGPT
- OpenCode
- Claude Code
- Gemini
- Any future contributor

The goal is to keep the codebase consistent, maintainable, and predictable over the lifetime of the project.

---

# Python

- Python 3.12
- Virtual environment required
- UTF-8 encoding
- LF line endings

---

# Formatting

- Black
- Ruff
- .editorconfig
- Format on Save enabled

Do not manually reformat files unless it is the purpose of the commit.

---

# Django

Every model should normally include:

- created_at
- updated_at

Only add is_active when it has real business meaning.

---

# Naming

Prefer descriptive names.

Examples:

Person
StudentProfile
StaffProfile
PersonRole

code
employee_id
created_at
updated_at

Avoid legacy names unless required for compatibility.

---

# Git

One feature per commit.

Do not mix:

- formatting
- refactoring
- feature implementation

in the same commit.

---

# AI Workflow

Architecture
↓

Implementation
↓

Review
↓

Commit
↓

Push

OpenCode may generate code.

ChatGPT performs architectural review before commit.

---

This document evolves with the project.
