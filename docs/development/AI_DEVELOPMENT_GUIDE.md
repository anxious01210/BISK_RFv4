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

# Long-Term Vision

BISK_RFv4 is intended to become a modular ERP platform.

AI should optimize for long-term maintainability rather than short-term convenience.

When in doubt, choose the solution that keeps the architecture clean and extensible.
