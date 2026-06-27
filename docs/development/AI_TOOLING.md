# BISK_RFv4 AI Tooling

## Current Tools

- ChatGPT: architecture, planning, review
- OpenCode 1.17.11: implementation harness
- OpenRouter: model provider
- GLM-5.2: default economical implementation model
- VS Code: planned lighter editor replacing PyCharm for daily work

## Current Policy

OpenCode should not be used as an uncontrolled coder.

Workflow:

1. ChatGPT designs the task
2. OpenCode performs read-only inspection
3. User approves
4. OpenCode implements one small change
5. OpenCode runs checks
6. Report is saved if useful
7. User reviews and commits

## Session Reports

Useful paid OpenCode reports should be saved under:

`.ai/sessions/opencode/YYYY-MM-DD-topic.md`

## Graphifyy Plan

Graphifyy should be evaluated after several OpenCode sessions.

Expected value:

- Convert project/session knowledge into structured JSON
- Reduce repeated scanning
- Make context cheaper and faster
- Improve AI recall across sessions

Do not add it until we define where its generated files live and what should be committed.

## Ponytail / Review Tool Plan

Use only as a reviewer, not architect.

Expected value:

- Migration review
- Diff review
- Risk detection
- Duplicate logic detection
