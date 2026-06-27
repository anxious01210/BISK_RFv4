# BISK_RFv4 AI Memory Pipeline

## Purpose

Preserve useful paid AI work so future sessions are cheaper, faster, and more accurate.

## Principle

Every useful AI session should become a reusable project asset.

## Storage

Session reports:

`.ai/sessions/opencode/YYYY-MM-DD-topic.md`

Future structured memory:

`.ai/memory/YYYY-MM-DD-topic.json`

## Required Session Metadata

Each saved session should include:

- Date
- Branch
- Tool
- Model
- Provider
- Approximate cost
- Approximate context tokens
- Mode: read-only / implementation / review
- Files inspected
- Files modified
- Checks run
- Key findings
- Decisions
- Risks
- Recommended next steps

## Workflow

1. Run AI task.
2. If output is useful, save it under `.ai/sessions/`.
3. Extract important facts into docs if they become permanent.
4. Later convert useful reports into structured JSON.
5. Later evaluate Graphifyy for indexing/querying this structured knowledge.

## Graphifyy Role

Graphifyy should eventually help convert project/session knowledge into searchable structured context.

It should not replace source documentation.

Expected future use:

- Build code/document relationship graph.
- Store structured JSON.
- Reduce repeated AI scanning.
- Improve long-term recall.
- Lower cost for repeated architecture questions.

## Rule

Do not save secrets, credentials, personal data, database dumps, or private student data in AI memory files.
