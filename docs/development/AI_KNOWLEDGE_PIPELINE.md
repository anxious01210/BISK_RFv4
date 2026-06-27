# BISK_RFv4 AI Knowledge Pipeline

## Purpose

Keep project knowledge synchronized with code so AI sessions become cheaper, faster, and more accurate.

## Permanent Documentation

Long-term knowledge lives in:

- `docs/`
- `AGENTS.md`
- `docs/ENGINEERING_HANDBOOK.md`

## AI Working Knowledge

AI-generated working knowledge lives in:

- `.ai/sessions/`
- `.ai/summaries/`
- `.ai/graph/`
- `.ai/decisions/`

## Required After Meaningful AI Work

After a meaningful AI-assisted task, update as needed:

1. `.ai/sessions/...` — raw session report
2. `.ai/summaries/...` — short reusable summary
3. `.ai/graph/*.json` — structured entities/relationships/workflows
4. `.ai/decisions/...` — decision record if architecture changed
5. `docs/agent/CHANGELOG_AI.md` — human-readable changelog

## Rule

AI-generated knowledge must not contain:

- secrets
- API keys
- database dumps
- student private data
- production credentials
- service account files

## Workflow

1. AI inspects or implements.
2. AI reports result.
3. User decides if the result is worth saving.
4. Save raw report in `.ai/sessions/`.
5. Extract reusable facts into `.ai/summaries/`.
6. Extract structured facts into `.ai/graph/`.
7. Commit useful knowledge with code or as its own commit.
