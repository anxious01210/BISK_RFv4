# OpenCode + OpenRouter Setup Guide for BISK_RFv4

Version: 1.0

## 1. Verify branch

```bash
cd ~/PycharmProjects/BISK_RFv4-1
git status
git branch --show-current
```

Expected:

```text
feature/person-architecture
nothing to commit, working tree clean
```

## 2. Add documentation bundle

Create folder:

```bash
mkdir -p docs/agent
```

Copy these files into `docs/agent/`:

```text
PROJECT_MASTER.md
ARCHITECTURE.md
CODING_RULES.md
ROADMAP.md
DECISIONS.md
CHANGELOG_AI.md
```

Copy `AGENTS.md` into the project root.

Then check:

```bash
ls -lh docs/agent
ls -lh AGENTS.md
```

## 3. Commit documentation

```bash
git add AGENTS.md docs/agent/
git status
git commit -m "Add AI agent project knowledge and roadmap"
git push
```

## 4. Install/Open OpenCode

Check if installed:

```bash
opencode --version
```

If not installed, follow current official OpenCode install docs from `https://opencode.ai/docs/`.

Do not paste API keys into chat.

## 5. Connect OpenCode to OpenRouter

From project root:

```bash
opencode
```

Inside OpenCode:

```text
/connect
```

Choose OpenRouter and paste your API key when prompted.

## 6. Select model

Inside OpenCode:

```text
/models
```

Recommended starting model:

```text
z-ai/glm-5.2
```

Use it for planning and bounded implementation.

## 7. First safe read-only test

Inside OpenCode, run:

```text
Read AGENTS.md and docs/agent/PROJECT_MASTER.md. Then inspect apps/attendance/models.py, apps/attendance/admin.py, apps/attendance/services.py, and apps/attendance/views.py. Do not edit files. Summarize the current meal/wallet/student architecture and propose the smallest safe first step for Person architecture.
```

Expected:

- It should not edit files.
- It should summarize.
- It should propose a small plan.
- It should mention checks.

## 8. First implementation prompt only after review

After manual review:

```text
Implement only the first step of the approved Person architecture plan. Add the minimal Person model and admin registration if needed. Do not update meal/wallet logic yet. Create migrations if required. Run python manage.py check and show git diff --stat. Do not commit.
```

## 9. Cost control rules

- Use read-only inspection prompts first.
- Avoid repeatedly asking the model to read the whole repo.
- Keep tasks small.
- Ask for plans before code.
- Use `git diff --stat` after each step.
- Commit stable checkpoints.
- Stop the agent if it starts broad rewrites.

## 10. Optional tools and skills

Start simple:

- OpenCode
- OpenRouter
- `z-ai/glm-5.2`
- `AGENTS.md`
- `docs/agent/`

Only add Graphify/Ponytail-style tools after the basic harness works.

## 11. Recovery commands

If unwanted unstaged edits:

```bash
git status
git diff
git restore <file>
```

If many unwanted edits:

```bash
git restore .
```

If unwanted untracked files:

```bash
git clean -n
git clean -fd
```

Always preview with `git clean -n` first.
