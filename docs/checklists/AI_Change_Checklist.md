# AI Change Checklist

Before editing:

- [ ] Confirm branch with `git branch --show-current`
- [ ] Confirm clean or understood state with `git status`
- [ ] Read `AGENTS.md`
- [ ] Read relevant docs under `docs/`

During editing:

- [ ] Keep change small
- [ ] Avoid unrelated files
- [ ] Do not add secrets, media, db files, pycache, virtualenvs, or backups
- [ ] Preserve existing behavior unless change is approved

After editing:

- [ ] Run `python manage.py check`
- [ ] Run `python manage.py makemigrations --check --dry-run`
- [ ] Show `git diff --stat`
- [ ] Update relevant docs if meaningful
- [ ] Do not commit unless user approves
