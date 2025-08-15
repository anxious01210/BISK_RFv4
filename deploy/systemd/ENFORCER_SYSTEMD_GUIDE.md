# BISK Enforcer — systemd Guide

This service keeps the **enforcer** running on the host via `systemd`.  
The enforcer calls `manage.py run_enforcer` (APScheduler) so camera schedules are enforced every **N** seconds and the dashboard `/dash/system/` shows **PID / Last run / Last OK**.

- Project: **BISK_RFv4**
- Interval: `ENFORCER_INTERVAL_SECONDS` (default **15s**)
- Single-instance lock: `/tmp/bisk_enforcer.lock`
- Timezone: `Asia/Baghdad` (set via env)

---

## 1) What differs between DEV and PROD?

Only host-specific values:

- **User** that runs the service (e.g., `rio` on dev, `bisk` on prod)
- **Project path** (e.g., `/home/rio/PycharmProjects/BISK_RFv4` vs `/srv/BISK_RFv4`)
- **Venv** Python path (e.g., `/…/.venv/bin/python`)
- Optional tuning: interval, lock path, PATH, TZ

Everything else is identical. We keep a **template + installer** in the repo to stamp per-host units in ~30 seconds.

---

## 2) Files in this folder

```text
deploy/systemd/
├─ bisk-enforcer.env.example          # example env for /etc/bisk/bisk-enforcer.env
├─ bisk-enforcer.service.template     # systemd unit template (with placeholders)
└─ install_enforcer.sh                # renders + installs + enables the service
```

> **Do not commit** real host env files. Only the example stays in git.  
> The real env file lives on each host at: `/etc/bisk/bisk-enforcer.env`.

---

## 3) Prerequisites (per host)

- Linux with systemd (Ubuntu 22.04+ recommended)
- Project folder with `.venv` and `manage.py`
- `ffmpeg`/`ffprobe` installed (usually `/usr/bin`)
- (Optional) NVIDIA driver + `nvidia-smi` for GPU stats

---

## 4) Quick Start (any host: DEV or PROD)

The installer detects your current **user**, **project path**, and **.venv/bin/python**.  
On PROD, override with env vars.

```bash
cd /path/to/BISK_RFv4
chmod +x deploy/systemd/install_enforcer.sh

# DEV (accept detected values)
./deploy/systemd/install_enforcer.sh

# PROD example (service account + path override)
USER_NAME=bisk PROJECT_DIR=/srv/BISK_RFv4 ./deploy/systemd/install_enforcer.sh
```

### What the installer does
- Seeds `/etc/bisk/bisk-enforcer.env` (if missing) from the example
- Renders `/etc/systemd/system/bisk-enforcer.service` from the template
  - Fills in **User**, **WorkingDirectory**, **ExecStart** (venv Python)
- Runs: `systemctl daemon-reload && systemctl enable --now bisk-enforcer.service`

### Verify it’s running
```bash
systemctl status bisk-enforcer.service
journalctl -u bisk-enforcer.service -f -n 100
```

Open `/dash/system/` and watch **PID / Last run / Last OK** tick every ~15s.

---

## 5) Daily Operations

```bash
# Restart after code or env changes
sudo systemctl restart bisk-enforcer.service

# Stop / Start
sudo systemctl stop bisk-enforcer.service
sudo systemctl start bisk-enforcer.service

# Disable + stop (e.g., on dev box)
sudo systemctl disable --now bisk-enforcer.service

# Status & logs
systemctl status bisk-enforcer.service
journalctl -u bisk-enforcer.service -f -n 200
```

> **Don’t** also run `python manage.py run_enforcer` while the service is active.  
> If you do, it exits with **“Another enforcer holds the lock”** — expected and safe.

---

## 6) When to Restart

**Restart is required after:**
- Pulling code that changes the enforcer (e.g., `enforce_schedules()` or scheduler helpers)
- Editing `bisk.settings` values used by the enforcer (e.g., interval)
- Editing `/etc/bisk/bisk-enforcer.env` (interval/lock/TZ/PATH)
- Upgrading Python packages in the venv

**Restart is *not* required after:**
- Normal admin changes (cameras / schedules / pause windows)
- Runner processes starting/stopping (the tick re-reads DB each run)

---

## 7) Environment File (per host)

**Real file:** `/etc/bisk/bisk-enforcer.env`  
**Example in repo:** `deploy/systemd/bisk-enforcer.env.example`

```ini
DJANGO_SETTINGS_MODULE=bisk.settings
PYTHONUNBUFFERED=1
TZ=Asia/Baghdad

# Tuning
ENFORCER_INTERVAL_SECONDS=15
ENFORCER_LOCK_FILE=/tmp/bisk_enforcer.lock

# Ensure tooling is visible to systemd (ffmpeg, nvidia-smi, etc.)
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin
```

After editing the env file:
```bash
sudo systemctl restart bisk-enforcer.service
```

---

## 8) How it Works (under the hood)

- Unit runs: `ExecStart=…/.venv/bin/python manage.py run_enforcer`
- Enforcer (APScheduler):
  - Schedules `enforce_schedules()` every **N** seconds
  - Writes cache breadcrumbs: `enforcer:last_run`, `enforcer:last_ok`, `enforcer:last_error`, `enforcer:running_pid`
  - Uses a file lock at `/tmp/bisk_enforcer.lock` to guarantee a single instance
- `/dash/system/` reads breadcrumbs + system stats (CPU/RAM/Disk/GPU) and runner tallies

---

## 9) Troubleshooting

**Service won’t start**
- Logs:
  ```bash
  journalctl -u bisk-enforcer.service -n 200 --no-pager
  ```
- Check unit paths:
  - `WorkingDirectory` must be the project root (where `manage.py` lives)
  - `ExecStart` must point to your **venv** Python
- Ensure service user can read the project and `.venv`

**`ffmpeg` / `nvidia-smi` command not found**
- Add them to `PATH` in `/etc/bisk/bisk-enforcer.env` (see example), then:
  ```bash
  sudo systemctl restart bisk-enforcer.service
  ```

**Permission errors (writing `media/` / `snapshots/`)**
- Make sure the service `User=` owns or has RW permissions on those directories

**“Another enforcer holds the lock”**
- A second instance was attempted. Stop the running one first:
  ```bash
  sudo systemctl stop bisk-enforcer.service
  ```
  (or just let the manual run exit — harmless)

**Edited the unit file but behavior didn’t change**
- Always:
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl restart bisk-enforcer.service
  ```

---

## 10) Uninstall

```bash
sudo systemctl disable --now bisk-enforcer.service
sudo rm -f /etc/systemd/system/bisk-enforcer.service
sudo systemctl daemon-reload
# optional:
# sudo rm -f /etc/bisk/bisk-enforcer.env
```

---

## 11) Installer Variables (advanced)

You can override defaults when running the installer:

| Variable      | Meaning                               | Default                         | Example                                |
|---------------|----------------------------------------|---------------------------------|----------------------------------------|
| `USER_NAME`   | Linux user to run the service          | current shell user              | `USER_NAME=bisk`                       |
| `PROJECT_DIR` | Absolute path to project root          | current working directory       | `PROJECT_DIR=/srv/BISK_RFv4`           |
| `ENV_FILE`    | Path to env file the unit will read    | `/etc/bisk/bisk-enforcer.env`   | `ENV_FILE=/etc/bisk/bisk.env`          |

Example:
```bash
USER_NAME=bisk PROJECT_DIR=/srv/BISK_RFv4 ./deploy/systemd/install_enforcer.sh
```

---

## 12) Template Placeholders (FYI)

`deploy/systemd/bisk-enforcer.service.template` contains placeholders:

- `__USER__` → rendered from `USER_NAME`
- `__PROJECT_DIR__` → rendered from `PROJECT_DIR`
- `__PYTHON__` → rendered from detected venv python (or `python3` as last resort)

The installer replaces these, writes `/etc/systemd/system/bisk-enforcer.service`, then enables & starts it.

---

## 13) Optional Git Hygiene

Add to `.gitignore` to prevent committing real env files:

```gitignore
# deployment artifacts
deploy/systemd/*.env
!deploy/systemd/bisk-enforcer.env.example
```

---

## 14) FAQ

**Q: Do I need to do this again on production?**  
A: Yes — systemd units are per-machine. Clone the repo on prod, create the venv, then run the installer with prod values.

**Q: How do I change the tick interval?**  
A: Edit `ENFORCER_INTERVAL_SECONDS` in `/etc/bisk/bisk-enforcer.env`, then restart the service.

**Q: How do I move the project path or switch venvs later?**  
A: Re-run the installer with new `PROJECT_DIR` (and ensure `.venv` exists there). It overwrites the unit and restarts it.

**Q: What if I accidentally start a second enforcer manually?**  
A: It will print **“Another enforcer holds the lock”** and exit. No harm done.

---

---
```Text
(.venv) rio@GP73:~/PycharmProjects/BISK_RFv4$ chmod +x deploy/systemd/install_enforcer.sh 

(.venv) rio@GP73:~/PycharmProjects/BISK_RFv4$ sudo systemctl stop bisk-enforcer.service
[sudo] password for rio: 
(.venv) rio@GP73:~/PycharmProjects/BISK_RFv4$ USER_NAME=rio ./deploy/systemd/install_enforcer.sh
User:        rio
Project dir: /home/rio/PycharmProjects/BISK_RFv4
Python:      /home/rio/PycharmProjects/BISK_RFv4/.venv/bin/python
Env file:    /etc/bisk/bisk-enforcer.env
Proceed? [y/N] y
Installed. Check status with: sudo systemctl status bisk-enforcer.service


(.venv) rio@GP73:~/PycharmProjects/BISK_RFv4$ systemctl status bisk-enforcer.service 
● bisk-enforcer.service - BISK Enforcer (APScheduler) - periodic enforce_schedules()
     Loaded: loaded (/etc/systemd/system/bisk-enforcer.service; enabled; preset: enabled)
     Active: active (running) since Fri 2025-08-15 19:10:12 +03; 2min 29s ago
   Main PID: 405842 (python)
      Tasks: 3 (limit: 38070)
     Memory: 38.7M (peak: 39.1M)
        CPU: 444ms
     CGroup: /system.slice/bisk-enforcer.service
             └─405842 /home/rio/PycharmProjects/BISK_RFv4/.venv/bin/python manage.py run_enforcer

Aug 15 19:10:12 GP73 systemd[1]: Started bisk-enforcer.service - BISK Enforcer (APScheduler) - periodic enforce_schedules().
Aug 15 19:10:13 GP73 python[405842]: Enforcer started. Press Ctrl+C to stop.
(.venv) rio@GP73:~/PycharmProjects/BISK_RFv4$ python3 manage.py run_enforcer
Another enforcer holds the lock: /tmp/bisk_enforcer.lock
(.venv) rio@GP73:~/PycharmProjects/BISK_RFv4$ journalctl -u bisk-enforcer.service -f -n 100
Aug 15 19:06:28 GP73 systemd[1]: Started bisk-enforcer.service - BISK Enforcer (APScheduler) - periodic enforce_schedules().
Aug 15 19:06:28 GP73 python[402809]: Enforcer started. Press Ctrl+C to stop.
Aug 15 19:09:53 GP73 systemd[1]: Stopping bisk-enforcer.service - BISK Enforcer (APScheduler) - periodic enforce_schedules()...
Aug 15 19:09:53 GP73 systemd[1]: bisk-enforcer.service: Deactivated successfully.
Aug 15 19:09:53 GP73 systemd[1]: Stopped bisk-enforcer.service - BISK Enforcer (APScheduler) - periodic enforce_schedules().
Aug 15 19:10:12 GP73 systemd[1]: Started bisk-enforcer.service - BISK Enforcer (APScheduler) - periodic enforce_schedules().
Aug 15 19:10:13 GP73 python[405842]: Enforcer started. Press Ctrl+C to stop.

```

# *) Daily Operations
### Restart after code or env changes
sudo systemctl restart bisk-enforcer.service

### Stop / Start
sudo systemctl stop bisk-enforcer.service
sudo systemctl start bisk-enforcer.service

### Disable + stop (e.g., on dev)
sudo systemctl disable --now bisk-enforcer.service

### Status and logs
systemctl status bisk-enforcer.service
journalctl -u bisk-enforcer.service -f -n 200
