# BISK Enforcer & Runner — Installation and Operations Guide (Updated)

This guide explains how to install, operate, and troubleshoot the **BISK Enforcer** service that automatically starts/stops camera recognition runners during active **PeriodOccurrences**. It also covers switching between the **All‑FFmpeg** runner and the legacy **single‑runner** implementation, and how to prune old heartbeats.

> **Update:** Environment variable names now align with `settings.py`:
> - `BISK_HEARTBEAT_URL`
> - `BISK_HEARTBEAT_KEY`
> (Previously shown `BISK_RUNNER_HB_URL/_KEY` are deprecated.)

---

## 1) What the installer does

The `install_enforcer.sh` script is an idempotent, production‑ready setup tool that:

- Creates a **system user/group** (defaults: `bisk:bisk`) if they do not already exist.
- Creates an environment file at **`/etc/bisk/bisk.env`** to hold runtime configuration (settings module, runner flavor, heartbeat URL/key, etc.).
- Installs **systemd** units:
  - `bisk-enforcer.service` — runs `manage.py run_enforcer` (APScheduler + Enforcer tick loop).
  - `bisk-prune-heartbeats.service` & `bisk-prune-heartbeats.timer` — prunes old `RunnerHeartbeat` rows hourly.
- Enables and **starts** the enforcer and prune timer.
- Leaves you with a single place (`/etc/bisk/bisk.env`) to switch runner flavor and tweak runtime variables.

> **Parallel runners per camera?** Yes. During active periods, the Enforcer launches **one runner process per allowed Camera/StreamProfile**. Each runner reports via the heartbeat API and is tracked in `RunningProcess` + `RunnerHeartbeat`.

---

## 2) Requirements

- Ubuntu/Debian with **systemd**.
- BISK app installed at some path (e.g. `/srv/bisk/app`), with a working **virtualenv** (e.g. `/srv/bisk/venv`).
- Django settings module (e.g. `bisk.settings`).
- Access to your BISK **heartbeat API** (e.g. `http://127.0.0.1:8000/api/runner/heartbeat/`).

---

## 3) Quick install

Run the installer as root (or with sudo). The example below uses typical production paths:

```bash
sudo bash install_enforcer.sh \
  --app-dir /srv/bisk/app \
  --venv /srv/bisk/venv \
  --user bisk --group bisk \
  --settings bisk.settings \
  --runner ffmpeg_all \
  --hb-url "http://127.0.0.1:8000/api/runner/heartbeat/" \
  --hb-key "dev-key-change-me"
```

**What username should I use?**  
- By default the script uses `--user bisk --group bisk`.  
- If the `bisk` user/group do **not** exist, the installer **creates** them as system accounts.  
- You **can** override these flags (e.g., `--user mysvc --group mysvc`). The script will create them for you if missing.

> You do **not** need to pre‑create the `bisk` user/group unless your security policy requires you to manage system users manually. The installer will handle it.

---

## 4) What gets installed

- **Environment file:** `/etc/bisk/bisk.env`  
  Example contents:
  ```ini
  DJANGO_SETTINGS_MODULE=bisk.settings
  BISK_RUNNER_IMPL=ffmpeg_all        # ffmpeg_all | ffmpeg_one
  BISK_STRICT_BINARIES=1

  # Heartbeat envs consumed by settings.py
  BISK_HEARTBEAT_URL=http://127.0.0.1:8000/api/runner/heartbeat/
  BISK_HEARTBEAT_KEY=dev-key-change-me

  # Optional overrides:
  # BISK_SNAPSHOT_DIR=
  # BISK_GPU_INDEX=
  # BISK_FFPROBE_BIN=
  # BISK_FFMPEG_BIN=
  ```

- **systemd units:**
  - `/etc/systemd/system/bisk-enforcer.service`  
  - `/etc/systemd/system/bisk-prune-heartbeats.service`  
  - `/etc/systemd/system/bisk-prune-heartbeats.timer`

---

## 5) Daily operations

### Start / Stop / Status
```bash
# Enforcer service
sudo systemctl start  bisk-enforcer.service
sudo systemctl stop   bisk-enforcer.service
sudo systemctl status bisk-enforcer.service

# Heartbeat prune (timer + on-demand service)
sudo systemctl start  bisk-prune-heartbeats.timer
sudo systemctl status bisk-prune-heartbeats.timer
sudo systemctl start  bisk-prune-heartbeats.service   # run prune now
```

### Enable at boot
```bash
sudo systemctl enable bisk-enforcer.service
sudo systemctl enable bisk-prune-heartbeats.timer
```

### Logs
```bash
# Live logs
journalctl -u bisk-enforcer.service -f
journalctl -u bisk-prune-heartbeats.service -f

# Historical (today)
journalctl -u bisk-enforcer.service --since today
```

---

## 6) Switching runner flavor

You can switch between the **All‑FFmpeg** runner and the legacy **single‑runner** without changing code:

1. Edit `/etc/bisk/bisk.env` and set:
   ```ini
   BISK_RUNNER_IMPL=ffmpeg_all   # or ffmpeg_one
   ```
2. Restart the enforcer:
   ```bash
   sudo systemctl restart bisk-enforcer.service
   ```
3. Verify:
   ```bash
   # should list processes started during active periods
   pgrep -af 'recognize_runner_all_ffmpeg.py|recognize_ffmpeg.py'
   ```

> The Admin UI and Enforcer logic will continue to track processes in `RunningProcess` and heartbeats in `RunnerHeartbeat` regardless of flavor.

---

## 7) Verification checklist

- **Admin → Enforcer/Runner dashboard** shows `RunningProcess` rows during active periods.
- Heartbeats populate in **RunnerHeartbeats**.
- Snapshots appear under your configured `SNAPSHOT_DIR` (usually `<MEDIA_ROOT>/snapshots`).
- System journal shows enforcer ticks and scheduler runs:
  ```bash
  journalctl -u bisk-enforcer.service -f
  ```

---

## 8) Troubleshooting

### Enforcer runs but no runners start
- Verify you **have active PeriodOccurrences** for the current time (timezone!).
- Check camera/profile **is_active** flags.
- Confirm heartbeat URL/key in `/etc/bisk/bisk.env` are correct and reachable.
- Tail enforcer logs:
  ```bash
  journalctl -u bisk-enforcer.service -f
  ```

### Runners start but don’t update status
- Check the **heartbeat endpoint** is reachable from the runner host:
  ```bash
  curl -i "$BISK_HEARTBEAT_URL"
  ```
- Review **Django server logs** for `/api/runner/heartbeat/` 5xx errors.
- Ensure `BISK_HEARTBEAT_KEY` in `/etc/bisk/bisk.env` matches your Django settings / expected key.

### Killing stuck runners
- Use the management command:
  ```bash
  source /srv/bisk/venv/bin/activate
  python /srv/bisk/app/manage.py stop_runners
  ```
- Confirm nothing is left:
  ```bash
  pgrep -af 'recognize_runner_all_ffmpeg.py|recognize_ffmpeg.py|ffmpeg' || echo "All clear"
  ```

### Snapshots not updating
- Ensure `SNAPSHOT_DIR` is writable by the service user.
- Confirm the runner was launched with `--snapshots` pointing to that directory.

### GPU saturation or contention
- Lower `fps` in your **StreamProfile** or adjust `det_set` for lighter detection.
- If multiple GPUs exist, pin profiles using `--gpu_index` (the Enforcer can pass this through).

### HTMX dashboard doesn’t auto-refresh fast enough
- Check your dashboard’s refresh settings and browser console.
- Server‑side: ensure the enforcer tick is running (logs), and heartbeats are landing.

---

## 9) Uninstall / Disable

```bash
sudo systemctl disable --now bisk-enforcer.service
sudo systemctl disable --now bisk-prune-heartbeats.timer

# Optional: remove units and env
sudo rm -f /etc/systemd/system/bisk-enforcer.service
sudo rm -f /etc/systemd/system/bisk-prune-heartbeats.service
sudo rm -f /etc/systemd/system/bisk-prune-heartbeats.timer
sudo rm -rf /etc/bisk
sudo systemctl daemon-reload
```

---

## 10) FAQ

**Q: Do I have to name the user `bisk`?**  
**A:** No. Pass `--user` and `--group` to the installer (e.g., `--user mysvc --group mysvc`). The script will create them for you if missing.

**Q: Is recognition parallel for multiple cameras?**  
**A:** Yes. The Enforcer launches one process per allowed camera/profile during active periods. Each runner is independent and reports via its own heartbeat.

**Q: Can I switch runner flavor without redeploying code?**  
**A:** Yes. Edit `/etc/bisk/bisk.env` and restart the enforcer.
