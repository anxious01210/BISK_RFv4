#!/usr/bin/env bash
# BISK_RFv4 — Enforcer systemd installer (with shared lock under /run/bisk)
# - Keeps PrivateTmp=yes for the service (hardening)
# - Uses RuntimeDirectory=bisk so the lock path is visible system-wide
# - Ensures /run/bisk exists even when the service is stopped (tmpfiles.d)
#
# Usage:
#   sudo ./deploy/systemd/install_enforcer.sh
#   # Or override defaults:
#   sudo APP_DIR=/srv/BISK_RFv4 APP_USER=bisk APP_GROUP=bisk VENV_PY=/srv/BISK_RFv4/.venv/bin/python ./deploy/systemd/install_enforcer.sh
#
# After successful install:
#   systemctl status bisk-enforcer.service --no-pager
#   ls -l /run/bisk/enforcer.lock && sudo lsof /run/bisk/enforcer.lock
#
# Optional (recommended) in settings.py:
#   import os
#   ENFORCER_LOCK_FILE = os.getenv('ENFORCER_LOCK_FILE', '/run/bisk/enforcer.lock')
#
# Quick end-to-end test (copy/paste after install):
#   sudo systemctl restart bisk-enforcer.service
#   pid=$(systemctl show -p MainPID --value bisk-enforcer.service)
#   sudo tr '\0' '\n' </proc/$pid/environ | grep ENFORCER_LOCK_FILE
#   ls -l /run/bisk/enforcer.lock && sudo lsof /run/bisk/enforcer.lock
#   python manage.py run_enforcer; echo "exit=$?"   # expect exit=1 (no-op) while service runs
#   sudo systemctl stop bisk-enforcer.service
#   ls -l /run/bisk/enforcer.lock || echo "no lock (expected)"
#   python manage.py run_enforcer &
#   sleep 1
#   ls -l /run/bisk/enforcer.lock && sudo lsof /run/bisk/enforcer.lock
#   pkill -f 'manage.py run_enforcer' || true
#   sudo systemctl start bisk-enforcer.service
#
set -euo pipefail

### --- CONFIG (override via env if needed) ---
APP_DIR="${APP_DIR:-/home/rio/PycharmProjects/BISK_RFv4}"
APP_USER="${APP_USER:-rio}"
APP_GROUP="${APP_GROUP:-rio}"
VENV_PY="${VENV_PY:-$APP_DIR/.venv/bin/python}"      # absolute path to venv python
UNIT_NAME="${UNIT_NAME:-bisk-enforcer.service}"
ENV_DIR="${ENV_DIR:-/etc/bisk}"
ENV_FILE="${ENV_FILE:-$ENV_DIR/bisk-enforcer.env}"
LOCK_PATH="${LOCK_PATH:-/run/bisk/enforcer.lock}"     # shared lock path (visible with PrivateTmp)
DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-bisk.settings}"
TMPFILES_CONF="${TMPFILES_CONF:-/etc/tmpfiles.d/bisk.conf}"
### --------------------------------------------

log() { printf "%s\n" "$*" ; }
ok()  { printf "\033[32m%s\033[0m\n" "$*" ; }
warn(){ printf "\033[33m%s\033[0m\n" "$*" ; }
err() { printf "\033[31m%s\033[0m\n" "$*" ; }

log "[1/7] Validating paths…"
[[ -x "$VENV_PY" ]] || { err "ERROR: VENV_PY not executable: $VENV_PY" ; exit 1; }
[[ -d "$APP_DIR" ]] || { err "ERROR: APP_DIR not found: $APP_DIR" ; exit 1; }

log "[2/7] Writing environment: $ENV_FILE"
install -d -m 0755 "$ENV_DIR"
cat >"$ENV_FILE" <<EOF
# Environment for $UNIT_NAME
DJANGO_SETTINGS_MODULE=$DJANGO_SETTINGS_MODULE
# Ensure service (and CLI) use the same shared lock path:
ENFORCER_LOCK_FILE=$LOCK_PATH
# Add more app-specific env here (e.g., TZ=Asia/Baghdad)
EOF
chmod 0644 "$ENV_FILE"
ok "Wrote $ENV_FILE"

log "[3/7] Ensuring /run/bisk exists via tmpfiles.d"
# This makes /run/bisk exist at boot AND while the service is stopped,
# so CLI runs can also create/lock $LOCK_PATH without sudo.
echo "d /run/bisk 0755 $APP_USER $APP_GROUP - -" > "$TMPFILES_CONF"
systemd-tmpfiles --create "$TMPFILES_CONF"
ok "tmpfiles rule installed at $TMPFILES_CONF"

log "[4/7] Installing systemd unit: /etc/systemd/system/$UNIT_NAME"
cat >/etc/systemd/system/$UNIT_NAME <<UNIT
[Unit]
Description=BISK Enforcer (APScheduler) - periodic enforce_schedules()
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_GROUP
WorkingDirectory=$APP_DIR

# Common env (includes ENFORCER_LOCK_FILE)
EnvironmentFile=$ENV_FILE

# Keep hardening, but use a shared runtime dir for the lock
PrivateTmp=yes
RuntimeDirectory=bisk
RuntimeDirectoryMode=0755
Environment=ENFORCER_LOCK_FILE=$LOCK_PATH

# Start the enforcer (idempotent; no-ops if lock is held)
ExecStart=$VENV_PY manage.py run_enforcer

Restart=always
RestartSec=5
TimeoutStopSec=15
KillMode=control-group

# Reasonable hardening
NoNewPrivileges=yes
ProtectSystem=full
ProtectHome=no

[Install]
WantedBy=multi-user.target
UNIT
ok "Unit installed"

log "[5/7] Reloading systemd & enabling service…"
systemctl daemon-reload
systemctl enable --now "$UNIT_NAME"
ok "Service enabled and started"

log "[6/7] Verifying environment & lock…"
pid="$(systemctl show -p MainPID --value "$UNIT_NAME" || true)"
if [[ -n "${pid:-}" && "$pid" != "0" ]]; then
  tr '\0' '\n' </proc/"$pid"/environ | grep -E '^ENFORCER_LOCK_FILE=' || true
else
  warn "Service PID not found yet."
fi

if [[ -e "$LOCK_PATH" ]]; then
  ls -l "$LOCK_PATH" || true
  command -v lsof >/dev/null 2>&1 && lsof "$LOCK_PATH" || warn "lsof not installed; skipping holder check"
else
  warn "Note: $LOCK_PATH not visible yet. It will appear once the enforcer acquires the lock."
fi

log "[7/7] Done."
cat <<'HELP'

Next steps & quick tests
------------------------
1) Check service health:
   systemctl status bisk-enforcer.service --no-pager

2) Verify the shared lock:
   ls -l /run/bisk/enforcer.lock && sudo lsof /run/bisk/enforcer.lock

3) Confirm the same lock path in the service env:
   pid=$(systemctl show -p MainPID --value bisk-enforcer.service)
   sudo tr '\0' '\n' </proc/$pid/environ | grep ENFORCER_LOCK_FILE

4) CLI should no-op while service holds the lock (exit=1):
   python manage.py run_enforcer; echo "exit=$?"

5) Stop the service -> lock disappears:
   sudo systemctl stop bisk-enforcer.service
   ls -l /run/bisk/enforcer.lock || echo "no lock (expected)"

6) Start from CLI -> acquires lock (dir exists thanks to tmpfiles.d):
   python manage.py run_enforcer &
   sleep 1
   ls -l /run/bisk/enforcer.lock && sudo lsof /run/bisk/enforcer.lock

7) Clean up:
   pkill -f 'manage.py run_enforcer' || true
   sudo systemctl start bisk-enforcer.service

Troubleshooting
---------------
- If 'Permission denied: /run/bisk': run 'systemd-tmpfiles --create /etc/tmpfiles.d/bisk.conf' and try again.
- If the file is missing but service is active: the lock may live in a PrivateTmp namespace (older unit). Re-run this installer and restart.
- Ensure your Django 'settings.py' includes:
    ENFORCER_LOCK_FILE = os.getenv('ENFORCER_LOCK_FILE', '/run/bisk/enforcer.lock')
  so CLI and service share the same path.

HELP
