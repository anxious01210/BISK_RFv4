#!/usr/bin/env bash
set -euo pipefail

# -------- Dev-friendly defaults (override via flags) --------
APP_DIR="${APP_DIR:-/home/rio/PycharmProjects/BISK_RFv4}"
VENV_DIR="${VENV_DIR:-${APP_DIR}/.venv}"
DJANGO_SETTINGS="${DJANGO_SETTINGS:-bisk.settings}"
APP_USER="${APP_USER:-rio}"
APP_GROUP="${APP_GROUP:-rio}"
RUNNER_IMPL="${RUNNER_IMPL:-ffmpeg_all}"   # ffmpeg_all | ffmpeg_one
HB_URL="${HB_URL:-http://127.0.0.1:8000/api/runner/heartbeat/}"
HB_KEY="${HB_KEY:-dev-key-change-me}"
ENV_DIR="${ENV_DIR:-/etc/bisk}"
ENV_FILE="${ENV_FILE:-${ENV_DIR}/bisk.env}"
UNIT_DIR="/etc/systemd/system"
ENFORCER_UNIT="bisk-enforcer.service"
PRUNE_SVC_UNIT="bisk-prune-heartbeats.service"
PRUNE_TMR_UNIT="bisk-prune-heartbeats.timer"
LOCK_PATH="${LOCK_PATH:-/run/bisk/enforcer.lock}"   # single-instance lock

# -------- Parse flags (optional) --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --app-dir) APP_DIR="$2"; shift 2;;
    --venv) VENV_DIR="$2"; shift 2;;
    --user) APP_USER="$2"; shift 2;;
    --group) APP_GROUP="$2"; shift 2;;
    --settings) DJANGO_SETTINGS="$2"; shift 2;;
    --runner) RUNNER_IMPL="$2"; shift 2;;
    --hb-url) HB_URL="$2"; shift 2;;
    --hb-key) HB_KEY="$2"; shift 2;;
    -h|--help)
      echo "Usage: sudo bash $0 [--app-dir DIR] [--venv DIR] [--user USER] [--group GROUP] [--settings MOD] [--runner ffmpeg_all|ffmpeg_one] [--hb-url URL] [--hb-key KEY]"
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

PYTHON_BIN="${VENV_DIR}/bin/python"
MANAGE_PY="${APP_DIR}/manage.py"

# -------- Sanity checks --------
[[ -d "$APP_DIR" ]] || { echo "ERROR: App dir not found: $APP_DIR"; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
[[ -f "$MANAGE_PY"  ]] || { echo "ERROR: manage.py not found: $MANAGE_PY"; exit 1; }

# -------- Ensure user/group --------
getent group "$APP_GROUP" >/dev/null || groupadd --system "$APP_GROUP"
id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --gid "$APP_GROUP" --home "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"

# -------- Ensure /run/bisk via tmpfiles.d (like the old script) --------
TMPFILES_CONF="/etc/tmpfiles.d/bisk.conf"
echo "d /run/bisk 0755 ${APP_USER} ${APP_GROUP} - -" | sudo tee "$TMPFILES_CONF" >/dev/null
systemd-tmpfiles --create "$TMPFILES_CONF"

# -------- Env file --------
mkdir -p "$ENV_DIR"; chmod 0755 "$ENV_DIR"
cat > "$ENV_FILE" <<EOF
# Managed by install_enforcer.sh
DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS}
BISK_RUNNER_IMPL=${RUNNER_IMPL}
BISK_STRICT_BINARIES=1

# Heartbeat (matches settings.py)
BISK_HEARTBEAT_URL=${HB_URL}
BISK_HEARTBEAT_KEY=${HB_KEY}

# Prefer RTSP sub-stream when possible
BISK_PREFER_SUBSTREAM=1

# Single-instance lock path (used by apps.scheduler.services.lock)
ENFORCER_LOCK_FILE=${LOCK_PATH}

# Optional CPU thread caps to avoid BLAS storms
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
EOF
chmod 0640 "$ENV_FILE"; chown ${APP_USER}:${APP_GROUP} "$ENV_FILE"

# -------- systemd units --------
cat > "${UNIT_DIR}/${ENFORCER_UNIT}" <<EOF
[Unit]
Description=BISK Enforcer (APScheduler) - periodic enforce_schedules()
After=network.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${PYTHON_BIN} ${MANAGE_PY} run_enforcer
User=${APP_USER}
Group=${APP_GROUP}
UMask=007
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

cat > "${UNIT_DIR}/${PRUNE_SVC_UNIT}" <<EOF
[Unit]
Description=BISK prune old runner heartbeats (service)
[Service]
Type=oneshot
WorkingDirectory=${APP_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${PYTHON_BIN} ${MANAGE_PY} prune_heartbeats --days 3
User=${APP_USER}
Group=${APP_GROUP}
EOF

cat > "${UNIT_DIR}/${PRUNE_TMR_UNIT}" <<EOF
[Unit]
Description=BISK prune old runner heartbeats (timer)
[Timer]
OnCalendar=hourly
Persistent=true
[Install]
WantedBy=timers.target
EOF

# -------- Reload + enable --------
systemctl daemon-reload
systemctl enable --now "${ENFORCER_UNIT}"
systemctl enable --now "${PRUNE_TMR_UNIT}"

echo
echo "✔ Enforcer + prune timer installed and started."
echo "  Service: systemctl status ${ENFORCER_UNIT}"
echo "  Timer:   systemctl list-timers | grep bisk-prune-heartbeats"
echo
echo "Defaults used:"
echo "  APP_DIR=${APP_DIR}"
echo "  VENV_DIR=${VENV_DIR}"
echo "  USER:GROUP=${APP_USER}:${APP_GROUP}"
echo "  LOCK=${LOCK_PATH}"
echo
echo "Sanity:"
echo "  pgrep -af 'recognize_runner_all_ffmpeg.py|recognize_ffmpeg.py'  # during active periods"
