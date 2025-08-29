#!/usr/bin/env bash
set -euo pipefail

#sudo bash deploy/systemd/install_enforcer.sh \
 #  --app-dir /home/rio/PycharmProjects/BISK_RFv4 \
 #  --venv /home/rio/PycharmProjects/BISK_RFv4/.venv \
 #  --user rio --group rio \
 #  --settings bisk.settings \
 #  --runner ffmpeg_all \
 #  --hb-url "http://127.0.0.1:8000/api/runner/heartbeat/" \
 #  --hb-key "dev-key-change-me"

# -------- Defaults (override via flags) --------
APP_DIR="/srv/bisk/app"
VENV_DIR="/srv/bisk/venv"
DJANGO_SETTINGS="bisk.settings"
APP_USER="bisk"
APP_GROUP="bisk"
RUNNER_IMPL="ffmpeg_all"      # ffmpeg_all | ffmpeg_one
RUNNER_HEARTBEAT_URL="http://127.0.0.1:8000/api/runner/heartbeat/"
RUNNER_HEARTBEAT_KEY="dev-key-change-me"
PYTHON_BIN=""
MANAGE_PY=""
ENV_DIR="/etc/bisk"
ENV_FILE="${ENV_DIR}/bisk.env"
UNIT_DIR="/etc/systemd/system"
ENFORCER_UNIT="bisk-enforcer.service"
PRUNE_SVC_UNIT="bisk-prune-heartbeats.service"
PRUNE_TMR_UNIT="bisk-prune-heartbeats.timer"

usage() {
  cat <<EOF
Usage:
  sudo bash $0 [--app-dir /srv/bisk/app] [--venv /srv/bisk/venv]
               [--user bisk] [--group bisk]
               [--settings bisk.settings]
               [--runner ffmpeg_all|ffmpeg_one]
               [--hb-url URL] [--hb-key KEY]

This will:
  • Create ${APP_USER}:${APP_GROUP} (if missing)
  • Create ${ENV_FILE} with runtime env (runner flavor, HB, settings)
  • Install systemd units: ${ENFORCER_UNIT}, ${PRUNE_SVC_UNIT}, ${PRUNE_TMR_UNIT}
  • Enable & start enforcer and prune timer

Example:
  sudo bash $0 \\
    --app-dir /srv/bisk/app --venv /srv/bisk/venv \\
    --user bisk --group bisk \\
    --settings bisk.settings \\
    --runner ffmpeg_all \\
    --hb-url "http://127.0.0.1:8000/api/runner/heartbeat/" \\
    --hb-key "dev-key-change-me"
EOF
}

# -------- Parse flags --------
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
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# -------- Sanity checks --------
if [[ ! -d "$APP_DIR" ]]; then
  echo "ERROR: App dir not found: $APP_DIR"
  exit 1
fi
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: Venv dir not found: $VENV_DIR"
  exit 1
fi
PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found at $PYTHON_BIN"
  exit 1
fi
MANAGE_PY="${APP_DIR}/manage.py"
if [[ ! -f "$MANAGE_PY" ]]; then
  echo "ERROR: manage.py not found at $MANAGE_PY"
  exit 1
fi

# -------- Create system user/group --------
if ! getent group "$APP_GROUP" >/dev/null; then
  groupadd --system "$APP_GROUP"
fi
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  useradd --system --gid "$APP_GROUP" --home "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
fi

# -------- Ensure /etc/bisk and env file --------
mkdir -p "$ENV_DIR"
chmod 0755 "$ENV_DIR"

cat > "$ENV_FILE" <<EOF
# Managed by install_enforcer.sh
DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS}
BISK_RUNNER_IMPL=${RUNNER_IMPL}
BISK_STRICT_BINARIES=1

# Heartbeat
BISK_RUNNER_HB_URL=${HB_URL}
BISK_RUNNER_HB_KEY=${HB_KEY}

# (Optional) override these if you need:
# BISK_SNAPSHOT_DIR=
# BISK_GPU_INDEX=
# BISK_FFPROBE_BIN=
# BISK_FFMPEG_BIN=
EOF

chmod 0640 "$ENV_FILE"
chown ${APP_USER}:${APP_GROUP} "$ENV_FILE"

# -------- Systemd units --------
# Enforcer service: runs "manage.py run_enforcer"
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

# Prune heartbeats service + timer
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

# -------- Reload + enable + start --------
systemctl daemon-reload
systemctl enable "${ENFORCER_UNIT}"
systemctl restart "${ENFORCER_UNIT}"

systemctl enable "${PRUNE_TMR_UNIT}"
systemctl restart "${PRUNE_TMR_UNIT}"

echo
echo "✔ Enforcer + prune timer installed and started."
echo "  - Service: systemctl status ${ENFORCER_UNIT}"
echo "  - Timer:   systemctl list-timers | grep bisk-prune-heartbeats"
echo
echo "Current runner flavor: ${RUNNER_IMPL}"
echo "To switch later: edit ${ENV_FILE} (BISK_RUNNER_IMPL) + 'systemctl restart ${ENFORCER_UNIT}'"
echo
echo "Sanity checks:"
echo "  pgrep -af 'recognize_runner_all_ffmpeg.py|recognize_ffmpeg.py'  # should show runners during active periods"
