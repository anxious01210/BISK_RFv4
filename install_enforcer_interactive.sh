#!/usr/bin/env bash
set -euo pipefail

# --- styling helpers ---
bold() { printf "\033[1m%s\033[0m" "$*"; }
green() { printf "\033[32m%s\033[0m" "$*"; }
yellow() { printf "\033[33m%s\033[0m" "$*"; }
red() { printf "\033[31m%s\033[0m" "$*"; }

require_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    red "This script must be run as root (use sudo).\n" >&2
    exit 1
  fi
}

ask_yn() {
  # $1 question, $2 default (y/n)
  local q="$1"; local def="${2:-y}"; local prompt
  if [[ "$def" == "y" ]]; then prompt=" [Y/n] "; else prompt=" [y/N] "; fi
  while true; do
    read -r -p "$(bold "$q")$prompt" ans || true
    ans="${ans:-$def}"
    case "$ans" in
      y|Y) return 0;;
      n|N) return 1;;
      *) yellow "Please answer y or n.
";;
    esac
  done
}

ask_val() {
  # $1 prompt, $2 default
  local q="$1"; local def="${2:-}"; local ans
  read -r -p "$(bold "$q") [${def}] : " ans || true
  echo "${ans:-$def}"
}

show_header() {
  echo
  bold "BISK Enforcer Installer (interactive)"
  echo
}

# --- main ---
require_root
show_header

# 1) Detect APP_DIR (project root with manage.py)
DEFAULT_APP_DIR="$(pwd)"
if [[ ! -f "$DEFAULT_APP_DIR/manage.py" ]]; then
  # try parent if called from deploy/systemd
  if [[ -f "$(pwd)/../manage.py" ]]; then
    DEFAULT_APP_DIR="$(cd .. && pwd)"
  fi
fi

echo "We will locate your Django project (must contain manage.py)."
APP_DIR="$(ask_val "Project directory" "$DEFAULT_APP_DIR")"
while [[ ! -f "$APP_DIR/manage.py" ]]; do
  red "manage.py not found in: $APP_DIR\n"
  APP_DIR="$(ask_val "Enter a valid project directory" "$DEFAULT_APP_DIR")"
done
echo -e "  Using APP_DIR = $(green "$APP_DIR")"

# 2) Detect venv python
DEFAULT_VENV="$APP_DIR/.venv/bin/python"
if [[ ! -x "$DEFAULT_VENV" ]]; then
  # try alternate
  [[ -x "$APP_DIR/venv/bin/python" ]] && DEFAULT_VENV="$APP_DIR/venv/bin/python" || true
fi
PYTHON_BIN="$(ask_val "Path to Python in your virtualenv" "$DEFAULT_VENV")"
while [[ ! -x "$PYTHON_BIN" ]]; do
  red "Python not found or not executable: $PYTHON_BIN\n"
  PYTHON_BIN="$(ask_val "Path to Python in your virtualenv" "$DEFAULT_VENV")"
done
echo -e "  Using PYTHON_BIN = $(green "$PYTHON_BIN")"

# 3) Settings module
DJANGO_SETTINGS="$(ask_val "Django settings module" "bisk.settings")"
echo -e "  Using DJANGO_SETTINGS = $(green "$DJANGO_SETTINGS")"

# 4) Detect local .env (informational only)
ENV_CANDIDATES=()
[[ -f "$APP_DIR/.env" ]] && ENV_CANDIDATES+=("$APP_DIR/.env")
[[ -f "$APP_DIR/.env.local" ]] && ENV_CANDIDATES+=("$APP_DIR/.env.local")
[[ -f "$APP_DIR/.env.prod" ]] && ENV_CANDIDATES+=("$APP_DIR/.env.prod")
if [[ ${#ENV_CANDIDATES[@]} -gt 0 ]]; then
  echo
  echo "$(bold "Found possible app .env files (for web workers, not the service):")"
  for p in "${ENV_CANDIDATES[@]}"; do echo "  - $p"; done
  ask_yn "Show first .env now?" n && { echo "----"; sed -n '1,120p' "${ENV_CANDIDATES[0]}" || true; echo "----"; }
fi

# 5) Enforcer env values
HB_URL="$(ask_val "Heartbeat URL" "http://127.0.0.1:8000/api/runner/heartbeat/")"
read -r -s -p "$(bold "Heartbeat KEY") [dev-key-change-me] : " HB_KEY; echo
HB_KEY="${HB_KEY:-dev-key-change-me}"

RUNNER_IMPL="$(ask_val "Runner flavor (ffmpeg_all | ffmpeg_one)" "ffmpeg_all")"
PREFER_SUB="$(ask_yn "Prefer RTSP sub-stream (/102) when available?" y && echo 1 || echo 0)"

# 6) Lock file & tmpfiles
USE_LOCK="$(ask_yn "Use a POSIX lock file to ensure a single enforcer instance?" y && echo yes || echo no)"
LOCK_PATH="/run/bisk/enforcer.lock"
if [[ "$USE_LOCK" == "yes" ]]; then
  LOCK_PATH="$(ask_val "Lock file path" "$LOCK_PATH")"
fi

# 7) Service user/group
SUG_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"
SUG_GROUP="$(id -gn "$SUG_USER" 2>/dev/null || echo "$SUG_USER")"
APP_USER="$(ask_val "Service user" "$SUG_USER")"
APP_GROUP="$(ask_val "Service group" "$SUG_GROUP")"

# 8) Start order hardening
ADD_AFTER="$(ask_yn "Delay Enforcer until DB/Redis are up (After=postgresql,redis)?" y && echo yes || echo no)"

# 9) Prune timer
USE_PRUNE="$(ask_yn "Install hourly prune timer for old heartbeats?" y && echo yes || echo no)"

# 10) Thread caps
CAP_THREADS="$(ask_yn "Set BLAS/OMP thread caps to 1 (reduce CPU spikes)?" y && echo yes || echo no)"

# Summarize
echo
bold "Summary"
cat <<EOF
  APP_DIR            = $APP_DIR
  PYTHON_BIN         = $PYTHON_BIN
  DJANGO_SETTINGS    = $DJANGO_SETTINGS
  Runner flavor      = $RUNNER_IMPL
  HB URL             = $HB_URL
  HB KEY             = (hidden)
  Prefer sub-stream  = $([ "$PREFER_SUB" -eq 1 ] && echo yes || echo no)
  POSIX lock         = $USE_LOCK ($LOCK_PATH)
  Service user:group = ${APP_USER}:${APP_GROUP}
  After DB/Redis     = $([ "$ADD_AFTER" == "yes" ] && echo yes || echo no)
  Prune timer        = $([ "$USE_PRUNE" == "yes" ] && echo yes || echo no)
  Thread caps        = $([ "$CAP_THREADS" == "yes" ] && echo yes || echo no)
EOF
ask_yn "Proceed to install and start the service?" y || { yellow "Aborted.\n"; exit 0; }

# Paths
ENV_DIR="/etc/bisk"
ENV_FILE="$ENV_DIR/bisk.env"
UNIT_DIR="/etc/systemd/system"
ENFORCER_UNIT="bisk-enforcer.service"
PRUNE_SVC_UNIT="bisk-prune-heartbeats.service"
PRUNE_TMR_UNIT="bisk-prune-heartbeats.timer"

# Ensure user/group exist (system user creation optional; if user exists, this is a no-op)
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  yellow "User $APP_USER does not exist; creating a system user."
  useradd --system --no-create-home --shell /usr/sbin/nologin "$APP_USER"
fi
if ! getent group "$APP_GROUP" >/dev/null 2>&1; then
  yellow "Group $APP_GROUP does not exist; creating."
  groupadd --system "$APP_GROUP"
fi

# Ensure /run/bisk via tmpfiles if lock is used
if [[ "$USE_LOCK" == "yes" ]]; then
  TMPFILES_CONF="/etc/tmpfiles.d/bisk.conf"
  install -d -m 0755 -o "$APP_USER" -g "$APP_GROUP" /run/bisk || true
  echo "d /run/bisk 0755 $APP_USER $APP_GROUP - -" >/etc/tmpfiles.d/bisk.conf
  systemd-tmpfiles --create "$TMPFILES_CONF"
fi

# Write env
install -d -m 0755 "$ENV_DIR"
cat >"$ENV_FILE" <<EOF
# Managed by install_enforcer.sh (interactive)
DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS}
BISK_RUNNER_IMPL=${RUNNER_IMPL}
BISK_HEARTBEAT_URL=${HB_URL}
BISK_HEARTBEAT_KEY=${HB_KEY}
BISK_PREFER_SUBSTREAM=${PREFER_SUB}
$( [[ "$USE_LOCK" == "yes" ]] && echo "ENFORCER_LOCK_FILE=${LOCK_PATH}" )

# Safety: keep linear algebra libs from over-subscribing CPU
$( [[ "$CAP_THREADS" == "yes" ]] && echo "OMP_NUM_THREADS=1" )
$( [[ "$CAP_THREADS" == "yes" ]] && echo "OPENBLAS_NUM_THREADS=1" )
$( [[ "$CAP_THREADS" == "yes" ]] && echo "MKL_NUM_THREADS=1" )
EOF
chmod 0640 "$ENV_FILE"; chown "$APP_USER:$APP_GROUP" "$ENV_FILE"

# Create systemd units
AFTER_BLOCK="After=network-online.target"
WANTS_BLOCK="Wants=network-online.target"
if [[ "$ADD_AFTER" == "yes" ]]; then
  AFTER_BLOCK="${AFTER_BLOCK} postgresql.service redis-server.service"
fi

cat >"$UNIT_DIR/$ENFORCER_UNIT" <<EOF
[Unit]
Description=BISK Enforcer (APScheduler) - periodic enforce_schedules()
${AFTER_BLOCK}
${WANTS_BLOCK}

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${PYTHON_BIN} ${APP_DIR}/manage.py run_enforcer
User=${APP_USER}
Group=${APP_GROUP}
UMask=007
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

if [[ "$USE_PRUNE" == "yes" ]]; then
  cat >"$UNIT_DIR/$PRUNE_SVC_UNIT" <<EOF
[Unit]
Description=BISK prune old runner heartbeats (service)

[Service]
Type=oneshot
WorkingDirectory=${APP_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${PYTHON_BIN} ${APP_DIR}/manage.py prune_heartbeats --days 3
User=${APP_USER}
Group=${APP_GROUP}
EOF

  cat >"$UNIT_DIR/$PRUNE_TMR_UNIT" <<EOF
[Unit]
Description=BISK prune old runner heartbeats (timer)

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
EOF
else
  # If timer exists from a previous run, disable it
  systemctl disable --now "$PRUNE_TMR_UNIT" 2>/dev/null || true
  rm -f "$UNIT_DIR/$PRUNE_TMR_UNIT" "$UNIT_DIR/$PRUNE_SVC_UNIT" 2>/dev/null || true
fi

# Reload systemd and enable
systemctl daemon-reload
systemctl enable --now "$ENFORCER_UNIT"
if [[ "$USE_PRUNE" == "yes" ]]; then
  systemctl enable --now "$PRUNE_TMR_UNIT"
fi

echo
green "✔ Enforcer installed and started."
echo "  Service:  systemctl status ${ENFORCER_UNIT} --no-pager"
if [[ "$USE_PRUNE" == "yes" ]]; then
  echo "  Timer:    systemctl list-timers | grep bisk-prune-heartbeats"
fi
echo
echo "Env file written to: $(green "$ENV_FILE")"
echo "You can edit it and run: sudo systemctl restart ${ENFORCER_UNIT}"
echo
echo "Quick checks:"
echo "  journalctl -u ${ENFORCER_UNIT} -f"
echo "  pgrep -af 'recognize_runner_all_ffmpeg.py|recognize_ffmpeg.py'"
echo


# from your project root
#sudo bash ~/Downloads/install_enforcer_interactive.sh