#!/usr/bin/env bash
## universal_web_app_installer.sh 
# chmod +x universal_web_app_installer.sh
# sudo ./universal_web_app_installer.sh --dry-run
# sudo ./universal_web_app_installer.sh
#
# Universal Django/Flask Web App Installer (Option C-style, with Redis & GPU detect)
#
# Features:
# - Works for Django or Flask projects
# - Interactive dev/prod/both setup
# - Auto create /srv/<project-name> if needed
# - Optional auto-clone from Git repo
# - Single Python venv per project (GPU libs etc. installed separately)
# - DB selection: SQLite / PostgreSQL / MySQL
# - Installs OS-level deps: Redis, ffmpeg, build tools, etc.
# - GPU section:
#     - Detect NVIDIA GPU (lspci)
#     - If found: recommend driver + CUDA, optional install
#     - If not found: recommend CPU-only stack
# - Python deps: after clone, detect requirements*.txt and let user choose
# - Systemd units:
#     - Gunicorn service (per env, configurable port)
#     - Optional Enforcer / Heartbeat / APScheduler (for Django-style commands)
# - Optional Nginx reverse proxy per env (Cloudflare-friendly)
# - Dry run mode: ./universal_web_app_installer.sh --dry-run
#
# NOTE:
#   - You can safely re-run this script; it will skip or overwrite as needed.
#   - For BISK_RFv4, you can use:
#       Project name:     BISK_RFv4
#       Project root:     /srv/BISK_RFv4
#       Project type:     Django
#       Django package:   bisk
#       Settings module:  bisk.settings
#       Git repo:         https://github.com/anxious01210/BISK_RFv4.git
#       Enforcer:         bisk_enforcer
#       Heartbeat:        bisk_heartbeat
#       APScheduler:      runapscheduler (prod only, usually)

set -euo pipefail

########################################
# 1. Helpers
########################################

DRY_RUN=0

if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
  echo ">>> DRY RUN MODE ENABLED (no changes will be applied)"
fi

run_cmd() {
  local cmd="$1"
  echo ">> $cmd"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    # shellcheck disable=SC2086
    eval $cmd
  fi
}

write_file() {
  local path="$1"
  local content="$2"

  echo ">> Writing file: $path"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$(dirname "$path")"
    printf "%s\n" "$content" > "$path"
  fi
}

ensure_root() {
  if [[ "$EUID" -ne 0 ]]; then
    echo "This script must be run as root (use: sudo $0 [--dry-run])"
    exit 1
  fi
}

is_dir_empty() {
  local d="$1"
  if [ -d "$d" ]; then
    [ -z "$(ls -A "$d")" ]
  else
    return 0
  fi
}

########################################
# 2. Basic info & generic defaults
########################################

ensure_root

DEFAULT_PROJECT_NAME="BISK_RFv4"
read -rp "Project name [${DEFAULT_PROJECT_NAME}]: " PROJECT_NAME
PROJECT_NAME="${PROJECT_NAME:-$DEFAULT_PROJECT_NAME}"

DEFAULT_PROJECT_ROOT="/srv/${PROJECT_NAME}"
read -rp "Project root directory [${DEFAULT_PROJECT_ROOT}]: " PROJECT_ROOT
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"

DEFAULT_GIT_REPO_URL="https://github.com/anxious01210/BISK_RFv4.git"
read -rp "Git repo URL (leave blank to skip auto-clone) [${DEFAULT_GIT_REPO_URL}]: " GIT_REPO_URL
GIT_REPO_URL="${GIT_REPO_URL:-$DEFAULT_GIT_REPO_URL}"

DEFAULT_APP_USER="${SUDO_USER:-rio}"
read -rp "Linux user that will run the app (non-root) [${DEFAULT_APP_USER}]: " APP_USER
APP_USER="${APP_USER:-$DEFAULT_APP_USER}"

DEFAULT_PYTHON_BIN="python3"
read -rp "Python binary for venv [${DEFAULT_PYTHON_BIN}]: " PYTHON_BIN
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"

echo
echo "Application type:"
echo "  1) Django"
echo "  2) Flask"
read -rp "Choose app type [1/2] (default 1): " APP_TYPE_CHOICE
APP_TYPE_CHOICE="${APP_TYPE_CHOICE:-1}"

APP_TYPE="django"
if [[ "$APP_TYPE_CHOICE" == "2" ]]; then
  APP_TYPE="flask"
fi

DJANGO_PROJECT_NAME=""
DJANGO_SETTINGS_MODULE=""
MANAGE_PY_PATH=""
FLASK_GUNICORN_APP=""

if [[ "$APP_TYPE" == "django" ]]; then
  DEFAULT_DJANGO_PROJECT_NAME="bisk"
  read -rp "Django project package name [${DEFAULT_DJANGO_PROJECT_NAME}]: " DJANGO_PROJECT_NAME
  DJANGO_PROJECT_NAME="${DJANGO_PROJECT_NAME:-$DEFAULT_DJANGO_PROJECT_NAME}"

  DEFAULT_DJANGO_SETTINGS_MODULE="${DJANGO_PROJECT_NAME}.settings"
  read -rp "DJANGO_SETTINGS_MODULE [${DEFAULT_DJANGO_SETTINGS_MODULE}]: " DJANGO_SETTINGS_MODULE
  DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-$DEFAULT_DJANGO_SETTINGS_MODULE}"

  DEFAULT_MANAGE_PATH="${PROJECT_ROOT}/manage.py"
  read -rp "Path to manage.py [${DEFAULT_MANAGE_PATH}]: " MANAGE_PY_PATH
  MANAGE_PY_PATH="${MANAGE_PY_PATH:-$DEFAULT_MANAGE_PATH}"
else
  echo
  echo "For Flask + Gunicorn, you need an import path like 'app:app' or 'myapp.wsgi:app'"
  DEFAULT_FLASK_APP="app:app"
  read -rp "Gunicorn app import (e.g. app:app) [${DEFAULT_FLASK_APP}]: " FLASK_GUNICORN_APP
  FLASK_GUNICORN_APP="${FLASK_GUNICORN_APP:-$DEFAULT_FLASK_APP}"
fi

echo
echo "Environment type:"
echo "  1) dev"
echo "  2) prod"
echo "  3) both"
read -rp "Choose environment type [1/2/3] (default 1): " ENV_CHOICE
ENV_CHOICE="${ENV_CHOICE:-1}"

APP_ENVS=()
case "$ENV_CHOICE" in
  1) APP_ENVS=("dev");;
  2) APP_ENVS=("prod");;
  3) APP_ENVS=("dev" "prod");;
  *) echo "Invalid choice, defaulting to dev"; APP_ENVS=("dev");;
esac

########################################
# 3. Database selection
########################################

echo
echo "Database choice:"
echo "  1) SQLite3 (simple, dev only or small installs)"
echo "  2) PostgreSQL (recommended for prod)"
echo "  3) MySQL/MariaDB"
read -rp "Choose database [1/2/3] (default 2): " DB_CHOICE
DB_CHOICE="${DB_CHOICE:-2}"

DB_TYPE=""
case "$DB_CHOICE" in
  1) DB_TYPE="sqlite";;
  2) DB_TYPE="postgres";;
  3) DB_TYPE="mysql";;
  *) DB_TYPE="postgres";;
esac

DB_NAME_DEFAULT="${PROJECT_NAME,,}_db"
DB_USER_DEFAULT="${PROJECT_NAME,,}_user"
DB_PASS_DEFAULT="${PROJECT_NAME,,}_pass"

if [[ "$DB_TYPE" != "sqlite" ]]; then
  echo
  echo "Database settings (for $DB_TYPE):"
  read -rp "Database name [${DB_NAME_DEFAULT}]: " DB_NAME
  DB_NAME="${DB_NAME:-$DB_NAME_DEFAULT}"

  read -rp "Database user [${DB_USER_DEFAULT}]: " DB_USER
  DB_USER="${DB_USER:-$DB_USER_DEFAULT}"

  read -rp "Database password [${DB_PASS_DEFAULT}]: " DB_PASS
  DB_PASS="${DB_PASS:-$DB_PASS_DEFAULT}"
fi

########################################
# 4. Optional management commands (Django-style)
########################################

echo
echo "Optional management commands (mostly for Django projects)."
echo "Leave blank to skip creating that service."

read -rp "Enforcer command name (manage.py <cmd>, e.g. 'bisk_enforcer') []: " ENFORCER_CMD
read -rp "Heartbeat command name (manage.py <cmd>, e.g. 'bisk_heartbeat') []: " HEARTBEAT_CMD
read -rp "APScheduler command name (manage.py <cmd>, e.g. 'runapscheduler') []: " APSCHEDULER_CMD

ENABLE_APSCHEDULER=0
if [[ -n "$APSCHEDULER_CMD" ]]; then
  echo
  echo "APScheduler service (manage.py ${APSCHEDULER_CMD}):"
  echo "  1) Enable"
  echo "  2) Disable"
  read -rp "Choose [1/2] (default 2): " APS_CHOICE
  APS_CHOICE="${APS_CHOICE:-2}"
  if [[ "$APS_CHOICE" == "1" ]]; then
    ENABLE_APSCHEDULER=1
  fi
fi

########################################
# 5. Heartbeat interval
########################################

HB_INTERVAL_MIN_DEFAULT=1
echo
read -rp "Heartbeat interval in minutes (if heartbeat command used) [${HB_INTERVAL_MIN_DEFAULT}]: " HB_INTERVAL_MIN
HB_INTERVAL_MIN="${HB_INTERVAL_MIN:-$HB_INTERVAL_MIN_DEFAULT}"

if ! [[ "$HB_INTERVAL_MIN" =~ ^[0-9]+$ ]]; then
  echo "Invalid interval, defaulting to ${HB_INTERVAL_MIN_DEFAULT} min."
  HB_INTERVAL_MIN="$HB_INTERVAL_MIN_DEFAULT"
fi

HEARTBEAT_INTERVAL_SEC=$((HB_INTERVAL_MIN * 60))

########################################
# 6. Nginx option (global)
########################################

echo
echo "Nginx reverse proxy:"
echo "  This will install Nginx (if needed) and optionally create per-env site configs."
read -rp "Install Nginx web server? [Y/n]: " INSTALL_NGINX
INSTALL_NGINX="${INSTALL_NGINX:-Y}"

INSTALL_NGINX=$(echo "$INSTALL_NGINX" | tr '[:upper:]' '[:lower:]')
ENABLE_NGINX=0
if [[ "$INSTALL_NGINX" == "y" || "$INSTALL_NGINX" == "yes" ]]; then
  ENABLE_NGINX=1
fi

########################################
# 7. System packages (including Redis, ffmpeg, build libs)
########################################

echo
echo "=== Installing system packages (Python, DB libs, build tools, git, Redis, ffmpeg) ==="

run_cmd "apt-get update"

# Basic build tools & Python utils + git
run_cmd "apt-get install -y build-essential git curl"
run_cmd "apt-get install -y ${PYTHON_BIN}-venv ${PYTHON_BIN}-dev python3-pip pkg-config libssl-dev libffi-dev libjpeg-dev zlib1g-dev"

# DB-specific packages
if [[ "$DB_TYPE" == "postgres" ]]; then
  run_cmd "apt-get install -y postgresql postgresql-contrib libpq-dev"
elif [[ "$DB_TYPE" == "mysql" ]]; then
  run_cmd "apt-get install -y mysql-server libmysqlclient-dev"
else
  echo "SQLite selected – no DB server install needed."
fi

# Redis packages
echo
echo "=== Installing and enabling Redis ==="
run_cmd "apt-get install -y redis-server redis-tools"
if [[ "$DRY_RUN" -eq 0 ]]; then
  systemctl enable redis-server
  systemctl restart redis-server || systemctl start redis-server
  echo ">> Testing Redis with: redis-cli ping"
  if redis-cli ping >/dev/null 2>&1; then
    echo "Redis is up (PING OK)."
  else
    echo "WARNING: redis-cli ping failed. Check 'sudo systemctl status redis-server'."
  fi
else
  echo ">> [DRY RUN] Would enable and start redis-server, then run redis-cli ping."
fi

# ffmpeg for media/streaming
echo
echo "=== Installing ffmpeg for any media/streaming scripts ==="
run_cmd "apt-get install -y ffmpeg"

# Nginx
if [[ "$ENABLE_NGINX" -eq 1 ]]; then
  run_cmd "apt-get install -y nginx"
fi

########################################
# 8. GPU / NVIDIA detection & recommendations
########################################

echo
echo "=== GPU / NVIDIA detection & recommendations ==="
GPU_PRESENT=0
if lspci | grep -i nvidia >/dev/null 2>&1; then
  GPU_PRESENT=1
  echo "NVIDIA GPU detected:"
  lspci | grep -i nvidia || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "nvidia-smi is present. Current GPU driver info:"
    nvidia-smi || true
  else
    echo
    echo "nvidia-smi not found – NVIDIA driver is likely NOT installed."
    DEFAULT_NVIDIA_DRIVER_PKG="nvidia-driver-550"
    read -rp "Install an NVIDIA driver package now? (e.g. nvidia-driver-550) [y/N]: " INSTALL_DRIVER
    INSTALL_DRIVER="${INSTALL_DRIVER:-N}"
    INSTALL_DRIVER=$(echo "$INSTALL_DRIVER" | tr '[:upper:]' '[:lower:]')
    if [[ "$INSTALL_DRIVER" == "y" || "$INSTALL_DRIVER" == "yes" ]]; then
      read -rp "Driver package name [${DEFAULT_NVIDIA_DRIVER_PKG}]: " NVIDIA_DRIVER_PKG
      NVIDIA_DRIVER_PKG="${NVIDIA_DRIVER_PKG:-$DEFAULT_NVIDIA_DRIVER_PKG}"
      run_cmd "apt-get install -y ${NVIDIA_DRIVER_PKG}"
      echo "NOTE: A reboot is usually required after installing NVIDIA drivers."
    else
      echo "Skipping NVIDIA driver install. You can manually install later if needed."
    fi
  fi

  echo
  DEFAULT_CUDA_TOOLKIT_PKG="nvidia-cuda-toolkit"
  read -rp "Install CUDA toolkit from Ubuntu repo? [y/N]: " INSTALL_CUDA
  INSTALL_CUDA="${INSTALL_CUDA:-N}"
  INSTALL_CUDA=$(echo "$INSTALL_CUDA" | tr '[:upper:]' '[:lower:]')
  if [[ "$INSTALL_CUDA" == "y" || "$INSTALL_CUDA" == "yes" ]]; then
    read -rp "CUDA toolkit package name [${DEFAULT_CUDA_TOOLKIT_PKG}]: " CUDA_PKG
    CUDA_PKG="${CUDA_PKG:-$DEFAULT_CUDA_TOOLKIT_PKG}"
    run_cmd "apt-get install -y ${CUDA_PKG}"
  else
    echo "Skipping CUDA toolkit install. You can install from NVIDIA repo or Ubuntu later."
  fi

  echo
  echo ">>> Python GPU recommendations:"
  echo "  - Use ONNXRuntime-GPU as backend, not PyTorch, unless you really need PyTorch."
  echo "  - Example GPU requirements for face recognition:"
  echo "      insightface"
  echo "      onnxruntime-gpu"
  echo "      ffmpeg-python"
  echo "      numpy"
  echo "  - Avoid installing torch/torchvision/torchaudio unless truly required."
else
  echo "No NVIDIA GPU detected (no 'lspci | grep -i nvidia' matches)."
  echo
  echo ">>> Python CPU-only recommendations:"
  echo "  - Use ONNXRuntime CPU backend:"
  echo "      onnxruntime"
  echo "      insightface  (configure providers=['CPUExecutionProvider'])"
  echo "  - Do NOT install GPU-only wheels (onnxruntime-gpu, torch with CUDA) on this host."
fi

########################################
# 9. Database setup
########################################

if [[ "$DB_TYPE" == "postgres" ]]; then
  echo
  echo "=== PostgreSQL: creating DB and user (if needed) ==="

  run_cmd "sudo -u postgres psql -tc \"SELECT 1 FROM pg_roles WHERE rolname = '${DB_USER}'\" | grep -q 1 || sudo -u postgres psql -c \"CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';\""
  run_cmd "sudo -u postgres psql -tc \"SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'\" | grep -q 1 || sudo -u postgres createdb -O ${DB_USER} ${DB_NAME}"
  run_cmd "sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};\""

elif [[ "$DB_TYPE" == "mysql" ]]; then
  echo
  echo "=== MySQL: creating DB and user (if needed) ==="

  MYSQL_CMD="mysql"
  run_cmd "$MYSQL_CMD -e \"CREATE DATABASE IF NOT EXISTS \\\`${DB_NAME}\\\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;\""
  run_cmd "$MYSQL_CMD -e \"CREATE USER IF NOT EXISTS '${DB_USER}'@'localhost' IDENTIFIED BY '${DB_PASS}';\""
  run_cmd "$MYSQL_CMD -e \"GRANT ALL PRIVILEGES ON \\\`${DB_NAME}\\\`.* TO '${DB_USER}'@'localhost'; FLUSH PRIVILEGES;\""
else
  echo
  echo "SQLite selected – remember to configure DATABASES['default'] / SQLALCHEMY_URL accordingly."
fi

########################################
# 10. Project directory, auto-create & optional auto-clone repo
########################################

echo
echo "=== Preparing project directory and optionally cloning repo ==="

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root ${PROJECT_ROOT} does not exist."
  run_cmd "mkdir -p '${PROJECT_ROOT}'"
fi

if is_dir_empty "$PROJECT_ROOT"; then
  echo "Project root ${PROJECT_ROOT} is empty."
  if [[ -n "$GIT_REPO_URL" ]]; then
    read -rp "Clone ${GIT_REPO_URL} into ${PROJECT_ROOT}? [Y/n]: " CLONE_EMPTY
    CLONE_EMPTY="${CLONE_EMPTY:-Y}"
    CLONE_EMPTY=$(echo "$CLONE_EMPTY" | tr '[:upper:]' '[:lower:]')

    if [[ "$CLONE_EMPTY" == "y" || "$CLONE_EMPTY" == "yes" ]]; then
      if [[ "$DRY_RUN" -eq 0 ]]; then
        echo ">> Cloning repo into ${PROJECT_ROOT} as ${APP_USER}"
        sudo -u "$APP_USER" -H bash -c "git clone '${GIT_REPO_URL}' '${PROJECT_ROOT}'"
      else
        echo ">> [DRY RUN] Would clone '${GIT_REPO_URL}' into '${PROJECT_ROOT}'"
      fi
    else
      echo "Skipping auto-clone. You can manually copy/clone your project into ${PROJECT_ROOT}."
    fi
  else
    echo "No Git URL provided, leaving ${PROJECT_ROOT} empty."
  fi
else
  echo "Project root ${PROJECT_ROOT} already has content."
fi

if [[ "$APP_TYPE" == "django" && ! -f "$MANAGE_PY_PATH" ]]; then
  echo "WARNING: manage.py not found at ${MANAGE_PY_PATH}. Adjust path or move your project."
fi

# Ownership
run_cmd "chown -R ${APP_USER}:${APP_USER} ${PROJECT_ROOT}"

########################################
# 11. Virtualenv
########################################

echo
echo "=== Creating virtualenv (if missing) ==="

VENV_PATH="${PROJECT_ROOT}/.venv"

if [[ "$DRY_RUN" -eq 0 ]]; then
  if [[ ! -d "$VENV_PATH" ]]; then
    echo ">> Creating venv at ${VENV_PATH}"
    sudo -u "$APP_USER" -H bash -c "cd '$PROJECT_ROOT' && $PYTHON_BIN -m venv .venv"
  else
    echo ">> venv already exists at ${VENV_PATH}, skipping creation."
  fi

  echo ">> Upgrading pip & wheel in venv"
  sudo -u "$APP_USER" -H bash -c "source '${VENV_PATH}/bin/activate' && pip install --upgrade pip wheel"
else
  echo ">> [DRY RUN] Skipping venv creation and pip upgrade."
fi

########################################
# 12. Python dependencies (requirements*.txt)
########################################

echo
echo "=== Python dependencies setup (project-specific) ==="

REQ_DEFAULT=""
REQ_CANDIDATES=()

if [[ -d "$PROJECT_ROOT" ]]; then
  while IFS= read -r f; do
    REQ_CANDIDATES+=("$f")
  done < <(find "$PROJECT_ROOT" -maxdepth 1 -type f -iname "requirements*.txt" 2>/dev/null || true)
fi

if (( ${#REQ_CANDIDATES[@]} > 0 )); then
  echo "Found possible requirements files in ${PROJECT_ROOT}:"
  idx=1
  for f in "${REQ_CANDIDATES[@]}"; do
    echo "  $idx) $(basename "$f")"
    ((idx++))
  done
  REQ_DEFAULT="${REQ_CANDIDATES[0]}"
  echo
  read -rp "Enter path to requirements file to install (press Enter for default: ${REQ_DEFAULT}, or leave blank to skip): " REQ_CHOICE
  if [[ -z "$REQ_CHOICE" ]]; then
    REQ_CHOICE="$REQ_DEFAULT"
  fi
else
  echo "No requirements*.txt files found in ${PROJECT_ROOT}."
  read -rp "Enter path to a requirements file to install (or leave empty to skip): " REQ_CHOICE
fi

if [[ -n "${REQ_CHOICE:-}" ]]; then
  if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ -f "$REQ_CHOICE" ]]; then
      echo ">> Installing Python dependencies from: $REQ_CHOICE"
      sudo -u "$APP_USER" -H bash -c "source '${VENV_PATH}/bin/activate' && pip install -r '$REQ_CHOICE'"
    else
      echo "WARNING: requirements file '$REQ_CHOICE' not found, skipping pip install."
    fi
  else
    echo ">> [DRY RUN] Would run: pip install -r '$REQ_CHOICE'"
  fi
else
  echo "Skipping Python dependency installation (no requirements file chosen)."
fi

########################################
# 13. Secret key / env hints
########################################

if [[ "$DRY_RUN" -eq 0 ]]; then
  SECRET_KEY="$(openssl rand -base64 48 | tr -d '\n' | tr '/' '_')"
else
  SECRET_KEY="<generated-secret-key-here>"
fi

echo
echo "Environment hints (you can put these in a .env file or systemd Environment= lines):"
if [[ "$APP_TYPE" == "django" ]]; then
  echo "  DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE}"
  echo "  DJANGO_SECRET_KEY=${SECRET_KEY}"
fi
echo "  REDIS_URL=redis://127.0.0.1:6379/0"
if [[ "$DB_TYPE" == "postgres" ]]; then
  echo "  DATABASE_URL=postgres://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}"
elif [[ "$DB_TYPE" == "mysql" ]]; then
  echo "  DATABASE_URL=mysql://${DB_USER}:${DB_PASS}@localhost:3306/${DB_NAME}"
fi
echo

########################################
# 14. Systemd units + per-env Gunicorn ports + optional Nginx
########################################

echo
echo "=== Creating systemd services (gunicorn, optional enforcer/heartbeat/apscheduler & nginx) ==="

DEV_PORT=""
PROD_PORT=""

for ENV in "${APP_ENVS[@]}"; do
  SERVICE_SUFFIX="${ENV}"

  # Gunicorn port per env
  if [[ "$ENV" == "dev" ]]; then
    DEFAULT_PORT=8000
  else
    DEFAULT_PORT=8001
  fi

  echo
  read -rp "Gunicorn HTTP port for ${ENV} environment [${DEFAULT_PORT}]: " GUNICORN_PORT
  GUNICORN_PORT="${GUNICORN_PORT:-$DEFAULT_PORT}"

  if [[ "$ENV" == "dev" ]]; then
    DEV_PORT="$GUNICORN_PORT"
  else
    PROD_PORT="$GUNICORN_PORT"
  fi

  GUNICORN_SERVICE="/etc/systemd/system/${PROJECT_NAME,,}-${SERVICE_SUFFIX}-gunicorn.service"
  ENFORCER_SERVICE="/etc/systemd/system/${PROJECT_NAME,,}-${SERVICE_SUFFIX}-enforcer.service"
  HEARTBEAT_SERVICE="/etc/systemd/system/${PROJECT_NAME,,}-${SERVICE_SUFFIX}-heartbeat.service"
  HEARTBEAT_TIMER="/etc/systemd/system/${PROJECT_NAME,,}-${SERVICE_SUFFIX}-heartbeat.timer"
  APSCHEDULER_SERVICE="/etc/systemd/system/${PROJECT_NAME,,}-${SERVICE_SUFFIX}-apscheduler.service"

  if [[ "$APP_TYPE" == "django" ]]; then
    GUNICORN_EXEC="${VENV_PATH}/bin/gunicorn ${DJANGO_PROJECT_NAME}.wsgi:application --bind 127.0.0.1:${GUNICORN_PORT} --workers 3"
  else
    GUNICORN_EXEC="${VENV_PATH}/bin/gunicorn ${FLASK_GUNICORN_APP} --bind 127.0.0.1:${GUNICORN_PORT} --workers 3"
  fi

  GUNICORN_CONTENT="[Unit]
Description=${PROJECT_NAME} (${ENV}) Gunicorn daemon
After=network.target

[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${PROJECT_ROOT}
Environment=\"PYTHONUNBUFFERED=1\"
ExecStart=${GUNICORN_EXEC}

Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"

  write_file "$GUNICORN_SERVICE" "$GUNICORN_CONTENT"

  # Optional Enforcer
  if [[ -n "$ENFORCER_CMD" && "$APP_TYPE" == "django" ]]; then
    ENFORCER_CONTENT="[Unit]
Description=${PROJECT_NAME} (${ENV}) Enforcer Service
After=network.target

[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${PROJECT_ROOT}
Environment=\"PYTHONUNBUFFERED=1\"
ExecStart=${VENV_PATH}/bin/python ${MANAGE_PY_PATH} ${ENFORCER_CMD}

Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"
    write_file "$ENFORCER_SERVICE" "$ENFORCER_CONTENT"
  fi

  # Optional Heartbeat
  if [[ -n "$HEARTBEAT_CMD" && "$APP_TYPE" == "django" ]]; then
    HEARTBEAT_SERVICE_CONTENT="[Unit]
Description=${PROJECT_NAME} (${ENV}) Heartbeat Service
After=network.target

[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${PROJECT_ROOT}
Environment=\"PYTHONUNBUFFERED=1\"
ExecStart=${VENV_PATH}/bin/python ${MANAGE_PY_PATH} ${HEARTBEAT_CMD}

[Install]
WantedBy=multi-user.target
"

    HEARTBEAT_TIMER_CONTENT="[Unit]
Description=${PROJECT_NAME} (${ENV}) Heartbeat Timer

[Timer]
OnBootSec=30
OnUnitActiveSec=${HEARTBEAT_INTERVAL_SEC}
Unit=$(basename "$HEARTBEAT_SERVICE")
Persistent=true

[Install]
WantedBy=timers.target
"

    write_file "$HEARTBEAT_SERVICE" "$HEARTBEAT_SERVICE_CONTENT"
    write_file "$HEARTBEAT_TIMER" "$HEARTBEAT_TIMER_CONTENT"
  fi

  # Optional APScheduler
  if [[ "$ENABLE_APSCHEDULER" -eq 1 && -n "$APSCHEDULER_CMD" && "$APP_TYPE" == "django" ]]; then
    APSCHEDULER_CONTENT="[Unit]
Description=${PROJECT_NAME} (${ENV}) APScheduler Service
After=network.target

[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${PROJECT_ROOT}
Environment=\"PYTHONUNBUFFERED=1\"
ExecStart=${VENV_PATH}/bin/python ${MANAGE_PY_PATH} ${APSCHEDULER_CMD}

Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"
    write_file "$APSCHEDULER_SERVICE" "$APSCHEDULER_CONTENT"
  fi

  ########################################
  # Optional Nginx config per env
  ########################################

  if [[ "$ENABLE_NGINX" -eq 1 ]]; then
    echo
    read -rp "Create Nginx reverse-proxy config for ${ENV}? [y/N]: " NGINX_ENV_CHOICE
    NGINX_ENV_CHOICE="${NGINX_ENV_CHOICE:-N}"
    NGINX_ENV_CHOICE=$(echo "$NGINX_ENV_CHOICE" | tr '[:upper:]' '[:lower:]')

    if [[ "$NGINX_ENV_CHOICE" == "y" || "$NGINX_ENV_CHOICE" == "yes" ]]; then
      DEFAULT_DOMAIN="${PROJECT_NAME,,}-${ENV}.example.com"
      read -rp "Domain name for ${ENV} (server_name) [${DEFAULT_DOMAIN}]: " NGINX_DOMAIN
      NGINX_DOMAIN="${NGINX_DOMAIN:-$DEFAULT_DOMAIN}"

      NGINX_CONF="/etc/nginx/sites-available/${PROJECT_NAME,,}-${ENV}.conf"

      NGINX_CONF_CONTENT="# ${PROJECT_NAME} (${ENV}) Nginx reverse proxy
#
# Upstream Gunicorn is on 127.0.0.1:${GUNICORN_PORT}
# Safe behind Cloudflare: point DNS for ${NGINX_DOMAIN} to this server's public IP.

server {
    listen 80;
    server_name ${NGINX_DOMAIN};

    location /static/ {
        alias ${PROJECT_ROOT}/static/;
        access_log off;
        expires 7d;
    }

    location /media/ {
        alias ${PROJECT_ROOT}/media/;
        access_log off;
        expires 7d;
    }

    location / {
        proxy_pass http://127.0.0.1:${GUNICORN_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
"

      write_file "$NGINX_CONF" "$NGINX_CONF_CONTENT"

      if [[ "$DRY_RUN" -eq 0 ]]; then
        run_cmd "ln -sf ${NGINX_CONF} /etc/nginx/sites-enabled/${PROJECT_NAME,,}-${ENV}.conf"
        run_cmd "nginx -t"
        run_cmd "systemctl reload nginx"
      else
        echo ">> [DRY RUN] Would enable Nginx site ${PROJECT_NAME,,}-${ENV} and reload nginx."
      fi
    fi
  fi

done

########################################
# 15. systemd reload + enable
########################################

echo
echo "=== Reloading systemd and enabling services ==="

run_cmd "systemctl daemon-reload"

for ENV in "${APP_ENVS[@]}"; do
  SERVICE_SUFFIX="${ENV}"
  GUNICORN_SERVICE="/etc/systemd/system/${PROJECT_NAME,,}-${SERVICE_SUFFIX}-gunicorn.service"
  ENFORCER_SERVICE="/etc/systemd/system/${PROJECT_NAME,,}-${SERVICE_SUFFIX}-enforcer.service"
  HEARTBEAT_TIMER="/etc/systemd/system/${PROJECT_NAME,,}-${SERVICE_SUFFIX}-heartbeat.timer"
  APSCHEDULER_SERVICE="/etc/systemd/system/${PROJECT_NAME,,}-${SERVICE_SUFFIX}-apscheduler.service"

  if [[ -f "$GUNICORN_SERVICE" ]]; then
    run_cmd "systemctl enable ${PROJECT_NAME,,}-${SERVICE_SUFFIX}-gunicorn.service"
  fi
  if [[ -f "$ENFORCER_SERVICE" ]]; then
    run_cmd "systemctl enable ${PROJECT_NAME,,}-${SERVICE_SUFFIX}-enforcer.service"
  fi
  if [[ -f "$HEARTBEAT_TIMER" ]]; then
    run_cmd "systemctl enable ${PROJECT_NAME,,}-${SERVICE_SUFFIX}-heartbeat.timer"
  fi
  if [[ -f "$APSCHEDULER_SERVICE" ]]; then
    run_cmd "systemctl enable ${PROJECT_NAME,,}-${SERVICE_SUFFIX}-apscheduler.service"
  fi
done

########################################
# 16. Summary
########################################

echo
echo "=== Setup finished (universal installer with GPU detection) ==="
echo "Summary:"
echo "  Project name:         ${PROJECT_NAME}"
echo "  Project root:         ${PROJECT_ROOT}"
echo "  Git repo:             ${GIT_REPO_URL}"
echo "  App user:             ${APP_USER}"
echo "  Venv:                 ${VENV_PATH}"
echo "  App type:             ${APP_TYPE}"
if [[ "$APP_TYPE" == "django" ]]; then
  echo "  Django package:       ${DJANGO_PROJECT_NAME}"
  echo "  Settings module:      ${DJANGO_SETTINGS_MODULE}"
  echo "  manage.py:            ${MANAGE_PY_PATH}"
else
  echo "  Gunicorn app:         ${FLASK_GUNICORN_APP}"
fi
echo "  DB type:              ${DB_TYPE}"
if [[ "$DB_TYPE" != "sqlite" ]]; then
  echo "  DB name/user/pass:    ${DB_NAME} / ${DB_USER} / ${DB_PASS}"
fi
if [[ -n "$ENFORCER_CMD" ]]; then
  echo "  Enforcer command:     ${ENFORCER_CMD}"
fi
if [[ -n "$HEARTBEAT_CMD" ]]; then
  echo "  Heartbeat command:    ${HEARTBEAT_CMD}"
fi
if [[ "$ENABLE_APSCHEDULER" -eq 1 ]]; then
  echo "  APScheduler command:  ${APSCHEDULER_CMD} (ENABLED)"
fi
echo "  Heartbeat interval:   ${HB_INTERVAL_MIN} minute(s)"
echo "  Environments:         ${APP_ENVS[*]}"
if [[ -n "$DEV_PORT" ]]; then
  echo "  Dev Gunicorn port:    ${DEV_PORT}"
fi
if [[ -n "$PROD_PORT" ]]; then
  echo "  Prod Gunicorn port:   ${PROD_PORT}"
fi
echo "  Redis:                Installed and enabled (redis-server)"
echo "  ffmpeg:               Installed"
if [[ "$ENABLE_NGINX" -eq 1 ]]; then
  echo "  Nginx:                Installed (per-env configs if chosen)"
else
  echo "  Nginx:                Not installed"
fi
if [[ "$GPU_PRESENT" -eq 1 ]]; then
  echo "  GPU:                  NVIDIA detected (see above for driver/CUDA status)"
else
  echo "  GPU:                  No NVIDIA GPU detected – CPU-only mode recommended"
fi
echo
echo "You can now start services with e.g.:"
for ENV in "${APP_ENVS[@]}"; do
  echo "  systemctl start ${PROJECT_NAME,,}-${ENV}-gunicorn.service"
  if [[ -n "$ENFORCER_CMD" && "$APP_TYPE" == "django" ]]; then
    echo "  systemctl start ${PROJECT_NAME,,}-${ENV}-enforcer.service"
  fi
  if [[ -n "$HEARTBEAT_CMD" && "$APP_TYPE" == "django" ]]; then
    echo "  systemctl start ${PROJECT_NAME,,}-${ENV}-heartbeat.timer"
  fi
  if [[ "$ENABLE_APSCHEDULER" -eq 1 && -n "$APSCHEDULER_CMD" && "$APP_TYPE" == "django" ]]; then
    echo "  systemctl start ${PROJECT_NAME,,}-${ENV}-apscheduler.service"
  fi
done
echo
echo "To run a dry summary only:"
echo "  sudo $0 --dry-run"
