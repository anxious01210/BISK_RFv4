#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Universal Web App Installer (Django/Flask)
# - Installs system deps (Python, DB, Redis, ffmpeg, nginx)
# - Sets up project under /srv/<project>
# - Creates venv, installs requirements
# - Creates systemd services (gunicorn + optional mgmt cmds)
# - Configures nginx reverse proxy
# - For Django:
#     * Fixes ownership on project root
#     * Ensures media/upload_tmp
#     * Appends DB + ALLOWED_HOSTS + STATIC_ROOT to settings.py
#     * Runs collectstatic
#     * Configures nginx /static/ and /media/
# =========================================================

if [[ $EUID -ne 0 ]]; then
  echo "Please run this script with sudo:"
  echo "  sudo $0"
  exit 1
fi

prompt_default() {
  local prompt="$1"
  local default="$2"
  local var
  read -r -p "$prompt [$default]: " var || true
  if [[ -z "$var" ]]; then
    echo "$default"
  else
    echo "$var"
  fi
}

yes_no_default() {
  local prompt="$1"
  local default="$2" # Y or N
  local ans
  read -r -p "$prompt [$default]: " ans || true
  ans="${ans:-$default}"
  case "$ans" in
    [Yy]*) echo "Y" ;;
    [Nn]*) echo "N" ;;
    *)     echo "$default" ;;
  esac
}

# --------------------------------------------
# Basic project info
# --------------------------------------------
PROJECT_NAME=$(prompt_default "Project name" "BISK_RFv4")
PROJECT_ROOT=$(prompt_default "Project root directory" "/srv/${PROJECT_NAME}")
DEFAULT_REPO="https://github.com/anxious01210/BISK_RFv4.git"
GIT_REPO=$(prompt_default "Git repo URL (leave blank to skip auto-clone)" "$DEFAULT_REPO")
APP_USER=$(prompt_default "Linux user that will run the app (non-root)" "rio")
PY_BIN=$(prompt_default "Python binary for venv" "python3")

# App type
echo
echo "Application type:"
echo "  1) Django"
echo "  2) Flask"
APP_TYPE_CHOICE=$(prompt_default "Choose app type [1/2] (default 1)" "1")
if [[ "$APP_TYPE_CHOICE" == "2" ]]; then
  APP_TYPE="flask"
else
  APP_TYPE="django"
fi

DJANGO_PACKAGE=""
DJANGO_SETTINGS_MODULE=""
MANAGE_PY_PATH=""

if [[ "$APP_TYPE" == "django" ]]; then
  DJANGO_PACKAGE=$(prompt_default "Django project package name" "bisk")
  DJANGO_SETTINGS_MODULE=$(prompt_default "DJANGO_SETTINGS_MODULE" "${DJANGO_PACKAGE}.settings")
  MANAGE_PY_PATH=$(prompt_default "Path to manage.py" "${PROJECT_ROOT}/manage.py")
fi

# Environment type
echo
echo "Environment type:"
echo "  1) dev"
echo "  2) prod"
echo "  3) both"
ENV_CHOICE=$(prompt_default "Choose environment type [1/2/3] (default 1)" "1")
ENV_DEV=false
ENV_PROD=false
case "$ENV_CHOICE" in
  2) ENV_PROD=true ;;
  3) ENV_DEV=true; ENV_PROD=true ;;
  *) ENV_DEV=true ;;
esac

# Database choice
echo
echo "Database choice:"
echo "  1) SQLite3 (simple, dev only or small installs)"
echo "  2) PostgreSQL (recommended for prod)"
echo "  3) MySQL/MariaDB"
DB_CHOICE=$(prompt_default "Choose database [1/2/3] (default 2)" "2")

DB_TYPE="postgres"
case "$DB_CHOICE" in
  1) DB_TYPE="sqlite" ;;
  3) DB_TYPE="mysql" ;;
  *) DB_TYPE="postgres" ;;
esac

DB_NAME=""
DB_USER=""
DB_PASS=""

if [[ "$DB_TYPE" == "postgres" ]]; then
  echo
  echo "Database settings (for postgres):"
  DB_NAME=$(prompt_default "Database name" "${PROJECT_NAME//-/_}_db")
  DB_USER=$(prompt_default "Database user" "${PROJECT_NAME//-/_}_user")
  DB_PASS=$(prompt_default "Database password" "${PROJECT_NAME//-/_}_pass")
fi

# Optional management commands (for Django)
ENFORCER_CMD=""
HEARTBEAT_CMD=""
APSCHEDULER_CMD=""
APSCHEDULER_ENABLED=false
HEARTBEAT_INTERVAL_MINUTES="1"

if [[ "$APP_TYPE" == "django" ]]; then
  echo
  echo "Optional management commands (mostly for Django projects)."
  echo "Leave blank to skip creating that service."
  read -r -p "Enforcer command name (manage.py <cmd>, e.g. 'bisk_enforcer') []: " ENFORCER_CMD || true
  read -r -p "Heartbeat command name (manage.py <cmd>, e.g. 'bisk_heartbeat') []: " HEARTBEAT_CMD || true
  read -r -p "APScheduler command name (manage.py <cmd>, e.g. 'runapscheduler') []: " APSCHEDULER_CMD || true

  if [[ -n "$APSCHEDULER_CMD" ]]; then
    echo
    echo "APScheduler service (manage.py ${APSCHEDULER_CMD}):"
    echo "  1) Enable"
    echo "  2) Disable"
    APS_CHOICE=$(prompt_default "Choose [1/2] (default 2)" "2")
    if [[ "$APS_CHOICE" == "1" ]]; then
      APSCHEDULER_ENABLED=true
    fi
  fi

  if [[ -n "$HEARTBEAT_CMD" ]]; then
    HEARTBEAT_INTERVAL_MINUTES=$(prompt_default "Heartbeat interval in minutes (if heartbeat command used)" "1")
  fi
fi

# Nginx
echo
echo "Nginx reverse proxy:"
echo "  This will install Nginx (if needed) and optionally create per-env site configs."
NGINX_INSTALL=$(yes_no_default "Install Nginx web server?" "Y")

DEV_DOMAIN=""
PROD_DOMAIN=""
DEV_GUNICORN_PORT="8000"
PROD_GUNICORN_PORT="8001"

if [[ "$ENV_DEV" == true ]]; then
  DEV_GUNICORN_PORT=$(prompt_default "Gunicorn HTTP port for dev environment" "8000")
fi
if [[ "$ENV_PROD" == true ]]; then
  PROD_GUNICORN_PORT=$(prompt_default "Gunicorn HTTP port for prod environment" "8001")
fi

# --------------------------------------------
# Install system packages
# --------------------------------------------
echo
echo "=== Installing system packages (Python, DB libs, build tools, git, Redis, ffmpeg) ==="
apt-get update
apt-get install -y build-essential git curl
apt-get install -y python3-venv python3-dev python3-pip pkg-config libssl-dev libffi-dev libjpeg-dev zlib1g-dev

if [[ "$DB_TYPE" == "postgres" ]]; then
  apt-get install -y postgresql postgresql-contrib libpq-dev
elif [[ "$DB_TYPE" == "mysql" ]]; then
  apt-get install -y mysql-server libmysqlclient-dev
fi

echo
echo "=== Installing and enabling Redis ==="
apt-get install -y redis-server redis-tools
systemctl enable redis-server
echo ">> Testing Redis with: redis-cli ping"
if redis-cli ping >/dev/null 2>&1; then
  echo "Redis is up (PING OK)."
else
  echo "WARNING: Redis ping failed; please check redis-server."
fi

echo
echo "=== Installing ffmpeg for any media/streaming scripts ==="
apt-get install -y ffmpeg

if [[ "$NGINX_INSTALL" == "Y" ]]; then
  apt-get install -y nginx
fi

# --------------------------------------------
# GPU / NVIDIA detection
# --------------------------------------------
echo
echo "=== GPU / NVIDIA detection & recommendations ==="
if lspci | grep -qi nvidia; then
  echo "Detected NVIDIA GPU:"
  lspci | grep -i nvidia || true
  echo
  echo ">>> Python GPU stack recommendations (you may already have them in requirements.txt):"
  echo "  - Use onnxruntime-gpu (with correct CUDA version) and/or PyTorch with CUDA."
  echo "  - Ensure NVIDIA driver + CUDA toolkit are installed on this host."
else
  echo "No NVIDIA GPU detected (no 'lspci | grep -i nvidia' matches)."
  echo
  echo ">>> Python CPU-only recommendations:"
  echo "  - Use ONNXRuntime CPU backend:"
  echo "      onnxruntime"
  echo "      insightface  (configure providers=['CPUExecutionProvider'])"
  echo "  - Do NOT install GPU-only wheels (onnxruntime-gpu, torch with CUDA) on this host."
fi

# --------------------------------------------
# Database creation (Postgres/MySQL)
# --------------------------------------------
if [[ "$DB_TYPE" == "postgres" ]]; then
  echo
  echo "=== PostgreSQL: creating DB and user (if needed) ==="
  sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname = '${DB_USER}'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';"

  sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'" | grep -q 1 || \
    sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}"

  sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};"
fi

# --------------------------------------------
# Prepare project root & clone
# --------------------------------------------
echo
echo "=== Preparing project directory and optionally cloning repo ==="
if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root $PROJECT_ROOT does not exist."
  echo ">> mkdir -p '$PROJECT_ROOT'"
  mkdir -p "$PROJECT_ROOT"
fi

# Fix ownership now (point 1)
chown -R "$APP_USER:$APP_USER" "$PROJECT_ROOT"

if [[ -n "$GIT_REPO" ]]; then
  # If directory is empty, clone; otherwise leave as-is
  if [[ -z "$(ls -A "$PROJECT_ROOT")" ]]; then
    echo "Project root $PROJECT_ROOT is empty."
    CLONE_ANS=$(yes_no_default "Clone $GIT_REPO into $PROJECT_ROOT?" "Y")
    if [[ "$CLONE_ANS" == "Y" ]]; then
      echo ">> Cloning repo into $PROJECT_ROOT as $APP_USER"
      sudo -u "$APP_USER" git clone "$GIT_REPO" "$PROJECT_ROOT"
    fi
  else
    echo "Project root $PROJECT_ROOT already has content."
  fi
else
  echo "No GIT_REPO specified; skipping clone."
fi

# Ensure final ownership
chown -R "$APP_USER:$APP_USER" "$PROJECT_ROOT"

# --------------------------------------------
# Create venv and install requirements
# --------------------------------------------
echo
echo "=== Creating virtualenv (if missing) ==="
if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
  echo ">> Creating venv at $PROJECT_ROOT/.venv"
  sudo -u "$APP_USER" "$PY_BIN" -m venv "$PROJECT_ROOT/.venv"
fi

echo ">> Upgrading pip & wheel in venv"
sudo -u "$APP_USER" "$PROJECT_ROOT/.venv/bin/pip" install --upgrade pip wheel

echo
echo "=== Python dependencies setup (project-specific) ==="
REQ_FILES=()
if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
  REQ_FILES+=("$PROJECT_ROOT/requirements.txt")
fi
if [[ -f "$PROJECT_ROOT/requirements-prod.txt" ]]; then
  REQ_FILES+=("$PROJECT_ROOT/requirements-prod.txt")
fi

if [[ "${#REQ_FILES[@]}" -gt 0 ]]; then
  echo "Found possible requirements files in $PROJECT_ROOT:"
  idx=1
  for f in "${REQ_FILES[@]}"; do
    echo "  $idx) $f"
    idx=$((idx + 1))
  done
  DEFAULT_REQ="${REQ_FILES[0]}"
  read -r -p "Enter path to requirements file to install (press Enter for default: $DEFAULT_REQ, or leave blank to skip): " REQ_CHOICE || true
  REQ_CHOICE="${REQ_CHOICE:-$DEFAULT_REQ}"
  if [[ -n "$REQ_CHOICE" ]]; then
    echo ">> Installing Python dependencies from: $REQ_CHOICE"
    sudo -u "$APP_USER" "$PROJECT_ROOT/.venv/bin/pip" install -r "$REQ_CHOICE"
  else
    echo "Skipping pip install (no requirements file chosen)."
  fi
else
  echo "No requirements*.txt found in $PROJECT_ROOT; skipping pip install."
fi

# --------------------------------------------
# Environment hints
# --------------------------------------------
echo
echo "Environment hints (you can put these in a .env file or systemd Environment= lines):"
if [[ "$APP_TYPE" == "django" ]]; then
  echo "  DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE}"
  echo "  DJANGO_SECRET_KEY=!!!GENERATE_A_SECRET_KEY!!!"
fi
if [[ "$DB_TYPE" == "postgres" ]]; then
  echo "  DATABASE_URL=postgres://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}"
fi
echo "  REDIS_URL=redis://127.0.0.1:6379/0"

# --------------------------------------------
# Django-specific auto fixes (media/upload_tmp)
# --------------------------------------------
SETTINGS_PY=""
if [[ "$APP_TYPE" == "django" ]]; then
  SETTINGS_PY="${PROJECT_ROOT}/${DJANGO_PACKAGE}/settings.py"
  echo
  echo "=== Django-specific setup ==="
  echo ">> Ensuring media/upload_tmp exists and is owned by $APP_USER"
  mkdir -p "${PROJECT_ROOT}/media/upload_tmp"
  chown -R "$APP_USER:$APP_USER" "${PROJECT_ROOT}/media"
fi

# --------------------------------------------
# Nginx site configuration
# --------------------------------------------
if [[ "$NGINX_INSTALL" == "Y" ]]; then
  echo
  echo "=== Creating Nginx reverse-proxy config(s) ==="

  if [[ "$ENV_DEV" == true ]]; then
    read -r -p "Create Nginx reverse-proxy config for dev? [y/N]: " DEV_NGINX || true
    DEV_NGINX="${DEV_NGINX:-N}"
    if [[ "$DEV_NGINX" =~ ^[Yy]$ ]]; then
      DEV_DOMAIN=$(prompt_default "Domain name for dev (server_name)" "${PROJECT_NAME,,}-dev.example.com")
      DEV_NGINX_CONF="/etc/nginx/sites-available/${PROJECT_NAME,,}-dev.conf"
      cat > "$DEV_NGINX_CONF" <<EOF
# ${PROJECT_NAME} (dev) Nginx reverse proxy
server {
    listen 80;
    server_name ${DEV_DOMAIN};

    location /static/ {
        alias ${PROJECT_ROOT}/staticfiles/;
        access_log off;
        expires 7d;
    }

    location /media/ {
        alias ${PROJECT_ROOT}/media/;
        access_log off;
        expires 7d;
    }

    location / {
        proxy_pass http://127.0.0.1:${DEV_GUNICORN_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
      ln -sf "$DEV_NGINX_CONF" "/etc/nginx/sites-enabled/$(basename "$DEV_NGINX_CONF")"
    fi
  fi

  if [[ "$ENV_PROD" == true ]]; then
    read -r -p "Create Nginx reverse-proxy config for prod? [y/N]: " PROD_NGINX || true
    PROD_NGINX="${PROD_NGINX:-N}"
    if [[ "$PROD_NGINX" =~ ^[Yy]$ ]]; then
      PROD_DOMAIN=$(prompt_default "Domain name for prod (server_name)" "${PROJECT_NAME,,}-prod.example.com")
      PROD_NGINX_CONF="/etc/nginx/sites-available/${PROJECT_NAME,,}-prod.conf"
      cat > "$PROD_NGINX_CONF" <<EOF
# ${PROJECT_NAME} (prod) Nginx reverse proxy
server {
    listen 80;
    server_name ${PROD_DOMAIN};

    location /static/ {
        alias ${PROJECT_ROOT}/staticfiles/;
        access_log off;
        expires 7d;
    }

    location /media/ {
        alias ${PROJECT_ROOT}/media/;
        access_log off;
        expires 7d;
    }

    location / {
        proxy_pass http://127.0.0.1:${PROD_GUNICORN_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
      ln -sf "$PROD_NGINX_CONF" "/etc/nginx/sites-enabled/$(basename "$PROD_NGINX_CONF")"
    fi
  fi

  # Disable default site (point 5)
  if [[ -f /etc/nginx/sites-enabled/default ]]; then
    echo ">> Disabling nginx default site"
    rm -f /etc/nginx/sites-enabled/default
  fi

  echo ">> Testing nginx config"
  nginx -t
  echo ">> Reloading nginx"
  systemctl reload nginx
fi

# --------------------------------------------
# Systemd services (gunicorn + optional mgmt)
# --------------------------------------------
echo
echo "=== Creating systemd services (gunicorn, optional enforcer/heartbeat/apscheduler) ==="

create_service() {
  local env="$1"  # dev or prod
  local gunicorn_port="$2"

  local svc_prefix="${PROJECT_NAME,,}-${env}"
  local gunicorn_svc="${svc_prefix}-gunicorn.service"

  local manage_cmd=""
  if [[ "$APP_TYPE" == "django" ]]; then
    manage_cmd="${PROJECT_ROOT}/manage.py"
  fi

  # Gunicorn service
  cat > "/etc/systemd/system/${gunicorn_svc}" <<EOF
[Unit]
Description=${PROJECT_NAME} (${env}) Gunicorn
After=network.target

[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${PROJECT_ROOT}
Environment="PATH=${PROJECT_ROOT}/.venv/bin"
EOF

  if [[ "$APP_TYPE" == "django" ]]; then
    cat >> "/etc/systemd/system/${gunicorn_svc}" <<EOF
Environment="DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE}"
ExecStart=${PROJECT_ROOT}/.venv/bin/gunicorn ${DJANGO_PACKAGE}.wsgi:application --bind 127.0.0.1:${gunicorn_port} --workers 3
EOF
  else
    cat >> "/etc/systemd/system/${gunicorn_svc}" <<EOF
ExecStart=${PROJECT_ROOT}/.venv/bin/gunicorn wsgi:app --bind 127.0.0.1:${gunicorn_port} --workers 3
EOF
  fi

  cat >> "/etc/systemd/system/${gunicorn_svc}" <<'EOF'
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

  # Enforcer
  if [[ -n "$ENFORCER_CMD" && "$APP_TYPE" == "django" ]]; then
    cat > "/etc/systemd/system/${svc_prefix}-enforcer.service" <<EOF
[Unit]
Description=${PROJECT_NAME} (${env}) Enforcer
After=network.target

[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${PROJECT_ROOT}
Environment="PATH=${PROJECT_ROOT}/.venv/bin"
Environment="DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE}"
ExecStart=${PROJECT_ROOT}/.venv/bin/python ${manage_cmd} ${ENFORCER_CMD}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  fi

  # Heartbeat
  if [[ -n "$HEARTBEAT_CMD" && "$APP_TYPE" == "django" ]]; then
    cat > "/etc/systemd/system/${svc_prefix}-heartbeat.service" <<EOF
[Unit]
Description=${PROJECT_NAME} (${env}) Heartbeat
After=network.target

[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${PROJECT_ROOT}
Environment="PATH=${PROJECT_ROOT}/.venv/bin"
Environment="DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE}"
ExecStart=${PROJECT_ROOT}/.venv/bin/python ${manage_cmd} ${HEARTBEAT_CMD}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    cat > "/etc/systemd/system/${svc_prefix}-heartbeat.timer" <<EOF
[Unit]
Description=${PROJECT_NAME} (${env}) Heartbeat timer

[Timer]
OnBootSec=30
OnUnitActiveSec=${HEARTBEAT_INTERVAL_MINUTES}min
Unit=${svc_prefix}-heartbeat.service

[Install]
WantedBy=timers.target
EOF
  fi

  # APScheduler
  if [[ "$APSCHEDULER_ENABLED" == true && -n "$APSCHEDULER_CMD" && "$APP_TYPE" == "django" ]]; then
    cat > "/etc/systemd/system/${svc_prefix}-apscheduler.service" <<EOF
[Unit]
Description=${PROJECT_NAME} (${env}) APScheduler
After=network.target

[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${PROJECT_ROOT}
Environment="PATH=${PROJECT_ROOT}/.venv/bin"
Environment="DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE}"
ExecStart=${PROJECT_ROOT}/.venv/bin/python ${manage_cmd} ${APSCHEDULER_CMD}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  fi
}

# create services per env
if [[ "$ENV_DEV" == true ]]; then
  create_service "dev" "$DEV_GUNICORN_PORT"
fi
if [[ "$ENV_PROD" == true ]]; then
  create_service "prod" "$PROD_GUNICORN_PORT"
fi

# --------------------------------------------
# Django: patch settings.py (DB, ALLOWED_HOSTS, STATIC_ROOT)
# --------------------------------------------
if [[ "$APP_TYPE" == "django" && -f "$SETTINGS_PY" ]]; then
  echo
  echo "=== Patching Django settings.py (DB, ALLOWED_HOSTS, STATIC_ROOT) ==="
  if ! grep -q "AUTO-GENERATED BY universal_web_app_installer" "$SETTINGS_PY"; then
    ROOT_DOMAIN=""
    # derive base domain from prod or dev domain
    DOMAIN_SRC="${PROD_DOMAIN:-$DEV_DOMAIN}"
    if [[ "$DOMAIN_SRC" == *.*.* ]]; then
      ROOT_DOMAIN="${DOMAIN_SRC#*.}"  # strip first label
    fi

    cat >> "$SETTINGS_PY" <<EOF

# === AUTO-GENERATED BY universal_web_app_installer - $(date) ===
EOF

    if [[ "$DB_TYPE" == "postgres" ]]; then
      cat >> "$SETTINGS_PY" <<EOF
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "${DB_NAME}",
        "USER": "${DB_USER}",
        "PASSWORD": "${DB_PASS}",
        "HOST": "127.0.0.1",
        "PORT": "5432",
    }
}
EOF
    elif [[ "$DB_TYPE" == "sqlite" ]]; then
      cat >> "$SETTINGS_PY" <<'EOF'
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}
EOF
    fi

    echo "" >> "$SETTINGS_PY"
    echo "ALLOWED_HOSTS = [" >> "$SETTINGS_PY"
    echo "    \"127.0.0.1\"," >> "$SETTINGS_PY"
    echo "    \"localhost\"," >> "$SETTINGS_PY"
    [[ -n "$DEV_DOMAIN" ]]  && echo "    \"${DEV_DOMAIN}\","  >> "$SETTINGS_PY"
    [[ -n "$PROD_DOMAIN" ]] && echo "    \"${PROD_DOMAIN}\"," >> "$SETTINGS_PY"
    [[ -n "$ROOT_DOMAIN" ]] && echo "    \".${ROOT_DOMAIN}\"," >> "$SETTINGS_PY"
    echo "]" >> "$SETTINGS_PY"

    cat >> "$SETTINGS_PY" <<'EOF'

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"
EOF
  else
    echo "Settings already contain installer marker; skipping re-append."
  fi
fi

# --------------------------------------------
# Django: collectstatic & chown staticfiles (point 7)
# --------------------------------------------
if [[ "$APP_TYPE" == "django" ]]; then
  echo
  echo "=== Running collectstatic and fixing staticfiles ownership ==="
  sudo -u "$APP_USER" "$PROJECT_ROOT/.venv/bin/python" "$MANAGE_PY_PATH" collectstatic --noinput || \
    echo "WARNING: collectstatic failed; check Django configuration."

  if [[ -d "$PROJECT_ROOT/staticfiles" ]]; then
    chown -R "$APP_USER:$APP_USER" "$PROJECT_ROOT/staticfiles"
  fi
fi

# --------------------------------------------
# Enable & restart services (point 4)
# --------------------------------------------
echo
echo "=== Reloading systemd and enabling services ==="
systemctl daemon-reload

enable_and_restart_env() {
  local env="$1"
  local svc_prefix="${PROJECT_NAME,,}-${env}"

  systemctl enable "${svc_prefix}-gunicorn.service"
  systemctl restart "${svc_prefix}-gunicorn.service" || \
    echo "WARNING: ${svc_prefix}-gunicorn.service failed to start."

  if [[ -n "$ENFORCER_CMD" && "$APP_TYPE" == "django" ]]; then
    systemctl enable "${svc_prefix}-enforcer.service"
    systemctl restart "${svc_prefix}-enforcer.service" || true
  fi

  if [[ -n "$HEARTBEAT_CMD" && "$APP_TYPE" == "django" ]]; then
    systemctl enable "${svc_prefix}-heartbeat.timer"
    systemctl restart "${svc_prefix}-heartbeat.timer" || true
  fi

  if [[ "$APSCHEDULER_ENABLED" == true && -n "$APSCHEDULER_CMD" && "$APP_TYPE" == "django" ]]; then
    systemctl enable "${svc_prefix}-apscheduler.service"
    systemctl restart "${svc_prefix}-apscheduler.service" || true
  fi
}

if [[ "$ENV_DEV" == true ]]; then
  enable_and_restart_env "dev"
fi
if [[ "$ENV_PROD" == true ]]; then
  enable_and_restart_env "prod"
fi

# --------------------------------------------
# Final summary
# --------------------------------------------
echo
echo "=== Setup finished (universal installer with GPU detection & Django fixes) ==="
echo "Summary:"
echo "  Project name:         ${PROJECT_NAME}"
echo "  Project root:         ${PROJECT_ROOT}"
echo "  Git repo:             ${GIT_REPO:-<none>}"
echo "  App user:             ${APP_USER}"
echo "  Venv:                 ${PROJECT_ROOT}/.venv"
echo "  App type:             ${APP_TYPE}"
if [[ "$APP_TYPE" == "django" ]]; then
  echo "  Django package:       ${DJANGO_PACKAGE}"
  echo "  Settings module:      ${DJANGO_SETTINGS_MODULE}"
  echo "  manage.py:            ${MANAGE_PY_PATH}"
fi
echo "  DB type:              ${DB_TYPE}"
if [[ "$DB_TYPE" == "postgres" ]]; then
  echo "  DB name/user/pass:    ${DB_NAME} / ${DB_USER} / ${DB_PASS}"
fi
if [[ "$APP_TYPE" == "django" ]]; then
  echo "  Enforcer command:     ${ENFORCER_CMD:-<none>}"
  echo "  Heartbeat command:    ${HEARTBEAT_CMD:-<none>}"
  echo "  APScheduler command:  ${APSCHEDULER_CMD:-<none>} (ENABLED=${APSCHEDULER_ENABLED})"
  echo "  Heartbeat interval:   ${HEARTBEAT_INTERVAL_MINUTES} minute(s)"
fi
echo "  Environments:         $( [[ \"$ENV_DEV\" == true ]] && echo -n 'dev ' )$( [[ \"$ENV_PROD\" == true ]] && echo -n 'prod' )"
if [[ "$ENV_DEV" == true ]]; then
  echo "  Dev Gunicorn port:    ${DEV_GUNICORN_PORT}"
fi
if [[ "$ENV_PROD" == true ]]; then
  echo "  Prod Gunicorn port:   ${PROD_GUNICORN_PORT}"
fi
echo "  Redis:                Installed and enabled (redis-server)"
echo "  ffmpeg:               Installed"
if [[ "$NGINX_INSTALL" == "Y" ]]; then
  echo "  Nginx:                Installed (default site disabled; per-env configs if chosen)"
fi

echo
echo "You can test Django (prod) locally with e.g.:"
if [[ "$ENV_PROD" == true ]]; then
  echo "  curl -I http://127.0.0.1:${PROD_GUNICORN_PORT}/"
fi
echo
echo "If you re-run this script on the same project, it will:"
echo "  - Keep existing DB and project files"
echo "  - Avoid re-adding settings if the marker comment is present"
echo "  - Overwrite systemd and nginx configs with the new choices"
