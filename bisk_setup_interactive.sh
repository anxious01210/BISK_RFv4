#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# BISK_RFv4 Interactive Setup Script (Ubuntu 22.04/24.04)
# - Installs & configures: Redis, Gunicorn systemd, Enforcer, Timers
# - Optional: Nginx site, PostgreSQL provisioning, Debug collector, Aliases
# - Safe to re-run: backs up files it edits
# ------------------------------------------------------------
# How to run the setu script:
# 	chmod +x bisk_setup_interactive.sh
# 	sudo ./bisk_setup_interactive.sh
# ------------------------------------------------------------

# ---------- helpers ----------
bold(){ echo -e "\033[1m$*\033[0m"; }
ok(){ echo -e "\033[32m[OK]\033[0m $*"; }
warn(){ echo -e "\033[33m[WARN]\033[0m $*"; }
err(){ echo -e "\033[31m[ERR]\033[0m $*" >&2; }
die(){ err "$*"; exit 1; }

have(){ command -v "$1" >/dev/null 2>&1; }

is_root(){ [[ "${EUID:-$(id -u)}" -eq 0 ]]; }

backup_file(){
  local f="$1"
  if [[ -f "$f" ]]; then
    local ts; ts="$(date +%Y%m%d-%H%M%S)"
    cp -a "$f" "${f}.bak.${ts}"
    ok "Backup: ${f} -> ${f}.bak.${ts}"
  fi
}

prompt_yn(){
  local prompt="$1" default="$2" ans
  while true; do
    read -r -p "${prompt} [${default}] " ans || true
    ans="${ans:-$default}"
    case "${ans,,}" in
      y|yes) echo "y"; return 0 ;;
      n|no)  echo "n"; return 0 ;;
      *) echo "Please answer y/n." ;;
    esac
  done
}

prompt_str(){
  local prompt="$1" default="$2" ans
  read -r -p "${prompt} [${default}] " ans || true
  echo "${ans:-$default}"
}

prompt_int(){
  local prompt="$1" default="$2" ans
  while true; do
    read -r -p "${prompt} [${default}] " ans || true
    ans="${ans:-$default}"
    [[ "$ans" =~ ^[0-9]+$ ]] && { echo "$ans"; return 0; }
    echo "Please enter a number."
  done
}

ensure_dir(){
  local d="$1" mode="${2:-755}" owner="${3:-root:root}"
  mkdir -p "$d"
  chmod "$mode" "$d" || true
  chown "$owner" "$d" || true
}

apt_install(){
  local pkgs=("$@")
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y "${pkgs[@]}"
}

systemd_reload(){ systemctl daemon-reload; }

systemd_enable_start(){
  local unit="$1"
  systemctl enable "$unit"
  systemctl restart "$unit" || systemctl start "$unit"
}

systemd_enable_start_timer(){
  local timer="$1"
  systemctl enable "$timer"
  systemctl start "$timer"
}

write_file(){
  local path="$1" content="$2"
  ensure_dir "$(dirname "$path")" 755 root:root
  backup_file "$path"
  printf "%s" "$content" > "$path"
  ok "Wrote: $path"
}

# ---------- preflight ----------
is_root || die "Please run as root: sudo $0"

bold "------------------------------------------------------------"
bold "BISK_RFv4 Interactive Setup"
bold "------------------------------------------------------------"

# ---------- defaults ----------
DEFAULT_PROJECT_ROOT="/srv/BISK_RFv4"
DEFAULT_RUN_USER="bisk"
DEFAULT_RUN_GROUP="bisk"
DEFAULT_VENV_REL=".venv"
DEFAULT_DJANGO_SETTINGS_MODULE="bisk.settings"

DEFAULT_BIND_HOST="127.0.0.1"
DEFAULT_BIND_PORT="8001"
DEFAULT_GUNICORN_WORKERS="3"
DEFAULT_GUNICORN_TIMEOUT="180"

DEFAULT_REDIS_PORT="6379"
DEFAULT_REDIS_DB="1"
DEFAULT_REDIS_BINDS="127.0.0.1 ::1"

DEFAULT_TZ="Asia/Baghdad"

# ---------- prompts ----------
PROJECT_ROOT="$(prompt_str "Project root path" "$DEFAULT_PROJECT_ROOT")"
RUN_USER="$(prompt_str "Linux service user" "$DEFAULT_RUN_USER")"
RUN_GROUP="$(prompt_str "Linux service group" "$DEFAULT_RUN_GROUP")"
VENV_PATH="${PROJECT_ROOT}/$(prompt_str "Virtualenv folder (relative to project root)" "$DEFAULT_VENV_REL")"
DJANGO_SETTINGS_MODULE="$(prompt_str "DJANGO_SETTINGS_MODULE" "$DEFAULT_DJANGO_SETTINGS_MODULE")"

ENABLE_POSTGRES="$(prompt_yn "Provision PostgreSQL locally (optional)?" "n")"
ENABLE_REDIS="$(prompt_yn "Install & configure Redis (recommended; used by Django cache)?" "y")"
ENABLE_NGINX="$(prompt_yn "Install & configure Nginx site (optional)?" "y")"
ENABLE_SYSTEMD_UNITS="$(prompt_yn "Install/Update BISK systemd services & timers?" "y")"
ENABLE_DEBUG_COLLECTOR="$(prompt_yn "Install bisk_collect_debug.sh command in /usr/local/bin?" "y")"
ENABLE_ALIASES="$(prompt_yn "Install helpful aliases into /etc/profile.d/bisk.sh (system-wide)?" "n")"

BIND_HOST="$(prompt_str "Gunicorn bind host" "$DEFAULT_BIND_HOST")"
BIND_PORT="$(prompt_str "Gunicorn bind port" "$DEFAULT_BIND_PORT")"
GUNICORN_WORKERS="$(prompt_int "Gunicorn workers" "$DEFAULT_GUNICORN_WORKERS")"
GUNICORN_TIMEOUT="$(prompt_int "Gunicorn timeout (seconds)" "$DEFAULT_GUNICORN_TIMEOUT")"

TZ="$(prompt_str "Server timezone (IANA name)" "$DEFAULT_TZ")"

# Redis prompts
REDIS_BINDS="$DEFAULT_REDIS_BINDS"
REDIS_PORT="$DEFAULT_REDIS_PORT"
REDIS_DB="$DEFAULT_REDIS_DB"
REDIS_REQUIREPASS=""
if [[ "$ENABLE_REDIS" == "y" ]]; then
  REDIS_BINDS="$(prompt_str "Redis bind addresses (space-separated)" "$DEFAULT_REDIS_BINDS")"
  REDIS_PORT="$(prompt_int "Redis port" "$DEFAULT_REDIS_PORT")"
  REDIS_DB="$(prompt_int "Redis DB index for BISK (0-15)" "$DEFAULT_REDIS_DB")"

  # If any bind is non-local, strongly recommend requirepass
  if echo "$REDIS_BINDS" | grep -Eqv '(^| )127\.0\.0\.1( |$)|(^| )::1( |$)'; then
    warn "You selected non-local bind(s). Exposing Redis on LAN can be dangerous."
    if [[ "$(prompt_yn "Require a Redis password (requirepass)?" "y")" == "y" ]]; then
      REDIS_REQUIREPASS="$(prompt_str "Redis password (stored in redis.conf)" "CHANGE_ME_STRONG_PASSWORD")"
    else
      warn "Proceeding WITHOUT Redis password. Ensure firewall restrictions!"
    fi
  fi
fi

# Nginx prompts
NGINX_SERVER_NAME=""
NGINX_SITE_NAME="bisk_rfv4"
NGINX_CLIENT_MAX_BODY="100m"
if [[ "$ENABLE_NGINX" == "y" ]]; then
  NGINX_SERVER_NAME="$(prompt_str "Nginx server_name (domain or IP; use '_' if unsure)" "_")"
  NGINX_SITE_NAME="$(prompt_str "Nginx site name (filename)" "$NGINX_SITE_NAME")"
  NGINX_CLIENT_MAX_BODY="$(prompt_str "Nginx client_max_body_size" "$NGINX_CLIENT_MAX_BODY")"
fi

bold ""
bold "Summary"
echo "  Project root:        $PROJECT_ROOT"
echo "  Service user/group:  $RUN_USER:$RUN_GROUP"
echo "  Virtualenv:          $VENV_PATH"
echo "  Gunicorn bind:       ${BIND_HOST}:${BIND_PORT}"
echo "  Gunicorn workers:    $GUNICORN_WORKERS"
echo "  Gunicorn timeout:    $GUNICORN_TIMEOUT"
echo "  Timezone:            $TZ"
echo "  Redis enabled:       $ENABLE_REDIS"
if [[ "$ENABLE_REDIS" == "y" ]]; then
  echo "    Redis bind:        $REDIS_BINDS"
  echo "    Redis port:        $REDIS_PORT"
  echo "    Redis db:          $REDIS_DB"
fi
echo "  Postgres provision:  $ENABLE_POSTGRES"
echo "  Nginx enabled:       $ENABLE_NGINX"
echo "  Systemd units:       $ENABLE_SYSTEMD_UNITS"
echo "  Debug collector:     $ENABLE_DEBUG_COLLECTOR"
echo "  Aliases:             $ENABLE_ALIASES"
bold ""

[[ "$(prompt_yn "Proceed with these actions?" "y")" == "y" ]] || die "Aborted."

# ---------- step 1: base packages ----------
bold "Step 1) Installing base packages"
apt_install ca-certificates curl git rsync unzip
apt_install python3 python3-venv python3-pip
apt_install build-essential pkg-config
apt_install sysstat iotop || true
[[ "$ENABLE_NGINX" == "y" ]] && apt_install nginx
[[ "$ENABLE_REDIS" == "y" ]] && apt_install redis-server
ok "Base packages installed."

# ---------- step 2: timezone ----------
bold "Step 2) Timezone"
if have timedatectl; then
  timedatectl set-timezone "$TZ" || true
  ok "Timezone set to $TZ"
else
  warn "timedatectl not found; skipping timezone setup."
fi

# ---------- step 3: user/group & folders ----------
bold "Step 3) User/group & folders"
getent group "$RUN_GROUP" >/dev/null || { groupadd --system "$RUN_GROUP"; ok "Created group: $RUN_GROUP"; }
id -u "$RUN_USER" >/dev/null 2>&1 || { useradd --system --gid "$RUN_GROUP" --home-dir "$PROJECT_ROOT" --shell /usr/sbin/nologin "$RUN_USER"; ok "Created user: $RUN_USER"; }

ensure_dir "$PROJECT_ROOT" 755 "$RUN_USER:$RUN_GROUP"
ensure_dir "$PROJECT_ROOT/media" 775 "$RUN_USER:$RUN_GROUP"
ensure_dir "$PROJECT_ROOT/media/logs/systemd" 775 "$RUN_USER:$RUN_GROUP"
ensure_dir "/etc/bisk" 755 root:root
ensure_dir "/var/log/bisk" 755 "$RUN_USER:$RUN_GROUP"

# /run is tmpfs; ensure on boot
write_file "/etc/tmpfiles.d/bisk.conf" "d /run/bisk 0755 ${RUN_USER} ${RUN_GROUP} -\n"
systemd-tmpfiles --create /etc/tmpfiles.d/bisk.conf || true
ensure_dir "/run/bisk" 755 "$RUN_USER:$RUN_GROUP"
ok "Folders prepared."

# ---------- step 4: python venv & deps ----------
bold "Step 4) Python venv & dependencies"
[[ -d "$PROJECT_ROOT" ]] || die "Project root not found: $PROJECT_ROOT"
[[ -f "${PROJECT_ROOT}/manage.py" ]] || warn "manage.py not found in $PROJECT_ROOT (clone/copy project first if this is a new server)."

if [[ ! -d "$VENV_PATH" ]]; then
  python3 -m venv "$VENV_PATH"
  ok "Created venv: $VENV_PATH"
else
  ok "Venv exists: $VENV_PATH"
fi

VENV_PY="${VENV_PATH}/bin/python"
VENV_PIP="${VENV_PATH}/bin/pip"
"$VENV_PIP" install -U pip wheel setuptools

if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  "$VENV_PIP" install -r "${PROJECT_ROOT}/requirements.txt"
  ok "Installed requirements.txt"
else
  warn "requirements.txt not found. Skipping pip install -r."
fi

# ---------- step 5: PostgreSQL (optional) ----------
if [[ "$ENABLE_POSTGRES" == "y" ]]; then
  bold "Step 5) PostgreSQL provisioning"
  apt_install postgresql postgresql-contrib
  systemctl enable postgresql
  systemctl start postgresql

  DB_NAME="$(prompt_str "Postgres DB name" "bisk_rfv4")"
  DB_USER="$(prompt_str "Postgres DB user" "bisk_rfv4")"
  DB_PASS="$(prompt_str "Postgres DB password" "CHANGE_ME_STRONG_PASSWORD")"

  sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1 || sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';"
  sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 || sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"
  ok "Postgres ensured: db=${DB_NAME} user=${DB_USER}"
fi

# ---------- step 6: Redis (recommended) ----------
if [[ "$ENABLE_REDIS" == "y" ]]; then
  bold "Step 6) Redis configuration"
  CONF="/etc/redis/redis.conf"
  [[ -f "$CONF" ]] || die "Redis config not found at $CONF"

  backup_file "$CONF"
  sed -i -E "s/^port[[:space:]]+[0-9]+/port ${REDIS_PORT}/" "$CONF"
  if grep -qE "^[[:space:]]*bind[[:space:]]+" "$CONF"; then
    sed -i -E "s/^[[:space:]]*bind[[:space:]].*/bind ${REDIS_BINDS}/" "$CONF"
  else
    sed -i "1i bind ${REDIS_BINDS}\n" "$CONF"
  fi
  if grep -qE "^[[:space:]]*protected-mode[[:space:]]+" "$CONF"; then
    sed -i -E "s/^[[:space:]]*protected-mode[[:space:]]+.*/protected-mode yes/" "$CONF"
  else
    echo "protected-mode yes" >> "$CONF"
  fi

  if [[ -n "$REDIS_REQUIREPASS" ]]; then
    if grep -qE "^[[:space:]]*requirepass[[:space:]]+" "$CONF"; then
      sed -i -E "s/^[[:space:]]*requirepass[[:space:]]+.*/requirepass ${REDIS_REQUIREPASS}/" "$CONF"
    else
      echo "requirepass ${REDIS_REQUIREPASS}" >> "$CONF"
    fi
    warn "Redis requirepass set in redis.conf (plain text)."
  fi

  systemctl enable redis-server || true
  systemctl restart redis-server || true

  if have redis-cli; then
    redis-cli -p "$REDIS_PORT" ping | grep -q PONG && ok "Redis responds (PONG)" || warn "Redis did not respond to PING."
  fi
else
  warn "Redis setup skipped. NOTE: your Django uses RedisCache; skipping may break cache/locks unless you change settings.py."
fi

# ---------- step 7: Nginx (optional) ----------
if [[ "$ENABLE_NGINX" == "y" ]]; then
  bold "Step 7) Nginx site"
  systemctl enable nginx
  systemctl start nginx

  STATIC_ROOT="${PROJECT_ROOT}/staticfiles"
  MEDIA_ROOT="${PROJECT_ROOT}/media"

  CONF="/etc/nginx/sites-available/${NGINX_SITE_NAME}"
  write_file "$CONF" "server {
    listen 80;
    server_name ${NGINX_SERVER_NAME};

    client_max_body_size ${NGINX_CLIENT_MAX_BODY};

    location /static/ {
        alias ${STATIC_ROOT}/;
        access_log off;
        expires 30d;
        add_header Cache-Control \"public, max-age=2592000\";
    }

    location /media/ {
        alias ${MEDIA_ROOT}/;
        expires 7d;
        add_header Cache-Control \"public, max-age=604800\";
    }

    location / {
        proxy_pass http://${BIND_HOST}:${BIND_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 30;
        proxy_send_timeout 300;
    }
}
"
  ln -sf "$CONF" "/etc/nginx/sites-enabled/${NGINX_SITE_NAME}"
  [[ -e /etc/nginx/sites-enabled/default ]] && rm -f /etc/nginx/sites-enabled/default || true

  nginx -t
  systemctl reload nginx
  ok "Nginx configured: ${NGINX_SITE_NAME}"
else
  ok "Nginx skipped."
fi

# ---------- step 8: systemd units ----------
if [[ "$ENABLE_SYSTEMD_UNITS" == "y" ]]; then
  bold "Step 8) Systemd services & timers"

  # env files
  if [[ ! -f /etc/bisk/bisk-web.env ]]; then
    write_file "/etc/bisk/bisk-web.env" "DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE}\nBISK_ENV=prod\n"
    chmod 600 /etc/bisk/bisk-web.env
  fi

  if [[ ! -f /etc/bisk/bisk-enforcer.env ]]; then
    write_file "/etc/bisk/bisk-enforcer.env" "DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE}\nBISK_ENV=prod\nENFORCER_LOCK_FILE=/run/bisk/enforcer.lock\nHEARTBEAT_KEY=CHANGE_ME_HEARTBEAT_KEY\nHEARTBEAT_URL=http://${BIND_HOST}:${BIND_PORT}/api/heartbeat/\n"
    chmod 600 /etc/bisk/bisk-enforcer.env
  fi

  LOGDIR="${PROJECT_ROOT}/media/logs/systemd"
  ensure_dir "$LOGDIR" 775 "$RUN_USER:$RUN_GROUP"

  # services/timers
  write_file "/etc/systemd/system/bisk-web.service" "[Unit]
Description=BISK_RFv4 (web) Gunicorn
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${PROJECT_ROOT}
EnvironmentFile=-/etc/bisk/bisk-web.env
ExecStart=${VENV_PATH}/bin/gunicorn bisk.wsgi:application --bind ${BIND_HOST}:${BIND_PORT} --workers ${GUNICORN_WORKERS} --timeout ${GUNICORN_TIMEOUT}
Restart=always
RestartSec=3
StandardOutput=append:${LOGDIR}/web.out.log
StandardError=append:${LOGDIR}/web.err.log

[Install]
WantedBy=multi-user.target
"

  write_file "/etc/systemd/system/bisk-enforcer.service" "[Unit]
Description=BISK_RFv4 (enforcer) Scheduler/Orchestrator
After=network.target redis-server.service
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${PROJECT_ROOT}
EnvironmentFile=-/etc/bisk/bisk-enforcer.env
ExecStart=${VENV_PATH}/bin/python ${PROJECT_ROOT}/manage.py enforcer
Restart=always
RestartSec=3
StandardOutput=append:${LOGDIR}/enforcer.out.log
StandardError=append:${LOGDIR}/enforcer.err.log

[Install]
WantedBy=multi-user.target
"

  write_file "/etc/systemd/system/bisk-prune-heartbeats.service" "[Unit]
Description=BISK_RFv4 - Prune stale heartbeats
After=network.target

[Service]
Type=oneshot
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${PROJECT_ROOT}
EnvironmentFile=-/etc/bisk/bisk-web.env
ExecStart=${VENV_PATH}/bin/python ${PROJECT_ROOT}/manage.py prune_heartbeats
"

  write_file "/etc/systemd/system/bisk-prune-heartbeats.timer" "[Unit]
Description=BISK_RFv4 - Prune heartbeats every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
"

  write_file "/etc/systemd/system/bisk-diag.service" "[Unit]
Description=BISK_RFv4 - Diagnostic snapshot

[Service]
Type=oneshot
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${PROJECT_ROOT}
EnvironmentFile=-/etc/bisk/bisk-web.env
ExecStart=${VENV_PATH}/bin/python ${PROJECT_ROOT}/manage.py diag_snapshot
"

  write_file "/etc/systemd/system/bisk-diag.timer" "[Unit]
Description=BISK_RFv4 - Diagnostic snapshot every 15 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min
Persistent=true

[Install]
WantedBy=timers.target
"

  write_file "/etc/systemd/system/bisk-diag-daily.service" "[Unit]
Description=BISK_RFv4 - Daily diagnostic snapshot

[Service]
Type=oneshot
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${PROJECT_ROOT}
EnvironmentFile=-/etc/bisk/bisk-web.env
ExecStart=${VENV_PATH}/bin/python ${PROJECT_ROOT}/manage.py diag_snapshot
"

  write_file "/etc/systemd/system/bisk-diag-daily.timer" "[Unit]
Description=BISK_RFv4 - Daily diagnostic snapshot

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
"

  systemd_reload

  [[ "$(prompt_yn "Enable & start bisk-web.service now?" "y")" == "y" ]] && systemd_enable_start bisk-web.service
  [[ "$(prompt_yn "Enable & start bisk-enforcer.service now?" "y")" == "y" ]] && systemd_enable_start bisk-enforcer.service

  if [[ "$(prompt_yn "Enable & start timers (prune/diag) now?" "y")" == "y" ]]; then
    systemd_enable_start_timer bisk-prune-heartbeats.timer
    systemd_enable_start_timer bisk-diag.timer
    systemd_enable_start_timer bisk-diag-daily.timer
  fi

else
  warn "Systemd units skipped."
fi

# ---------- step 9: debug collector ----------
if [[ "$ENABLE_DEBUG_COLLECTOR" == "y" ]]; then
  bold "Step 9) Debug collector command"
  write_file "/usr/local/bin/bisk_collect_debug.sh" "#!/usr/bin/env bash
set -euo pipefail

ROOT=\"${PROJECT_ROOT}\"
OUT_BASE=\"\${ROOT}/media/logs/diag_bundles\"
TS=\"\$(date +%Y%m%d-%H%M%S)\"
OUT_DIR=\"\${OUT_BASE}/\${TS}\"
mkdir -p \"\${OUT_DIR}\"

uptime > \"\${OUT_DIR}/uptime.txt\" || true
free -h > \"\${OUT_DIR}/free.txt\" || true
df -h > \"\${OUT_DIR}/df.txt\" || true
ps -eo pid,ppid,cmd,%cpu,%mem --sort=-%cpu | head -n 60 > \"\${OUT_DIR}/ps_top_cpu.txt\" || true
ps -eo pid,ppid,cmd,%cpu,%mem --sort=-%mem | head -n 60 > \"\${OUT_DIR}/ps_top_mem.txt\" || true

iostat -xz 1 3 > \"\${OUT_DIR}/iostat_xz.txt\" 2>&1 || true
iotop -b -n 3 -o > \"\${OUT_DIR}/iotop_top_io.txt\" 2>&1 || true
du -h -d 2 \"\${ROOT}/media\" | sort -hr > \"\${OUT_DIR}/du_media_top.txt\" 2>&1 || true

nvidia-smi > \"\${OUT_DIR}/nvidia_smi.txt\" 2>&1 || true
nvidia-smi dmon -s u -c 3 > \"\${OUT_DIR}/nvidia_dmon.txt\" 2>&1 || true

dmesg -T | egrep -i \"killed process|oom|out of memory|segfault|cgroup\" > \"\${OUT_DIR}/dmesg_kills.txt\" || true
dmesg -T | tail -n 400 > \"\${OUT_DIR}/dmesg_tail.txt\" || true

journalctl -u bisk-web.service -n 500 --no-pager > \"\${OUT_DIR}/journal_bisk_web.txt\" || true
journalctl -u bisk-enforcer.service -n 500 --no-pager > \"\${OUT_DIR}/journal_enforcer.txt\" || true
journalctl -u bisk-prune-heartbeats.service -n 200 --no-pager > \"\${OUT_DIR}/journal_prune_heartbeats.txt\" || true

LOGDIR=\"\${ROOT}/media/logs/systemd\"
for f in web.out.log web.err.log gunicorn.access.log gunicorn.error.log enforcer.out.log enforcer.err.log; do
  if [ -f \"\${LOGDIR}/\${f}\" ]; then
    tail -n 2000 \"\${LOGDIR}/\${f}\" > \"\${OUT_DIR}/tail_\${f}.txt\" || true
  fi
done

pgrep -a ffmpeg > \"\${OUT_DIR}/pgrep_ffmpeg.txt\" 2>&1 || true

tail -n 4000 /var/log/nginx/access.log > \"\${OUT_DIR}/nginx_access_tail.txt\" 2>/dev/null || true
tail -n 1000 /var/log/nginx/error.log  > \"\${OUT_DIR}/nginx_error_tail.txt\" 2>/dev/null || true
tail -n 2000 /var/log/nginx/bisk_access.log > \"\${OUT_DIR}/nginx_bisk_access_tail.txt\" 2>/dev/null || true
tail -n 1000 /var/log/nginx/bisk_error.log  > \"\${OUT_DIR}/nginx_bisk_error_tail.txt\" 2>/dev/null || true

\"${VENV_PATH}/bin/python\" \"${PROJECT_ROOT}/manage.py\" check --deploy > \"\${OUT_DIR}/django_check_deploy.txt\" 2>&1 || true

tar -czf \"\${OUT_DIR}.tar.gz\" -C \"\${OUT_BASE}\" \"\${TS}\" || true

find \"\${OUT_BASE}\" -maxdepth 1 -name \"*.tar.gz\" -mtime +7 -delete || true
find \"\${OUT_BASE}\" -mindepth 1 -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \\; || true

echo \"OK: \${OUT_DIR}.tar.gz\"
"
  chmod 755 /usr/local/bin/bisk_collect_debug.sh
  ok "Installed: /usr/local/bin/bisk_collect_debug.sh"
fi

# ---------- step 10: aliases ----------
if [[ "$ENABLE_ALIASES" == "y" ]]; then
  bold "Step 10) Aliases"
  write_file "/etc/profile.d/bisk.sh" "# BISK shortcuts (system-wide)
alias bisk-enable='sudo systemctl enable bisk-web.service bisk-enforcer.service bisk-prune-heartbeats.timer bisk-diag.timer bisk-diag-daily.timer'
alias bisk-disable='sudo systemctl disable bisk-web.service bisk-enforcer.service bisk-prune-heartbeats.timer bisk-diag.timer bisk-diag-daily.timer'
alias bisk-start='sudo systemctl start bisk-web.service bisk-enforcer.service && sudo systemctl start bisk-prune-heartbeats.timer bisk-diag.timer bisk-diag-daily.timer'
alias bisk-stop='sudo systemctl stop bisk-web.service bisk-enforcer.service && sudo systemctl stop bisk-prune-heartbeats.timer bisk-diag.timer bisk-diag-daily.timer'
alias bisk-status='systemctl status bisk-web.service bisk-enforcer.service bisk-prune-heartbeats.timer bisk-diag.timer bisk-diag-daily.timer --no-pager'
"
  chmod 644 /etc/profile.d/bisk.sh
  ok "Aliases installed (new shells will load them)."
fi

bold ""
bold "------------------------------------------------------------"
ok "Setup complete."
bold "Next steps (recommended):"
echo "1) Ensure your project code exists at: $PROJECT_ROOT"
echo "2) Review env files:"
echo "   - /etc/bisk/bisk-web.env"
echo "   - /etc/bisk/bisk-enforcer.env  (SET HEARTBEAT_KEY!)"
echo "3) Run migrations & collectstatic:"
echo "   sudo -u ${RUN_USER} ${VENV_PATH}/bin/python ${PROJECT_ROOT}/manage.py migrate"
echo "   sudo -u ${RUN_USER} ${VENV_PATH}/bin/python ${PROJECT_ROOT}/manage.py collectstatic --noinput"
echo "4) Check services:"
echo "   systemctl status bisk-web.service bisk-enforcer.service --no-pager"
echo "5) Debug bundle:"
echo "   bisk_collect_debug.sh"
bold "------------------------------------------------------------"
