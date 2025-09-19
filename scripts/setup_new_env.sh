#!/usr/bin/env bash
# make it executable, and run it from the project root
set -euo pipefail

# ────────────────────────────────────────────────────────────────────────────────
# Pretty output
# ────────────────────────────────────────────────────────────────────────────────
BOLD="\033[1m"; DIM="\033[2m"; RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"; BLUE="\033[34m"; NC="\033[0m"
info(){ echo -e "${BLUE}ℹ${NC} $*"; }
ok(){   echo -e "${GREEN}✔${NC} $*"; }
warn(){ echo -e "${YELLOW}⚠${NC} $*"; }
err(){  echo -e "${RED}✖${NC} $*"; }

# ────────────────────────────────────────────────────────────────────────────────
# Usage / flags
# ────────────────────────────────────────────────────────────────────────────────
usage(){
cat <<'USAGE'
Guided setup for BISK_RFv4 on Ubuntu 24.04

ENV OVERRIDES (optional):
  PYTHON_BIN=python3.12       # which python to use to create venv
  VENV_DIR=.venv              # virtualenv path
  REQS_FILE=requirements.txt  # requirements file to install
  CPU_ORT_PIN=onnxruntime==1.22.0    # CPU fallback if no GPU present
  RUN_DJANGO=true             # run check/makemigrations/migrate
  DJANGO_SETTINGS_MODULE=bisk.settings
  FILE_UPLOAD_TEMP_DIR=<auto> # defaults to ./media/upload_tmp
  INSTALL_ENFORCER=false      # run deploy/systemd/install_enforcer.sh
  CONFIGURE_REDIS=true        # ensure Redis listens on 127.0.0.1:6379
  REDIS_DB=1                  # Django cache DB index

Examples:
  ./scripts/setup_new_env.sh
  PYTHON_BIN=python3.12 RUN_DJANGO=false ./scripts/setup_new_env.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi

# ────────────────────────────────────────────────────────────────────────────────
# Defaults (overridable by ENV)
# ────────────────────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv}"
REQS_FILE="${REQS_FILE:-requirements.txt}"
CPU_ORT_PIN="${CPU_ORT_PIN:-onnxruntime==1.22.0}"
RUN_DJANGO="${RUN_DJANGO:-true}"
DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-bisk.settings}"
FILE_UPLOAD_TEMP_DIR="${FILE_UPLOAD_TEMP_DIR:-$(pwd)/media/upload_tmp}"
INSTALL_ENFORCER="${INSTALL_ENFORCER:-false}"
ENFORCER_SCRIPT="${ENFORCER_SCRIPT:-$(pwd)/deploy/systemd/install_enforcer.sh}"
CONFIGURE_REDIS="${CONFIGURE_REDIS:-true}"
REDIS_DB="${REDIS_DB:-1}"

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
ask(){
  # ask "Question" "default" -> sets REPLY
  local q="$1"; local d="$2"
  read -r -p "$(echo -e "${BOLD}?${NC} ${q} ${DIM}[${d}]${NC} ")" REPLY
  if [[ -z "$REPLY" ]]; then REPLY="$d"; fi
}

require_file(){
  [[ -f "$1" ]] || { err "Missing file: $1"; exit 1; }
}

require_cmd(){
  command -v "$1" >/dev/null 2>&1 || { err "Missing command: $1"; exit 1; }
}

# ────────────────────────────────────────────────────────────────────────────────
# Preflight: project root check
# ────────────────────────────────────────────────────────────────────────────────
require_file "manage.py"
require_file "$REQS_FILE"

echo -e "${BOLD}BISK_RFv4 Guided Setup${NC}"
info "Project root: $(pwd)"
info "Using Python: ${PYTHON_BIN}"
info "Venv path   : ${VENV_DIR}"
info "Reqs file   : ${REQS_FILE}"

# Interactive confirmations (only if running on a TTY)
if [[ -t 0 ]]; then
  ask "Create or reuse virtualenv at ${VENV_DIR}?" "yes"
  VENV_CREATE="$REPLY"
  ask "Install/enable Redis and ensure it listens on 127.0.0.1:6379?" "${CONFIGURE_REDIS}"
  CONFIGURE_REDIS="$REPLY"
  ask "Run Django check/makemigrations/migrate?" "${RUN_DJANGO}"
  RUN_DJANGO="$REPLY"
  ask "Seed 1 active GlobalResourceSettings if none exists?" "yes"
  SEED_GRS="$REPLY"
  ask "Install/enable systemd enforcer after migrate?" "${INSTALL_ENFORCER}"
  INSTALL_ENFORCER="$REPLY"
fi

# ────────────────────────────────────────────────────────────────────────────────
# System packages
# ────────────────────────────────────────────────────────────────────────────────
info "Installing required system packages (g++, ffmpeg/ffprobe, redis, etc.)"
sudo apt update -y
sudo apt install -y \
  build-essential g++ make cmake pkg-config \
  $([ -x /usr/bin/python3.12 ] || echo python3.12) \
  python3.12-venv python3.12-dev \
  libgl1 libglib2.0-0 \
  ffmpeg git curl

# Redis
if [[ "$CONFIGURE_REDIS" == "yes" || "$CONFIGURE_REDIS" == "true" ]]; then
  sudo apt install -y redis-server
  # Force TCP 6379 on localhost (Ubuntu’s default may be socket-only)
  sudo sed -i 's/^port .*/port 6379/' /etc/redis/redis.conf
  sudo sed -i 's/^# *bind .*/bind 127.0.0.1 ::1/' /etc/redis/redis.conf
  sudo sed -i 's/^supervised .*/supervised systemd/' /etc/redis/redis.conf
  sudo systemctl enable --now redis-server
  sudo systemctl restart redis-server
  if redis-cli -n "$REDIS_DB" ping | grep -q PONG; then
    ok "Redis on 127.0.0.1:6379 (DB ${REDIS_DB}) is responding"
  else
    warn "Redis did not respond on DB ${REDIS_DB}; check: sudo journalctl -u redis-server -e"
  fi
else
  warn "Skipping Redis setup by request. Make sure it’s reachable at 127.0.0.1:6379."
fi

# ffprobe sanity (your settings check this)
require_cmd ffprobe
ok "ffprobe present: $(command -v ffprobe)"

# ────────────────────────────────────────────────────────────────────────────────
# Application filesystem
# ────────────────────────────────────────────────────────────────────────────────
info "Ensuring FILE_UPLOAD_TEMP_DIR exists → ${FILE_UPLOAD_TEMP_DIR}"
mkdir -p "${FILE_UPLOAD_TEMP_DIR}"
sudo chown -R "${SUDO_USER:-$USER}:${SUDO_USER:-$USER}" "$(pwd)/media" || true
ok "media/ ready"

# Optional lock dir for enforcer
sudo mkdir -p /run/bisk || true
sudo chown -R "${SUDO_USER:-$USER}:${SUDO_USER:-$USER}" /run/bisk || true

# ────────────────────────────────────────────────────────────────────────────────
# Virtualenv
# ────────────────────────────────────────────────────────────────────────────────
if [[ "${VENV_CREATE,,}" == "yes" || "${VENV_CREATE,,}" == "y" ]]; then
  info "Creating venv at ${VENV_DIR} if missing"
  [[ -d "$VENV_DIR" ]] || $PYTHON_BIN -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
ok "venv activated: $(python -V)"

python -m pip install -U pip setuptools wheel

# ────────────────────────────────────────────────────────────────────────────────
# GPU detection → decide ORT flavor
# ────────────────────────────────────────────────────────────────────────────────
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_GPU=1
fi
info "NVIDIA GPU detected? ${HAS_GPU}"

# Prepare requirements file (swap ORT to CPU if no GPU)
REQS_TMP="$(mktemp)"
cp "$REQS_FILE" "$REQS_TMP"
if [[ "$HAS_GPU" -eq 0 ]]; then
  info "No GPU → replace any onnxruntime-gpu pin with ${CPU_ORT_PIN}"
  grep -vi '^onnxruntime-gpu' "$REQS_TMP" > "${REQS_TMP}.cpu" || true
  echo "$CPU_ORT_PIN" >> "${REQS_TMP}.cpu"
  mv "${REQS_TMP}.cpu" "$REQS_TMP"
fi

info "Installing Python packages (this may take a while)…"
python -m pip install -r "$REQS_TMP"
ok "Python dependencies installed"

# ────────────────────────────────────────────────────────────────────────────────
# InsightFace sanity (CPU/GPU)
# ────────────────────────────────────────────────────────────────────────────────
info "Running InsightFace sanity check…"
python - <<'PY' || warn "InsightFace sanity check had warnings; continue if expected."
import subprocess, sys
try:
    import onnxruntime as ort
    print("onnxruntime:", ort.__version__, "providers:", ort.get_available_providers())
    from insightface.app import FaceAnalysis
    has_gpu = subprocess.call("nvidia-smi -L >/dev/null 2>&1", shell=True) == 0
    ctx = 0 if has_gpu else -1
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=ctx)
    print("InsightFace model prepared on", "GPU" if has_gpu else "CPU")
except Exception as e:
    print("Sanity check warning:", e, file=sys.stderr)
PY
ok "Sanity check done"

# ────────────────────────────────────────────────────────────────────────────────
# Django: check / migrate / seed
# ────────────────────────────────────────────────────────────────────────────────
if [[ "${RUN_DJANGO,,}" == "true" || "${RUN_DJANGO,,}" == "yes" ]]; then
  info "Django checks and migrations…"
  export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE}"
  # Custom checks are migration-safe (your guarded checks.py)
  python manage.py check || true
  python manage.py makemigrations
  python manage.py migrate

  if [[ "${SEED_GRS,,}" == "yes" || "${SEED_GRS,,}" == "true" ]]; then
    info "Seeding exactly one active GlobalResourceSettings (if none)…"
    python manage.py shell -c "
from django.db import transaction
from apps.scheduler.models import GlobalResourceSettings
with transaction.atomic():
    qs = GlobalResourceSettings.objects.all()
    if qs.filter(is_active=True).count() > 1:
        qs.update(is_active=False)
    if not qs.filter(is_active=True).exists():
        obj = qs.first() or GlobalResourceSettings()
        obj.is_active = True
        obj.save()
print('OK: exactly one active GlobalResourceSettings is present.')
"
  fi
fi

# ────────────────────────────────────────────────────────────────────────────────
# Optional: systemd enforcer
# ────────────────────────────────────────────────────────────────────────────────
if [[ "${INSTALL_ENFORCER,,}" == "true" || "${INSTALL_ENFORCER,,}" == "yes" ]]; then
  if [[ -x "$ENFORCER_SCRIPT" ]]; then
    info "Installing/starting systemd enforcer via ${ENFORCER_SCRIPT}"
    sudo "$ENFORCER_SCRIPT" || warn "Enforcer installer returned non-zero"
    sudo systemctl status bisk-enforcer.service --no-pager || true
  else
    warn "ENFORCER_SCRIPT not found or not executable: ${ENFORCER_SCRIPT}"
  fi
fi

echo
ok "All done!"
echo -e "Next steps:\n  ${DIM}source ${VENV_DIR}/bin/activate${NC}\n  python manage.py runserver 0.0.0.0:8000"
