#!/usr/bin/env bash
set -euo pipefail

# --- inputs (change or pass as env) ---
USER_NAME="${USER_NAME:-$USER}"  # default to current user
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
ENV_FILE="${ENV_FILE:-/etc/bisk/bisk-enforcer.env}"
SERVICE_FILE="/etc/systemd/system/bisk-enforcer.service"

# Try to auto-detect venv python
if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  VENV_PY="${PROJECT_DIR}/.venv/bin/python"
else
  # last resort: system python in PATH (not recommended)
  VENV_PY="$(command -v python3)"
fi

echo "User:        ${USER_NAME}"
echo "Project dir: ${PROJECT_DIR}"
echo "Python:      ${VENV_PY}"
echo "Env file:    ${ENV_FILE}"
read -p "Proceed? [y/N] " YN
[[ "${YN}" == "y" || "${YN}" == "Y" ]] || exit 1

sudo mkdir -p /etc/bisk

# If no env file exists, seed from repo example (you can edit later)
if [[ ! -f "${ENV_FILE}" ]]; then
  if [[ -f "${PROJECT_DIR}/deploy/systemd/bisk-enforcer.env.example" ]]; then
    sudo cp "${PROJECT_DIR}/deploy/systemd/bisk-enforcer.env.example" "${ENV_FILE}"
  else
    # minimal env
    sudo tee "${ENV_FILE}" >/dev/null <<EOF
DJANGO_SETTINGS_MODULE=bisk.settings
PYTHONUNBUFFERED=1
TZ=Asia/Baghdad
ENFORCER_INTERVAL_SECONDS=15
ENFORCER_LOCK_FILE=/tmp/bisk_enforcer.lock
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin
EOF
  fi
  echo "Wrote ${ENV_FILE} (edit if you need)."
fi

# Render service from template
TMP_SERVICE="$(mktemp)"
sed \
  -e "s|__USER__|${USER_NAME}|g" \
  -e "s|__PROJECT_DIR__|${PROJECT_DIR}|g" \
  -e "s|__PYTHON__|${VENV_PY}|g" \
  "${PROJECT_DIR}/deploy/systemd/bisk-enforcer.service.template" > "${TMP_SERVICE}"

sudo mv "${TMP_SERVICE}" "${SERVICE_FILE}"
sudo chmod 644 "${SERVICE_FILE}"

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now bisk-enforcer.service

echo "Installed. Check status with: sudo systemctl status bisk-enforcer.service"
