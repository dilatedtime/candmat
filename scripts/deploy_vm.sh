#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/candmat}"
BRANCH="${BRANCH:-main}"
REPO_URL="${REPO_URL:-https://github.com/dilatedtime/candmat.git}"
SERVICE_NAME="${SERVICE_NAME:-candmat-sandbox}"
PORT="${PORT:-8501}"
RUN_USER="$(id -un)"

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git python3 python3-venv python3-pip curl

if [ ! -d "$APP_DIR/.git" ]; then
  sudo rm -rf "$APP_DIR"
  sudo mkdir -p "$APP_DIR"
  sudo chown "$RUN_USER:$RUN_USER" "$APP_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
else
  sudo chown -R "$RUN_USER:$RUN_USER" "$APP_DIR"
  git -C "$APP_DIR" remote set-url origin "$REPO_URL"
  git -C "$APP_DIR" fetch origin "$BRANCH"
  git -C "$APP_DIR" checkout "$BRANCH"
  git -C "$APP_DIR" reset --hard "origin/$BRANCH"
fi

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/python" -m pip install --upgrade pip
"$APP_DIR/.venv/bin/python" -m pip install -r "$APP_DIR/sandbox/requirements.txt"

sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<SERVICE
[Unit]
Description=candmat Streamlit sandbox
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${APP_DIR}
Environment=PATH=${APP_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${APP_DIR}/.venv/bin/streamlit run sandbox/app.py --server.address 0.0.0.0 --server.port ${PORT} --server.headless true --browser.gatherUsageStats false
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/_stcore/health" >/dev/null; then
    sudo systemctl --no-pager --full status "$SERVICE_NAME"
    exit 0
  fi
  sleep 2
done

sudo journalctl -u "$SERVICE_NAME" -n 120 --no-pager
exit 1
