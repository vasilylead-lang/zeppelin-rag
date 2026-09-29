#!/usr/bin/env bash
# Installs or updates the Telegram bot as a systemd service. Run on the server as root:
#   bash deploy/install.sh
# Expects the repository at /opt/zeppelin-rag and a filled /opt/zeppelin-rag/.env.
set -euo pipefail

APP_DIR=/opt/zeppelin-rag
SERVICE=zeppelin-bot

cd "$APP_DIR"

if [[ ! -f .env ]]; then
    echo "Missing $APP_DIR/.env: copy .env.example and fill in the keys first." >&2
    exit 1
fi

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi

id zeppelin >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin zeppelin
chown -R zeppelin:zeppelin "$APP_DIR"
chmod 600 .env

# Install dependencies and download the embedding model once, as the service user.
runuser -u zeppelin -- env HOME=/home/zeppelin FASTEMBED_CACHE_PATH="$APP_DIR/.cache/fastembed" \
    /usr/local/bin/uv sync --locked --no-dev
runuser -u zeppelin -- env HOME=/home/zeppelin FASTEMBED_CACHE_PATH="$APP_DIR/.cache/fastembed" \
    /usr/local/bin/uv run --locked --no-dev python -c \
    "from fastembed import TextEmbedding; from zeppelin_rag.config import Settings; TextEmbedding(Settings().embedding_model)"

cp deploy/$SERVICE.service /etc/systemd/system/$SERVICE.service
systemctl daemon-reload
systemctl enable --now $SERVICE
systemctl restart $SERVICE
systemctl --no-pager status $SERVICE | head -n 12
