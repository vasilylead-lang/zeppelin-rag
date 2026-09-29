FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.14 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

# Dependencies first, so code and knowledge-base edits reuse this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY data ./data
RUN uv sync --locked --no-dev

# Bake the embedding model (~220 MB) into the image instead of downloading it on start.
RUN uv run --no-sync python -c \
    "from fastembed import TextEmbedding; from zeppelin_rag.config import Settings; TextEmbedding(Settings().embedding_model)"

# Secrets (.env) are passed at run time, never copied into the image.
CMD ["uv", "run", "--no-sync", "zeppelin-bot"]
