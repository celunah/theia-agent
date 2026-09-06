# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.11.2 AS uv
FROM python:3.12-slim-bookworm

ARG THEIA_COMMIT=unknown
ARG THEIA_VERSION=1.0.2

LABEL org.opencontainers.image.version="${THEIA_VERSION}"
LABEL org.opencontainers.image.revision="${THEIA_COMMIT}"

ENV HOME=/home/theia \
    PATH=/app/.venv/bin:/app/node_modules/.bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    THEIA_COMMIT=${THEIA_COMMIT} \
    THEIA_HOME=/data/theia \
    THEIA_STATE=/data/theia/sessions.json \
    CODEX_CWD=/workspace \
    CODEX_LOG_COLORS=false

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        ffmpeg \
        libopus0 \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 theia \
    && useradd --create-home --gid 10001 --uid 10001 --shell /usr/sbin/nologin theia

WORKDIR /app

# Keep dependency layers stable when application source changes.
COPY pyproject.toml uv.lock LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund

COPY --chown=theia:theia main.py ./
COPY --chown=theia:theia theia ./theia
RUN printf '%s\n' "${THEIA_COMMIT}" > ./theia/build-revision.txt \
    && uv sync --frozen --no-dev \
    && mkdir --parents /data/theia /workspace \
    && chown --recursive theia:theia /data /workspace

VOLUME ["/data", "/workspace"]

USER theia
ENTRYPOINT ["python", "main.py"]
