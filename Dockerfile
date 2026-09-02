# =============================================================================
# BotTrade — Production container
# =============================================================================
# Build:   docker build -t bottrade:latest .
# Run:     docker run -p 8501:8501 --env-file .env -v $(pwd)/data:/app/data \
#              -v $(pwd)/logs:/app/logs bottrade:latest
# Compose: docker compose up -d
# =============================================================================

FROM python:3.11-slim AS base

# --- System deps ---
# build-essential is needed by some wheels (chromadb / sentence-transformers
# fall back to source build on platforms without prebuilt wheels).
# curl is for the HEALTHCHECK; tini is a lightweight init that reaps
# orphaned children spawned by Streamlit's hot-reloader.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

WORKDIR /app

# --- Python deps (cached layer) ---
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- App code ---
COPY . /app

# Create non-root user with the same UID we expect from a typical host bind
# mount. Switching to it after copying keeps the build fast.
RUN chmod +x /app/docker/entrypoint.sh \
    && useradd -ms /bin/bash --uid 1000 bottrade \
    && mkdir -p /app/data /app/logs \
    && chown -R bottrade:bottrade /app
USER bottrade

EXPOSE 8501

# Streamlit-specific runtime tuning
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Healthcheck — Streamlit exposes a 200 on /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# tini handles signal forwarding cleanly so docker stop doesn't hang.
# entrypoint.sh runs first — on most hosts (docker-compose with the secrets
# bind-mount) it's a no-op and falls straight through to CMD; on a host with
# no way to bind-mount a host file (Fly, Railway, Render) it materialises
# .streamlit/secrets.toml from BOTTRADE_STREAMLIT_SECRETS first. See
# docker/entrypoint.sh and DEPLOY.md section 2b.
ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/entrypoint.sh"]

# `python -X utf8` keeps Hebrew / emoji clean on the file paths during ingest.
CMD ["python", "-X", "utf8", "-m", "streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
