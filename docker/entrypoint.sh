#!/bin/sh
# =============================================================================
# BotTrade container entrypoint.
#
# Materialises .streamlit/secrets.toml from an env var, then execs the real
# command. Exists for hosts with no bind-mount from the machine you're
# deploying from — Fly, Railway, Render — where there is no host filesystem
# to mount .streamlit/secrets.toml out of the way docker-compose does it.
#
# docker-compose is unaffected: it doesn't set BOTTRADE_STREAMLIT_SECRETS, so
# this is a no-op there and the existing bind-mount (docker-compose.yml, the
# commented-out `.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro`
# line) keeps working exactly as before.
#
# Usage (Fly):
#   fly secrets set BOTTRADE_STREAMLIT_SECRETS="$(cat .streamlit/secrets.toml)"
# =============================================================================
set -eu

# Overridable so tests/test_entrypoint.py can point this at a scratch
# directory instead of the real /app. Production never sets it, so this
# defaults to exactly the path the app expects.
APP_DIR="${BOTTRADE_APP_DIR:-/app}"

if [ -n "${BOTTRADE_STREAMLIT_SECRETS:-}" ]; then
    mkdir -p "$APP_DIR/.streamlit"
    # Written with 600 perms before content lands, not after — a reader
    # racing the write should never see the file world-readable even briefly.
    install -m 600 /dev/null "$APP_DIR/.streamlit/secrets.toml"
    printf '%s' "$BOTTRADE_STREAMLIT_SECRETS" > "$APP_DIR/.streamlit/secrets.toml"
fi

exec "$@"
