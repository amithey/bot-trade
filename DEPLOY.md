# BotTrade — Deployment Guide

This guide covers running BotTrade outside your laptop: Docker, cloud
hosts, password auth, and crash reporting. For local dev, the
`README.md` is enough.

---

## 1. Prerequisites

- An Anthropic API key (`sk-ant-…`)
- One of:
  - **Docker 24+** with the Compose plugin (recommended)
  - Or Python 3.11+ on the target host

---

## 2. Configure environment

Copy the example and fill in your secrets:

```bash
cp .env.example .env
```

Required keys:

| Key                        | Purpose                                       |
| -------------------------- | --------------------------------------------- |
| `ANTHROPIC_API_KEY`        | Claude access for the decision engine         |
| `BOTTRADE_ENV`             | `dev` / `staging` / `production`              |
| `BOTTRADE_PORT`            | Host port (default 8501)                      |

Optional but strongly recommended for any deployment that's reachable
from the internet:

| Key                              | Purpose                          |
| -------------------------------- | -------------------------------- |
| `BOTTRADE_AUTH_PASSWORD_HASH`    | SHA-256 hash — gates the UI       |
| `BOTTRADE_AUTH_HASH_SALT`        | Per-deploy salt (any random str) |

Generate a password hash:

```bash
python -c "import hashlib; print(hashlib.sha256(b'MYSALT' + b'mypassword').hexdigest())"
```

Then put the digest into `BOTTRADE_AUTH_PASSWORD_HASH` and the same
salt into `BOTTRADE_AUTH_HASH_SALT`. The dashboard now shows a login
form before any page renders.

> **Local dev** — leave both unset. Auth is automatically skipped.

---

## 3. Docker (recommended)

### Build & run

```bash
docker compose up -d --build
docker compose logs -f bottrade   # tail logs
```

The dashboard is served at `http://localhost:${BOTTRADE_PORT:-8501}`.

### Persistent state

Two host-mounted volumes survive container rebuilds:

```text
./data   → /app/data    (portfolios, RAG index, ML models, notif config)
./logs   → /app/logs    (rotating bottrade.log + crashes.log)
```

A named volume caches HuggingFace embedding weights so cold-start
doesn't redownload `all-MiniLM-L6-v2`.

### Updating

```bash
git pull
docker compose up -d --build      # rebuild + recreate
```

### Stopping

```bash
docker compose down               # stops and removes the container
docker compose down -v            # ALSO drops the HF cache volume
```

### Resource limits

`docker-compose.yml` ships with a 2 GB memory ceiling. Tune for your
host; ChromaDB + sentence-transformers comfortably fit in 1 GB once
warm.

---

## 4. Cloud hosts

### Streamlit Community Cloud

Free, easiest path, but **no Telegram outbound from the free tier** —
the bot still works, you just won't get pings.

1. Push the repo to GitHub.
2. https://share.streamlit.io → New app → pick `dashboard/app.py`.
3. In *Advanced settings → Secrets*, paste your `.env` content (TOML
   format — convert `KEY=value` to `KEY = "value"`).
4. Save. The first cold start takes 3–5 minutes (downloads embedding
   weights).

### Fly.io

Fly handles a single-container deploy gracefully:

```bash
fly launch --no-deploy             # creates fly.toml — pick "yes" to use Dockerfile
fly secrets set ANTHROPIC_API_KEY=sk-ant-... BOTTRADE_ENV=production
fly secrets set BOTTRADE_AUTH_PASSWORD_HASH=...  BOTTRADE_AUTH_HASH_SALT=...
fly volumes create bottrade_data --size 5 --region <your-region>
# Edit fly.toml to mount the volume at /app/data
fly deploy
```

Fly auto-provisions HTTPS at `https://<app>.fly.dev`. **Always set
the auth hash** — Fly URLs are guessable.

### Railway / Render

Both auto-detect the `Dockerfile`. Steps:

1. Connect your repo.
2. Add env vars from `.env` in the dashboard UI.
3. Add a **persistent disk** mounted at `/app/data` (≥ 5 GB).
4. Deploy.

### VPS (DigitalOcean / Hetzner / your own)

Use the included `docker-compose.yml`. Stick **Caddy** or **Traefik**
in front for TLS:

```
# Caddyfile (optional — uncomment the caddy service in docker-compose.yml)
your-domain.example.com {
    reverse_proxy bottrade:8501
}
```

Caddy auto-provisions Let's Encrypt certs. Done.

---

## 5. Operations

### Where are the logs?

| Path                  | Contents                                          |
| --------------------- | ------------------------------------------------- |
| `logs/bottrade.log`   | Daily-rotated app log (30-day retention)          |
| `logs/crashes.log`    | Uncaught exceptions, main thread + worker threads |

The crash logger fans out a one-liner to your Telegram / webhook
notifier when configured (see Settings → Notifications).

### Where is portfolio state?

`data/virtual_portfolio.json` is written atomically on every trade.
**Back this file up** (any restic / borg / cron-rsync) — it's your
audit trail.

### Resetting the password

Stop the container, edit `.env`, restart:

```bash
docker compose stop
# update BOTTRADE_AUTH_PASSWORD_HASH in .env
docker compose up -d
```

Sessions are not server-side, so old sessions die when the container
restarts.

### Health check

`http://<host>:8501/_stcore/health` returns `200 OK` while
Streamlit is alive. Compose runs it every 30 s and restarts the
container if it fails 3 times in a row.

---

## 6. Production checklist

Before pointing real eyeballs at the URL:

- [ ] `BOTTRADE_ENV=production` set
- [ ] `BOTTRADE_AUTH_PASSWORD_HASH` + `BOTTRADE_AUTH_HASH_SALT` set
- [ ] Telegram or webhook notifications configured under *Settings*
- [ ] HTTPS in front of the container (Caddy / Cloudflare Tunnel /
      managed-host TLS)
- [ ] `data/` and `logs/` mounted on persistent storage
- [ ] Backup script for `data/virtual_portfolio.json`
- [ ] Tested *Panic Stop* on a fresh trade (it really closes positions)
- [ ] Reviewed Conservative / Balanced / Aggressive risk envelopes —
      they cap position size + circuit breakers
- [ ] Monitored the first 24 h via the Notifications channel

---

## 7. Troubleshooting

**“Cannot find module `dashboard`”** — Streamlit not run from the
project root. The `Dockerfile` already handles this; for ad-hoc runs:

```bash
python -m streamlit run dashboard/app.py
```

**ChromaDB warns about `tokenizers` parallelism** — harmless. Already
silenced in `utils/hf_quiet.py`.

**Cold start takes minutes** — first run downloads the embedding model
(~90 MB). Subsequent starts use the `hf_cache` volume.

**Dashboard renders blank with login screen** — auth is enabled but
the password hash is wrong. Double-check the salt.

**Live engine not running after restart** — the engine is started by
clicking the play button on the Live page. By design — never wants
to silently run on a bare boot. If you want auto-start on container
boot, plumb a flag in `dashboard/app.py`.
