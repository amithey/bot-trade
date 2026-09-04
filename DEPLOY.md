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

Running from source rather than Docker:

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scriptsctivate
pip install -r requirements.txt                  # runtime
pip install -r requirements-dev.txt              # tests + linters, optional
```

Use a virtualenv. `requirements.txt` pins exact versions of the stack the
test suite runs on, and installing it into a shared global Python will fight
with whatever else lives there.

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
| `BOTTRADE_AUTH_PASSWORD_HASH`    | PBKDF2 hash — gates the UI        |
| `BOTTRADE_AUTH_HASH_SALT`        | Legacy hashes only (see below)   |

Generate a password hash:

```bash
python -m dashboard._auth
```

It prompts for the password twice and prints the line to paste into `.env`.
The hash is PBKDF2-HMAC-SHA256 with a random salt baked into it, so
`BOTTRADE_AUTH_HASH_SALT` is not needed for new hashes. The dashboard then
shows a login form before any page renders.

> **Upgrading from the old hash?** Bare SHA-256 digests (64 hex characters,
> with the optional `BOTTRADE_AUTH_HASH_SALT`) still verify, so nothing breaks
> on deploy — but a single unsalted SHA-256 round is cheap to brute-force. The
> login screen flags it; rerun the command above to rotate.

> **Local dev** — leave both unset. Auth is automatically skipped.

> **One password is one account.** This gate controls *access*, not
> *identity*: everyone who knows the password shares a single account,
> a single trading profile, and a single free-tier budget. For a real
> multi-user product, use section 2b instead.

---

## 2b. Per-person accounts (required for the free tier)

BotTrade uses Streamlit's built-in OIDC sign-in (`st.login` / `st.user`).
Streamlit owns the login cookie; BotTrade just reads the identity. This is
what makes per-user budgets, per-user profiles and per-user billing real —
identity survives a refresh, a new tab, and a restart.

Without it the app still runs, but every visitor is one account, so the
free-tier trial budget is a single shared pool that resets whenever someone
clears their session. The Settings page says so out loud when this is the
case.

**1. Install the dependency**

```bash
pip install "Authlib>=1.3.2"
```

**2. Register an OAuth client**

Google (free, ~10 minutes): console.cloud.google.com → *APIs & Services* →
*Credentials* → *Create OAuth client ID* → *Web application*. Under
*Authorised redirect URIs* add exactly the `redirect_uri` you will configure
below — it must end in `/oauth2callback`.

Want email+password signup as well as social login? Use Auth0 instead. It
speaks OIDC, so nothing on the BotTrade side changes.

**3. Write the secrets file**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
python -c "import secrets; print(secrets.token_urlsafe(48))"   # cookie_secret
```

Fill in `client_id`, `client_secret`, `cookie_secret`, and set `redirect_uri`
to your real origin in production (`https://yourdomain/oauth2callback`).
`.streamlit/secrets.toml` is gitignored — keep it that way.

**4. Restart**

Running under Docker? `.streamlit/secrets.toml` is excluded from the image on
purpose, so mount it at runtime — uncomment the secrets volume in
`docker-compose.yml`. Create the file *before* starting: Docker silently
creates a directory at a missing bind-mount source, which then breaks
Streamlit.


The dashboard now opens on a sign-in screen. After sign-in, each person gets:

| Per user | Stored at |
| --- | --- |
| Trading profile — capital, risk, watchlist | `data/profiles/<account>.json` |
| Virtual portfolio — cash, positions, trade history | `data/portfolios/<account>.json` |
| API spend + trial budget | `data/usage.db`, keyed by account |
| BYOK Anthropic key | session memory only — never written to disk |

All of it survives a refresh, a new tab, and a process restart. Back up
`data/` and you have backed up your users.

**Migration note.** An existing single-user deployment has one
`data/user_profile.json`. The first account to sign in adopts it, and the
file is then renamed to `.json.migrated` so the second person to sign up
starts from defaults instead of inheriting a stranger's capital and
watchlist.

### Live bots and capacity

A running bot is a background thread owned by the *process*, not by a browser
session (`trading/registry.py`), keyed by account. A refresh reattaches to the
bot that is already trading instead of orphaning the thread and starting a
second one on the same portfolio. The portfolio is checkpointed to disk after
every cycle and once more when the loop exits, so a restart resumes with
positions, cash and trade history intact.

`BOTTRADE_MAX_LIVE_ENGINES` (default 100) caps concurrent live bots. When
full, a **new** bot is refused — an existing one is never evicted, because
evicting means silently stopping someone's trading and leaving their stop-loss
unwatched. Users already holding an engine always reattach, however full the
process is. Everything except the live loop (backtests, Committee Lab,
analytics) keeps working at capacity.

Measure before you change it:

```bash
python -m tools.loadtest_engines
```

It runs the CPU-bound half of a cycle (the 38-indicator vote plus portfolio
bookkeeping) at several concurrency levels, using COMMITTEE mode so it costs
nothing in API calls. On the development machine it sustained ~7 cycles per
second regardless of bot count — the GIL, since the indicator maths is pure
pandas/numpy — and 55 KB of memory per idle engine. At a 30-second interval
that is ~210 bots before CPU saturates.

So CPU and memory are *not* the first limits. What will bite first, and what
the tool does not measure: Yahoo Finance rate-limiting a host fetching for a
hundred symbols, Anthropic rate limits in the AI modes, and thread scheduling
under real network load. Run the tool on the actual host, then set the cap
below the level where p95 starts climbing.

---

## 2c. Billing (Paddle)

Without this, every account stays on the Free plan forever — nothing is
broken, there's just no way to actually get paid. `saas/billing.py` is the
integration; this section is what to do in the Paddle Dashboard to make it
work.

**Why Paddle, not Stripe:** Stripe does not support Israel as a seller's
business location — confirmed against the real onboarding flow, not just
read about it. Paddle is a merchant of record (Paddle is legally the seller
for every transaction and handles global VAT/tax compliance, so BotTrade
carries zero sales-tax liability), and it does accept Israel as a business
location. That also changes the integration shape: there's no Stripe-style
hosted Checkout URL to redirect to — Paddle Checkout runs client-side via
Paddle.js, embedded in the Settings page.

**1. Account setup.** Sign up as a vendor at [paddle.com](https://www.paddle.com/).
Paddle's own onboarding walks through business verification; a plain static
page describing the product satisfies the "business website" requirement if
you don't have one — see `landing/` in this repo, a zero-cost landing page
deployable to a free Vercel subdomain. BotTrade itself can't be that page:
it's a Streamlit app with a persistent background thread and a websocket
connection, which serverless hosts like Vercel don't support.

**2. Create two recurring Prices** in the Paddle Dashboard (Catalog →
Products → New product) — a "BotTrade Pro" product priced at $29/mo, and a
**separate** "BotTrade Desk" product priced at $99/mo. Don't add a second
price to the same product — Paddle prices belong to one product each, and
mixing them up misattributes the subscription's plan on reconciliation. The
Free plan needs no Paddle price; it's the default for every new account.
Copy each Price ID (`pri_...`, not the Product ID).

**3. Create a scoped API key** (Developer tools → Authentication → API keys
→ Create API key) with only what `saas/billing.py` actually calls: Customers
(Read + Write), Customer portal sessions (Write), Customer authentication
tokens (Write), Prices (Read), Products (Read), Subscriptions (Read),
Transactions (Read). Also grab the **client-side token** (Developer tools →
Authentication → Client-side tokens) — unlike the API key, this one is
public by design; it's what ships to the browser to open the Checkout
overlay.

**4. Set the environment variables** (see `.env.example`):

```bash
PADDLE_API_KEY=pdl_sdbx_apikey_...       # secret — server-side only
PADDLE_ENVIRONMENT=sandbox               # 'sandbox' or 'production'
PADDLE_CLIENT_TOKEN=test_...             # public — ships to the browser
PADDLE_PRICE_ID_PRO=pri_...
PADDLE_PRICE_ID_DESK=pri_...
BOTTRADE_BASE_URL=https://yourdomain.example   # or http://localhost:8501 for local testing
```

**Test in sandbox first.** Sandbox and production are entirely separate
Paddle accounts — separate keys, separate product catalog, separate
dashboard. With `PADDLE_ENVIRONMENT=sandbox` and sandbox keys/Price IDs,
confirm the whole loop — Settings → Upgrade → Paddle's Checkout overlay
(Paddle's sandbox accepts test card `4242 4242 4242 4242`, any future
expiry, any CVC) → redirected back with the new plan active — before
switching to `PADDLE_ENVIRONMENT=production` and live keys/Price IDs
(promoting a Paddle account from sandbox to production/live is a separate
step in Paddle's own dashboard: "Switch your account to live").

**How it works once configured:**

* **Checkout** — clicking "Upgrade" in Settings opens Paddle's Checkout
  overlay client-side via Paddle.js. BotTrade never sees a card number.
* **Confirmation** — Paddle's overlay returns to `/Settings?paddle_return=1`
  with no transaction id to look up (unlike Stripe's `?session_id=`); the
  return instead triggers an immediate forced reconciliation against
  Paddle, using the customer id that was already known before checkout
  opened.
* **Cancellation / payment method / invoices** — a "Manage subscription"
  button opens Paddle's hosted Customer Portal. No custom UI to maintain.
* **No webhook receiver.** The textbook way to learn about a cancellation
  the instant it happens is a webhook, but Streamlit has no clean way to
  expose an HTTP route for one, and standing up a second service just for
  this is a bigger lift than a subscription business needs at this stage.
  Instead, `sync_subscription_status` re-checks Paddle on a 5-minute TTL
  whenever Settings loads. **Known limitation:** a cancellation made
  directly in Paddle (not through the app) can take up to 5 minutes, or
  until the user next opens Settings, to downgrade their access here.

**5. Production checklist addition:** confirm `BOTTRADE_BASE_URL` is the
real HTTPS origin before going live — Paddle's Checkout overlay redirecting
to a wrong or unreachable URL leaves a paying customer stranded with no idea
whether it worked.

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

`fly.toml` in the repo root is already written for this app — it builds from
the Dockerfile, mounts a persistent volume at `/app/data` (this is the whole
point of Fly over Streamlit Community Cloud: accounts, portfolios, the
billing ledger and the RAG index survive a redeploy instead of resetting to
empty), and disables auto-stop so the background trading loop keeps running
between page views.

**1. Install the CLI and sign in** (opens a browser — this step is yours,
not something to script):

```bash
# Windows PowerShell
iwr https://fly.io/install.ps1 -useb | iex
# macOS / Linux
curl -L https://fly.io/install.sh | sh

fly auth login
```

**2. Claim the app name.** `bottrade` in `fly.toml` is a placeholder — Fly
app names are global across every account, so it is almost certainly taken.
`fly launch` detects the existing `fly.toml` and will prompt to confirm or
change the name; accept its suggestion or pick your own, but say **no** when
it asks to overwrite `fly.toml` — the one in the repo already has the volume
mount and the auto-stop setting configured.

```bash
fly launch --no-deploy
```

**3. Create the persistent volume** — same region as `primary_region` in
`fly.toml`:

```bash
fly volumes create bottrade_data --size 5 --region iad
```

**4. Set secrets** — every value from your `.env` that isn't already a
plain default in `fly.toml`'s `[env]` block. Names must match
`config/settings.py` exactly (case-insensitive, but keep it consistent):

```bash
fly secrets set \
  ANTHROPIC_API_KEY=sk-ant-... \
  PADDLE_API_KEY=pdl_apikey_... \
  PADDLE_ENVIRONMENT=production \
  PADDLE_CLIENT_TOKEN=live_... \
  PADDLE_PRICE_ID_PRO=pri_... \
  PADDLE_PRICE_ID_DESK=pri_... \
  BOTTRADE_BASE_URL=https://<your-app>.fly.dev
```

Add `BOTTRADE_AUTH_PASSWORD_HASH` / `BOTTRADE_AUTH_HASH_SALT` too if you're
running password mode instead of OIDC (see section 2b) — Fly URLs are
guessable, so **something** must gate the app before real users touch it.

OIDC's `client_secret` and `cookie_secret` live in
`.streamlit/secrets.toml`, not `.env`, and Fly has no host bind-mount to
carry that file in the way `docker-compose.yml`'s commented-out volume line
does. Once you've written a real `.streamlit/secrets.toml` locally (section
2b, steps 1–3), ship its *contents* as one secret and the container writes
the file itself on boot (`docker/entrypoint.sh`, wired into the
`ENTRYPOINT` in the `Dockerfile`):

```bash
fly secrets set --app <your-app> \
  BOTTRADE_STREAMLIT_SECRETS="$(cat .streamlit/secrets.toml)"
fly deploy --app <your-app>
```

This only matters once you actually enable OIDC — skip it for an initial
deploy in open or password mode, exactly as this repo's own first Fly
deploy did.

**5. Deploy:**

```bash
fly deploy
```

Fly auto-provisions HTTPS at `https://<your-app>.fly.dev`. **Update**
`BOTTRADE_BASE_URL` (step 4) and `landing/index.html`'s `APP_URL` **to
match this exact URL** once you know it — Paddle's Checkout return and the
landing page's pricing buttons both depend on it being right.

**Cost note:** the `[[vm]]` block in `fly.toml` requests 2 GB RAM (ChromaDB
+ sentence-transformers need it warm), and `min_machines_running = 1` keeps
it up around the clock for the trading loop. That combination is past what
Fly's free allowance covers — expect a small monthly charge, not a free
deployment.

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
# regenerate with: python -m dashboard._auth, then update .env
docker compose up -d
```

Sessions are not server-side, so old sessions die when the container
restarts.

### Health check

`http://<host>:8501/_stcore/health` returns `200 OK` while
Streamlit is alive. Compose runs it every 30 s and restarts the
container if it fails 3 times in a row.

### Shipping an update

Once the product is live, `main` is what's deployed — nobody commits to it
directly. Day-to-day fixes and features land on `develop`:

```bash
git checkout develop
git pull
# ... make changes, commit, push to develop ...
```

`develop` has no branch protection, so it's fine to push straight to it
mid-task. `main` does (GitHub requires the `pytest` check from
`.github/workflows/tests.yml` to pass on it) — the only way changes reach
`main`, and therefore production, is by merging `develop` into it once
the batch of changes is something you'd actually want live:

```bash
git checkout main
git pull
git merge --no-ff develop
git push origin main
# then, when you're ready to actually ship it:
fly deploy    # or your host's equivalent — see section 4
```

`--no-ff` keeps a merge commit marking exactly which changes went out in
each release, rather than rewriting `develop`'s history onto `main`. A
push to `develop` is *not* a deploy — it never reaches the running
container until someone merges it to `main` and runs `fly deploy` (or
redeploys via whichever host section 4 describes). This mirrors how a
real product ships: work happens on an integration branch, and only a
deliberate merge-and-deploy step touches what users are running.

---

## 6. Production checklist

Before pointing real eyeballs at the URL:

- [ ] `BOTTRADE_ENV=production` set
- [ ] Sign-in configured — `[auth]` in `.streamlit/secrets.toml`
      (section 2b). Required before exposing the free tier; the shared
      password gate alone cannot enforce a per-user budget
- [ ] `redirect_uri` points at the real HTTPS origin, and the same URI is
      registered with the OAuth provider
- [ ] `cookie_secret` is a fresh random string, not the example value
- [ ] `.streamlit/secrets.toml` and `data/usage.db` excluded from the image
      and from git
- [ ] `BOTTRADE_FREE_BUDGET_USD` set deliberately (`0` disables trial credit)
- [ ] Telegram or webhook notifications configured under *Settings*
- [ ] HTTPS in front of the container (Caddy / Cloudflare Tunnel /
      managed-host TLS)
- [ ] `data/` and `logs/` mounted on persistent storage
- [ ] Backup covers `data/profiles/`, `data/portfolios/` and
      `data/usage.db` — these are your users' accounts and billing
- [ ] `BOTTRADE_MAX_LIVE_ENGINES` matched to the host's capacity
- [ ] If accepting payments: `PADDLE_API_KEY`, `PADDLE_CLIENT_TOKEN` and the
      price IDs are the **live** ones with `PADDLE_ENVIRONMENT=production`,
      not sandbox — verified with a real card, not `4242 4242 4242 4242`
- [ ] `BOTTRADE_BASE_URL` is the real HTTPS origin — a wrong redirect
      strands a paying customer with no idea whether checkout worked
- [ ] `docker build` succeeded from a clean checkout — the image installs
      exactly `requirements.txt`, so anything not pinned there is not
      in production
- [ ] Confirmed `.streamlit/secrets.toml` is NOT inside the built image
      (`docker run --rm <image> ls -la /app/.streamlit`)
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
