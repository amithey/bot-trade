# BotTrade

An AI trading bot with a RAG-backed knowledge base, four selectable strategy
modes, a live Streamlit dashboard, and a multi-tenant layer that lets other
people run it on their own Anthropic key.

Paper-trading only — every position is virtual, tracked in a local portfolio,
never touching a real brokerage.

## What it does

- **Ingests trading knowledge** from YouTube transcripts and web articles into
  a local ChromaDB vector store, and retrieves the relevant strategy for the
  current market state before every decision.
- **Reads the market** — price, volume and a 38-indicator technical
  committee (trend / momentum / volatility / volume) computed in pure
  pandas/numpy, no external TA library.
- **Decides via one of four strategy modes**, selectable per session:

  | Mode | What decides | Calls Claude? |
  | --- | --- | --- |
  | `COMMITTEE` | 38 indicators vote; majority rules | No — runs free forever |
  | `AI` | Claude alone, with RAG context | 1 call/cycle |
  | `HYBRID` | Committee votes, Claude reviews the tally | 1 call/cycle |
  | `BOARDROOM` | 8 analyst personas + a chairman, in parallel | ~9 calls/cycle |

- **Manages risk automatically** — stop-loss, take-profit, a consecutive-loss
  circuit breaker, a tilt cooldown after a bad trade, a daily loss/profit
  halt, and a slippage model so paper fills aren't unrealistically clean.
- **Backtests** the committee strategy over years of daily data in under a
  second (fully vectorized, zero API calls), with a parameter-sweep optimizer.
- **Runs live** in a Streamlit dashboard: portfolio, analytics, sector
  heatmap, watchlist scanner, and a knowledge-base manager — see
  [Dashboard pages](#dashboard-pages) below.

### Multi-tenant / hosting a shared instance

BotTrade can run as a single-user tool on your own machine, or be hosted for
other people:

- **Bring-your-own-key** — a visitor can supply their own Anthropic API key
  in Settings; their tokens bill to their own account, not yours.
- **Shared decision cache** — a verdict on one symbol for one bar is
  identical for every user watching it, so it's computed once and reused.
  API cost scales with symbols under watch, not with headcount.
- **Plans and a usage ledger** — Free / Pro / Desk tiers with per-mode and
  per-interval limits, and a full spend breakdown per user (`saas/`).
- **Per-account portfolios and identity** — each signed-in user gets their
  own virtual portfolio and trading profile, persisted to disk and
  reattached automatically after a refresh or restart.

See [DEPLOY.md](DEPLOY.md) for hosting this for other people, including
setting up per-person sign-in.

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/amithey/bot-trade.git
cd bot-trade

python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY (get one at console.anthropic.com)

python -X utf8 -m streamlit run dashboard/app.py
```

Open `http://localhost:8501`. No `ANTHROPIC_API_KEY`? `COMMITTEE` mode still
works fully — it makes no API calls at all.

### Optional: backtesting extras

`vectorbt`-based backtesting (`backtesting/backtest_runner.py`) needs a
separate numpy/pandas generation the rest of the project doesn't use:

```bash
pip install -r requirements-backtesting.txt
```

The dashboard's own **Committee Lab** page backtests without this — it's
pure pandas/numpy and needs no extra install.

### Optional: dev tools

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Command-line usage

`main.py` exposes the pipeline directly, without the dashboard:

```bash
# Ingest a YouTube video into the knowledge base
python main.py ingest --url "https://www.youtube.com/watch?v=XXX" --title "RSI Strategy"

# Run one live decision on a ticker
python main.py decide --ticker QQQ

# Run a historical backtest
python main.py backtest --ticker QQQ --start 2023-01-01 --end 2023-12-31 --capital 10000

# Show knowledge base status
python main.py status
```

## Dashboard pages

| Page | What it's for |
| --- | --- |
| **app** (live) | Start/stop the bot, watch it trade in real time, KPI strip |
| Portfolio | Positions, trade history, safety status |
| Settings | Plan, API key, trading profile, watchlist |
| Knowledge | Ingest YouTube/articles into the RAG knowledge base |
| Market Research | Free-form macro research from Claude |
| Sector Heatmap | Cross-sector performance at a glance |
| Watchlist Scanner | Scan a list of tickers for setups |
| ML Lab | Train/evaluate the ML models under `ml/` |
| Analytics | Sharpe/Sortino/Calmar, drawdown, attribution |
| Committee Lab | Backtest the 38-indicator strategy, parameter sweep — no API calls |
| Usage and Billing | Per-account spend, cache savings, cost estimator |

## Project layout

```
config/            Settings (pydantic-settings) and risk-profile envelopes
market_data/        Price fetching + hand-rolled technical indicators
strategy/           The 38-indicator committee + its backtester
rag/                ChromaDB retrieval over the ingested knowledge base
knowledge_ingestion/ YouTube + article scrapers that feed the knowledge base
decision_engine/    AITradingEngine (Claude) + the analyst boardroom
risk/               Circuit breakers, cooldowns, slippage modeling
portfolio/          Virtual account: positions, trade log, P&L
trading/            The live engine loop + the per-account engine registry
saas/               Plans, BYOK, usage ledger, the shared decision cache
dashboard/          Streamlit app, pages, identity/auth, shared UI helpers
analytics/          Performance metrics for the Analytics page
backtesting/        vectorbt-based backtester (opt-in extra)
ml/                 ML models used by the ML Lab page
news/               News/sentiment feed
notifications/      Telegram / webhook / in-app toast dispatch
tools/              Operational scripts (e.g. load-testing live engines)
tests/              pytest suite, run in CI on every push and PR
```

## Testing

```bash
pytest -q
```

189 tests, no network calls, no real API key required — every test that
touches an Anthropic key uses a fake one. Runs automatically on every push
and pull request via GitHub Actions (`.github/workflows/tests.yml`); a PR
with failing tests cannot merge to `main`.

## Deployment

See [DEPLOY.md](DEPLOY.md) — Docker, environment variables, per-person
sign-in (OIDC), the shared-password gate, and a production checklist.

## License

MIT — see [LICENSE](LICENSE).
