# COMMITTEE ADX25 production deployment

Deployed at the user's explicit request on 2026-09-18.

- Application: https://bottrade.fly.dev/
- Image: `registry.fly.io/bottrade:committee-v5-adx25-20260918`
- Image digest: `sha256:51ba83368dea374a547a7dd146acbafc78e3fb6ce2a739fa3a2baeb72eafbb64`
- Machine: `830377b7490638`, revision 27, started; one health check passing.
- Source: tracked commit `6dd2a05146efc41af0304d0694f0d0dc9191eaf6`, with exactly two production file overrides: `strategy/committee_policy.py` and `trading/live_engine.py`. Built from an isolated snapshot so unrelated local tools/research/data were not included.

## Active behavior

COMMITTEE entry now requires ADX(14) >= 25 on the most recent completed five-minute candle, in addition to all existing v4 conditions. Missing/nonfinite ADX blocks entry. The report includes the ADX value, named admission check, reason and `committee-v5-adx25` version. Existing-position exits are not gated by ADX. No Chandelier experiment or forward-observation scheduler was enabled. Other modes retain their existing behavior.

109 targeted tests passed, including live-cycle threshold boundary, missing ADX and existing-position exit tests. Fly deployment, smoke, health and DNS checks succeeded. Remote file digests matched the local tested snapshot:

```text
fe8b16dc9933f69342228635e304533dedbacc8ab7d7e879c51ff20593ae8a33 strategy/committee_policy.py
eb3b006b20d90fae8c12eecbee0bd76051b5d517a2b9a7376cf00282dd864bdb trading/live_engine.py
```

The SSH command printed both matching digests, then the Windows client emitted `The handle is invalid` during teardown. Deployment independently completed with exit code 0 and passing server checks.

## User session

Deployment restarts the process. Portfolio data is persistent, but the engine registry is process-local and does not auto-start account trading loops after restart. The available browser session displayed the sign-in page, so account-level bot operation could not be resumed or verified. The user must sign in, select COMMITTEE and press **Start agent** to run it. Do not conflate a healthy web server with an active account trading loop.

This live-policy change modifies the code fingerprint used by the previously prepared local forward study. That frozen study correctly refuses further evaluation under changed code; continuing forward research requires a new named study rather than editing its frozen fingerprint.

Previous deployment for rollback: `registry.fly.io/bottrade:main-6dd2a05`. Research results support reduced historical losses with ADX, not established profitability.
