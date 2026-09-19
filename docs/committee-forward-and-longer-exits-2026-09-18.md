# COMMITTEE follow-up: longer exits and frozen forward observation

Research only; live trading code has not been changed by these experiments.

## Longer exits: completed historical comparison

Same saved six-asset data and costs as the [initial study](committee-adx-chandelier-2026-09-18.md). Fixed additional variants use Chandelier(22,3) on completed 30-minute or hourly candles. ADX remains on five-minute candles. Incomplete higher-timeframe buckets are excluded, including partial equity-session boundary hours. Each band's value becomes available only after the final component candle closes. Position-specific ratcheting and the existing hard stop remain active.

The hourly `trend` variant additionally removes the maximum holding time and fixed profit target. It therefore changes more than the stop's timeframe; this is an explicit additional comparison, not evidence isolating a single cause.

Net account returns (%):

| Asset | ADX only, late | ADX + 30m exit, late | ADX + 1h exit, late | ADX + 1h trend, late | ADX + 1h trend, early |
|---|---:|---:|---:|---:|---:|
| BTC | -0.545 | -0.572 | -0.481 | +0.333 | -1.248 |
| ETH | -0.333 | -0.120 | -0.306 | +0.173 | -0.269 |
| META | -0.187 | -0.510 | -0.510 | -1.075 | +3.103 |
| NVDA | +0.286 | +0.286 | +0.286 | +0.863 | -2.743 |
| SPY | -0.161 | -0.106 | -0.156 | -0.156 | -0.664 |
| MSFT | -0.042 | -0.598 | -0.598 | -0.869 | -2.664 |

Doubling slippage in the late hourly-trend variant produces BTC +0.200%, ETH +0.293%, META -1.046%, NVDA +0.749%, SPY 0% (no trades), MSFT -0.879%. Cost-aware admissions can change the trades, explaining why some returns improve under higher assumed costs. NVDA's late result has only two trades. No variant establishes robust profitability across assets and periods. Selecting the best asset-specific exit after seeing these results would introduce selection bias.

A replay timestamp bug was also corrected: the close of a five-minute candle after a session gap is open + five minutes, rather than open + the preceding overnight/weekend gap. Baseline and ADX were rerun alongside the new variants; reported returns in these windows were unchanged. This fix affects the research replay, not production execution.

Evidence: `data/research_reviews/committee_20260918/adx_chandelier/longer_exits_{early|late}_slip{1|2}.json`. Reproduce with `python -m tools.compare_trend_controls`, `--result-prefix longer_exits`, and variants `baseline adx25 adx25_chandelier_30min adx25_chandelier_1h adx25_chandelier_1h_trend`; use `--segment early` or `--stress 2` for corresponding checks.

## Forward observation prepared and initialized

`tools/forward_committee.py` compares only baseline and ADX >= 25, each in a separate simulated account, at normal and doubled slippage. It has no broker API or order submission. It replays recorded observations from a frozen start boundary and marks open positions instead of inventing an end-of-report liquidation. Open-position equity includes paid entry costs, but excludes hypothetical future exit costs until an exit occurs.

The active local study is `data/research_reviews/committee_20260918/forward_adx25_v2/`. Its fixed start is **2026-09-18 17:25 UTC (20:25 Israel)**. The first observation contains no forward trades and is explicitly marked `insufficient_forward_evidence`. This is an initialized experiment, not a completed forward validation.

The protocol freezes settings, source-code fingerprints, pandas/numpy versions, seed data hashes and the start boundary. Each run records a separate observation and input hashes. Changing code/settings requires a separately named study. Previously recorded prices are never overwritten. Initial provider checks found revisions to two pre-study SPY cells; pre-study revisions are now counted and ignored in favor of the originally observed values, while revisions inside the forward period stop evaluation. The initial `forward_adx25` attempt is retained for audit and superseded by v2 before collecting forward trades.

The procedural review floor is 30 calendar days of observed coverage across all assets and 50 completed ADX trades in total at normal costs. This is not a statistical guarantee or an automatic profitability approval. Passing it only permits manual review of costs, net returns, drawdown, asset concentration and samples. No automatic deployment is implemented.

Run one collection/evaluation:

```powershell
python -m tools.forward_committee --directory data/research_reviews/committee_20260918/forward_adx25_v2 --update
```

Omit `--update` to evaluate only already recorded candles. **No recurring scheduler or background process is running.** Further observations require invoking this command. Delayed public OHLCV and simulated fills are not broker paper execution. Funding, stock borrow, latency and market impact remain unmodeled. Intrabar simulation limitations from the initial study still apply.

## Validation

104 tests passed across trend controls, forward observation, committee policy/indicators, entry research and trade management. Focused tests were rerun after handling pre-study provider corrections (15 passed). Tests cover higher-timeframe prefix invariance, incomplete candles, ratcheting, adverse gaps, session-close timestamps, unliquidated open-position accounting, immutable seeds/configuration and provider revisions.

[Freqtrade's lookahead-analysis documentation](https://www.freqtrade.io/en/stable/lookahead-analysis/) explains why causal feature validation matters when a backtest has access to a full dataframe. [StockCharts' Chandelier reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit) supplies the base formula; neither source validates this implementation's profitability.
