# COMMITTEE: ADX / Chandelier experiment — 2026-09-18

## Decision

Do not activate these experimental controls in live trading yet. ADX >= 25 reduced losses relative to the deployed committee-v4 baseline across both tested segments (one late-segment tie), but did not establish positive expectancy. Chandelier (22, 3) was inconsistent relative to ADX alone and sometimes cut profitable trends short. Reducing losses is not evidence of a profitable strategy.

The implementation is restricted to the offline replay. Live entry/exit code and other trading modes are unchanged by this experiment.

## Method

- Six independent $10,000 accounts: BTC-USD, ETH-USD, META, NVDA, SPY, MSFT. These are not a shared-capital portfolio.
- Public Yahoo Finance adjusted 5-minute OHLCV, requested 60 days, last returned candle removed. Saved CSVs and SHA-256 hashes permit repeatable offline runs.
- Fixed variants: deployed v4 baseline; mandatory ADX(14) >= 25; Chandelier(22, 3); both. Additional sensitivity checks use ADX 20, or remove the fixed target and maximum holding time alongside Chandelier.
- The primary Chandelier variant replaces percentage profit protection and signal-based soft exits. It retains the 0.8% hard stop, 2% target and Balanced maximum holding rule. It rejects entries already beyond the Chandelier band.
- Long stop = highest high over 22 bars minus 3 × ATR(22); short stop is the symmetric lowest-low formula. A position-specific ratchet prevents loosening, resets on each new position, and never loosens the hard stop.
- Signals, ADX and stop levels use completed prior bars; execution is at the next open. Intrabar stops use high/low, adverse gap fills and stop-first ordering if stop and target both occur.
- Commission 0.1% per side; baseline slippage uses the existing Balanced model (5 bps plus its volatility term). Existing v4 exposure, risk-sizing and entry/cooldown gates remain active.
- Later evaluation uses the last 30% of bars. Earlier evaluation uses the first 70%, after 720 warm-up bars. No parameter search was performed. This is a retrospective comparison, **not an untouched out-of-sample test**: portions of the data were already examined during prior investigations.

Later crypto evaluation begins 2026-08-31 18:40 UTC; equities begin 2026-08-24 17:50 UTC. Data ends 2026-09-18 16:55 UTC.

## Later segment: net account return, percent

| Asset | Baseline | ADX 25 | Chandelier | ADX 25 + Chandelier |
|---|---:|---:|---:|---:|
| BTC | -2.378 | -0.545 | -1.484 | -0.422 |
| ETH | -0.677 | -0.333 | -0.286 | -0.010 |
| META | -1.428 | -0.187 | -0.895 | -0.045 |
| NVDA | -0.691 | +0.286 | -0.642 | -0.354 |
| SPY | -0.161 | -0.161 | -0.039 | -0.039 |
| MSFT | -1.020 | -0.042 | -1.014 | +0.042 |

Baseline generated 78 completed trades, ADX alone 25, and the combined variant 28. The combined variant still lost money on five of six assets. NVDA's positive ADX result consists of just two trades; SPY has one trade. These samples cannot establish statistical significance.

### Concrete premature-exit example

On NVDA, both variants opened the same short on September 9 at 14:05 UTC, filled at $224.1010. ADX alone closed the next session for +$30.98 net. Adding Chandelier stopped that trade at 14:25 for -$17.18 net; the variant subsequently re-entered and incurred another loss. This is a simulated comparison, not a claim that the profitable outcome was predictable at entry.

On five-minute data, 22 bars represent 110 trading minutes, not 22 days. A standard daily-chart parameter transferred unchanged can produce a relatively tight intraday stop. In this experiment Chandelier often reduced holding time instead of extending it.

## Earlier segment: net account return, percent

| Asset | Baseline | ADX 25 | Chandelier | ADX 25 + Chandelier |
|---|---:|---:|---:|---:|
| BTC | -3.050 | -1.509 | -1.969 | -0.889 |
| ETH | -1.814 | -0.766 | -1.286 | -0.403 |
| META | -0.498 | +0.538 | -1.691 | -0.244 |
| NVDA | -2.416 | -1.689 | -1.829 | -1.798 |
| SPY | -0.570 | -0.454 | -0.560 | -0.404 |
| MSFT | -2.407 | -1.878 | -0.980 | -0.501 |

ADX alone improved all six relative to baseline. The combined variant lost money on all six. Adding Chandelier to ADX worsened META and NVDA. Chandelier alone substantially worsened META relative to baseline.

## Sensitivity and costs

With doubled slippage in the later segment, combined returns were BTC -0.429%, ETH -0.061%, META -0.147%, NVDA -0.388%, SPY 0% (no trades), MSFT -0.046%. None were positive. Cost-aware entry gates and sizing change the admitted trades, so this is a complete policy rerun, not simply repricing identical trades.

Lowering the combined ADX threshold to 20 produced mixed results: ETH improved to +0.172%, but NVDA fell to -1.138% and MSFT to -0.643%. Removing the fixed target and maximum holding rule did not rescue the combined strategy: ETH and META worsened, and the other four were unchanged in the later segment. There is no basis here to pick a universal winning threshold or stop variant.

## Research interpretation and limitations

[Fidelity's ADX guide](https://www.fidelity.com/viewpoints/active-investor/average-directional-index-ADX) describes values below 20 as weak and above 25 as strong trend conditions. ADX measures strength, not direction. This source does not establish a universal 30–40% reduction in false signals; this experiment does not measure that claim either.

[StockCharts' Chandelier explanation](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit) documents the rolling-extreme/ATR formula. The raw band can move backwards as the rolling extreme or ATR changes; the explicit position ratchet is what prevents a stop from loosening.

The replay does not reconstruct ten-second live quotes or true intrabar paths. Existing percentage profit protection observes open/close prices, while hard/Chandelier stops observe bar extremes. Results therefore compare this replay's implementations, not a tick-accurate live replica. Stock overnight gaps affect wall-clock holding times. Funding, borrow availability/cost, market impact and live account-level daily safety guards are not modeled. Fundamental/news evidence is not tested. Selection of six familiar assets creates selection bias. Buy-and-hold uses full exposure and is not risk-matched to the bot's capped positions.

The next useful validation is an untouched forward paper-trading window with fixed ADX settings, realistic costs and a risk-matched benchmark. Separately test longer-timeframe exits before considering Chandelier for live use. These are future research steps, not completed validations or a recommendation to trade real funds.

## Reproduction and checks

Saved evidence directory: `data/research_reviews/committee_20260918/adx_chandelier/` (ignored local research data).

- `results_slip1.json`: initial later-segment run, all six variants.
- `results_early_slip1.json`: earlier segment, four primary variants.
- `results_late_slip2.json`: doubled slippage, four primary variants.

Offline reproduction:

```powershell
python -m tools.compare_trend_controls --output-dir data/research_reviews/committee_20260918/adx_chandelier
python -m tools.compare_trend_controls --output-dir data/research_reviews/committee_20260918/adx_chandelier --segment early --variants baseline adx25 chandelier adx25_chandelier
python -m tools.compare_trend_controls --output-dir data/research_reviews/committee_20260918/adx_chandelier --stress 2 --variants baseline adx25 chandelier adx25_chandelier
```

The current script names the reproduced first output `results_late_slip1.json`. Use `--download` only to deliberately replace the frozen inputs with a new provider snapshot.

Validation: 98 tests passed across trend controls, committee policy, committee indicators, entry research and trade management. Tests cover causal features, mandatory ADX admission, stop ratcheting, prior-bar levels, adverse gaps, hard-stop preservation and disabled-control defaults. Passing tests establish implementation checks, not profitability.
