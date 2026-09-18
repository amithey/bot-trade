# Committee investigation — 18 September 2026

Status: experimental candidate. **No profitable edge established; do not describe this as a profitable replacement.** No running service was restarted or deployed.

## Diagnosis

The supplied trade export contained six closed BTC trades, four losses, and net P&L of -$2,202.30. All reasons were truncated to 120 characters. Chart markers omitted SHORT/COVER. Those defects prevented a full trade-by-trade reconstruction; the supplied export is not an OHLCV backtest dataset.

The hourly slope had been measured after expansion to five-minute rows, erasing direction within each hour. The previous patch corrected it. Model exits previously bypassed the soft-exit confirmation policy. Those fixes alone do not establish an edge.

Committee technical votes are not independent analyst opinions or calibrated profit probabilities. Financial/company facts and news in the report are contextual and do not constitute a tested committee valuation model. Company P/E and cash flow do not value BTC. No funding, order-book or on-chain feed was added or invented.

## Candidate implementation

Only COMMITTEE uses the new admission and envelope policy. Other model modes retain their prior directional exit behavior.

- A confirmed trend setup, explicit completed-hour direction, quorum, directional margin >= 0.16, and positive direction-adjusted Trend/Momentum/Volume group scores are all required. Unknown hourly direction fails closed.
- Estimated round-trip fees and slippage must fit within a six-ATR movement screen. This screen is a heuristic volatility budget, not an expected return.
- Gross target is at least 2.5 times the existing profile stop. Net target / stop-plus-cost ratio must be at least one. This is a distance constraint, not a win probability.
- Planned stop plus estimated costs limits each entry's nominal equity risk to 0.25%, further constrained by existing account/profile exposure caps. Gaps can exceed that budget.
- No committee pyramiding. Fifteen-minute cooldown after an exit. Existing cash, confidence and safety checks remain active.
- Soft exits honor confirmation and include costs in the positive-P&L threshold. Hard stops, maximum holding time and profit protection remain independent.
- The default committee trailing protection arms at max(existing activation, half the profile stop distance) in net percentage points; explicitly supplied protection configs remain unchanged. Previously armed stops are not loosened.
- Fetch one month of five-minute bars for committee hourly warm-up, including assets with shorter trading sessions.
- Research displays admission evidence and an explicit experimental/loss-making validation warning. Trade export retains full reasons; chart shows all trade directions.

## Dataset and procedure

Public Yahoo Finance BTC-USD five-minute OHLCV downloaded through yfinance: 21 July–18 September 2026. The last potentially open candle was removed (17,147 completed rows). A 70/30 chronological split reserved the later segment for evaluation. No optimizer or parameter grid was used. Development alternatives at 15 minutes and one hour were also negative and were not integrated.

The replay uses an initial $10,000, Balanced profile, a 30% exposure ceiling for both policies, 0.1% fee each side and the repository's adverse slippage model. The candidate additionally uses the risk-budget ceiling. The baseline includes the earlier hourly and soft-exit fixes; it is a reference replay, not an exact reconstruction of production before this investigation.

Both policies share the same replay/accounting machinery. Signals use the prior completed candle and fill at the next open. Existing positions are not closed and reopened within the same decision. Stops use adverse opening gaps; stop takes precedence when both stop and target occur in one bar. Trailing protection observes open and close quotes only. Remaining exposure is liquidated at evaluation end with exit costs.

## Later-period evaluation

31 August 2026 16:10 UTC through 18 September 2026 13:25 UTC:

| Metric | Reference policy | Candidate |
| --- | ---: | ---: |
| Net return | -5.22% | -2.38% |
| Maximum drawdown | -5.35% | -2.61% |
| Closed trades | 50 | 20 |
| Win rate | 30% | 15% |
| Profit factor | 0.147 | 0.100 |
| Fees | $291.56 | $89.84 |
| Net P&L | -$521.76 | -$237.81 |

Buy-and-hold's uncosted price change over the same window was -0.68%, with different exposure/risk. The candidate's smaller loss comes with fewer trades and smaller sizing; its worse win rate and profit factor do **not** demonstrate improved predictive ability.

With doubled slippage, reference return was -7.78% and candidate return -1.76%. Candidate trades fell to 16 because higher cost estimates blocked additional entries. This is an adaptive-policy stress result, not evidence that worse fills improve identical trades.

The fee-aware soft-exit correction was rerun after the first evaluation; reported results were unchanged. Treat this as a development audit, not an independently certified untouched holdout. Future changes require fresh data rather than tuning to this segment.

## Reproduction and limits

257 targeted tests passed, covering committee admission, causal hourly direction, long/short execution, cost-aware exits, protection, exposure sizing, chart markers and preservation of AI exit behavior. Compilation and Git whitespace checks passed. These verify implementation, not trading profitability.

Run `python -m tools.validate_committee PATH_TO_COMPLETED_OHLCV.csv --output result.json`. The output records the source SHA-256, dates, costs, trades and blocked checks. The downloaded data and full local outputs are under `data/research_reviews/committee_20260918/` and excluded from Git along with provider cookies/cache.

The replay is not a production-equivalent simulator: it lacks the ten-second quote path, daily account halts, all safety/cooldown rules, liquidity, funding and latency. It uses longer continuous warm-up than a rolling live fetch. Historical financial/news snapshots were unavailable. Reduced losses are useful risk-control evidence, **not acceptance evidence for profitable deployment**. More independent market regimes and forward paper performance are needed before calling this a replacement strategy.
