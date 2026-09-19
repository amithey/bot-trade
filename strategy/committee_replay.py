"""Costed, chronological committee comparison; no fitting or network calls.

Signals use completed bars and fill at the following open. Stop/target ambiguity
uses stop first. Trailing protection observes opens/closes only: OHLC cannot
reproduce the live ten-second quote path. No funding, latency or market impact.
"""
import numpy as np
import pandas as pd

from config.user_profile import RISK_ENVELOPES
from risk.profit_protection import ProfitProtectionConfig, assess_profit_protection
from risk.slippage import SlippageConfig, apply_slippage
from strategy.committee import IndicatorCommittee
from strategy.committee_policy import assess_committee_entry, committee_profit_config, committee_envelope
from strategy.research import entry_features
from strategy.trade_management import assess_intraday_exit


def replay_committee(df, *, enhanced, evaluation_start, risk_profile="Balanced",
                     fee_rate=.001, slippage_multiplier=1., user_cap_pct=30.,
                     features=None, votes=None, adx_min=None, chandelier=False,
                     trend_holding=False, controls=None, finalize=True, bar_minutes=5,
                     volatility_stop=False, profit_config=None):
    if bar_minutes not in (5, 30, 60):
        raise ValueError("Replay supports explicit 5, 30 or 60 minute candles")
    if adx_min is not None and (not np.isfinite(adx_min) or not 0 <= adx_min <= 100):
        raise ValueError("ADX threshold must be between zero and 100")
    if trend_holding and not chandelier:
        raise ValueError("Trend holding requires Chandelier protection")
    if volatility_stop and not trend_holding:
        raise ValueError("Volatility stops are restricted to the swing research variant")
    if (not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing
            or df.index.has_duplicates or not 200 <= evaluation_start < len(df) - 1):
        raise ValueError("Need sorted, unique bars, 200 warm-up bars and an evaluation window")
    if not 0 <= fee_rate < .1 or not np.isfinite(slippage_multiplier) or slippage_multiplier < 0:
        raise ValueError("Invalid execution costs")
    if not 0 < user_cap_pct <= 100:
        raise ValueError("Invalid exposure cap")
    com = IndicatorCommittee()
    features = entry_features(df, risk_profile) if features is None else features
    votes = com.vote_matrix(df) if votes is None else votes
    from strategy.trend_controls import trend_control_features, ratchet_chandelier
    if controls is None and (adx_min is not None or chandelier):
        controls = trend_control_features(df)
    if controls is not None and not controls.index.equals(df.index):
        raise ValueError("Trend controls must align with price bars")
    if not features.index.equals(df.index) or not votes.index.equals(df.index):
        raise ValueError("Evidence must align with the price bars")
    scores = votes.sum(axis=1) / len(com.agents)
    quorum = votes.ne(0).sum(axis=1) >= com.config.min_quorum
    categories = {cat: votes[[a.name for a in com.agents if a.category == cat]].mean(axis=1)
                  for cat in {a.category for a in com.agents}}
    env = RISK_ENVELOPES[risk_profile]
    if enhanced:
        env = committee_envelope(env)
    slip_cfg = SlippageConfig.for_profile(risk_profile)
    protection = ProfitProtectionConfig()
    if enhanced:
        protection = committee_profit_config(protection, env.stop_loss_pct)
    if profit_config is not None:
        protection = profit_config
    cash, pos, last_exit, paid, slipped = 10000., None, None, 0., 0.
    trades, equity, blocked = [], [], {}
    exposed_bars = 0

    def bps_for(price, atr):
        return apply_slippage(price, "BUY", cfg=slip_cfg,
                              atr_pct=100 * atr / price)[1] * slippage_multiplier

    def close_position(price, stamp, reason, bps):
        nonlocal cash, pos, last_exit, paid, slipped
        sign = pos["sign"]
        fill = price * (1 - sign * bps / 10000)
        slipped += sign * (price - fill) * pos["quantity"] + pos["entry_slip"]
        fee = pos["quantity"] * fill * fee_rate
        pnl = sign * (fill - pos["entry"]) * pos["quantity"] - fee - pos["entry_fee"]
        cash += pos["notional"] + sign * (fill - pos["entry"]) * pos["quantity"] - fee
        paid += fee
        trades.append({"entry_time": str(pos["opened"]), "exit_time": str(stamp),
                       "side": "LONG" if sign == 1 else "SHORT", "entry": pos["entry"],
                       "exit": fill, "pnl": pnl, "reason": reason})
        pos, last_exit = None, stamp

    def protect(price, stamp, bps):
        nonlocal pos
        if pos is None or chandelier:
            return
        assessment = assess_profit_protection(
            side="LONG" if pos["sign"] == 1 else "SHORT", entry_price=pos["entry"],
            quantity=pos["quantity"], entry_fees=pos["entry_fee"], price=price,
            best_price=pos["best"], stop_price=pos["trail"], armed=pos["armed"],
            fee_rate=fee_rate, exit_slippage_bps=bps, config=protection,
        )
        pos.update(best=assessment.best_price, trail=assessment.stop_price, armed=assessment.armed)
        if assessment.should_exit:
            close_position(price, stamp, "profit protection", bps)

    # First evaluation bar is warm-up for the first evaluated signal, just as
    # the old tester does; no pre-window position leaks into the comparison.
    for i in range(evaluation_start + 1, len(df)):
        stamp = df.index[i]
        row, f = df.iloc[i], features.iloc[i - 1]
        price, score = float(row.Open), float(scores.iloc[i - 1])
        q = bool(quorum.iloc[i - 1])
        bps = bps_for(price, float(f.atr))
        had_position = pos is not None
        protect(price, stamp, bps)
        if pos is not None:
            sign = pos["sign"]
            signal_exit = (q and sign * score <= -.16) or bool(
                f.exit_recommended if sign == 1 else f.short_exit_recommended)
            assessment = assess_intraday_exit(
                risk_profile=risk_profile, opened_at=pos["opened"].to_pydatetime(),
                now=stamp.to_pydatetime(), pnl_pct=sign * (price / pos["entry"] - 1) * 100,
                signal_exit=signal_exit and not chandelier,
                round_trip_cost_pct=2 * (fee_rate * 100 + bps / 100) if enhanced else 0.,
            )
            if assessment.should_exit and not trend_holding:
                close_position(price, stamp, assessment.reason, bps)
        if pos is None and not had_position:
            report = {"entry_allowed": bool(f.eligible), "setup": f.setup, "regime": f.regime,
                      "signal_side": f.signal_side, "timeframe_bias": f.higher_timeframe_bias,
                      "metrics": {key: float(f[key]) for key in
                                  ("close", "atr", "support", "resistance")}}
            from strategy.briefing import execution_entry_check
            execution_ok, _ = execution_entry_check(report, price, side=f.signal_side)
            sign = -1 if f.signal_side == "SHORT" else 1
            threshold = sign * float(f.required_committee_score)
            admitted = (bool(f.eligible) and q and sign * score >= threshold
                        and f.signal_confidence >= env.conf_threshold and execution_ok)
            size = min(env.size_max_pct, user_cap_pct)
            entry_stop = max(env.stop_loss_pct, 250 * float(f.atr) / price) if volatility_stop else env.stop_loss_pct
            entry_target = max(env.take_profit_pct, 2.5 * entry_stop) if volatility_stop else env.take_profit_pct
            if volatility_stop and entry_stop > 5.:
                admitted = False
                blocked["Swing stop exceeds 5 percent"] = blocked.get("Swing stop exceeds 5 percent", 0) + 1
            if enhanced:
                admission = assess_committee_entry(
                    report, score=score, quorum=q,
                    categories={cat: float(series.iloc[i - 1]) for cat, series in categories.items()},
                    fee_rate=fee_rate, slippage_bps=bps, stop_loss_pct=entry_stop,
                    take_profit_pct=entry_target,
                    minutes_since_exit=(stamp - last_exit).total_seconds() / 60 if last_exit is not None else None,
                )
                admitted = admitted and admission.allowed
                size = min(size, admission.size_pct)
                if f.eligible:
                    for check in admission.checks:
                        if not check["passed"]:
                            blocked[check["name"]] = blocked.get(check["name"], 0) + 1
            if admitted and adx_min is not None:
                adx = float(controls.adx.iloc[i - 1])
                if not np.isfinite(adx) or adx < adx_min:
                    admitted = False
                    blocked["ADX"] = blocked.get("ADX", 0) + 1
            if admitted and chandelier:
                candidate = float(controls["chandelier_long" if sign == 1 else "chandelier_short"].iloc[i - 1])
                if not np.isfinite(candidate) or sign * (price - candidate) <= 0:
                    admitted = False
                    blocked["Chandelier entry invalidated"] = blocked.get("Chandelier entry invalidated", 0) + 1
            if admitted:
                fill = price * (1 + sign * bps / 10000)
                notional = cash * size / 100 / (1 + fee_rate * size / 100)
                entry_fee = notional * fee_rate
                cash -= notional + entry_fee
                paid += entry_fee
                pos = {"sign": sign, "entry": fill, "quantity": notional / fill,
                       "entry_slip": sign * (fill - price) * notional / fill,
                       "notional": notional, "entry_fee": entry_fee, "opened": stamp,
                       "best": fill, "trail": None, "armed": False, "chandelier_stop": None,
                       "stop_loss_pct": entry_stop, "take_profit_pct": entry_target}
        if pos is not None:
            exposed_bars += 1
            sign = pos["sign"]
            stop = pos["entry"] * (1 - sign * pos["stop_loss_pct"] / 100)
            stop_reason = "hard stop"
            if chandelier:
                candidate = float(controls["chandelier_long" if sign == 1 else "chandelier_short"].iloc[i - 1])
                pos["chandelier_stop"] = ratchet_chandelier(
                    "LONG" if sign == 1 else "SHORT", pos["chandelier_stop"], candidate)
                if pos["chandelier_stop"] is not None and sign * (pos["chandelier_stop"] - stop) > 0:
                    stop, stop_reason = pos["chandelier_stop"], "chandelier stop"
            target = pos["entry"] * (1 + sign * pos["take_profit_pct"] / 100)
            stop_hit = row.Low <= stop if sign == 1 else row.High >= stop
            target_hit = row.High >= target if sign == 1 else row.Low <= target
            if stop_hit:
                # Gaps fill at the worse open, never magically at the stop.
                fill = min(price, stop) if sign == 1 else max(price, stop)
                close_position(fill, stamp, stop_reason, bps)
            elif target_hit and not trend_holding:
                close_position(target, stamp, "target", bps)
        # A Monday open's candle closes five minutes later, not one weekend
        # later. Session gaps must never become the current bar's duration.
        close_stamp = stamp + pd.Timedelta(minutes=bar_minutes)
        protect(float(row.Close), close_stamp, bps)
        if finalize and i == len(df) - 1 and pos is not None:
            close_position(float(row.Close), close_stamp, "end of evaluation", bps)
        mark = cash if pos is None else cash + pos["notional"] + pos["sign"] * (row.Close - pos["entry"]) * pos["quantity"]
        equity.append(float(mark))
    curve = pd.Series([10000.] + equity)
    market_returns = df.Close.iloc[evaluation_start:].reset_index(drop=True).pct_change().fillna(0.)
    strategy_returns = curve.pct_change().fillna(0.)
    up, down = market_returns > 0, market_returns < 0
    upside_capture = (100 * strategy_returns[up].sum() / market_returns[up].sum()) if up.any() else 0.
    downside_capture = (100 * strategy_returns[down].sum() / market_returns[down].sum()) if down.any() else 0.
    pnl = [t["pnl"] for t in trades]
    wins, losses = sum(p for p in pnl if p > 0), -sum(p for p in pnl if p < 0)
    return {"policy": "committee-v4" if enhanced else "research-v3 with hourly/soft-exit fixes",
            "adx_min": adx_min, "chandelier": chandelier, "trend_holding": trend_holding,
            "start": str(df.index[evaluation_start]), "end": str(df.index[-1]),
            "return_pct": (equity[-1] / 10000 - 1) * 100,
            "equity": equity[-1], "cash": cash,
            "open_position": None if pos is None else {**pos, "opened": str(pos["opened"])},
            "finalized": finalize, "bar_minutes": bar_minutes, "volatility_stop": volatility_stop,
            "max_drawdown_pct": float((curve / curve.cummax() - 1).min() * 100),
            "trades": len(trades), "win_rate_pct": 100 * sum(p > 0 for p in pnl) / len(pnl) if pnl else 0.,
            "profit_factor": wins / losses if losses else None,
            "fees": paid, "slippage_cost": slipped, "net_pnl": sum(pnl),
            "gross_pnl": sum(pnl) + paid + slipped,
            "time_in_market_pct": 100 * exposed_bars / len(equity) if equity else 0.,
            "upside_capture_pct": float(upside_capture), "downside_capture_pct": float(downside_capture),
            "mean_hold_minutes": float(np.mean([(pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds() / 60 for t in trades])) if trades else 0.,
            "buy_hold_pct": (df.Close.iloc[-1] / df.Close.iloc[evaluation_start] - 1) * 100,
            "blocked": blocked, "trade_log": trades}
