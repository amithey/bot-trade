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
                     features=None, votes=None):
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
    cash, pos, last_exit, paid = 10000., None, None, 0.
    trades, equity, blocked = [], [], {}

    def bps_for(price, atr):
        return apply_slippage(price, "BUY", cfg=slip_cfg,
                              atr_pct=100 * atr / price)[1] * slippage_multiplier

    def close_position(price, stamp, reason, bps):
        nonlocal cash, pos, last_exit, paid
        sign = pos["sign"]
        fill = price * (1 - sign * bps / 10000)
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
        if pos is None:
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
                signal_exit=signal_exit,
                round_trip_cost_pct=2 * (fee_rate * 100 + bps / 100) if enhanced else 0.,
            )
            if assessment.should_exit:
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
            if enhanced:
                admission = assess_committee_entry(
                    report, score=score, quorum=q,
                    categories={cat: float(series.iloc[i - 1]) for cat, series in categories.items()},
                    fee_rate=fee_rate, slippage_bps=bps, stop_loss_pct=env.stop_loss_pct,
                    take_profit_pct=env.take_profit_pct,
                    minutes_since_exit=(stamp - last_exit).total_seconds() / 60 if last_exit is not None else None,
                )
                admitted = admitted and admission.allowed
                size = min(size, admission.size_pct)
                if f.eligible:
                    for check in admission.checks:
                        if not check["passed"]:
                            blocked[check["name"]] = blocked.get(check["name"], 0) + 1
            if admitted:
                fill = price * (1 + sign * bps / 10000)
                notional = cash * size / 100 / (1 + fee_rate * size / 100)
                entry_fee = notional * fee_rate
                cash -= notional + entry_fee
                paid += entry_fee
                pos = {"sign": sign, "entry": fill, "quantity": notional / fill,
                       "notional": notional, "entry_fee": entry_fee, "opened": stamp,
                       "best": fill, "trail": None, "armed": False}
        if pos is not None:
            sign = pos["sign"]
            stop = pos["entry"] * (1 - sign * env.stop_loss_pct / 100)
            target = pos["entry"] * (1 + sign * env.take_profit_pct / 100)
            stop_hit = row.Low <= stop if sign == 1 else row.High >= stop
            target_hit = row.High >= target if sign == 1 else row.Low <= target
            if stop_hit:
                # Gaps fill at the worse open, never magically at the stop.
                fill = min(price, stop) if sign == 1 else max(price, stop)
                close_position(fill, stamp, "hard stop", bps)
            elif target_hit:
                close_position(target, stamp, "target", bps)
        close_stamp = stamp + (df.index[i] - df.index[i - 1])
        protect(float(row.Close), close_stamp, bps)
        if i == len(df) - 1 and pos is not None:
            close_position(float(row.Close), close_stamp, "end of evaluation", bps)
        mark = cash if pos is None else cash + pos["notional"] + pos["sign"] * (row.Close - pos["entry"]) * pos["quantity"]
        equity.append(float(mark))
    curve = pd.Series([10000.] + equity)
    pnl = [t["pnl"] for t in trades]
    wins, losses = sum(p for p in pnl if p > 0), -sum(p for p in pnl if p < 0)
    return {"policy": "committee-v4" if enhanced else "research-v3 with hourly/soft-exit fixes",
            "start": str(df.index[evaluation_start]), "end": str(df.index[-1]),
            "return_pct": (cash / 10000 - 1) * 100,
            "max_drawdown_pct": float((curve / curve.cummax() - 1).min() * 100),
            "trades": len(trades), "win_rate_pct": 100 * sum(p > 0 for p in pnl) / len(pnl) if pnl else 0.,
            "profit_factor": wins / losses if losses else None,
            "fees": paid, "net_pnl": sum(pnl),
            "buy_hold_pct": (df.Close.iloc[-1] / df.Close.iloc[evaluation_start] - 1) * 100,
            "blocked": blocked, "trade_log": trades}
