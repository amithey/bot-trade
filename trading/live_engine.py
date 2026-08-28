"""
LiveTradingEngine — a background-threaded, pulse-emitting trader.

Drives one continuous analyze→decide→execute loop per selected ticker.
Emits PulseEvents at every stage so the UI can show a live "heartbeat".
Enforces stop-loss / take-profit / daily-loss risk controls automatically.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class PulseStage(str, Enum):
    IDLE        = "IDLE"
    FETCH       = "FETCH"        # pulling market data
    INDICATORS  = "INDICATORS"   # technical indicators computed
    RAG         = "RAG"          # querying knowledge base
    AI          = "AI"           # Claude reasoning
    DECISION    = "DECISION"     # decision produced
    RISK        = "RISK"         # running risk checks on open positions
    EXECUTE     = "EXECUTE"      # placing a BUY / SELL
    REFLECT     = "REFLECT"      # post-trade lesson written to RAG
    SLEEP       = "SLEEP"        # waiting for next cycle
    ERROR       = "ERROR"
    STOPPED     = "STOPPED"


@dataclass
class PulseEvent:
    ts: datetime
    stage: PulseStage
    message: str
    ticker: str = ""
    level: str = "INFO"       # INFO, DECISION, TRADE, WARN, ERROR
    meta: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "ts":      self.ts,
            "stage":   self.stage.value,
            "level":   self.level,
            "ticker":  self.ticker,
            "message": self.message,
        }


class LiveTradingEngine:
    """
    Runs on a background daemon thread; loops until stop() is called.

    The caller (Streamlit page) reads `snapshot()` every refresh tick to
    render the live state, and `drain_events()` to append to a log stream.
    Thread-safety: all mutable state is guarded by `_lock`.
    """

    def __init__(
        self,
        portfolio,
        fetcher,
        retriever,
        engine,
        max_events: int = 500,
        tenant=None,
    ) -> None:
        self.portfolio   = portfolio
        self._fetcher    = fetcher
        self._retriever  = retriever
        self._engine     = engine

        # Commercial context (saas.tenant.Tenant) — decides which strategy
        # modes this user may run, how fast, and whose API key pays. ``None``
        # means single-user mode: no gating, operator's key, no metering.
        self._tenant = tenant
        # Shared-decision-cache accounting, surfaced in snapshot() so the UI
        # can show what the cache is saving.
        self._api_cycles:    int = 0    # cycles that actually called Claude
        self._cached_cycles: int = 0    # cycles served from the shared cache

        # Set by the owner (see dashboard/_shared.get_live_engine) to persist
        # the portfolio after every cycle. None → no persistence.
        self._persist_cb = None

        # Quiet-market gate — skip paying for a fresh model call when the
        # market has not moved enough for the answer to change.
        self._quiet_skip_enabled: bool = True
        self._quiet_price_pct: float = 0.10     # % move that forces a call
        self._quiet_score_delta: float = 0.05   # committee-score move likewise
        self._max_decision_age_sec: float = 900.0   # never coast past 15 min
        self._last_ai_context: dict = {}
        self._quiet_skips: int = 0

        # Config (settable while running via set_config)
        self._ticker         = "BTC-USD"
        self._strategy_mode  = "AI"   # AI | COMMITTEE | HYBRID | BOARDROOM
        self._interval_sec   = 30
        self._risk_profile   = "Balanced"
        self._trade_size_pct = 20.0
        self._stop_loss_pct  = 2.0
        self._take_profit_pct = 4.0
        self._daily_loss_limit_pct = 5.0    # halt trading if daily P&L <= -5%
        self._daily_target_pct     = 0.0    # 0 = disabled; e.g. 1.5 = stop at +1.5%

        # Live state
        self._stage:    PulseStage = PulseStage.IDLE
        self._activity: str = "Idle"
        self._last_decision = None
        self._last_committee: Optional[dict] = None
        self._committee = None               # lazy IndicatorCommittee
        self._boardroom = None               # lazy AnalystBoardroom
        self._boardroom_llm = None           # LLM the boardroom was built on
        self._last_boardroom: Optional[dict] = None
        # Mark-to-market equity history — one point per cycle, ~24h at 30s
        self._equity_history: deque[tuple[datetime, float]] = deque(maxlen=2880)
        self._last_price: float = 0.0
        self._last_snapshot_df = None
        self._cycle_count: int = 0
        self._last_cycle_started: Optional[datetime] = None
        self._next_cycle_at:     Optional[datetime] = None
        self._halt_reason: str = ""

        # Event stream (for UI logs)
        self._events: "queue.Queue[PulseEvent]" = queue.Queue(maxsize=max_events)
        self._event_history: deque[PulseEvent] = deque(maxlen=max_events)

        # Rolling buffer of per-trade lessons used to trigger a batched
        # "meta-lesson" every META_LESSON_EVERY round trips.
        self._META_LESSON_EVERY: int = 5
        self._recent_lessons: deque[dict] = deque(maxlen=50)

        # Safety + slippage controllers — defaults match the current risk
        # profile and refresh whenever set_config(risk_profile=...) fires.
        from risk import SafetyConfig, SafetyController, SlippageConfig
        self._safety = SafetyController(
            SafetyConfig.for_profile(self._risk_profile)
        )
        self._slippage_cfg = SlippageConfig.for_profile(self._risk_profile)
        self._enable_slippage: bool = True

        # Notifications — Telegram / webhook / in-app toasts.
        from notifications import NotificationConfig, NotificationDispatcher
        self._notifier = NotificationDispatcher(NotificationConfig.load())

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        # Set when the user changes ticker / config so the sleep loop can
        # short-circuit and start the next cycle immediately.
        self._wake_event = threading.Event()
        self._lock = threading.RLock()

    # ── Public control ──────────────────────────────────────────────────────

    def start(self) -> None:
        # If stop() was just requested, the old worker may still be unwinding.
        # Do not clear its shared stop flag by starting a second loop.
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._halt_reason = ""
        self._thread = threading.Thread(
            target=self._run_loop, name="bt_live_engine", daemon=True,
        )
        self._thread.start()
        self._emit(PulseStage.IDLE, f"Engine started — watching {self._ticker}",
                   level="INFO")

    def stop(self) -> None:
        self._stop_flag.set()
        self._emit(PulseStage.STOPPED, "Engine stopped by user", level="WARN")

    # ── Safety controls ─────────────────────────────────────────────────────

    def panic_stop(self, reason: str = "User panic-stop") -> dict:
        """Force-close every open position at last known price and halt
        the engine. Returns a summary dict.

        Used as the big red button in the UI when the trader wants to
        bail out of everything immediately.
        """
        closed: list[dict] = []
        errors: list[str] = []
        with self._lock:
            tickers = list(self.portfolio.positions.keys())

        for tk in tickers:
            pos = self.portfolio.positions.get(tk)
            if pos is None:
                continue
            price = float(pos.current_price or pos.avg_entry_price)
            try:
                tr = self.portfolio.sell(
                    tk, price, action_label="FORCE_CLOSE",
                    reasoning=f"PANIC STOP — {reason}",
                )
                closed.append({
                    "ticker": tk, "price": price,
                    "realized_pnl": tr.realized_pnl,
                })
                self._emit(PulseStage.EXECUTE,
                           f"🛑 PANIC FORCE-CLOSE {tk} @ ${price:,.2f} "
                           f"P&L ${tr.realized_pnl:+,.2f}",
                           level="WARN")
            except Exception as exc:
                errors.append(f"{tk}: {exc}")
                self._emit(PulseStage.ERROR,
                           f"Panic close failed for {tk}: {exc}",
                           level="ERROR")

        # Block any further BUYs until manually cleared
        self._safety.manual_block(reason)
        self._stop_flag.set()
        self._halt_reason = f"PANIC STOP: {reason}"
        self._emit(PulseStage.STOPPED,
                   f"🛑 PANIC STOP engaged — {len(closed)} position(s) closed",
                   level="ERROR")

        return {"closed": closed, "errors": errors, "reason": reason}

    def clear_panic(self) -> None:
        """Lift the manual block set by ``panic_stop``."""
        self._safety.manual_block(None)
        self._halt_reason = ""
        self._emit(PulseStage.IDLE, "Panic block lifted", level="INFO")

    def safety_override(self, minutes: float) -> None:
        """Temporarily allow BUYs even when a circuit breaker is active.
        Pass 0 to clear. Useful when the user has reviewed the situation
        and accepts the risk."""
        from datetime import timedelta
        if minutes and minutes > 0:
            self._safety.override_until(
                datetime.utcnow() + timedelta(minutes=float(minutes))
            )
            self._emit(PulseStage.IDLE,
                       f"Safety override active for {minutes:.0f} min",
                       level="WARN")
        else:
            self._safety.override_until(None)
            self._emit(PulseStage.IDLE, "Safety override cleared", level="INFO")

    def get_safety_status(self):
        """Return the current SafetyStatus (live-computed from the trade log)."""
        return self._safety.check(
            self.portfolio.trade_log,
            initial_capital=self.portfolio.initial_capital,
        )

    # ── Notifications ───────────────────────────────────────────────────────

    def update_notifier_config(self, cfg) -> None:
        """Hot-swap the notifier config (called from Settings page)."""
        self._notifier.update_config(cfg)

    def get_notifier(self):
        """Expose the dispatcher for ad-hoc test pings."""
        return self._notifier

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() \
               and not self._stop_flag.is_set()

    # ── Multi-tenancy: entitlements, key routing, shared decisions ─────────

    def set_tenant(self, tenant) -> None:
        """Attach (or replace) the commercial context.

        Re-coerces the current mode and interval immediately, so a user who
        exhausts their trial budget mid-session is stepped down to COMMITTEE
        on the very next cycle rather than at the next config change.
        """
        with self._lock:
            self._tenant = tenant
            mode, note = self._coerce_mode(self._strategy_mode)
            if note:
                self._emit(PulseStage.IDLE, note, level="WARN")
            self._strategy_mode = mode
            self._interval_sec, ivl_note = self._clamp_interval(self._interval_sec)
            if ivl_note:
                self._emit(PulseStage.IDLE, ivl_note, level="WARN")

    def _entitlement(self):
        """This user's entitlement, or ``None`` in single-user mode."""
        t = self._tenant
        if t is None:
            return None
        try:
            return t.entitlement
        except Exception as exc:                               # noqa: BLE001
            self._emit(PulseStage.ERROR,
                       f"Entitlement check failed ({exc}) — assuming free tier",
                       level="WARN")
            from saas.plans import resolve as _resolve
            return _resolve("FREE", has_own_key=False, platform_spent_usd=1e9)

    def _coerce_mode(self, mode: str) -> tuple[str, str]:
        ent = self._entitlement()
        if ent is None:
            return mode, ""
        return ent.coerce_mode(mode)

    def _clamp_interval(self, seconds) -> tuple[int, str]:
        ent = self._entitlement()
        if ent is None:
            return max(5, int(seconds)), ""
        return ent.clamp_interval(seconds)

    def _ai_engine(self):
        """The engine whose API key funds this cycle.

        With a tenant, that is the engine bound to their own key (metered and
        attributed to them).  Without one, the process-wide engine built from
        the operator's ``.env``.
        """
        t = self._tenant
        if t is None:
            return self._engine
        eng = t.engine()
        return eng if eng is not None else self._engine

    def _get_boardroom(self):
        """The analyst boardroom bound to the *current* funding engine.

        Rebuilt whenever the engine changes — a user who pastes their own key
        mid-session must not keep convening a boardroom wired to the previous
        key's LLM.
        """
        engine = self._ai_engine()
        llm = getattr(engine, "llm", None)
        if llm is None:
            raise RuntimeError("AI engine exposes no LLM")

        # Analysts and chairman may run on different models — see
        # AnalystBoardroom and the LLM_MODEL_ANALYST / LLM_MODEL_CHAIR
        # settings. Both default to the engine's own model.
        from config.settings import settings
        make = getattr(engine, "make_llm", None)
        if make is not None:
            analyst_llm = make(settings.llm_model_analyst)
            chair_llm = make(settings.llm_model_chair)
        else:                       # a stand-in engine in tests
            analyst_llm = chair_llm = llm

        if self._boardroom is None or self._boardroom_llm is not analyst_llm:
            from decision_engine.boardroom import AnalystBoardroom
            self._boardroom = AnalystBoardroom(analyst_llm,
                                               chair_llm=chair_llm)
            self._boardroom_llm = analyst_llm
        return self._boardroom

    @staticmethod
    def _bar_stamp(snap) -> str:
        """Identity of the most recent bar — the unit of decision sharing.

        Two users looking at the same symbol on the same bar are looking at
        exactly the same market state, so they should never pay for two calls.
        """
        try:
            return str(snap.data.index[-1])
        except Exception:                                      # noqa: BLE001
            return "unknown"

    def _is_quiet_since_last_decision(self, price: float, verdict
                                      ) -> tuple[bool, str]:
        """Has anything material moved since the last model call?

        Returns ``(quiet, reason)``. ``quiet`` means the market state is close
        enough to the last decision that paying for a fresh one is waste.

        Three ways to fail the check, any of which forces a real call:

        * no prior decision to lean on;
        * the price or the 38-indicator score moved past its threshold;
        * the last decision is older than ``_max_decision_age_sec`` — a
          staleness ceiling so a flat market can never leave the bot running
          on an opinion from hours ago.
        """
        if not self._quiet_skip_enabled:
            return False, ""

        with self._lock:
            ctx = self._last_ai_context
            last = self._last_decision
            max_age = self._max_decision_age_sec
            price_tol = self._quiet_price_pct
            score_tol = self._quiet_score_delta

        if last is None or not ctx:
            return False, ""

        age = time.time() - float(ctx.get("ts", 0))
        if age >= max_age:
            return False, ""

        prev_price = float(ctx.get("price") or 0)
        if prev_price <= 0:
            return False, ""
        move_pct = abs(price / prev_price - 1.0) * 100.0
        if move_pct >= price_tol:
            return False, ""

        # A committee swing is a regime change even at a flat price.
        if verdict is not None and ctx.get("score") is not None:
            if abs(float(verdict.score) - float(ctx["score"])) >= score_tol:
                return False, ""
            # An outright change of side always deserves a fresh look.
            if ctx.get("committee_action") and \
                    verdict.action != ctx["committee_action"]:
                return False, ""

        return True, (f"Quiet market — price {move_pct:+.3f}% since the last "
                      f"call {age/60:.1f}m ago")

    def _remember_ai_context(self, price: float, verdict) -> None:
        """Record what the market looked like when a decision was paid for."""
        with self._lock:
            self._last_ai_context = {
                "ts": time.time(),
                "price": float(price),
                "score": (float(verdict.score) if verdict is not None else None),
                "committee_action": (verdict.action if verdict is not None
                                     else None),
            }

    def _shared_decision(self, cache_key: str, ttl: float, compute, *,
                         mode: str, ticker: str):
        """Run ``compute`` unless another user already ran it for this bar.

        Returns whatever ``compute`` returns.  On a hit, emits a pulse so the
        saving is visible in the log and records it against the tenant's
        ledger, which is what lets the operator prove the cache is working
        rather than assert it.
        """
        from decision_engine.decision_cache import get_decision_cache

        # Tag whatever the meter records next, so the usage page can break
        # spend down by strategy mode and symbol.
        meter = getattr(self._ai_engine(), "usage_meter", None)
        if meter is not None:
            meter.set_context(mode=mode, ticker=ticker)

        cache = get_decision_cache()
        value, was_cached = cache.get_or_compute(cache_key, compute, ttl=ttl)
        with self._lock:
            if was_cached:
                self._cached_cycles += 1
            else:
                self._api_cycles += 1
        if was_cached:
            self._emit(PulseStage.AI,
                       f"⚡ Reused this bar's {mode} verdict from the shared "
                       f"cache — no API call", level="INFO")
            t = self._tenant
            if t is not None:
                try:
                    t.record_saving(mode=mode, ticker=ticker)
                except Exception:                              # noqa: BLE001
                    pass
        return value

    def set_config(
        self,
        ticker: Optional[str] = None,
        strategy_mode: Optional[str] = None,
        interval_sec: Optional[int] = None,
        risk_profile: Optional[str] = None,
        trade_size_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        daily_loss_limit_pct: Optional[float] = None,
        daily_target_pct: Optional[float] = None,
        quiet_skip: Optional[bool] = None,
    ) -> None:
        with self._lock:
            if ticker is not None and ticker.upper() != self._ticker:
                self._ticker = ticker.upper()
                self._last_decision = None
                self._last_snapshot_df = None
                self._emit(PulseStage.IDLE,
                           f"Ticker changed → {self._ticker}", level="INFO")
                # Wake the sleep loop so the next cycle starts immediately
                # instead of waiting out the remaining interval.
                self._wake_event.set()
            if strategy_mode is not None \
                    and strategy_mode in ("AI", "COMMITTEE", "HYBRID",
                                          "BOARDROOM"):
                # A mode the plan does not cover degrades to COMMITTEE rather
                # than failing — the user keeps trading, just deterministically.
                mode, note = self._coerce_mode(strategy_mode)
                if note:
                    self._emit(PulseStage.IDLE, note, level="WARN")
                if mode != self._strategy_mode:
                    self._strategy_mode = mode
                    self._last_decision = None
                    self._last_committee = None
                    self._emit(PulseStage.IDLE,
                               f"Strategy mode → {mode}", level="INFO")
                    self._wake_event.set()
            if interval_sec is not None:
                secs, note = self._clamp_interval(interval_sec)
                if note:
                    self._emit(PulseStage.IDLE, note, level="WARN")
                self._interval_sec = max(5, secs)
            if risk_profile is not None and risk_profile != self._risk_profile:
                self._risk_profile = risk_profile
                from risk import SafetyConfig, SlippageConfig
                self._safety.set_config(SafetyConfig.for_profile(risk_profile))
                self._slippage_cfg = SlippageConfig.for_profile(risk_profile)
            if trade_size_pct is not None:
                self._trade_size_pct = float(trade_size_pct)
            if stop_loss_pct is not None:
                self._stop_loss_pct = float(stop_loss_pct)
            if take_profit_pct is not None:
                self._take_profit_pct = float(take_profit_pct)
            if daily_loss_limit_pct is not None:
                self._daily_loss_limit_pct = float(daily_loss_limit_pct)
            if daily_target_pct is not None:
                self._daily_target_pct = float(daily_target_pct)
            if quiet_skip is not None:
                self._quiet_skip_enabled = bool(quiet_skip)

    # ── Public read ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Compact state snapshot for the UI."""
        with self._lock:
            return {
                "running":         self.is_running(),
                "stage":           self._stage.value,
                "activity":        self._activity,
                "ticker":          self._ticker,
                "strategy_mode":   self._strategy_mode,
                "last_committee":  self._last_committee,
                "last_boardroom":  self._last_boardroom,
                "equity_history":  list(self._equity_history),
                "interval_sec":    self._interval_sec,
                "risk_profile":    self._risk_profile,
                "trade_size_pct":  self._trade_size_pct,
                "stop_loss_pct":   self._stop_loss_pct,
                "take_profit_pct": self._take_profit_pct,
                "daily_target_pct": self._daily_target_pct,
                "daily_loss_limit_pct": self._daily_loss_limit_pct,
                "last_decision":   self._last_decision,
                "last_price":      self._last_price,
                "last_df":         self._last_snapshot_df,
                "cycle_count":     self._cycle_count,
                "cycle_started":   self._last_cycle_started,
                "next_cycle_at":   self._next_cycle_at,
                "halt_reason":     self._halt_reason,
                "safety":          self.get_safety_status().to_dict(),
                "slippage_enabled": self._enable_slippage,
                "api_cycles":      self._api_cycles,
                "cached_cycles":   self._cached_cycles,
                "quiet_skips":     self._quiet_skips,
                "tenant":          self._tenant_snapshot(),
            }

    def _tenant_snapshot(self) -> Optional[dict]:
        """Plan / funding / entitlement summary for the UI, or ``None``."""
        t = self._tenant
        if t is None:
            return None
        try:
            return t.to_dict()
        except Exception:                                      # noqa: BLE001
            return None

    def drain_events(self) -> list[PulseEvent]:
        """Drain all queued events (for log append in UI)."""
        out: list[PulseEvent] = []
        try:
            while True:
                out.append(self._events.get_nowait())
        except queue.Empty:
            pass
        return out

    def history(self, limit: int = 200) -> list[PulseEvent]:
        """Recent event history (for log panel)."""
        with self._lock:
            return list(self._event_history)[-limit:]

    # ── Internal helpers ────────────────────────────────────────────────────

    def _emit(
        self,
        stage: PulseStage,
        message: str,
        level: str = "INFO",
        meta: Optional[dict] = None,
    ) -> None:
        evt = PulseEvent(
            ts=datetime.now(),
            stage=stage,
            message=message,
            ticker=self._ticker,
            level=level,
            meta=meta or {},
        )
        with self._lock:
            self._stage = stage
            self._activity = message
            self._event_history.append(evt)
        try:
            self._events.put_nowait(evt)
        except queue.Full:
            try: self._events.get_nowait()
            except queue.Empty: pass
            try: self._events.put_nowait(evt)
            except queue.Full: pass

        # Fan-out to notification channels (Telegram / webhook / toasts).
        # Map our internal levels to notification categories and let the
        # dispatcher handle filters + dedup. Always best-effort.
        try:
            cat = None
            if level == "TRADE":
                cat = "TRADE"
            elif level == "ERROR":
                cat = "ERROR"
            elif stage == PulseStage.RISK or "PANIC" in message.upper() \
                    or "🛡" in message or "🛑" in message:
                cat = "RISK"
            elif level == "DECISION":
                cat = "DECISION"
            elif stage == PulseStage.REFLECT:
                cat = "REFLECT"
            if cat is not None:
                self._notifier.notify(
                    cat, message,
                    meta=evt.meta, ticker=evt.ticker, level=level,
                )
        except Exception:  # noqa: BLE001
            pass  # Notifications never break the trading loop

    def _reflect_on_sell(self, sell_record, exit_reasoning: str) -> None:
        """
        Best-effort post-trade reflection: pair this SELL with the most
        recent BUY on the same ticker from ``portfolio.trade_log``, ask
        Claude for a 2-3 sentence lesson, and persist it to the RAG
        collection via ``retriever.add_lesson``. Never raises.
        """
        try:
            from decision_engine.reflector import reflect_on_trade
            ticker = sell_record.ticker
            # Find matching entry: last BUY for this ticker before the SELL
            entry = None
            for rec in reversed(list(self.portfolio.trade_log)):
                if rec is sell_record:
                    continue
                if rec.ticker == ticker and rec.action == "BUY":
                    entry = rec
                    break
            if entry is None:
                return

            exit_price  = float(sell_record.price)
            entry_price = float(entry.price)
            pnl         = float(sell_record.realized_pnl)
            pnl_pct     = (exit_price - entry_price) / entry_price * 100.0 \
                          if entry_price else None

            llm = getattr(self._engine, "llm", None)
            if llm is None:
                return
            lesson = reflect_on_trade(
                llm,
                ticker=ticker,
                risk_profile=self._risk_profile,
                entry_price=entry_price,
                exit_price=exit_price,
                realized_pnl=pnl,
                entry_reasoning=entry.reasoning or "",
                exit_reasoning=exit_reasoning or sell_record.reasoning or "",
                pnl_pct=pnl_pct,
            )
            if not lesson:
                return
            try:
                self._retriever.add_lesson(
                    lesson,
                    metadata={
                        "ticker": ticker,
                        "action": "round_trip",
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else 0.0,
                        "risk_profile": self._risk_profile,
                    },
                )
            except Exception as exc:
                self._emit(PulseStage.ERROR,
                           f"Lesson persist failed: {exc}", level="WARN")
                return
            preview = (lesson[:120] + "…") if len(lesson) > 120 else lesson
            self._emit(PulseStage.REFLECT,
                       f"📘 Lesson learned · {ticker} · {preview}",
                       level="INFO",
                       meta={"lesson": lesson, "pnl": pnl})

            # Buffer for meta-lesson roll-up; trigger every N round trips
            self._recent_lessons.append({
                "ticker":  ticker,
                "pnl":     pnl,
                "pnl_pct": pnl_pct,
                "lesson":  lesson,
            })
            if len(self._recent_lessons) >= self._META_LESSON_EVERY \
               and len(self._recent_lessons) % self._META_LESSON_EVERY == 0:
                self._roll_up_meta_lesson()

            # Refresh the personal win-probability classifier so the next
            # BUY decision benefits from the just-closed round trip.
            self._maybe_retrain_journal_ml()
        except Exception as exc:
            # Reflection must never break the trading loop
            self._emit(PulseStage.REFLECT,
                       f"Reflection skipped: {exc}", level="WARN")

    # ------------------------------------------------------------------ #
    # Journal-ML auto-retrain
    # ------------------------------------------------------------------ #

    def _maybe_retrain_journal_ml(self) -> None:
        """Refit the TradeJournalML classifier on the portfolio trade log.

        Called after every closed round trip. The fit itself is cheap
        (RandomForest on ≤ a few hundred rows), but the per-trade history
        fetch can hit yfinance, so we throttle: only refit when the total
        number of closed (BUY+SELL) round trips is a multiple of 3 AND ≥10.
        Wraps everything in try/except — never breaks the loop.
        """
        try:
            from ml.trade_journal_ml import TradeJournalML
        except Exception as exc:
            self._emit(PulseStage.REFLECT,
                       f"Journal-ML import failed: {exc}", level="WARN")
            return

        # Count closed round trips in the trade log.
        sells = sum(
            1 for r in self.portfolio.trade_log
            if r.action in ("SELL", "FORCE_CLOSE")
        )
        if sells < 10 or sells % 3 != 0:
            return

        def _fetch_history(ticker: str, end_dt):
            """Return enriched OHLCV up to end_dt for a single trade."""
            try:
                snap = self._fetcher.fetch_latest(ticker, lookback_days=180)
                df = snap.data
                if df is None or df.empty:
                    return None
                # Keep only rows at-or-before the trade timestamp
                if end_dt is not None:
                    try:
                        idx = df.index
                        # tz-aware vs naive handling
                        if hasattr(idx, "tz") and idx.tz is not None:
                            from datetime import timezone
                            cutoff = end_dt.replace(tzinfo=timezone.utc) \
                                if end_dt.tzinfo is None else end_dt
                        else:
                            cutoff = end_dt.replace(tzinfo=None) \
                                if end_dt.tzinfo is not None else end_dt
                        df = df.loc[df.index <= cutoff]
                    except Exception:
                        pass
                return df if len(df) >= 30 else None
            except Exception:
                return None

        try:
            jm = TradeJournalML()
            n = jm.fit_from_portfolio(
                list(self.portfolio.trade_log), _fetch_history
            )
            self._emit(PulseStage.REFLECT,
                       f"🎓 Journal-ML refit on {n} historical trades",
                       level="INFO",
                       meta={"n_train": n})
        except Exception as exc:
            self._emit(PulseStage.REFLECT,
                       f"Journal-ML retrain skipped: {exc}", level="WARN")

    def _roll_up_meta_lesson(self) -> None:
        """
        Summarise the last ``_META_LESSON_EVERY`` per-trade lessons into a
        single meta-rule and persist it to RAG with ``type='meta_lesson'``.
        Best-effort — must never break the trading loop.
        """
        try:
            from decision_engine.reflector import summarize_meta_lesson
            batch = list(self._recent_lessons)[-self._META_LESSON_EVERY:]
            llm = getattr(self._engine, "llm", None)
            if llm is None or not batch:
                return
            meta_text = summarize_meta_lesson(
                llm, risk_profile=self._risk_profile, round_trips=batch,
            )
            if not meta_text:
                return
            wins  = sum(1 for r in batch if r["pnl"] > 0)
            losses = sum(1 for r in batch if r["pnl"] < 0)
            total = sum(r["pnl"] for r in batch)
            try:
                self._retriever.add_lesson(
                    meta_text,
                    metadata={
                        "type": "meta_lesson",
                        "batch_size": len(batch),
                        "wins": wins,
                        "losses": losses,
                        "total_pnl": round(total, 2),
                        "risk_profile": self._risk_profile,
                    },
                )
            except Exception as exc:
                self._emit(PulseStage.ERROR,
                           f"Meta-lesson persist failed: {exc}", level="WARN")
                return
            preview = (meta_text[:140] + "…") \
                      if len(meta_text) > 140 else meta_text
            self._emit(PulseStage.REFLECT,
                       f"🧠 Meta-lesson ({wins}W/{losses}L · "
                       f"${total:+,.2f}) — {preview}",
                       level="INFO",
                       meta={"meta_lesson": meta_text,
                             "batch_size": len(batch)})
        except Exception as exc:
            self._emit(PulseStage.REFLECT,
                       f"Meta-lesson skipped: {exc}", level="WARN")

    @staticmethod
    def _combine_hybrid(ticker: str, verdict, ai_dec):
        """
        Merge the committee verdict with Claude's opinion into one decision.

        Rules (long-only, exit-to-cash):
        * BUY  — committee says BUY and the AI does not actively object
                 (anything but SELL). Both agreeing earns a confidence
                 bonus; an AI HOLD keeps the committee's confidence.
        * SELL — committee says SELL (it owns the systematic exit), OR
                 the AI says SELL while the committee leans bearish
                 (score ≤ 0). The AI alone can never panic-sell against
                 a bullish committee.
        * HOLD — everything else.
        """
        com_dec = verdict.to_trading_decision(ticker)
        ai_ok = not ai_dec.is_fallback

        if verdict.action == "BUY" and (not ai_ok or ai_dec.action != "SELL"):
            action = "BUY"
            if ai_ok and ai_dec.action == "BUY":
                conf = min(1.0, max(com_dec.confidence_score,
                                    ai_dec.confidence_score) + 0.08)
            else:
                conf = com_dec.confidence_score
        elif verdict.action == "SELL" or \
                (ai_ok and ai_dec.action == "SELL" and verdict.score <= 0):
            action = "SELL"
            conf = min(1.0, max(com_dec.confidence_score,
                                ai_dec.confidence_score
                                if ai_ok and ai_dec.action == "SELL"
                                else 0.0))
        else:
            action = "HOLD"
            conf = 0.5

        ai_part = ("(AI opinion unavailable — fallback)" if not ai_ok else
                   f"AI says {ai_dec.action} "
                   f"(conf {ai_dec.confidence_score:.0%}): "
                   f"{ai_dec.reasoning[:400]}")
        reasoning = (
            f"HYBRID DECISION — committee × Claude.\n"
            f"Committee: {verdict.action} — {verdict.bulls} bull / "
            f"{verdict.bears} bear of {verdict.total} "
            f"(score {verdict.score:+.2f}).\n"
            f"{ai_part}\n"
            f"Final: {action}."
        )

        merged_keys = list(dict.fromkeys(
            (com_dec.key_indicators or []) + (ai_dec.key_indicators or [])
        ))[:6]
        base = ai_dec if ai_ok else com_dec
        return base.model_copy(update={
            "action": action,
            "confidence_score": round(conf, 3),
            "reasoning": reasoning,
            "key_indicators": merged_keys,
            "is_fallback": False,
        })

    def _liquidate_all(self, reason: str) -> None:
        """Close every open position at its last known price (used on halt)."""
        for tk, pos in list(self.portfolio.positions.items()):
            try:
                tr = self.portfolio.sell(
                    tk, pos.current_price, action_label="FORCE_CLOSE",
                    reasoning=reason,
                )
                emoji = "🟢" if tr.realized_pnl >= 0 else "🔴"
                self._emit(PulseStage.EXECUTE,
                           f"{emoji} FORCE_CLOSE {tk} @ "
                           f"${pos.current_price:,.2f} "
                           f"P&L ${tr.realized_pnl:+,.2f} — {reason}",
                           level="TRADE")
                self._reflect_on_sell(tr, exit_reasoning=reason)
            except Exception as exc:
                self._emit(PulseStage.ERROR,
                           f"Force close {tk} failed: {exc}", level="ERROR")

    # ── Core loop ───────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        try:
            while not self._stop_flag.is_set():
                self._cycle_once()
                # Checkpoint here rather than inside _cycle_once: the
                # stop-loss and take-profit paths sell and then return early,
                # so a save at the end of the cycle body would miss exactly
                # the trades that matter most.
                self._checkpoint()
                if self._stop_flag.is_set():
                    break
                # Sleep in short ticks so stop() is responsive
                with self._lock:
                    self._stage = PulseStage.SLEEP
                    self._next_cycle_at = datetime.now().fromtimestamp(
                        time.time() + self._interval_sec)
                # Note: do NOT clear _wake_event here. If set_config was
                # called DURING the cycle, wake_event is already set and
                # we want to skip sleep entirely. Clearing happens at the
                # top of the next cycle.
                if self._wake_event.is_set():
                    with self._lock:
                        self._activity = "Ticker/config changed — starting next cycle immediately"
                    self._wake_event.clear()
                    continue
                end_time = time.time() + self._interval_sec
                while time.time() < end_time and not self._stop_flag.is_set():
                    if self._wake_event.is_set():
                        with self._lock:
                            self._activity = "Waking up — ticker/config changed"
                        self._wake_event.clear()
                        break
                    remaining = int(end_time - time.time())
                    with self._lock:
                        self._activity = f"Next cycle in {remaining}s"
                    # Wait up to 1s but exit early if wake_event fires
                    self._wake_event.wait(timeout=min(1.0,
                                                      end_time - time.time()))
        except Exception as exc:
            self._emit(PulseStage.ERROR,
                       f"Loop crashed: {type(exc).__name__}: {exc}",
                       level="ERROR",
                       meta={"traceback": traceback.format_exc()})
        finally:
            # A crashed or stopped loop still owes the user their trades.
            self._checkpoint()
            self._emit(PulseStage.STOPPED, "Loop exited", level="WARN")

    def _checkpoint(self) -> None:
        """Persist the portfolio, if the owner gave us somewhere to put it.

        Never raises: a disk problem must not take down a running bot, and
        the next cycle will try again a few seconds later.
        """
        cb = self._persist_cb
        if cb is None:
            return
        try:
            cb(self.portfolio)
        except Exception as exc:                               # noqa: BLE001
            self._emit(PulseStage.ERROR,
                       f"Could not save portfolio: {type(exc).__name__}",
                       level="WARN")

    def set_persist_callback(self, cb) -> None:
        """Install the function that writes the portfolio to durable storage.

        Called after every cycle and once more when the loop exits.  ``None``
        disables persistence (single-user / test use).
        """
        with self._lock:
            self._persist_cb = cb

    def _cycle_once(self) -> None:
        with self._lock:
            ticker           = self._ticker
            risk_profile     = self._risk_profile
            trade_size_pct   = self._trade_size_pct
            daily_limit      = self._daily_loss_limit_pct
            self._cycle_count += 1
            self._last_cycle_started = datetime.now()

        # SL / TP come from the profile envelope so they track the risk setting
        from config.user_profile import RISK_ENVELOPES
        env = RISK_ENVELOPES.get(risk_profile, RISK_ENVELOPES["Balanced"])
        stop_loss_pct   = env.stop_loss_pct
        take_profit_pct = env.take_profit_pct
        with self._lock:
            self._stop_loss_pct   = stop_loss_pct
            self._take_profit_pct = take_profit_pct

        # Daily loss guard
        daily_pct = self.portfolio.get_daily_pnl_pct()
        if daily_pct <= -abs(daily_limit):
            self._halt_reason = (f"Daily loss limit hit ({daily_pct:+.2f}% "
                                 f"≤ -{daily_limit:.1f}%)")
            self._emit(PulseStage.ERROR, self._halt_reason, level="ERROR")
            self._stop_flag.set()
            return

        # Daily profit target (0 = disabled)
        target = self._daily_target_pct
        if target > 0 and daily_pct >= target:
            self._halt_reason = (f"🎯 Daily profit target reached "
                                 f"({daily_pct:+.2f}% ≥ +{target:.2f}%) — "
                                 f"bot will halt to lock in gains")
            self._emit(PulseStage.DECISION, self._halt_reason, level="TRADE")
            # Close any open positions at current price before halting
            self._liquidate_all("Daily target reached")
            self._stop_flag.set()
            return

        # ── Fetch ───────────────────────────────────────────────────────────
        # Try 5m intraday first. If that fails (stock market closed / weekend
        # → insufficient bars for SMA_200), fall back once to daily bars.
        # Fundamentals are cached inside the fetcher (1-hour TTL) so only
        # the first cycle per ticker pays the yfinance.info latency.
        self._emit(PulseStage.FETCH,
                   f"Fetching 5-minute candles for {ticker}…", level="INFO")
        snap = None
        used_interval = "5m"
        last_exc: Optional[Exception] = None
        try:
            snap = self._fetcher.fetch_with_fundamentals(
                ticker, period="5d", interval="5m",
            )
        except Exception as exc:
            last_exc = exc

        if snap is None or snap.data is None or len(snap.data) == 0:
            self._emit(PulseStage.FETCH,
                       f"5m bars unavailable — falling back to 1d "
                       f"(market likely closed)", level="WARN")
            try:
                snap = self._fetcher.fetch_with_fundamentals(
                    ticker, period="1y", interval="1d",
                )
                used_interval = "1d"
            except Exception as exc:
                last_exc = exc
                snap = None

        if snap is None or len(snap.data) == 0:
            msg = (f"No market data for {ticker}: {last_exc}" if last_exc
                   else f"No market data for {ticker}")
            self._emit(PulseStage.ERROR, msg, level="ERROR")
            return

        try:
            price = float(snap.data["Close"].iloc[-1])
        except Exception:
            self._emit(PulseStage.ERROR, "Empty market data",
                       level="ERROR")
            return

        with self._lock:
            self._last_price = price
            self._last_snapshot_df = snap.data

        # ── Indicators ──────────────────────────────────────────────────────
        last = snap.data.iloc[-1]
        def _col(prefix: str):
            c = next((c for c in snap.data.columns if c.startswith(prefix)), None)
            return float(last[c]) if c else None

        rsi_val   = _col("RSI_")
        atr_val   = _col("ATR_")
        bbw_val   = _col("BB_Width_")
        vwap_val  = _col("VWAP_")

        ind_parts = [f"Price ${price:,.2f}"]
        if rsi_val  is not None: ind_parts.append(f"RSI {rsi_val:.1f}")
        if atr_val  is not None: ind_parts.append(f"ATR ${atr_val:,.2f}")
        if bbw_val  is not None: ind_parts.append(f"BBw {bbw_val:.2f}%")
        if vwap_val is not None: ind_parts.append(f"VWAP ${vwap_val:,.2f}")
        self._emit(PulseStage.INDICATORS, " · ".join(ind_parts), level="INFO")

        # ── Risk management on existing position ────────────────────────────
        self.portfolio.update_price(ticker, price)
        with self._lock:
            self._equity_history.append(
                (datetime.now(), float(self.portfolio.get_total_value()))
            )

        # Re-check daily guards after mark-to-market refresh. The early guard
        # above protects already-realized limits; this one catches fresh price
        # moves in the current cycle and closes using the latest known mark.
        daily_pct = self.portfolio.get_daily_pnl_pct()
        if daily_pct <= -abs(daily_limit):
            self._halt_reason = (f"Daily loss limit hit ({daily_pct:+.2f}% "
                                 f"<= -{daily_limit:.1f}%)")
            self._emit(PulseStage.ERROR, self._halt_reason, level="ERROR")
            self._liquidate_all("Daily loss limit hit")
            self._stop_flag.set()
            return

        target = self._daily_target_pct
        if target > 0 and daily_pct >= target:
            self._halt_reason = (f"Daily profit target reached "
                                 f"({daily_pct:+.2f}% >= +{target:.2f}%) - "
                                 f"bot will halt to lock in gains")
            self._emit(PulseStage.DECISION, self._halt_reason, level="TRADE")
            self._liquidate_all("Daily target reached")
            self._stop_flag.set()
            return

        pos = self.portfolio.positions.get(ticker)
        if pos is not None:
            pnl_pct = pos.unrealized_pnl_pct
            self._emit(PulseStage.RISK,
                       f"Position open: {pos.quantity:.4f} @ "
                       f"${pos.avg_entry_price:,.2f} · P&L {pnl_pct:+.2f}%",
                       level="INFO")
            if pnl_pct <= -abs(stop_loss_pct):
                self._emit(PulseStage.EXECUTE,
                           f"🛑 STOP-LOSS hit ({pnl_pct:+.2f}% ≤ "
                           f"-{stop_loss_pct:.1f}%) — force closing",
                           level="WARN")
                try:
                    if self._enable_slippage:
                        from risk import apply_slippage
                        sl_px, _ = apply_slippage(
                            price, "SELL", cfg=self._slippage_cfg)
                    else:
                        sl_px = price
                    tr = self.portfolio.sell(
                        ticker, sl_px, action_label="SELL",
                        reasoning=f"Auto stop-loss at {pnl_pct:+.2f}%",
                    )
                    self._emit(PulseStage.EXECUTE,
                               f"🔴 SELL {ticker} @ ${price:,.2f} "
                               f"P&L ${tr.realized_pnl:+,.2f}",
                               level="TRADE")
                    self._reflect_on_sell(
                        tr, exit_reasoning=f"Auto stop-loss at {pnl_pct:+.2f}%")
                except Exception as exc:
                    self._emit(PulseStage.ERROR,
                               f"Stop-loss sell failed: {exc}", level="ERROR")
                return
            if pnl_pct >= abs(take_profit_pct):
                self._emit(PulseStage.EXECUTE,
                           f"🎯 TAKE-PROFIT hit ({pnl_pct:+.2f}% ≥ "
                           f"{take_profit_pct:.1f}%) — locking in gains",
                           level="WARN")
                try:
                    if self._enable_slippage:
                        from risk import apply_slippage
                        tp_px, _ = apply_slippage(
                            price, "SELL", cfg=self._slippage_cfg)
                    else:
                        tp_px = price
                    tr = self.portfolio.sell(
                        ticker, tp_px, action_label="SELL",
                        reasoning=f"Auto take-profit at {pnl_pct:+.2f}%",
                    )
                    self._emit(PulseStage.EXECUTE,
                               f"🟢 SELL {ticker} @ ${price:,.2f} "
                               f"P&L ${tr.realized_pnl:+,.2f}",
                               level="TRADE")
                    self._reflect_on_sell(
                        tr, exit_reasoning=f"Auto take-profit at {pnl_pct:+.2f}%")
                except Exception as exc:
                    self._emit(PulseStage.ERROR,
                               f"Take-profit sell failed: {exc}", level="ERROR")
                return

        # ── Decision — committee vote, RAG + Claude, or both (hybrid) ──────
        # Re-check entitlement every cycle, not just on config change: a trial
        # budget can run out mid-session, and the right response is a quiet
        # step down to COMMITTEE rather than a stream of API errors.
        with self._lock:
            strategy_mode = self._strategy_mode
        strategy_mode, gate_note = self._coerce_mode(strategy_mode)
        if gate_note:
            self._emit(PulseStage.AI, gate_note, level="WARN")
            with self._lock:
                self._strategy_mode = strategy_mode

        # Everything downstream of the same bar is the same decision, so the
        # cache key is the market state plus only those user-side inputs that
        # genuinely change the answer.
        from decision_engine.decision_cache import (
            make_key, pnl_bucket, position_bucket,
        )
        _pos = self.portfolio.positions.get(ticker)
        _cache_ttl = 270.0 if used_interval == "5m" else 1800.0
        _shared_key_parts = dict(
            v=1,
            ticker=ticker,
            interval=used_interval,
            bar=self._bar_stamp(snap),
            mode=strategy_mode,
            risk=risk_profile,
        )

        verdict = None
        if strategy_mode in ("COMMITTEE", "HYBRID", "BOARDROOM"):
            self._emit(PulseStage.AI,
                       "Committee of 38 indicators voting…", level="INFO")
            try:
                if self._committee is None:
                    from strategy.committee import IndicatorCommittee
                    self._committee = IndicatorCommittee()
                verdict = self._committee.vote_latest(
                    snap.data,
                    in_position=ticker in self.portfolio.positions,
                )
            except Exception as exc:
                if strategy_mode == "BOARDROOM":
                    # The quant seat will simply abstain — the meeting
                    # can still convene on the other three packets.
                    self._emit(PulseStage.AI,
                               f"Quant packet unavailable ({exc}) — "
                               f"boardroom convenes without it",
                               level="WARN")
                else:
                    self._emit(PulseStage.ERROR, f"Committee failed: {exc}",
                               level="ERROR")
                    return
            if verdict is not None:
                with self._lock:
                    self._last_committee = verdict.summary_dict()

        # ── Quiet-market gate ───────────────────────────────────────────────
        # The shared cache already collapses repeat cycles *within* one bar.
        # This covers the next case up: a new bar where nothing actually
        # moved. Re-asking a nine-seat boardroom because the price ticked
        # 0.02% is money for an answer that was never going to change.
        #
        # Crucially this SKIPS THE CYCLE rather than re-running the old
        # verdict through execution: acting on a stale BUY every quiet cycle
        # would pyramid into the position again and again. Stop-loss and
        # take-profit already ran above and are unaffected — they are
        # deterministic and never consult the model.
        #
        # COMMITTEE is exempt: it costs nothing, so there is nothing to save.
        if strategy_mode != "COMMITTEE":
            quiet, why = self._is_quiet_since_last_decision(price, verdict)
            if quiet:
                with self._lock:
                    self._quiet_skips += 1
                self._emit(PulseStage.AI,
                           f"😴 {why} — holding the last {strategy_mode} "
                           f"verdict, no API call this cycle", level="INFO")
                t = self._tenant
                if t is not None:
                    try:
                        t.record_saving(mode=strategy_mode, ticker=ticker)
                    except Exception:                          # noqa: BLE001
                        pass
                return

        if strategy_mode == "COMMITTEE":
            decision = verdict.to_trading_decision(ticker)

            with self._lock:
                self._last_decision = decision

            self._emit(
                PulseStage.DECISION,
                f"🗳 {decision.action} · {verdict.bulls}🐂/{verdict.bears}🐻 "
                f"of {verdict.total} · score {verdict.score:+.2f} · "
                f"conf {decision.confidence_score:.0%}",
                level="DECISION",
                meta={"decision": decision.model_dump(),
                      "committee": verdict.summary_dict()},
            )
        elif strategy_mode == "BOARDROOM":
            self._emit(PulseStage.AI,
                       "🪑 Boardroom convening — 4 analysts studying "
                       "their briefing packets…", level="INFO")
            try:
                pos = self.portfolio.positions.get(ticker)
                daily_pnl = self.portfolio.get_daily_pnl_pct()

                def _convene():
                    board = self._get_boardroom()
                    return board.convene(
                        snap, verdict=verdict,
                        in_position=pos is not None,
                        entry_price=(float(pos.avg_entry_price)
                                     if pos is not None else None),
                        risk_profile=risk_profile,
                        daily_pnl_pct=daily_pnl,
                    )

                # Nine calls a cycle makes this the one path where sharing
                # matters most: every user on this symbol and bar rides the
                # same meeting.
                ruling = self._shared_decision(
                    make_key(
                        **_shared_key_parts,
                        pos=position_bucket(
                            pos is not None,
                            float(pos.unrealized_pnl_pct) if pos else None),
                        dpnl=pnl_bucket(daily_pnl),
                        votes=(verdict.score if verdict is not None else None),
                    ),
                    ttl=_cache_ttl,
                    compute=_convene,
                    mode="BOARDROOM",
                    ticker=ticker,
                )
                decision = ruling.decision
            except Exception as exc:
                self._emit(PulseStage.ERROR, f"Boardroom failed: {exc}",
                           level="ERROR")
                return

            with self._lock:
                self._last_decision = decision
                self._last_boardroom = ruling.summary_dict()
            self._remember_ai_context(price, verdict)

            t = ruling.tally()
            chair_tag = "⚠ majority fallback" if ruling.chair_is_fallback \
                        else f"chair {ruling.chair_name.split()[0]}"
            self._emit(
                PulseStage.DECISION,
                f"🪑 {decision.action} · panel "
                f"{t.get('BUY', 0)}B/{t.get('SELL', 0)}S/"
                f"{t.get('HOLD', 0)}H · {chair_tag} · "
                f"conf {decision.confidence_score:.0%}",
                level="DECISION",
                meta={"decision": decision.model_dump(),
                      "boardroom": ruling.summary_dict()},
            )
        elif strategy_mode == "HYBRID":
            # ── RAG + Claude, with the committee tally in the prompt ───────
            self._emit(PulseStage.RAG, "Searching strategy knowledge base…",
                       level="INFO")
            try:
                retrieval = self._retriever.get_relevant_strategies(snap)
            except Exception as exc:
                self._emit(PulseStage.ERROR, f"RAG failed: {exc}", level="ERROR")
                return

            self._emit(PulseStage.AI,
                       f"Hybrid: Claude reviewing the committee's "
                       f"{verdict.bulls}🐂/{verdict.bears}🐻 vote…",
                       level="INFO")
            committee_ctx = (
                f"An ensemble committee of {verdict.total} technical "
                f"indicators just voted on {ticker}:\n"
                f"  BULL votes: {verdict.bulls}\n"
                f"  BEAR votes: {verdict.bears}\n"
                f"  NEUTRAL:    {verdict.neutrals}\n"
                f"  Net score:  {verdict.score:+.2f}  (range -1..+1)\n"
                f"  Committee verdict: {verdict.action}\n"
                f"  Leading agreeing indicators: "
                f"{', '.join(verdict.top_contributors(5)) or '—'}\n"
                f"Treat this as a strong systematic signal. Agree or "
                f"disagree explicitly in your reasoning, and explain why."
            )
            try:
                ai_dec = self._shared_decision(
                    make_key(**_shared_key_parts,
                             votes=(verdict.score if verdict is not None else None)),
                    ttl=_cache_ttl,
                    compute=lambda: self._ai_engine().evaluate_market(
                        snap, retrieval, risk_profile=risk_profile,
                        extra_context=committee_ctx,
                    ),
                    mode="HYBRID",
                    ticker=ticker,
                )
            except Exception as exc:
                self._emit(PulseStage.ERROR, f"AI engine failed: {exc}",
                           level="ERROR")
                return

            decision = self._combine_hybrid(ticker, verdict, ai_dec)

            with self._lock:
                self._last_decision = decision
            self._remember_ai_context(price, verdict)

            agree = "AGREE" if ai_dec.action == verdict.action else \
                    f"committee {verdict.action} / AI {ai_dec.action}"
            self._emit(
                PulseStage.DECISION,
                f"🤝 {decision.action} · {agree} · "
                f"score {verdict.score:+.2f} · AI conf "
                f"{ai_dec.confidence_score:.0%} → final "
                f"{decision.confidence_score:.0%}",
                level="DECISION",
                meta={"decision": decision.model_dump(),
                      "committee": verdict.summary_dict(),
                      "ai_action": ai_dec.action},
            )
        else:
            # ── RAG ─────────────────────────────────────────────────────────
            self._emit(PulseStage.RAG, "Searching strategy knowledge base…",
                       level="INFO")
            try:
                retrieval = self._retriever.get_relevant_strategies(snap)
            except Exception as exc:
                self._emit(PulseStage.ERROR, f"RAG failed: {exc}", level="ERROR")
                return

            # ── AI decision ─────────────────────────────────────────────────
            self._emit(PulseStage.AI,
                       f"Claude analysing ({len(retrieval.chunks)} chunks, "
                       f"{risk_profile})…", level="INFO")
            try:
                decision = self._shared_decision(
                    make_key(**_shared_key_parts),
                    ttl=_cache_ttl,
                    compute=lambda: self._ai_engine().evaluate_market(
                        snap, retrieval, risk_profile=risk_profile,
                    ),
                    mode="AI",
                    ticker=ticker,
                )
            except Exception as exc:
                self._emit(PulseStage.ERROR, f"AI engine failed: {exc}",
                           level="ERROR")
                return

            with self._lock:
                self._last_decision = decision
            self._remember_ai_context(price, verdict)

            tag = "⚠" if decision.is_fallback else "✓"
            self._emit(
                PulseStage.DECISION,
                f"{tag} {decision.action} · conf {decision.confidence_score:.0%} · "
                f"risk {decision.risk_level} · RAG {decision.rag_context_quality}",
                level="DECISION",
                meta={"decision": decision.model_dump()},
            )

        # ── Execute ─────────────────────────────────────────────────────────
        # Compute slippage-adjusted ATR% (for size-aware fills below)
        atr_pct_for_slip = None
        try:
            for c in snap.data.columns:
                if c.startswith("ATR_"):
                    atr_v = float(snap.data[c].iloc[-1])
                    if atr_v > 0 and price > 0:
                        atr_pct_for_slip = (atr_v / price) * 100.0
                    break
        except Exception:
            atr_pct_for_slip = None

        def _fill_price(side: str) -> float:
            if not self._enable_slippage:
                return price
            from risk import apply_slippage
            eff, _bps = apply_slippage(
                price, side, cfg=self._slippage_cfg, atr_pct=atr_pct_for_slip,
            )
            return eff

        if decision.action == "BUY":
            # Safety gate — circuit breakers / cooldown / daily cap
            sstat = self.get_safety_status()
            if sstat.is_blocked:
                self._emit(PulseStage.RISK,
                           f"🛡 BUY blocked by safety: {sstat.reason}",
                           level="WARN",
                           meta={"safety": sstat.to_dict()})
                return
            if sstat.severity == "WARN":
                self._emit(PulseStage.RISK,
                           f"⚠ Safety warning: {sstat.reason}",
                           level="WARN")

            # Confidence gate — per-profile threshold
            from config.user_profile import RISK_ENVELOPES
            env = RISK_ENVELOPES.get(risk_profile, RISK_ENVELOPES["Balanced"])
            if decision.confidence_score < env.conf_threshold:
                self._emit(PulseStage.EXECUTE,
                           f"BUY gated — confidence "
                           f"{decision.confidence_score:.0%} < threshold "
                           f"{env.conf_threshold:.0%} ({risk_profile})",
                           level="INFO")
                return

            already_held = ticker in self.portfolio.positions
            if already_held and not env.allow_pyramiding:
                self._emit(PulseStage.EXECUTE,
                           f"BUY skipped — already holding {ticker} "
                           f"(no pyramiding on {risk_profile})",
                           level="INFO")
                return

            # Final position size: AI suggestion clamped to envelope, fallback
            # to user's ceiling when AI didn't provide one.
            ai_sugg = decision.suggested_position_size_pct
            if ai_sugg is None:
                chosen = min(trade_size_pct, env.size_max_pct)
            else:
                chosen = max(env.size_min_pct,
                             min(env.size_max_pct, float(ai_sugg)))
            # Also respect the user's global ceiling from the top-bar
            chosen = min(chosen, trade_size_pct)

            if already_held:
                # Pyramiding: only add if we have at least env.size_min_pct
                # headroom left in cash; otherwise skip.
                available_pct = (self.portfolio.cash
                                 / max(1.0, self.portfolio.get_total_value())
                                 * 100.0)
                if available_pct < env.size_min_pct:
                    self._emit(PulseStage.EXECUTE,
                               f"Pyramid skipped — cash too low "
                               f"({available_pct:.1f}%)", level="INFO")
                    return

            cash_to_use = self.portfolio.get_total_value() * (chosen / 100.0)

            # Position-size sanity check — never overshoot cash
            from risk.safety import SafetyController as _SC
            ok, why = _SC.validate_buy_size(
                cash_to_use, self.portfolio.cash, self.portfolio.fee_rate,
            )
            if not ok:
                self._emit(PulseStage.RISK,
                           f"BUY size sanity failed: {why}", level="WARN")
                # Trim to a safe amount instead of failing entirely
                cash_to_use = self.portfolio.cash / (1.0 + self.portfolio.fee_rate)
                if cash_to_use <= 0:
                    return

            buy_px = _fill_price("BUY")
            try:
                self.portfolio.buy(
                    ticker, buy_px, cash_amount=cash_to_use,
                    reasoning=decision.reasoning[:250],
                )
                pyramid_tag = " (pyramid)" if already_held else ""
                slip_tag = (f" [slip {(buy_px-price)/price*1e4:+.1f}bps]"
                            if self._enable_slippage and price else "")
                self._emit(PulseStage.EXECUTE,
                           f"🟢 BUY {ticker} @ ${buy_px:,.2f} "
                           f"({chosen:.0f}% · ${cash_to_use:,.0f}){pyramid_tag}{slip_tag}",
                           level="TRADE")
            except Exception as exc:
                self._emit(PulseStage.ERROR,
                           f"BUY failed: {exc}", level="ERROR")
        elif decision.action == "SELL":
            if ticker not in self.portfolio.positions:
                self._emit(PulseStage.EXECUTE,
                           f"SELL skipped — no open {ticker} position",
                           level="INFO")
            else:
                sell_px = _fill_price("SELL")
                try:
                    tr = self.portfolio.sell(
                        ticker, sell_px,
                        reasoning=decision.reasoning[:250],
                    )
                    emoji = "🟢" if tr.realized_pnl >= 0 else "🔴"
                    slip_tag = (f" [slip {(sell_px-price)/price*1e4:+.1f}bps]"
                                if self._enable_slippage and price else "")
                    self._emit(PulseStage.EXECUTE,
                               f"{emoji} SELL {ticker} @ ${sell_px:,.2f} "
                               f"P&L ${tr.realized_pnl:+,.2f}{slip_tag}",
                               level="TRADE")
                    self._reflect_on_sell(tr, exit_reasoning=decision.reasoning[:250])
                except Exception as exc:
                    self._emit(PulseStage.ERROR,
                               f"SELL failed: {exc}", level="ERROR")
        else:
            self._emit(PulseStage.EXECUTE,
                       f"HOLD — no action (conf "
                       f"{decision.confidence_score:.0%})",
                       level="INFO")
