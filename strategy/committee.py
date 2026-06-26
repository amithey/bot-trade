"""
Indicator Committee — a deterministic 38-agent voting strategy.
================================================================

Instead of a single AI opinion per cycle, a *committee* of 38 classic
technical indicators each casts a vote on every bar:

    +1  BULL    (this indicator says the asset is going up)
     0  NEUTRAL (no clear signal)
    -1  BEAR    (this indicator says the asset is going down)

The committee then tallies the votes and produces a single long-only
verdict per time window:

    BUY   — bull majority exceeds the entry threshold (go / stay long)
    SELL  — bear majority exceeds the exit threshold (exit to CASH;
            never short)
    HOLD  — no clear majority; keep whatever position you have

This works identically for crypto (BTC-USD, SOL-USD, …) and stocks
(AAPL, QQQ, TEVA.TA, …) because every agent only needs OHLCV bars.

All vote computation is fully vectorized over the whole DataFrame, so
the same code powers both the live engine (vote on the latest bar) and
the offline backtester (votes for every historical bar at once) with
zero look-ahead bias — each agent uses only data at-or-before bar *t*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Vote constants
# ---------------------------------------------------------------------------

BULL, NEUTRAL, BEAR = 1, 0, -1

VoteFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Agent:
    """One committee member: a named, categorised vote function."""
    name: str
    category: str          # Trend / Momentum / Volatility / Volume
    fn: VoteFn


@dataclass(frozen=True)
class AgentVote:
    """A single agent's vote on the latest bar."""
    name: str
    category: str
    vote: int              # +1 / 0 / -1


@dataclass
class CommitteeConfig:
    """
    Tunable thresholds for the committee verdict.

    score = (bulls - bears) / total_agents   ∈ [-1, +1]

    enter_score:  minimum score required to open / keep a long.
    exit_score:   score at-or-below which we exit to cash.
                  Asymmetric by default — easy in, slow out — so noise
                  doesn't shake the position (the FB-post behaviour:
                  "buy and hold the coin, sell only on a real bear turn").
    min_quorum:   minimum number of non-neutral votes for any signal.
    """
    enter_score: float = 0.16    # ≈ bulls lead by 6+ of 38
    exit_score: float = -0.16    # ≈ bears lead by 6+ of 38
    min_quorum: int = 10


@dataclass
class CommitteeVerdict:
    """The committee's decision for one bar."""
    action: str                       # BUY / SELL / HOLD
    score: float                      # net vote score [-1, +1]
    bulls: int
    bears: int
    neutrals: int
    total: int
    quorum_met: bool
    votes: list[AgentVote] = field(default_factory=list)
    category_scores: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    @property
    def confidence(self) -> float:
        """Map vote margin to a [0.5, 1.0] confidence figure."""
        return round(min(1.0, 0.5 + abs(self.score)), 3)

    def top_contributors(self, n: int = 5) -> list[str]:
        """Names of agents voting WITH the verdict direction."""
        want = BULL if self.score >= 0 else BEAR
        return [v.name for v in self.votes if v.vote == want][:n]

    def summary_dict(self) -> dict:
        """JSON-safe snapshot for the dashboard."""
        return {
            "action": self.action,
            "score": round(self.score, 4),
            "bulls": self.bulls,
            "bears": self.bears,
            "neutrals": self.neutrals,
            "total": self.total,
            "quorum_met": self.quorum_met,
            "confidence": self.confidence,
            "category_scores": {k: round(v, 3)
                                for k, v in self.category_scores.items()},
            "votes": [{"name": v.name, "category": v.category,
                       "vote": v.vote} for v in self.votes],
            "reasoning": self.reasoning,
        }

    def to_trading_decision(self, ticker: str):
        """
        Adapt the verdict to the ``TradingDecision`` schema used by the
        live engine and the dashboard, so committee mode plugs into the
        existing execution path unchanged.
        """
        from decision_engine.ai_engine import TradingDecision
        outlook = ("BULLISH" if self.score > 0.05 else
                   "BEARISH" if self.score < -0.05 else "NEUTRAL")
        attr = ("ATTRACTIVE" if self.action == "BUY" else
                "UNATTRACTIVE" if self.action == "SELL" else "NEUTRAL")
        return TradingDecision(
            action=self.action,
            confidence_score=self.confidence,
            reasoning=self.reasoning if len(self.reasoning) >= 20 else (
                f"Indicator committee verdict for {ticker}: {self.action} "
                f"({self.bulls} bull / {self.bears} bear of {self.total})."
            ),
            risk_level="LOW" if abs(self.score) > 0.35 else
                       "MEDIUM" if abs(self.score) > 0.15 else "HIGH",
            key_indicators=self.top_contributors(5),
            attractiveness_score=round(min(1.0, abs(self.score) * 2), 3),
            attractiveness_label=attr,
            price_outlook=outlook,
            rag_context_quality="NONE",
            is_fallback=False,
        )


# ---------------------------------------------------------------------------
# Indicator primitives (pure pandas / numpy, no TA library)
# ---------------------------------------------------------------------------

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
    loss = (-delta).clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100.0).where(loss.notna(), np.nan)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, min_periods=n, adjust=False).mean()


def _stoch(df: pd.DataFrame, n: int = 14, d: int = 3):
    lo = df["Low"].rolling(n, min_periods=n).min()
    hi = df["High"].rolling(n, min_periods=n).max()
    k = (df["Close"] - lo) / (hi - lo).replace(0, np.nan) * 100
    return k, k.rolling(d, min_periods=d).mean()


def _adx(df: pd.DataFrame, n: int = 14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = _atr(df, n).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(com=n - 1, min_periods=n, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(com=n - 1, min_periods=n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(com=n - 1, min_periods=n, adjust=False).mean()
    return adx, plus_di, minus_di


def _supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """Returns +1 when price is above the supertrend line, -1 below."""
    atr = _atr(df, n)
    mid = (df["High"] + df["Low"]) / 2
    upper = (mid + mult * atr).to_numpy()
    lower = (mid - mult * atr).to_numpy()
    close = df["Close"].to_numpy()
    n_rows = len(df)
    trend = np.zeros(n_rows)
    fub = np.full(n_rows, np.nan)   # final upper band
    flb = np.full(n_rows, np.nan)   # final lower band
    start = int(np.argmax(~np.isnan(upper))) if not np.all(np.isnan(upper)) else n_rows
    for i in range(start, n_rows):
        if i == start:
            fub[i], flb[i], trend[i] = upper[i], lower[i], 1
            continue
        fub[i] = upper[i] if (upper[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lower[i] if (lower[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
        if trend[i - 1] == 1:
            trend[i] = -1 if close[i] < flb[i] else 1
        else:
            trend[i] = 1 if close[i] > fub[i] else -1
    out = pd.Series(trend, index=df.index)
    out.iloc[:start] = np.nan
    return out


def _psar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Returns +1 when PSAR is below price (uptrend), -1 above."""
    high, low = df["High"].to_numpy(), df["Low"].to_numpy()
    n = len(df)
    if n < 3:
        return pd.Series(np.nan, index=df.index)
    trend = np.ones(n)
    sar = np.zeros(n)
    sar[0] = low[0]
    ep, af = high[0], af_step
    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if trend[i - 1] == 1:
            sar[i] = min(sar[i], low[i - 1], low[max(0, i - 2)])
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep
                ep, af = low[i], af_step
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep, af = high[i], min(af_max, af + af_step)
        else:
            sar[i] = max(sar[i], high[i - 1], high[max(0, i - 2)])
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep
                ep, af = high[i], af_step
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep, af = low[i], min(af_max, af + af_step)
    return pd.Series(trend, index=df.index)


def _vote_from_bool(bull: pd.Series, bear: pd.Series) -> pd.Series:
    """Combine boolean bull/bear masks into a -1/0/+1 vote series."""
    v = pd.Series(NEUTRAL, index=bull.index, dtype=float)
    v[bull.fillna(False)] = BULL
    v[bear.fillna(False)] = BEAR
    # Where the underlying indicator was NaN (warm-up), stay neutral.
    return v


# ---------------------------------------------------------------------------
# The 38 agents
# ---------------------------------------------------------------------------

def _build_agents() -> list[Agent]:
    A: list[Agent] = []

    def add(name: str, cat: str, fn: VoteFn) -> None:
        A.append(Agent(name, cat, fn))

    # ── Trend (12) ─────────────────────────────────────────────────────
    for p in (20, 50, 200):
        def f(df, p=p):
            sma = _sma(df["Close"], p)
            return _vote_from_bool(df["Close"] > sma, df["Close"] < sma)
        add(f"Price vs SMA{p}", "Trend", f)

    def sma_cross(df, fast=20, slow=50):
        a, b = _sma(df["Close"], fast), _sma(df["Close"], slow)
        return _vote_from_bool(a > b, a < b)
    add("SMA20 > SMA50", "Trend", lambda df: sma_cross(df, 20, 50))
    add("SMA50 > SMA200", "Trend", lambda df: sma_cross(df, 50, 200))

    def ema_cross(df):
        a, b = _ema(df["Close"], 8), _ema(df["Close"], 21)
        return _vote_from_bool(a > b, a < b)
    add("EMA8 > EMA21", "Trend", ema_cross)

    add("Supertrend (10,3)", "Trend",
        lambda df: _supertrend(df).fillna(NEUTRAL))
    add("Parabolic SAR", "Trend", lambda df: _psar(df))

    def adx_di(df):
        adx, pdi, mdi = _adx(df)
        strong = adx > 20
        return _vote_from_bool(strong & (pdi > mdi), strong & (mdi > pdi))
    add("ADX + DI", "Trend", adx_di)

    def aroon(df, n=25):
        idx_hi = df["High"].rolling(n + 1, min_periods=n + 1) \
            .apply(lambda x: float(np.argmax(x)), raw=True)
        idx_lo = df["Low"].rolling(n + 1, min_periods=n + 1) \
            .apply(lambda x: float(np.argmin(x)), raw=True)
        up = idx_hi / n * 100
        dn = idx_lo / n * 100
        return _vote_from_bool(up > 70, dn > 70)
    add("Aroon (25)", "Trend", aroon)

    def ichimoku_cloud(df):
        tenkan = (df["High"].rolling(9).max() + df["Low"].rolling(9).min()) / 2
        kijun = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2
        span_a = ((tenkan + kijun) / 2).shift(26)
        span_b = ((df["High"].rolling(52).max()
                   + df["Low"].rolling(52).min()) / 2).shift(26)
        top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
        return _vote_from_bool(df["Close"] > top, df["Close"] < bot)
    add("Ichimoku Cloud", "Trend", ichimoku_cloud)

    def ichimoku_tk(df):
        tenkan = (df["High"].rolling(9).max() + df["Low"].rolling(9).min()) / 2
        kijun = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2
        return _vote_from_bool(tenkan > kijun, tenkan < kijun)
    add("Ichimoku TK Cross", "Trend", ichimoku_tk)

    # ── Momentum (14) ──────────────────────────────────────────────────
    def rsi_trend(df):
        r = _rsi(df["Close"])
        return _vote_from_bool(r > 55, r < 45)
    add("RSI-14 Trend", "Momentum", rsi_trend)

    def rsi_reversal(df):
        r = _rsi(df["Close"])
        return _vote_from_bool(r < 30, r > 70)
    add("RSI-14 Reversal", "Momentum", rsi_reversal)

    def stoch_kd(df):
        k, d = _stoch(df)
        return _vote_from_bool((k > d) & (k < 80), (k < d) & (k > 20))
    add("Stochastic %K/%D", "Momentum", stoch_kd)

    def stoch_rsi(df, n=14):
        r = _rsi(df["Close"], n)
        lo = r.rolling(n, min_periods=n).min()
        hi = r.rolling(n, min_periods=n).max()
        sr = (r - lo) / (hi - lo).replace(0, np.nan)
        return _vote_from_bool(sr > 0.8, sr < 0.2)
    add("Stoch RSI", "Momentum", stoch_rsi)

    def macd_parts(df):
        macd = _ema(df["Close"], 12) - _ema(df["Close"], 26)
        sig = _ema(macd, 9)
        return macd, sig

    add("MACD Cross", "Momentum",
        lambda df: _vote_from_bool(*(lambda m, s: (m > s, m < s))(*macd_parts(df))))
    add("MACD Zero Line", "Momentum",
        lambda df: _vote_from_bool(*(lambda m, s: (m > 0, m < 0))(*macd_parts(df))))

    def macd_hist_dir(df):
        m, s = macd_parts(df)
        h = m - s
        return _vote_from_bool(h > h.shift(1), h < h.shift(1))
    add("MACD Histogram Slope", "Momentum", macd_hist_dir)

    def roc(df, n=12):
        r = df["Close"].pct_change(n, fill_method=None) * 100
        return _vote_from_bool(r > 0.5, r < -0.5)
    add("ROC-12", "Momentum", roc)

    def momentum(df, n=10):
        m = df["Close"] - df["Close"].shift(n)
        return _vote_from_bool(m > 0, m < 0)
    add("Momentum-10", "Momentum", momentum)

    def williams(df, n=14):
        hi = df["High"].rolling(n, min_periods=n).max()
        lo = df["Low"].rolling(n, min_periods=n).min()
        wr = (hi - df["Close"]) / (hi - lo).replace(0, np.nan) * -100
        return _vote_from_bool(wr > -20, wr < -80)
    add("Williams %R", "Momentum", williams)

    def cci(df, n=20):
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        ma = tp.rolling(n, min_periods=n).mean()
        md = (tp - ma).abs().rolling(n, min_periods=n).mean()
        c = (tp - ma) / (0.015 * md.replace(0, np.nan))
        return _vote_from_bool(c > 100, c < -100)
    add("CCI-20", "Momentum", cci)

    def tsi(df, r=25, s=13):
        m = df["Close"].diff()
        num = _ema(_ema(m, r), s)
        den = _ema(_ema(m.abs(), r), s).replace(0, np.nan)
        t = 100 * num / den
        return _vote_from_bool(t > 5, t < -5)
    add("TSI", "Momentum", tsi)

    def ultimate(df):
        pc = df["Close"].shift(1)
        bp = df["Close"] - pd.concat([df["Low"], pc], axis=1).min(axis=1)
        tr = pd.concat([df["High"], pc], axis=1).max(axis=1) \
            - pd.concat([df["Low"], pc], axis=1).min(axis=1)
        tr = tr.replace(0, np.nan)
        def avg(n): return bp.rolling(n).sum() / tr.rolling(n).sum()
        uo = 100 * (4 * avg(7) + 2 * avg(14) + avg(28)) / 7
        return _vote_from_bool(uo > 60, uo < 40)
    add("Ultimate Oscillator", "Momentum", ultimate)

    def awesome(df):
        mid = (df["High"] + df["Low"]) / 2
        ao = _sma(mid, 5) - _sma(mid, 34)
        return _vote_from_bool(ao > 0, ao < 0)
    add("Awesome Oscillator", "Momentum", awesome)

    # ── Volatility / Channels (6) ──────────────────────────────────────
    def bb(df, n=20, k=2.0):
        mid = _sma(df["Close"], n)
        sd = df["Close"].rolling(n, min_periods=n).std()
        return mid, mid + k * sd, mid - k * sd

    def bb_pctb(df):
        mid, up, lo = bb(df)
        pctb = (df["Close"] - lo) / (up - lo).replace(0, np.nan)
        return _vote_from_bool(pctb > 0.55, pctb < 0.45)
    add("Bollinger %B", "Volatility", bb_pctb)

    def bb_breakout(df):
        _, up, lo = bb(df)
        return _vote_from_bool(df["Close"] > up, df["Close"] < lo)
    add("Bollinger Breakout", "Volatility", bb_breakout)

    def keltner(df, n=20, mult=2.0):
        mid = _ema(df["Close"], n)
        atr = _atr(df, n)
        return _vote_from_bool(df["Close"] > mid + mult * atr,
                               df["Close"] < mid - mult * atr)
    add("Keltner Channel", "Volatility", keltner)

    def donchian(df, n):
        hi = df["High"].rolling(n, min_periods=n).max().shift(1)
        lo = df["Low"].rolling(n, min_periods=n).min().shift(1)
        return _vote_from_bool(df["Close"] > hi, df["Close"] < lo)
    add("Donchian-20 Breakout", "Volatility", lambda df: donchian(df, 20))
    add("Donchian-55 Breakout", "Volatility", lambda df: donchian(df, 55))

    def chandelier(df, n=22, mult=3.0):
        atr = _atr(df, n)
        long_stop = df["High"].rolling(n, min_periods=n).max() - mult * atr
        short_stop = df["Low"].rolling(n, min_periods=n).min() + mult * atr
        return _vote_from_bool(df["Close"] > short_stop, df["Close"] < long_stop)
    add("Chandelier Exit", "Volatility", chandelier)

    # ── Volume (6) ─────────────────────────────────────────────────────
    def obv(df):
        direction = np.sign(df["Close"].diff()).fillna(0)
        line = (direction * df["Volume"]).cumsum()
        ma = _sma(line, 20)
        return _vote_from_bool(line > ma, line < ma)
    add("OBV Trend", "Volume", obv)

    def mfi(df, n=14):
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        flow = tp * df["Volume"]
        pos = flow.where(tp > tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
        neg = flow.where(tp < tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
        m = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
        return _vote_from_bool(m < 20, m > 80)
    add("MFI-14", "Volume", mfi)

    def cmf(df, n=20):
        rng = (df["High"] - df["Low"]).replace(0, np.nan)
        mult = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
        mfv = mult * df["Volume"]
        c = mfv.rolling(n, min_periods=n).sum() / \
            df["Volume"].rolling(n, min_periods=n).sum().replace(0, np.nan)
        return _vote_from_bool(c > 0.05, c < -0.05)
    add("Chaikin Money Flow", "Volume", cmf)

    def vwap(df, n=20):
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        vol = df["Volume"].astype(float)
        v = (tp * vol).rolling(n, min_periods=n).sum() / \
            vol.rolling(n, min_periods=n).sum().replace(0, np.nan)
        return _vote_from_bool(df["Close"] > v, df["Close"] < v)
    add("Price vs VWAP-20", "Volume", vwap)

    def force(df, n=13):
        fi = _ema(df["Close"].diff() * df["Volume"], n)
        return _vote_from_bool(fi > 0, fi < 0)
    add("Force Index", "Volume", force)

    def adl(df):
        rng = (df["High"] - df["Low"]).replace(0, np.nan)
        mult = (((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng).fillna(0)
        line = (mult * df["Volume"]).cumsum()
        ma = _sma(line, 20)
        return _vote_from_bool(line > ma, line < ma)
    add("A/D Line Trend", "Volume", adl)

    assert len(A) == 38, f"Committee must have exactly 38 agents, got {len(A)}"
    return A


# ---------------------------------------------------------------------------
# The committee
# ---------------------------------------------------------------------------

class IndicatorCommittee:
    """
    Runs all 38 agents over an OHLCV DataFrame.

    ``vote_matrix(df)`` → DataFrame of per-bar votes (one column per agent)
    ``vote_latest(df)`` → CommitteeVerdict for the most recent bar
    ``score_series(df)`` → net score per bar (used by the backtester)
    """

    def __init__(self, config: Optional[CommitteeConfig] = None) -> None:
        self.config = config or CommitteeConfig()
        self.agents = _build_agents()

    # -- core ------------------------------------------------------------

    def vote_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-bar votes for every agent. Failing agents vote NEUTRAL."""
        out = {}
        for agent in self.agents:
            try:
                v = agent.fn(df)
                out[agent.name] = v.fillna(NEUTRAL).clip(-1, 1)
            except Exception:
                out[agent.name] = pd.Series(NEUTRAL, index=df.index, dtype=float)
        return pd.DataFrame(out, index=df.index)

    def score_series(self, df: pd.DataFrame,
                     votes: Optional[pd.DataFrame] = None) -> pd.Series:
        """Net score per bar: (bulls − bears) / total agents ∈ [-1, 1]."""
        if votes is None:
            votes = self.vote_matrix(df)
        return votes.sum(axis=1) / len(self.agents)

    def verdict_for_row(self, votes_row: pd.Series,
                        in_position: bool = False) -> CommitteeVerdict:
        """Tally one bar's votes into a long-only verdict."""
        cfg = self.config
        bulls = int((votes_row == BULL).sum())
        bears = int((votes_row == BEAR).sum())
        neutrals = int((votes_row == NEUTRAL).sum())
        total = len(self.agents)
        score = (bulls - bears) / total
        quorum = (bulls + bears) >= cfg.min_quorum

        if not quorum:
            action = "HOLD"
        elif score >= cfg.enter_score:
            action = "BUY"
        elif score <= cfg.exit_score:
            action = "SELL"
        else:
            action = "HOLD"

        by_cat: dict[str, list[int]] = {}
        agent_votes: list[AgentVote] = []
        for agent in self.agents:
            v = int(votes_row.get(agent.name, NEUTRAL))
            agent_votes.append(AgentVote(agent.name, agent.category, v))
            by_cat.setdefault(agent.category, []).append(v)
        cat_scores = {c: (sum(vs) / len(vs)) for c, vs in by_cat.items()}

        cat_txt = " · ".join(
            f"{c} {s:+.2f}" for c, s in sorted(cat_scores.items())
        )
        reasoning = (
            f"Committee of {total} indicators voted {bulls} BULL / "
            f"{bears} BEAR / {neutrals} NEUTRAL (net score {score:+.2f}). "
            f"Category scores: {cat_txt}. "
            f"Verdict: {action}"
            + ("" if quorum else " (quorum not met — defaulting to HOLD)")
            + ". Long-only policy: SELL means exit to cash, never short."
        )

        return CommitteeVerdict(
            action=action, score=score, bulls=bulls, bears=bears,
            neutrals=neutrals, total=total, quorum_met=quorum,
            votes=agent_votes, category_scores=cat_scores,
            reasoning=reasoning,
        )

    def vote_latest(self, df: pd.DataFrame,
                    in_position: bool = False) -> CommitteeVerdict:
        """Verdict for the most recent bar of *df*."""
        if df is None or len(df) < 60:
            raise ValueError(
                f"Committee needs at least 60 bars of OHLCV data, "
                f"got {0 if df is None else len(df)}."
            )
        votes = self.vote_matrix(df)
        return self.verdict_for_row(votes.iloc[-1], in_position=in_position)
