"""
UsageMeter — a LangChain callback that prices every Claude call.

Attached to the ``ChatAnthropic`` instance at construction, so it sees each
call the AI engine, the hybrid path and all nine boardroom seats make —
including the ones fired concurrently from the boardroom's thread pool.

Attribution works because engines are built per key: ``engine_for_key()``
hands each distinct API key its own ``AITradingEngine``, and each engine gets
its own meter carrying that account's id and funding source.  Nothing has to
be threaded through call arguments, and no thread-local state is involved.

The meter never raises into the trading loop.  A metering failure must not
cost a user a trade, so every handler body is wrapped.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

from saas.ledger import UsageLedger, get_ledger
from utils.logger import get_logger

logger = get_logger(__name__)


def _usage_from_response(response: Any) -> Optional[dict]:
    """Pull token counts out of an ``LLMResult``, whatever shape it takes.

    LangChain surfaces Anthropic usage in two places depending on version and
    call style: normalised on ``message.usage_metadata``, or raw on
    ``llm_output``/``response_metadata``.  Try both.
    """
    # 1) Normalised usage_metadata on the generated message.
    try:
        for gen_list in (getattr(response, "generations", None) or []):
            for gen in (gen_list or []):
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None)
                if um:
                    details = dict(um.get("input_token_details") or {})
                    return {
                        "input_tokens":  int(um.get("input_tokens") or 0),
                        "output_tokens": int(um.get("output_tokens") or 0),
                        "cache_read":    int(details.get("cache_read") or 0),
                        "cache_write":   int(details.get("cache_creation") or 0),
                    }
    except Exception:                                          # noqa: BLE001
        pass

    # 2) Raw Anthropic usage dict.
    raw = None
    try:
        out = getattr(response, "llm_output", None) or {}
        raw = out.get("usage") or out.get("token_usage")
        if raw is None:
            for gen_list in (getattr(response, "generations", None) or []):
                for gen in (gen_list or []):
                    msg = getattr(gen, "message", None)
                    meta = getattr(msg, "response_metadata", None) or {}
                    raw = meta.get("usage")
                    if raw:
                        break
                if raw:
                    break
    except Exception:                                          # noqa: BLE001
        raw = None

    if not raw:
        return None
    get = raw.get if isinstance(raw, dict) else (lambda k, d=0: getattr(raw, k, d))
    return {
        "input_tokens":  int(get("input_tokens", 0) or 0),
        "output_tokens": int(get("output_tokens", 0) or 0),
        "cache_read":    int(get("cache_read_input_tokens", 0) or 0),
        "cache_write":   int(get("cache_creation_input_tokens", 0) or 0),
    }


def _model_from_response(response: Any, default: str) -> str:
    try:
        out = getattr(response, "llm_output", None) or {}
        if out.get("model"):
            return str(out["model"])
        for gen_list in (getattr(response, "generations", None) or []):
            for gen in (gen_list or []):
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "response_metadata", None) or {}
                if meta.get("model"):
                    return str(meta["model"])
                if meta.get("model_name"):
                    return str(meta["model_name"])
    except Exception:                                          # noqa: BLE001
        pass
    return default


class UsageMeter(BaseCallbackHandler):
    """Records token spend for one account into the ledger.

    Parameters
    ----------
    account_id:
        Ledger account the spend belongs to.  For BYOK users this is the key
        fingerprint; for platform-funded users, their session/account id.
    funding:
        ``"BYOK"`` or ``"PLATFORM"`` — keeps a subscriber's own spend from
        counting against the operator's trial budget.
    default_model:
        Used when a response carries no model id of its own.
    """

    raise_error = False        # LangChain must swallow anything we throw

    def __init__(
        self,
        account_id: str,
        funding: str = "PLATFORM",
        default_model: str = "",
        ledger: Optional[UsageLedger] = None,
    ) -> None:
        super().__init__()
        self.account_id = account_id
        self.funding = funding
        self.default_model = default_model
        self._ledger = ledger or get_ledger()
        self._ctx_lock = threading.Lock()
        self._mode = ""
        self._ticker = ""
        # Live counters for the current process, cheap for the UI to read.
        self.calls = 0
        self.cost_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0

    # -- context -----------------------------------------------------------
    def set_context(self, mode: str = "", ticker: str = "") -> None:
        """Tag subsequent calls with the strategy mode and symbol in play."""
        with self._ctx_lock:
            self._mode = mode or self._mode
            self._ticker = ticker or self._ticker

    # -- callback ----------------------------------------------------------
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:   # noqa: D102
        try:
            usage = _usage_from_response(response)
            if not usage:
                return
            model = _model_from_response(response, self.default_model)
            with self._ctx_lock:
                mode, ticker = self._mode, self._ticker
            cost = self._ledger.record(
                account_id=self.account_id,
                model=model,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cache_read_tokens=usage["cache_read"],
                cache_write_tokens=usage["cache_write"],
                funding=self.funding,
                mode=mode,
                ticker=ticker,
            )
            with self._ctx_lock:
                self.calls += 1
                self.cost_usd += cost
                self.input_tokens += usage["input_tokens"] + usage["cache_read"]
                self.output_tokens += usage["output_tokens"]
        except Exception as exc:                               # noqa: BLE001
            logger.debug(f"[UsageMeter] metering skipped: {exc}")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:  # noqa: D102
        # A failed call still costs nothing, so there is nothing to record.
        return

    # -- reporting ---------------------------------------------------------
    def snapshot(self) -> dict:
        with self._ctx_lock:
            return {
                "account_id":    self.account_id,
                "funding":       self.funding,
                "calls":         self.calls,
                "cost_usd":      round(self.cost_usd, 6),
                "input_tokens":  self.input_tokens,
                "output_tokens": self.output_tokens,
            }
