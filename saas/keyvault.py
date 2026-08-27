"""
Bring-your-own-key handling.

A subscriber's Anthropic key is the most sensitive thing this platform ever
touches, so the rules here are deliberately strict:

* **Never persisted.**  The key lives in the user's Streamlit session and in
  the engine object bound to that session.  Nothing writes it to ``.env``, to
  the ledger, or to any log line.
* **Never displayed.**  The UI shows :func:`mask` output only.
* **Identified by fingerprint.**  A SHA-256 prefix of the key doubles as the
  account id, so usage can be attributed across sessions without the key
  itself ever being stored.
* **Verified before use.**  :func:`verify_live` makes one free ``models.list``
  call so a typo surfaces in Settings rather than as a failed trading cycle
  forty minutes later.

The operator's own key (from ``.env``) is handled by the same code path — it
is simply the fallback when the user has not supplied one.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

#: Anthropic keys look like ``sk-ant-api03-...``.  Deliberately loose on the
#: middle segment so a new key format does not lock users out.
_KEY_RE = re.compile(r"^sk-ant-[A-Za-z0-9_\-]{20,}$")


def normalise(key: Optional[str]) -> str:
    """Trim whitespace users pick up when copying out of the console."""
    return (key or "").strip()


def validate_format(key: Optional[str]) -> tuple[bool, str]:
    """Cheap local sanity check — no network, no cost."""
    k = normalise(key)
    if not k:
        return False, "No key provided."
    if " " in k or "\n" in k:
        return False, "Key contains whitespace — copy it again."
    if not k.startswith("sk-ant-"):
        return False, "Anthropic keys start with 'sk-ant-'."
    if not _KEY_RE.match(k):
        return False, "That does not look like a complete Anthropic key."
    return True, "Format looks right."


def fingerprint(key: Optional[str]) -> str:
    """Stable, non-reversible id for a key.

    Used as the ledger account id for BYOK users: the same key always maps to
    the same account, and the account id reveals nothing about the key.
    """
    k = normalise(key)
    if not k:
        return ""
    return "byok_" + hashlib.sha256(k.encode("utf-8")).hexdigest()[:16]


def mask(key: Optional[str]) -> str:
    """Render a key for display: ``sk-ant-...4f2a``."""
    k = normalise(key)
    if not k:
        return "—"
    if len(k) <= 12:
        return "sk-ant-…"
    return f"{k[:7]}…{k[-4:]}"


def verify_live(key: Optional[str], timeout: float = 12.0) -> tuple[bool, str]:
    """Confirm the key actually authenticates.

    Uses ``models.list`` — an authenticated call that consumes no tokens, so
    validating a key costs the user nothing.
    """
    ok, msg = validate_format(key)
    if not ok:
        return False, msg
    k = normalise(key)
    try:
        import anthropic
    except ImportError:
        return True, "Format valid (anthropic SDK not installed — skipped live check)."

    try:
        client = anthropic.Anthropic(api_key=k, timeout=timeout, max_retries=0)
        models = client.models.list(limit=1)
    except Exception as exc:                                   # noqa: BLE001
        name = type(exc).__name__
        if "Authentication" in name:
            return False, "Anthropic rejected this key (authentication failed)."
        if "PermissionDenied" in name:
            return False, "This key lacks permission to call the Messages API."
        if "RateLimit" in name:
            return True, "Key is valid (rate-limited right now, but authenticated)."
        if "Connection" in name or "Timeout" in name:
            return False, "Could not reach Anthropic — check your connection."
        return False, f"Key check failed: {name}"

    n = len(getattr(models, "data", []) or [])
    return True, f"Key verified — {n or 'no'} model(s) visible to this account."


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KeyHandle:
    """A resolved API key plus the metadata safe to show and store."""
    key: str
    source: str            # "user" | "platform" | "none"
    account_id: str

    @property
    def is_user_key(self) -> bool:
        return self.source == "user"

    @property
    def available(self) -> bool:
        return bool(self.key)

    @property
    def masked(self) -> str:
        return mask(self.key)

    @property
    def funding(self) -> str:
        """The ``funding`` value the ledger should record for this key."""
        if self.source == "user":
            return "BYOK"
        if self.source == "platform":
            return "PLATFORM"
        return "NONE"

    def __repr__(self) -> str:      # never leak the key through a traceback
        return f"KeyHandle(source={self.source!r}, masked={self.masked!r})"


def platform_key() -> str:
    """The operator's own key, used only to fund trial budgets."""
    return normalise(os.environ.get("ANTHROPIC_API_KEY", ""))


def resolve_key(
    user_key: Optional[str] = None,
    allow_platform: bool = True,
    anon_account_id: str = "anon",
) -> KeyHandle:
    """Pick which key funds this user's calls.

    A user key always wins, so a subscriber who has supplied one never
    silently spends the operator's budget.
    """
    uk = normalise(user_key)
    if uk:
        return KeyHandle(key=uk, source="user", account_id=fingerprint(uk))
    if allow_platform:
        pk = platform_key()
        if pk:
            return KeyHandle(key=pk, source="platform", account_id=anon_account_id)
    return KeyHandle(key="", source="none", account_id=anon_account_id)
