"""Bounded local decision audit for diagnosing HOLDs and shadow disagreements.

No credentials, identity strings or account balances are stored. The database
is local to the application's persistent data directory; it sends no data out.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "committee_observations.sqlite3"


def record_observation(account_id, report, *, path=DEFAULT_PATH):
    if not account_id or not report.get("bar_closed_at"):
        return False
    account = hashlib.sha256(str(account_id).encode()).hexdigest()
    # Explicit allowlist, not arbitrary provider text or account objects.
    packet = {key: report.get(key) for key in (
        "ticker", "bar_closed_at", "committee_policy_version", "setup", "regime",
        "signal_side", "entry_allowed", "committee_admission", "committee_evidence")}
    payload = json.dumps(packet, ensure_ascii=True, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=.2) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS observations (account TEXT, ticker TEXT, bar TEXT, version TEXT, packet TEXT, PRIMARY KEY(account,ticker,bar,version))")
        connection.execute("INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?)",
            (account, str(report.get("ticker", "")), str(report["bar_closed_at"]), str(report.get("committee_policy_version", "")), payload))
        # At most 2,000 observations per account. This is a rolling decision
        # diagnostic, not an immutable long-term research dataset.
        connection.execute("DELETE FROM observations WHERE account=? AND rowid NOT IN (SELECT rowid FROM observations WHERE account=? ORDER BY rowid DESC LIMIT 2000)", (account, account))
    return True
