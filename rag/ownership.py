"""Who a knowledge chunk belongs to, and who is allowed to retrieve it.

There is one ChromaDB collection for the whole deployment, and three very
different kinds of text end up in it:

``seed``, ``seed_chart_analysis``
    Playbooks the operator ships and re-seeds. Curated, identical for
    everyone, and genuinely meant to be shared.

``web_article``, ``youtube``
    Whatever a user pasted into the Knowledge page. Arbitrary text from a
    URL the user chose.

``lesson``
    Post-trade reflections written back by the engine — derived from one
    account's own trades.

Only the first group is safe to share. The other two were shared anyway,
which meant one user's ingested document (or trade history) could surface in
another user's retrieval and land in their Claude prompt. Every chunk written
from now on therefore carries an ``owner``.

Why the filter has two arms
---------------------------
Chunks written before this existed have no ``owner`` field at all, and a
``where`` clause on a missing field matches nothing. Filtering on ``owner``
alone would make the entire seeded playbook — the thing the whole RAG step
depends on — silently vanish on upgrade.

So the filter also admits the operator-curated ``source`` values directly.
That arm is a *rule*, not a migration artifact: seed content is shared because
of what it is, regardless of when it was written. Legacy user-ingested chunks
match neither arm and drop out of retrieval, which is the safe direction — and
``tools/backfill_chunk_owner.py`` exists to adopt them deliberately.
"""
from __future__ import annotations

from typing import Optional

#: Owner value for content every account may retrieve.
SHARED_OWNER = "__shared__"

#: ``source`` values produced by the operator-run seed scripts. Trusted
#: because the operator controls them, not because of who ran them.
#:
#: Keep this in step with every seeder. ``seed_script`` comes from
#: ``seed_knowledge.py`` at the repository root rather than from
#: ``knowledge_ingestion/`` — it was missed on the first pass here and only
#: surfaced by running the backfill tool against a real collection, so check
#: for stragglers the same way before trusting this list.
TRUSTED_SHARED_SOURCES: tuple[str, ...] = (
    "seed", "seed_chart_analysis", "seed_script",
)


def normalise_owner(raw: Optional[str]) -> str:
    """Coerce an account identifier into a stable metadata value.

    Empty or missing means "not attributed", which callers treat as shared —
    a single-user deployment has nobody to isolate from.
    """
    cleaned = (raw or "").strip()
    return cleaned or SHARED_OWNER


def owner_filter(owner: Optional[str]) -> Optional[dict]:
    """ChromaDB ``where`` clause limiting retrieval to what *owner* may see.

    Returns ``None`` when no owner is supplied, meaning "do not filter". That
    is the single-user case: with one account there is nothing to isolate, and
    filtering would only hide that account's own legacy chunks.

    With an owner, retrieval is limited to shared content plus their own.
    """
    if not owner or owner == SHARED_OWNER:
        return None
    return {
        "$or": [
            {"owner": {"$in": [SHARED_OWNER, owner]}},
            {"source": {"$in": list(TRUSTED_SHARED_SOURCES)}},
        ]
    }


def is_visible_to(chunk_metadata: Optional[dict], owner: Optional[str]) -> bool:
    """Python-side mirror of :func:`owner_filter`.

    ChromaDB applies the ``where`` clause, so this is not in the retrieval
    path. It exists so the rule can be asserted directly in tests and reused
    by any caller filtering an already-fetched list.
    """
    if not owner or owner == SHARED_OWNER:
        return True
    meta = chunk_metadata or {}
    if meta.get("source") in TRUSTED_SHARED_SOURCES:
        return True
    return meta.get("owner") in (SHARED_OWNER, owner)
