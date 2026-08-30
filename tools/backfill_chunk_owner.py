"""Adopt knowledge chunks written before ownership existed.

Chunks ingested before ``rag.ownership`` carry no ``owner`` field. Retrieval
handles the important half of that on its own: operator-curated playbooks are
admitted by their ``source`` value, so the seeded knowledge keeps working with
no migration at all.

Legacy *user-ingested* chunks (``web_article``, ``youtube``, ``lesson``) match
neither arm of the filter and drop out of retrieval. That is the safe default —
the alternative is leaving content of unknown provenance readable by every
account — but it means articles you ingested yourself go quiet until you claim
them. This tool is how you claim them.

    # See what would change, touching nothing:
    python -m tools.backfill_chunk_owner --owner <slug> --dry-run

    # Apply:
    python -m tools.backfill_chunk_owner --owner <slug>

Find <slug> in the dashboard: it is the per-account slug used for your
portfolio and profile files under data/portfolios/.

Only chunks with no ``owner`` at all are touched. Anything already attributed
is left alone, so re-running is safe and a second account's content can never
be reassigned by mistake.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import settings
from rag.ownership import SHARED_OWNER, TRUSTED_SHARED_SOURCES

_BATCH = 200


def _open_collection():
    import chromadb
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )
    from utils.hf_quiet import quiet_model_load

    with quiet_model_load():
        embed_fn = SentenceTransformerEmbeddingFunction(
            # No trust_remote_code — the flag lets a model repo run code.
            model_name=settings.embedding_model,
        )
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def backfill(owner: str, dry_run: bool = False) -> dict:
    """Stamp ``owner`` onto every chunk that has none. Returns a summary."""
    collection = _open_collection()
    got = collection.get(include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []

    to_update_ids: list[str] = []
    to_update_metas: list[dict] = []
    by_source: dict[str, int] = {}
    already = 0

    for chunk_id, meta in zip(ids, metas):
        meta = dict(meta or {})
        if meta.get("owner"):
            already += 1
            continue
        source = str(meta.get("source") or "unknown")
        # Seed content is shared by rule, so stamp it as such rather than
        # handing the operator's playbooks to one account.
        meta["owner"] = (
            SHARED_OWNER if source in TRUSTED_SHARED_SOURCES else owner
        )
        by_source[source] = by_source.get(source, 0) + 1
        to_update_ids.append(chunk_id)
        to_update_metas.append(meta)

    if to_update_ids and not dry_run:
        for start in range(0, len(to_update_ids), _BATCH):
            sl = slice(start, start + _BATCH)
            collection.update(ids=to_update_ids[sl],
                              metadatas=to_update_metas[sl])

    return {
        "total": len(ids),
        "already_owned": already,
        "updated": len(to_update_ids),
        "by_source": by_source,
        "dry_run": dry_run,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Attribute pre-ownership knowledge chunks to an account.")
    ap.add_argument("--owner", required=True,
                    help="Account slug to adopt untagged user content.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    result = backfill(args.owner, dry_run=args.dry_run)
    verb = "would update" if result["dry_run"] else "updated"
    print(f"{result['total']} chunk(s) in collection")
    print(f"  {result['already_owned']} already attributed — left alone")
    print(f"  {verb} {result['updated']}:")
    for source, n in sorted(result["by_source"].items()):
        dest = SHARED_OWNER if source in TRUSTED_SHARED_SOURCES else args.owner
        print(f"    {source:<22} {n:>5}  ->  {dest}")
    if result["dry_run"]:
        print("\nDry run — nothing written. Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
