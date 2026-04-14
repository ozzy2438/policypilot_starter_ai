"""
ASIC Regulatory Guide Ingestion
--------------------------------
Reads ASIC regulatory snippets from JSON, generates embeddings via OpenAI,
and upserts them into a Qdrant collection for hybrid search.

Usage:
    python -m app.ingest.load_asic            # full pipeline
    python -m app.ingest.load_asic --dry-run  # show snippets only, no Qdrant write
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

settings = get_settings()


def load_snippets() -> list[dict[str, Any]]:
    """Load ASIC regulatory snippets from JSON file."""
    json_path = settings.data_raw / "asic_rg271_snippets.json"
    log.info("Reading %s", json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        snippets = json.load(f)

    log.info("Loaded %d regulatory snippets", len(snippets))
    return snippets


def print_summary(snippets: list[dict[str, Any]]) -> None:
    """Print summary of loaded snippets."""
    log.info("=" * 60)
    log.info("ASIC Regulatory Snippets – Summary")
    log.info("=" * 60)

    # By guide
    guides: dict[str, int] = {}
    categories: dict[str, int] = {}
    for s in snippets:
        guides[s["guide"]] = guides.get(s["guide"], 0) + 1
        categories[s["category"]] = categories.get(s["category"], 0) + 1

    log.info("\n📚 Snippets by Guide:")
    for guide, count in sorted(guides.items(), key=lambda x: -x[1]):
        log.info("   %-45s %d", guide, count)

    log.info("\n🏷️  Snippets by Category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        log.info("   %-30s %d", cat, count)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via OpenAI API (batch)."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    log.info("Generating embeddings for %d texts (model: %s)",
             len(texts), settings.OPENAI_EMBEDDING_MODEL)

    response = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = [item.embedding for item in response.data]
    log.info("✅ Generated %d embeddings (dim=%d)", len(embeddings), len(embeddings[0]))
    return embeddings


def upsert_to_qdrant(
    snippets: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> None:
    """Upsert policy snippets with embeddings into Qdrant."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
    )

    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    collection = settings.QDRANT_COLLECTION
    dim = len(embeddings[0])

    # Recreate collection
    if client.collection_exists(collection):
        client.delete_collection(collection)
        log.info("Deleted existing collection '%s'", collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    log.info("Created collection '%s' (dim=%d, cosine)", collection, dim)

    # Build points
    points = []
    for i, (snippet, embedding) in enumerate(zip(snippets, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, snippet["section_id"])),
                vector=embedding,
                payload={
                    "section_id": snippet["section_id"],
                    "guide": snippet["guide"],
                    "title": snippet["title"],
                    "text": snippet["text"],
                    "category": snippet["category"],
                    "effective_date": snippet["effective_date"],
                },
            )
        )

    client.upsert(collection_name=collection, points=points)
    log.info("✅ Upserted %d points into Qdrant collection '%s'", len(points), collection)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load ASIC policy snippets into Qdrant")
    parser.add_argument("--dry-run", action="store_true", help="Only show summary, skip embedding + Qdrant")
    args = parser.parse_args()

    snippets = load_snippets()
    print_summary(snippets)

    if args.dry_run:
        log.info("🔸 Dry-run mode – skipping embeddings and Qdrant write")
        return

    # Combine title + text for richer embedding
    texts = [f"{s['title']}\n\n{s['text']}" for s in snippets]
    embeddings = generate_embeddings(texts)
    upsert_to_qdrant(snippets, embeddings)


if __name__ == "__main__":
    main()
