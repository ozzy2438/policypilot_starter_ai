"""
PolicyPilot – Qdrant Hybrid Retriever
--------------------------------------
Retrieves relevant ASIC regulatory snippets from Qdrant using:
  • Dense vector search (cosine similarity via OpenAI embeddings)
  • Optional score threshold filtering
  • Metadata-based post-filtering (guide, category)
  • Mock mode for development without Qdrant
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.config import get_settings
from app.core.embeddings import PolicyEmbeddings

log = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedDoc:
    """A single retrieved document with its metadata and score."""
    section_id: str
    guide: str
    title: str
    text: str
    category: str
    score: float
    effective_date: str = ""

    @property
    def reference(self) -> str:
        """Human-readable reference string, e.g. 'RG 271.28'."""
        return f"{self.guide} — {self.section_id}"

    def to_context_str(self) -> str:
        """Format for injection into LLM prompt context."""
        return (
            f"[{self.section_id}] {self.title}\n"
            f"Source: {self.guide} | Category: {self.category}\n"
            f"{self.text}"
        )


class PolicyRetriever:
    """Retrieves ASIC policy snippets from Qdrant vector store."""

    def __init__(
        self,
        embeddings: Optional[PolicyEmbeddings] = None,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ):
        self.embeddings = embeddings or PolicyEmbeddings()
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._mock = False
        self._mock_data: list[dict[str, Any]] = []

        # Try connecting to Qdrant
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=5,
            )
            # Verify collection exists
            if not self.client.collection_exists(settings.QDRANT_COLLECTION):
                log.warning(
                    "Qdrant collection '%s' not found – switching to MOCK mode",
                    settings.QDRANT_COLLECTION,
                )
                self._mock = True
                self._load_mock_data()
            else:
                log.info("Connected to Qdrant collection '%s'", settings.QDRANT_COLLECTION)
        except Exception as e:
            log.warning("Cannot connect to Qdrant (%s) – switching to MOCK mode", e)
            self._mock = True
            self._load_mock_data()

    # ── Public API ───────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_guide: Optional[str] = None,
        filter_category: Optional[str] = None,
    ) -> list[RetrievedDoc]:
        """
        Retrieve relevant policy documents for a complaint query.

        Args:
            query: The complaint text to search for
            top_k: Override default number of results
            filter_guide: Filter by specific guide (e.g. "RG 271")
            filter_category: Filter by category (e.g. "Hardship")

        Returns:
            List of RetrievedDoc sorted by relevance score (descending)
        """
        k = top_k or self.top_k

        if self._mock:
            return self._mock_retrieve(query, k, filter_guide, filter_category)

        # Embed query
        query_vector = self.embeddings.embed_text(query)

        # Build Qdrant filter
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        if filter_guide:
            conditions.append(
                FieldCondition(key="guide", match=MatchValue(value=filter_guide))
            )
        if filter_category:
            conditions.append(
                FieldCondition(key="category", match=MatchValue(value=filter_category))
            )

        qdrant_filter = Filter(must=conditions) if conditions else None

        # Search
        results = self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            score_threshold=self.score_threshold,
        )

        docs = []
        for hit in results:
            payload = hit.payload or {}
            docs.append(RetrievedDoc(
                section_id=payload.get("section_id", ""),
                guide=payload.get("guide", ""),
                title=payload.get("title", ""),
                text=payload.get("text", ""),
                category=payload.get("category", ""),
                effective_date=payload.get("effective_date", ""),
                score=hit.score,
            ))

        log.info(
            "Retrieved %d docs for query (top_k=%d, threshold=%.2f)",
            len(docs), k, self.score_threshold,
        )
        return docs

    # ── Mock mode ────────────────────────────────────────────────────

    def _load_mock_data(self) -> None:
        """Load ASIC snippets from JSON for mock retrieval."""
        import json
        # Prefer the real parsed RG 271 document; fall back to synthetic snippets
        json_path = settings.data_raw / "asic_rg271_real.json"
        if not json_path.exists():
            json_path = settings.data_raw / "asic_rg271_snippets.json"
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self._mock_data = json.load(f)
            log.info("MOCK: Loaded %d snippets from %s", len(self._mock_data), json_path)
        except FileNotFoundError:
            log.warning("MOCK: %s not found – mock retrieval will return empty results", json_path)

    def _mock_retrieve(
        self,
        query: str,
        top_k: int,
        filter_guide: Optional[str],
        filter_category: Optional[str],
    ) -> list[RetrievedDoc]:
        """Keyword-based mock retrieval for development without Qdrant."""
        if not self._mock_data:
            return []

        query_lower = query.lower()
        scored: list[tuple[float, dict]] = []

        for snippet in self._mock_data:
            # Apply filters
            if filter_guide and snippet["guide"] != filter_guide:
                continue
            if filter_category and snippet["category"] != filter_category:
                continue

            # Simple keyword matching score
            combined = f"{snippet['title']} {snippet['text']} {snippet['category']}".lower()
            query_words = set(query_lower.split())
            matched = sum(1 for w in query_words if w in combined)
            score = matched / max(len(query_words), 1)

            if score > 0:
                scored.append((score, snippet))

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]

        # If no keyword matches, return most relevant defaults based on category
        if not top:
            defaults = self._mock_data[:min(top_k, 3)]
            top = [(0.5, s) for s in defaults]

        docs = []
        for score, snippet in top:
            docs.append(RetrievedDoc(
                section_id=snippet["section_id"],
                guide=snippet["guide"],
                title=snippet["title"],
                text=snippet["text"],
                category=snippet["category"],
                effective_date=snippet.get("effective_date", ""),
                score=round(score, 4),
            ))

        log.info("MOCK: Retrieved %d docs for query", len(docs))
        return docs
