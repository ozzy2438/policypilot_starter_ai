"""
PolicyPilot – OpenAI Embedding Wrapper
--------------------------------------
Provides a thin abstraction over the OpenAI Embeddings API with:
  • Batch embedding support
  • Dimension tracking
  • Token-cost estimation
  • Fallback mock mode (no API key required)
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


class PolicyEmbeddings:
    """Wraps OpenAI text-embedding models with batch support."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or settings.OPENAI_EMBEDDING_MODEL
        self.api_key = api_key or settings.OPENAI_API_KEY
        self._dimension: Optional[int] = None
        self._mock = not bool(self.api_key) or self.api_key.startswith("sk-your")

        if self._mock:
            log.warning(
                "No valid OPENAI_API_KEY detected – running in MOCK mode. "
                "Embeddings will be random vectors (dim=1536)."
            )
            self._dimension = 1536

    # ── Public API ───────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string. Returns a float vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        if self._mock:
            return self._mock_embed(texts)

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        log.info("Embedding %d texts (model=%s)", len(texts), self.model)
        response = client.embeddings.create(model=self.model, input=texts)

        vectors = [item.embedding for item in response.data]
        self._dimension = len(vectors[0])

        # Log cost estimate
        total_tokens = response.usage.total_tokens
        cost_usd = total_tokens * 0.00002  # text-embedding-3-small pricing
        log.info(
            "✅ Embedded %d texts → dim=%d | tokens=%d | est_cost=$%.4f",
            len(vectors), self._dimension, total_tokens, cost_usd,
        )
        return vectors

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (populated after first call)."""
        if self._dimension is None:
            # Make a probe call to determine dimension
            self.embed_text("dimension probe")
        return self._dimension

    # ── Mock mode ────────────────────────────────────────────────────

    def _mock_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic pseudo-random vectors for testing."""
        import hashlib
        import struct

        dim = 1536
        vectors = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            # Expand hash to fill the vector deterministically
            seed = struct.unpack("<I", h[:4])[0]
            import random
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(dim)]
            # L2-normalise
            norm = sum(v * v for v in vec) ** 0.5
            vectors.append([v / norm for v in vec])

        log.debug("MOCK: Generated %d vectors (dim=%d)", len(vectors), dim)
        return vectors
