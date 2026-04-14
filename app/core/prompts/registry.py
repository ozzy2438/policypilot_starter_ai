"""
Prompt Registry
----------------
Manages prompt versions and provides a unified interface to build prompts.

Supported versions:
  • v1 – Naive baseline
  • v2 – Chain-of-Thought
  • v3 – Few-shot + CoT (production default)
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.prompts import v1_naive, v2_cot, v3_fewshot

log = logging.getLogger(__name__)

# ── Registry ─────────────────────────────────────────────────────────

_VERSIONS = {
    "v1": v1_naive,
    "v2": v2_cot,
    "v3": v3_fewshot,
}

_DESCRIPTIONS = {
    "v1": "Naive baseline – simple direct prompt",
    "v2": "Chain-of-Thought – structured step-by-step reasoning",
    "v3": "Few-shot + CoT – production prompt with examples (default)",
}


def get_prompt(
    version: str,
    complaint_text: str,
    docs: list[Any],
    entities: Any,
) -> dict[str, str]:
    """
    Build a prompt using the specified version.

    Args:
        version: Prompt version ("v1", "v2", or "v3")
        complaint_text: The complaint text
        docs: Retrieved regulatory documents
        entities: Extracted entities

    Returns:
        Dict with "system" and "user" keys

    Raises:
        ValueError: If version is not registered
    """
    if version not in _VERSIONS:
        available = ", ".join(_VERSIONS.keys())
        raise ValueError(f"Unknown prompt version '{version}'. Available: {available}")

    module = _VERSIONS[version]
    prompt = module.build_prompt(complaint_text, docs, entities)

    log.info("Built prompt version '%s': %s", version, _DESCRIPTIONS[version])
    return prompt


def list_versions() -> list[dict[str, str]]:
    """Return list of available prompt versions with descriptions."""
    return [
        {"version": k, "description": v}
        for k, v in _DESCRIPTIONS.items()
    ]
