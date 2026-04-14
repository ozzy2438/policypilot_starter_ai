"""
Prompt v1 – Naive (Baseline)
-----------------------------
Simple, direct prompt with minimal instruction.
Used as a baseline for comparison with more sophisticated approaches.
"""

from __future__ import annotations

from typing import Any


def build_prompt(
    complaint_text: str,
    context_docs: list[Any],
    entities: Any,
) -> dict[str, str]:
    """Build v1 naive prompt."""

    # Build context from retrieved docs
    context = ""
    if context_docs:
        context_parts = []
        for doc in context_docs:
            context_parts.append(f"- {doc.section_id}: {doc.text}")
        context = "\n".join(context_parts)

    system = (
        "You are a complaint triage assistant for an Australian financial services company. "
        "Classify the complaint and suggest a response."
    )

    user = f"""Complaint:
{complaint_text}

Relevant regulations:
{context if context else 'No specific regulations retrieved.'}

Please classify this complaint and provide:
1. Category
2. Sub-category
3. Relevant policy references
4. A draft response to the complainant
5. Your reasoning"""

    return {"system": system, "user": user}
