"""
Prompt v2 – Chain-of-Thought (CoT)
------------------------------------
Structured step-by-step reasoning prompt that guides the LLM through
a systematic triage process. Inspired by ASIC's actual IDR workflow.

Improvements over v1:
  • Explicit reasoning steps
  • Role-specific persona
  • Structured output format
  • Regulatory awareness
"""

from __future__ import annotations

from typing import Any


def build_prompt(
    complaint_text: str,
    context_docs: list[Any],
    entities: Any,
) -> dict[str, str]:
    """Build v2 Chain-of-Thought prompt."""

    # Build context from retrieved docs
    context = ""
    if context_docs:
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(
                f"[{i}] {doc.section_id} — {doc.title}\n    {doc.text}"
            )
        context = "\n\n".join(context_parts)

    # Build entity summary
    entity_summary = entities.summary if hasattr(entities, 'summary') else "No entities extracted"

    system = """You are a Senior Compliance Analyst at an Australian financial services firm, specialising in AFCA complaint triage and ASIC regulatory compliance.

Your role is to:
1. Accurately classify incoming complaints according to AFCA categories
2. Identify applicable ASIC regulations and industry codes
3. Assess risk and urgency based on regulatory timeframes
4. Draft professional, compliant responses

You must reason step-by-step through each complaint, following the IDR (Internal Dispute Resolution) framework outlined in ASIC Regulatory Guide 271.

Always cite specific regulatory sections. Never fabricate regulations."""

    user = f"""## Complaint for Triage

**Complaint Text:**
{complaint_text}

**Extracted Entities:**
{entity_summary}

---

## Relevant Regulatory Context

{context if context else 'No specific regulations retrieved from the knowledge base.'}

---

## Instructions

Analyse this complaint step-by-step:

**Step 1 – Classification:** What is the primary complaint category and sub-category? Consider AFCA's standard taxonomy (Banking, Insurance, Superannuation, Credit, Investments).

**Step 2 – Regulatory Assessment:** Which specific ASIC regulatory guides, legislation sections, or industry codes apply? Reference the regulatory context provided above.

**Step 3 – Risk Factors:** What are the key risk factors? Consider monetary value, regulatory timeline obligations (RG 271 timeframes), potential for systemic issues, and reputational risk.

**Step 4 – Draft Response:** Write a professional IDR response that:
  - Acknowledges receipt of the complaint
  - Summarises the complainant's key concerns
  - Outlines the investigation process
  - Provides the applicable IDR timeframe
  - Informs about AFCA escalation rights

**Step 5 – Reasoning Summary:** Provide a concise summary of your reasoning tying together the classification, regulations, and response."""

    return {"system": system, "user": user}
