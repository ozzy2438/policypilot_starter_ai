"""
PolicyPilot – RAG Triage Chain
-------------------------------
End-to-end LangChain pipeline that:
  1. Accepts a complaint text
  2. Extracts entities (regex + NER)
  3. Retrieves relevant ASIC policy docs from Qdrant
  4. Generates structured triage output via LLM (function calling)
  5. Calculates risk score

Supports:
  • Multiple prompt versions (v1/v2/v3) via prompt registry
  • Structured JSON output via OpenAI function calling
  • Full mock mode for development without API keys
  • Token usage and cost tracking
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.config import get_settings
from app.core.embeddings import PolicyEmbeddings
from app.core.entities import EntityExtractor, ExtractedEntities
from app.core.retriever import PolicyRetriever, RetrievedDoc
from app.core.risk import RiskScorer, RiskAssessment

log = logging.getLogger(__name__)
settings = get_settings()


# ── Output Schema ────────────────────────────────────────────────────

@dataclass
class TriageResult:
    """Complete triage output for a single complaint."""
    complaint_id: Optional[str]
    category: str
    sub_category: str
    risk_assessment: RiskAssessment
    policy_refs: list[str]
    entities: ExtractedEntities
    draft_response: str
    reasoning: str
    retrieved_docs: list[RetrievedDoc]
    prompt_version: str
    model: str
    latency_ms: float
    tokens_used: int
    cost_usd: float

    def to_dict(self) -> dict:
        """Serialise to dict for API response."""
        return {
            "complaint_id": self.complaint_id,
            "category": self.category,
            "sub_category": self.sub_category,
            "risk_score": self.risk_assessment.overall_score,
            "risk_level": self.risk_assessment.risk_level,
            "risk_factors": self.risk_assessment.factors,
            "recommended_priority": self.risk_assessment.recommended_priority,
            "idr_deadline_days": self.risk_assessment.idr_deadline_days,
            "policy_refs": self.policy_refs,
            "entities": self.entities.to_dict(),
            "draft_response": self.draft_response,
            "reasoning": self.reasoning,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "tokens_used": self.tokens_used,
            "cost_usd": round(self.cost_usd, 6),
        }


# ── LLM Output Schema (for function calling) ────────────────────────

TRIAGE_FUNCTIONS = [
    {
        "name": "triage_complaint",
        "description": "Triage an AFCA financial complaint by classifying it and generating a response",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Primary complaint category (e.g. Unauthorised Transaction, Claim Denial, Hardship)",
                },
                "sub_category": {
                    "type": "string",
                    "description": "Specific sub-issue (e.g. Fraudulent charge, Home and contents, Repayment difficulty)",
                },
                "policy_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of relevant ASIC/regulatory references (e.g. RG 271.28, Section 54 ICA)",
                },
                "draft_response": {
                    "type": "string",
                    "description": "Professional draft response to the complainant acknowledging their complaint and outlining next steps",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning explaining the classification and applicable regulations",
                },
            },
            "required": ["category", "sub_category", "policy_refs", "draft_response", "reasoning"],
        },
    }
]


class TriageChain:
    """
    End-to-end RAG triage pipeline.

    Usage:
        chain = TriageChain(prompt_version="v3")
        result = chain.run("I noticed three transactions on my Visa...")
    """

    def __init__(
        self,
        prompt_version: str = "v3",
        model: Optional[str] = None,
        retriever: Optional[PolicyRetriever] = None,
    ):
        self.prompt_version = prompt_version
        self.model = model or settings.OPENAI_MODEL
        self.retriever = retriever or PolicyRetriever()
        self.entity_extractor = EntityExtractor()
        self.risk_scorer = RiskScorer()
        self._mock = not bool(settings.OPENAI_API_KEY) or settings.OPENAI_API_KEY.startswith("sk-your")

        if self._mock:
            log.warning("TriageChain running in MOCK mode (no OpenAI API key)")

    def run(
        self,
        complaint_text: str,
        complaint_id: Optional[str] = None,
        product: Optional[str] = None,
        source: str = "email",
    ) -> TriageResult:
        """
        Run the full triage pipeline on a complaint.

        Args:
            complaint_text: The raw complaint text
            complaint_id: Optional complaint reference
            product: Known product type (if available)
            source: Complaint source channel

        Returns:
            TriageResult with all analysis
        """
        t_start = time.perf_counter()

        # Step 1: Extract entities
        entities = self.entity_extractor.extract(complaint_text)
        log.info("Step 1: Extracted entities → %s", entities.summary)

        # Step 2: Retrieve relevant policies
        docs = self.retriever.retrieve(complaint_text, top_k=5)
        log.info("Step 2: Retrieved %d policy documents", len(docs))

        # Step 3: Build context + call LLM (or mock)
        if self._mock:
            llm_output, tokens, cost = self._mock_llm(complaint_text, docs, entities)
        else:
            llm_output, tokens, cost = self._call_llm(complaint_text, docs, entities)

        category = llm_output.get("category", "Unknown")
        sub_category = llm_output.get("sub_category", "Unknown")

        # Step 4: Calculate risk score
        risk = self.risk_scorer.score(
            category=category,
            sub_category=sub_category,
            entities=entities,
            product=product,
            complaint_text=complaint_text,
        )
        log.info("Step 4: Risk score = %.2f (%s)", risk.overall_score, risk.risk_level)

        latency_ms = (time.perf_counter() - t_start) * 1000

        result = TriageResult(
            complaint_id=complaint_id,
            category=category,
            sub_category=sub_category,
            risk_assessment=risk,
            policy_refs=llm_output.get("policy_refs", []),
            entities=entities,
            draft_response=llm_output.get("draft_response", ""),
            reasoning=llm_output.get("reasoning", ""),
            retrieved_docs=docs,
            prompt_version=self.prompt_version,
            model=self.model if not self._mock else "mock",
            latency_ms=latency_ms,
            tokens_used=tokens,
            cost_usd=cost,
        )

        log.info(
            "✅ Triage complete: %s → %s (%s) | %.0fms | %d tokens | $%.4f",
            complaint_id or "N/A", category, risk.risk_level,
            latency_ms, tokens, cost,
        )
        return result

    # ── LLM Call ─────────────────────────────────────────────────────

    def _call_llm(
        self,
        complaint_text: str,
        docs: list[RetrievedDoc],
        entities: ExtractedEntities,
    ) -> tuple[dict[str, Any], int, float]:
        """Call OpenAI with function calling for structured output."""
        from openai import OpenAI

        # Build prompt
        prompt = self._build_prompt(complaint_text, docs, entities)

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            functions=TRIAGE_FUNCTIONS,
            function_call={"name": "triage_complaint"},
            temperature=settings.OPENAI_TEMPERATURE,
        )

        # Parse function call response
        message = response.choices[0].message
        if message.function_call:
            output = json.loads(message.function_call.arguments)
        else:
            output = {
                "category": "Unknown",
                "sub_category": "Unknown",
                "policy_refs": [],
                "draft_response": message.content or "",
                "reasoning": "LLM did not return structured output",
            }

        # Token tracking
        usage = response.usage
        tokens = usage.total_tokens if usage else 0
        # Pricing: gpt-4o-mini input=$0.15/1M, output=$0.60/1M (approx $0.0004/1K)
        cost = tokens * 0.0004 / 1000

        return output, tokens, cost

    def _build_prompt(
        self,
        complaint_text: str,
        docs: list[RetrievedDoc],
        entities: ExtractedEntities,
    ) -> dict[str, str]:
        """Build the prompt using the selected version."""
        # Import prompt registry
        from app.core.prompts.registry import get_prompt
        return get_prompt(
            version=self.prompt_version,
            complaint_text=complaint_text,
            docs=docs,
            entities=entities,
        )

    # ── Mock LLM ─────────────────────────────────────────────────────

    def _mock_llm(
        self,
        complaint_text: str,
        docs: list[RetrievedDoc],
        entities: ExtractedEntities,
    ) -> tuple[dict[str, Any], int, float]:
        """Generate mock triage output based on keyword analysis."""
        text_lower = complaint_text.lower()

        # Simple keyword-based classification
        category_map = {
            "unauthorised": ("Unauthorised Transaction", "Fraudulent charge"),
            "fraud": ("Fraud", "Identity fraud"),
            "scam": ("Scam Loss", "Authorised push payment"),
            "hardship": ("Financial Hardship", "Repayment difficulty"),
            "claim denied": ("Claim Denial", "Policy exclusion"),
            "claim delay": ("Claim Delay", "Processing delay"),
            "denied": ("Claim Denial", "Coverage dispute"),
            "insurance": ("Insurance Dispute", "Claim assessment"),
            "interest rate": ("Interest Rate Dispute", "Rate increase"),
            "fee": ("Fees and Charges", "Excessive fees"),
            "privacy": ("Privacy Breach", "Data exposure"),
            "home loan": ("Home Loan Dispute", "Lending practice"),
            "credit card": ("Credit Card Dispute", "Transaction dispute"),
            "superannuation": ("Superannuation Dispute", "Account administration"),
            "responsible lending": ("Responsible Lending", "Unaffordable loan"),
        }

        category, sub_category = "General Complaint", "Other"
        for keyword, (cat, sub) in category_map.items():
            if keyword in text_lower:
                category, sub_category = cat, sub
                break

        # Build policy refs from retrieved docs
        policy_refs = [doc.section_id for doc in docs[:3]] if docs else ["RG 271.28"]

        # Generate mock draft response
        draft = (
            f"Thank you for your complaint regarding {category.lower()}. "
            f"We acknowledge receipt of your complaint and have assigned it priority status. "
            f"Under ASIC Regulatory Guide 271, we are required to provide you with a response "
            f"within 30 calendar days. We will investigate the matters you have raised and "
            f"contact you with our findings. If you are not satisfied with our response, "
            f"you may escalate your complaint to the Australian Financial Complaints Authority (AFCA)."
        )

        reasoning = (
            f"1. Classification: The complaint relates to {category} based on the described issues.\n"
            f"2. Regulatory framework: {', '.join(policy_refs)} are applicable.\n"
            f"3. Key entities: {entities.summary}\n"
            f"4. Timeline: IDR response required within the applicable ASIC timeframe.\n"
            f"5. Draft response prepared acknowledging the complaint and outlining the IDR process."
        )

        output = {
            "category": category,
            "sub_category": sub_category,
            "policy_refs": policy_refs,
            "draft_response": draft,
            "reasoning": reasoning,
        }

        # Simulate token usage
        mock_tokens = len(complaint_text.split()) * 3
        mock_cost = mock_tokens * 0.0004 / 1000

        return output, mock_tokens, mock_cost
