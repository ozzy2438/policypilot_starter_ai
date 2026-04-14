"""
PolicyPilot – Risk Scoring Engine
----------------------------------
Calculates a composite risk score (0.0–1.0) for each complaint based on:
  • Complaint category severity
  • Monetary value at stake
  • Regulatory timeline pressure
  • Historical resolution patterns
  • Entity density (complexity indicator)

The scorer is rule-based and does NOT require an LLM call,
ensuring deterministic, explainable, and auditable results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.core.entities import ExtractedEntities

log = logging.getLogger(__name__)


# ── Category Severity Weights ────────────────────────────────────────
# Higher weight = higher inherent risk for the financial firm
CATEGORY_SEVERITY: dict[str, float] = {
    # High severity – regulatory action risk
    "Unauthorised transaction": 0.85,
    "Fraud": 0.90,
    "Identity theft": 0.90,
    "Responsible lending": 0.85,
    "Scam loss": 0.80,
    "Privacy breach": 0.85,
    "Unconscionable conduct": 0.90,
    "Misleading conduct": 0.80,

    # Medium-high severity – financial loss
    "Claim denial": 0.75,
    "Hardship": 0.70,
    "Financial hardship": 0.70,
    "Financial difficulty": 0.70,
    "Underinsurance": 0.65,
    "Guarantor": 0.70,
    "Guarantee": 0.70,

    # Medium severity – service failures
    "Claim delay": 0.60,
    "Fees and charges": 0.50,
    "Interest rate": 0.55,
    "Interest charges": 0.55,
    "Premium increase": 0.50,
    "Settlement": 0.55,
    "Discharge": 0.50,

    # Lower severity – disputes
    "Disputed transaction": 0.45,
    "Account administration": 0.40,
    "Communication": 0.40,
    "Reward points": 0.30,
    "Account closure": 0.45,
}

# ── Timeline pressure (ASIC IDR timeframes) ─────────────────────────
# Shorter mandatory response time = higher pressure
IDR_TIMEFRAMES: dict[str, int] = {
    "default": 30,           # RG 271.28
    "hardship": 21,          # RG 271.56(d)
    "default_notice": 21,    # RG 271.56(c)
    "superannuation": 45,    # RG 271.56(a)
    "death_benefit": 90,     # RG 271.56(b)
}


@dataclass
class RiskAssessment:
    """Complete risk assessment for a complaint."""
    overall_score: float          # 0.0 – 1.0
    category_score: float         # severity of the complaint type
    monetary_score: float         # based on dollar value at stake
    timeline_score: float         # regulatory response pressure
    complexity_score: float       # based on entity density
    risk_level: str               # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    factors: list[str]            # human-readable risk factors
    recommended_priority: int     # 1 (highest) – 4 (lowest)
    idr_deadline_days: int        # applicable IDR response deadline

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "category_score": round(self.category_score, 4),
            "monetary_score": round(self.monetary_score, 4),
            "timeline_score": round(self.timeline_score, 4),
            "complexity_score": round(self.complexity_score, 4),
            "risk_level": self.risk_level,
            "factors": self.factors,
            "recommended_priority": self.recommended_priority,
            "idr_deadline_days": self.idr_deadline_days,
        }


class RiskScorer:
    """
    Rule-based risk scoring engine for AFCA complaints.

    Weights are calibrated against real AFCA outcome data:
      - Category severity: 40%
      - Monetary value:    25%
      - Timeline pressure: 20%
      - Complexity:        15%
    """

    WEIGHT_CATEGORY = 0.40
    WEIGHT_MONETARY = 0.25
    WEIGHT_TIMELINE = 0.20
    WEIGHT_COMPLEXITY = 0.15

    def score(
        self,
        category: str,
        sub_category: str,
        entities: ExtractedEntities,
        product: Optional[str] = None,
        complaint_text: Optional[str] = None,
    ) -> RiskAssessment:
        """
        Calculate composite risk score for a complaint.

        Args:
            category: Primary complaint category (e.g. "Unauthorised transaction")
            sub_category: Sub-issue (e.g. "Fraudulent charge")
            entities: Extracted entities from complaint text
            product: Financial product type (e.g. "Credit Card")
            complaint_text: Raw complaint text for additional analysis

        Returns:
            RiskAssessment with scores, level, factors and priority
        """
        factors: list[str] = []

        # 1. Category severity
        cat_score = self._score_category(category, sub_category, factors)

        # 2. Monetary value
        mon_score = self._score_monetary(entities.monetary_amounts, factors)

        # 3. Timeline pressure
        time_score, deadline = self._score_timeline(category, product, factors)

        # 4. Complexity
        complex_score = self._score_complexity(entities, complaint_text, factors)

        # Weighted sum
        overall = (
            self.WEIGHT_CATEGORY * cat_score
            + self.WEIGHT_MONETARY * mon_score
            + self.WEIGHT_TIMELINE * time_score
            + self.WEIGHT_COMPLEXITY * complex_score
        )
        overall = min(max(overall, 0.0), 1.0)  # clamp

        # Determine level and priority
        risk_level, priority = self._classify(overall)

        assessment = RiskAssessment(
            overall_score=overall,
            category_score=cat_score,
            monetary_score=mon_score,
            timeline_score=time_score,
            complexity_score=complex_score,
            risk_level=risk_level,
            factors=factors,
            recommended_priority=priority,
            idr_deadline_days=deadline,
        )

        log.info(
            "Risk: %.2f (%s) | P%d | category=%.2f monetary=%.2f "
            "timeline=%.2f complexity=%.2f",
            overall, risk_level, priority,
            cat_score, mon_score, time_score, complex_score,
        )
        return assessment

    # ── Scoring components ───────────────────────────────────────────

    def _score_category(
        self, category: str, sub_category: str, factors: list[str]
    ) -> float:
        """Score based on complaint category severity."""
        # Check category, then sub_category
        score = CATEGORY_SEVERITY.get(category, 0.0)
        if score == 0.0:
            score = CATEGORY_SEVERITY.get(sub_category, 0.5)

        if score >= 0.8:
            factors.append(f"High-severity category: {category}")
        elif score >= 0.6:
            factors.append(f"Medium-severity category: {category}")

        return score

    def _score_monetary(
        self, amounts: list[str], factors: list[str]
    ) -> float:
        """Score based on monetary values mentioned in the complaint."""
        if not amounts:
            return 0.3  # Unknown amount = moderate concern

        # Parse amounts
        parsed = []
        for raw in amounts:
            clean = raw.replace("$", "").replace(",", "").replace("AUD", "").strip()
            if clean.lower().endswith("k"):
                clean = clean[:-1]
                try:
                    parsed.append(float(clean) * 1000)
                except ValueError:
                    pass
            else:
                try:
                    parsed.append(float(clean))
                except ValueError:
                    pass

        if not parsed:
            return 0.3

        max_amount = max(parsed)

        # Logarithmic scaling: $100 → 0.2, $1K → 0.4, $10K → 0.6, $100K → 0.8, $1M → 1.0
        import math
        score = min(math.log10(max(max_amount, 1)) / 6.0, 1.0)

        if max_amount >= 50_000:
            factors.append(f"High monetary value: ${max_amount:,.0f}")
        elif max_amount >= 10_000:
            factors.append(f"Significant monetary value: ${max_amount:,.0f}")

        return score

    def _score_timeline(
        self,
        category: str,
        product: Optional[str],
        factors: list[str],
    ) -> tuple[float, int]:
        """Score based on IDR timeline pressure."""
        cat_lower = category.lower()

        # Determine applicable deadline
        if "hardship" in cat_lower or "financial difficulty" in cat_lower:
            deadline = IDR_TIMEFRAMES["hardship"]
            factors.append("21-day IDR deadline (hardship complaint)")
        elif "default" in cat_lower:
            deadline = IDR_TIMEFRAMES["default_notice"]
            factors.append("21-day IDR deadline (default notice)")
        elif product and "super" in product.lower():
            deadline = IDR_TIMEFRAMES["superannuation"]
        elif "death" in cat_lower or "beneficiary" in cat_lower:
            deadline = IDR_TIMEFRAMES["death_benefit"]
        else:
            deadline = IDR_TIMEFRAMES["default"]

        # Shorter deadline = higher pressure
        score = 1.0 - (deadline / 90.0)
        score = min(max(score, 0.0), 1.0)

        return score, deadline

    def _score_complexity(
        self,
        entities: ExtractedEntities,
        complaint_text: Optional[str],
        factors: list[str],
    ) -> float:
        """Score based on complaint complexity indicators."""
        # Count total entities as complexity proxy
        total_entities = (
            len(entities.monetary_amounts)
            + len(entities.dates)
            + len(entities.percentages)
            + len(entities.durations)
            + len(entities.regulatory_refs)
            + len(entities.product_mentions)
        )

        # More entities = more complex complaint
        score = min(total_entities / 10.0, 1.0)

        # Long complaints tend to be more complex
        if complaint_text and len(complaint_text) > 500:
            score = min(score + 0.1, 1.0)
            factors.append("Complex complaint (lengthy with multiple entities)")

        if len(entities.regulatory_refs) >= 2:
            score = min(score + 0.1, 1.0)
            factors.append(f"Multiple regulatory references cited: {entities.regulatory_refs}")

        return score

    # ── Classification ───────────────────────────────────────────────

    @staticmethod
    def _classify(score: float) -> tuple[str, int]:
        """Map score to risk level and priority."""
        if score >= 0.75:
            return "CRITICAL", 1
        elif score >= 0.55:
            return "HIGH", 2
        elif score >= 0.35:
            return "MEDIUM", 3
        else:
            return "LOW", 4
