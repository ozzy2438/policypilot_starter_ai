"""
PolicyPilot – Entity Extraction
--------------------------------
Extracts structured entities from complaint text using a hybrid approach:
  • Regex patterns for deterministic entities (dates, amounts, account numbers)
  • LLM-based extraction for semantic entities (product type, company, names)

This module does NOT require an API key – regex extraction works standalone.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    """Container for entities extracted from complaint text."""
    monetary_amounts: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    account_numbers: list[str] = field(default_factory=list)
    percentages: list[str] = field(default_factory=list)
    durations: list[str] = field(default_factory=list)
    regulatory_refs: list[str] = field(default_factory=list)
    product_mentions: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "monetary_amounts": self.monetary_amounts,
            "dates": self.dates,
            "account_numbers": self.account_numbers,
            "percentages": self.percentages,
            "durations": self.durations,
            "regulatory_refs": self.regulatory_refs,
            "product_mentions": self.product_mentions,
            "states": self.states,
        }

    @property
    def summary(self) -> str:
        """One-line summary of extracted entities."""
        parts = []
        if self.monetary_amounts:
            parts.append(f"amounts={self.monetary_amounts}")
        if self.dates:
            parts.append(f"dates={self.dates}")
        if self.regulatory_refs:
            parts.append(f"refs={self.regulatory_refs}")
        if self.product_mentions:
            parts.append(f"products={self.product_mentions}")
        return " | ".join(parts) if parts else "no entities found"


# ── Regex Patterns ───────────────────────────────────────────────────

# Australian dollar amounts: $4,280 or $45,000.00 or AUD 150k
_AMOUNT_RE = re.compile(
    r'(?:\$|AUD\s?)[\d,]+(?:\.\d{2})?(?:k)?',
    re.IGNORECASE,
)

# Dates: 15 November 2022, January 2023, 2022-23, dd/mm/yyyy
_DATE_RE = re.compile(
    r'\b\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b'
    r'|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    r'|\d{4}-\d{2}'
    r'|\d{1,2}/\d{1,2}/\d{4}',
    re.IGNORECASE,
)

# Account / reference numbers
_ACCOUNT_RE = re.compile(r'\b[A-Z]{2,4}[-\s]?\d{4,}[-\s]?\d{0,6}\b')

# Percentages: 0.50%, 21.99%, 38%
_PERCENT_RE = re.compile(r'\b\d+(?:\.\d+)?%\b')

# Durations: 3 months, 14 days, 2 years, 6 weeks
_DURATION_RE = re.compile(
    r'\b\d+\s?(?:day|days|week|weeks|month|months|year|years|business\sdays)\b',
    re.IGNORECASE,
)

# ASIC / regulatory references: RG 271.28, ASIC RG 209, Section 54
_REG_REF_RE = re.compile(
    r'(?:ASIC\s)?RG\s?\d+(?:\.\d+)?'
    r'|Section\s\d+[A-Z]?'
    r'|Part\s\d+[A-Z.]+'
    r'|APP\s\d+',
    re.IGNORECASE,
)

# Australian states
_STATE_RE = re.compile(
    r'\b(?:VIC|NSW|QLD|WA|SA|TAS|ACT|NT'
    r'|Victoria|New South Wales|Queensland|Western Australia'
    r'|South Australia|Tasmania)\b',
    re.IGNORECASE,
)

# Financial product keywords
_PRODUCT_KEYWORDS = {
    "credit card", "home loan", "mortgage", "personal loan",
    "superannuation", "super fund", "life insurance", "income protection",
    "general insurance", "car insurance", "home insurance", "travel insurance",
    "term deposit", "savings account", "business loan", "overdraft",
    "buy now pay later", "bnpl", "offset account", "redraw",
}


class EntityExtractor:
    """Extracts structured entities from complaint text."""

    def extract(self, text: str) -> ExtractedEntities:
        """
        Extract all entities from complaint text using regex patterns.

        Args:
            text: Raw complaint text

        Returns:
            ExtractedEntities with all found entities
        """
        entities = ExtractedEntities(
            monetary_amounts=_unique(_AMOUNT_RE.findall(text)),
            dates=_unique(_DATE_RE.findall(text)),
            account_numbers=_unique(_ACCOUNT_RE.findall(text)),
            percentages=_unique(_PERCENT_RE.findall(text)),
            durations=_unique(_DURATION_RE.findall(text)),
            regulatory_refs=_unique(_REG_REF_RE.findall(text)),
            product_mentions=self._extract_products(text),
            states=_unique(_STATE_RE.findall(text)),
        )

        log.debug("Extracted entities: %s", entities.summary)
        return entities

    def _extract_products(self, text: str) -> list[str]:
        """Find financial product mentions via keyword matching."""
        text_lower = text.lower()
        found = []
        for keyword in _PRODUCT_KEYWORDS:
            if keyword in text_lower:
                found.append(keyword)
        return sorted(set(found))


def _unique(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalised = item.strip()
        if normalised not in seen:
            seen.add(normalised)
            result.append(normalised)
    return result
