"""
PolicyPilot – Pydantic API Models
----------------------------------
Request/response schemas for all API endpoints.
Structured for automatic OpenAPI (Swagger) documentation.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────────

class ComplaintRequest(BaseModel):
    """Single complaint triage request."""
    text: str = Field(
        ...,
        min_length=10,
        description="The complaint text to triage",
        examples=["I noticed three transactions on my Visa credit card totalling $4,280 that I did not authorise."],
    )
    complaint_id: Optional[str] = Field(
        None,
        description="Optional complaint reference ID",
        examples=["AFCA-2023-00001"],
    )
    product: Optional[str] = Field(
        None,
        description="Known financial product type",
        examples=["Credit Card", "Home Loan", "General Insurance"],
    )
    source: str = Field(
        "email",
        description="Complaint source channel",
        examples=["email", "phone", "web", "letter"],
    )
    prompt_version: str = Field(
        "v3",
        description="Prompt version to use (v1, v2, v3)",
        examples=["v1", "v2", "v3"],
    )


# ── Response Models ──────────────────────────────────────────────────

class EntitiesResponse(BaseModel):
    """Extracted entities from complaint text."""
    monetary_amounts: list[str] = []
    dates: list[str] = []
    account_numbers: list[str] = []
    percentages: list[str] = []
    durations: list[str] = []
    regulatory_refs: list[str] = []
    product_mentions: list[str] = []
    states: list[str] = []


class TriageResponse(BaseModel):
    """Complete triage result for a single complaint."""
    complaint_id: Optional[str] = None
    category: str
    sub_category: str
    risk_score: float = Field(ge=0.0, le=1.0, description="Composite risk score")
    risk_level: str = Field(description="CRITICAL, HIGH, MEDIUM, or LOW")
    risk_factors: list[str] = []
    recommended_priority: int = Field(ge=1, le=4, description="1=highest, 4=lowest")
    idr_deadline_days: int = Field(description="Applicable ASIC IDR deadline")
    policy_refs: list[str] = []
    entities: EntitiesResponse
    draft_response: str
    reasoning: str
    prompt_version: str
    model: str
    latency_ms: float
    tokens_used: int
    cost_usd: float


class BatchTriageRequest(BaseModel):
    """Batch triage request for multiple complaints."""
    complaints: list[ComplaintRequest] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of complaints to triage (max 50)",
    )


class BatchTriageResponse(BaseModel):
    """Batch triage results."""
    results: list[TriageResponse]
    total: int
    total_latency_ms: float
    total_tokens: int
    total_cost_usd: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    environment: str
    components: dict[str, str]


class PromptVersionResponse(BaseModel):
    """Available prompt version info."""
    version: str
    description: str
