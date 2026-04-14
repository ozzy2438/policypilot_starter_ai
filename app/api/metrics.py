"""
PolicyPilot – Prometheus Metrics
---------------------------------
Defines application metrics for monitoring AI performance,
latency, cost, and operational health.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, Info


# ── Application Info ─────────────────────────────────────────────────
APP_INFO = Info("policypilot", "PolicyPilot application information")
APP_INFO.info({
    "version": "1.0.0",
    "description": "AFCA Complaint Triage & Compliance Copilot",
})

# ── Request Metrics ──────────────────────────────────────────────────
TRIAGE_REQUESTS = Counter(
    "policypilot_triage_requests_total",
    "Total number of triage requests",
    ["endpoint", "prompt_version", "status"],
)

TRIAGE_LATENCY = Histogram(
    "policypilot_triage_latency_seconds",
    "Triage request latency in seconds",
    ["endpoint", "prompt_version"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# ── AI Metrics ───────────────────────────────────────────────────────
LLM_TOKENS = Counter(
    "policypilot_llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "prompt_version"],
)

LLM_COST = Counter(
    "policypilot_llm_cost_usd_total",
    "Total LLM cost in USD",
    ["model", "prompt_version"],
)

RISK_SCORE = Histogram(
    "policypilot_risk_score",
    "Distribution of triage risk scores",
    ["risk_level"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

RETRIEVAL_DOCS = Histogram(
    "policypilot_retrieval_docs_count",
    "Number of documents retrieved per query",
    buckets=[0, 1, 2, 3, 4, 5, 10],
)

# ── Category Metrics ─────────────────────────────────────────────────
COMPLAINT_CATEGORY = Counter(
    "policypilot_complaint_category_total",
    "Complaint count by category",
    ["category", "risk_level"],
)

# ── System Metrics ───────────────────────────────────────────────────
ACTIVE_REQUESTS = Gauge(
    "policypilot_active_requests",
    "Number of currently active triage requests",
)


def record_triage(
    endpoint: str,
    prompt_version: str,
    model: str,
    latency_s: float,
    tokens: int,
    cost: float,
    risk_score: float,
    risk_level: str,
    category: str,
    status: str = "success",
) -> None:
    """Record all metrics for a completed triage request."""
    TRIAGE_REQUESTS.labels(
        endpoint=endpoint, prompt_version=prompt_version, status=status
    ).inc()

    TRIAGE_LATENCY.labels(
        endpoint=endpoint, prompt_version=prompt_version
    ).observe(latency_s)

    LLM_TOKENS.labels(model=model, prompt_version=prompt_version).inc(tokens)
    LLM_COST.labels(model=model, prompt_version=prompt_version).inc(cost)

    RISK_SCORE.labels(risk_level=risk_level).observe(risk_score)
    COMPLAINT_CATEGORY.labels(category=category, risk_level=risk_level).inc()
