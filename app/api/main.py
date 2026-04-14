"""
PolicyPilot – FastAPI Application
-----------------------------------
Production-ready API for AFCA complaint triage.

Endpoints:
  POST /triage          – Triage a single complaint
  POST /batch           – Triage multiple complaints
  GET  /health          – Health check (DB + Qdrant status)
  GET  /prompts         – List available prompt versions
  GET  /metrics         – Prometheus metrics endpoint
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.api.middleware import RequestLoggingMiddleware
from app.api.models import (
    BatchTriageRequest,
    BatchTriageResponse,
    ComplaintRequest,
    HealthResponse,
    PromptVersionResponse,
    TriageResponse,
    EntitiesResponse,
)
from app.api import metrics as m
from app.core.config import get_settings

log = logging.getLogger("policypilot.api")
settings = get_settings()

# ── Shared chain instance ────────────────────────────────────────────
_chain = None


def _get_chain(prompt_version: str = "v3"):
    """Lazy-initialise the triage chain (singleton per prompt version)."""
    global _chain
    from app.core.chain import TriageChain
    if _chain is None or _chain.prompt_version != prompt_version:
        _chain = TriageChain(prompt_version=prompt_version)
    return _chain


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup / shutdown events."""
    log.info("🚀 PolicyPilot API starting up (env=%s)", settings.ENVIRONMENT)
    # Pre-warm chain
    _get_chain()
    yield
    log.info("PolicyPilot API shutting down")


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="PolicyPilot API",
    description=(
        "AFCA Complaint Triage & Compliance Copilot – "
        "AI-powered complaint classification, risk scoring, "
        "and regulatory response drafting for Australian financial services."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────

@app.post(
    "/triage",
    response_model=TriageResponse,
    summary="Triage a single complaint",
    tags=["Triage"],
)
async def triage(payload: ComplaintRequest) -> TriageResponse:
    """
    Triage a single AFCA complaint.

    The pipeline:
    1. Extracts entities (amounts, dates, products, regulations)
    2. Retrieves relevant ASIC policy documents
    3. Classifies the complaint and generates a draft response
    4. Calculates a composite risk score
    """
    m.ACTIVE_REQUESTS.inc()
    t_start = time.perf_counter()

    try:
        chain = _get_chain(payload.prompt_version)
        result = chain.run(
            complaint_text=payload.text,
            complaint_id=payload.complaint_id,
            product=payload.product,
            source=payload.source,
        )

        response = TriageResponse(
            complaint_id=result.complaint_id,
            category=result.category,
            sub_category=result.sub_category,
            risk_score=round(result.risk_assessment.overall_score, 4),
            risk_level=result.risk_assessment.risk_level,
            risk_factors=result.risk_assessment.factors,
            recommended_priority=result.risk_assessment.recommended_priority,
            idr_deadline_days=result.risk_assessment.idr_deadline_days,
            policy_refs=result.policy_refs,
            entities=EntitiesResponse(**result.entities.to_dict()),
            draft_response=result.draft_response,
            reasoning=result.reasoning,
            prompt_version=result.prompt_version,
            model=result.model,
            latency_ms=round(result.latency_ms, 1),
            tokens_used=result.tokens_used,
            cost_usd=round(result.cost_usd, 6),
        )

        # Record metrics
        latency_s = (time.perf_counter() - t_start)
        m.record_triage(
            endpoint="/triage",
            prompt_version=result.prompt_version,
            model=result.model,
            latency_s=latency_s,
            tokens=result.tokens_used,
            cost=result.cost_usd,
            risk_score=result.risk_assessment.overall_score,
            risk_level=result.risk_assessment.risk_level,
            category=result.category,
        )

        return response

    except Exception as exc:
        latency_s = (time.perf_counter() - t_start)
        m.TRIAGE_REQUESTS.labels(
            endpoint="/triage",
            prompt_version=payload.prompt_version,
            status="error",
        ).inc()
        log.error("Triage failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Triage failed: {str(exc)}")

    finally:
        m.ACTIVE_REQUESTS.dec()


@app.post(
    "/batch",
    response_model=BatchTriageResponse,
    summary="Batch triage multiple complaints",
    tags=["Triage"],
)
async def batch_triage(payload: BatchTriageRequest) -> BatchTriageResponse:
    """
    Triage multiple complaints in a single request (max 50).
    Returns individual results plus aggregate statistics.
    """
    t_start = time.perf_counter()
    results = []
    total_tokens = 0
    total_cost = 0.0

    for complaint in payload.complaints:
        try:
            chain = _get_chain(complaint.prompt_version)
            result = chain.run(
                complaint_text=complaint.text,
                complaint_id=complaint.complaint_id,
                product=complaint.product,
                source=complaint.source,
            )

            results.append(TriageResponse(
                complaint_id=result.complaint_id,
                category=result.category,
                sub_category=result.sub_category,
                risk_score=round(result.risk_assessment.overall_score, 4),
                risk_level=result.risk_assessment.risk_level,
                risk_factors=result.risk_assessment.factors,
                recommended_priority=result.risk_assessment.recommended_priority,
                idr_deadline_days=result.risk_assessment.idr_deadline_days,
                policy_refs=result.policy_refs,
                entities=EntitiesResponse(**result.entities.to_dict()),
                draft_response=result.draft_response,
                reasoning=result.reasoning,
                prompt_version=result.prompt_version,
                model=result.model,
                latency_ms=round(result.latency_ms, 1),
                tokens_used=result.tokens_used,
                cost_usd=round(result.cost_usd, 6),
            ))

            total_tokens += result.tokens_used
            total_cost += result.cost_usd

        except Exception as exc:
            log.error("Batch item %s failed: %s", complaint.complaint_id, exc)
            continue

    total_latency_ms = (time.perf_counter() - t_start) * 1000

    return BatchTriageResponse(
        results=results,
        total=len(results),
        total_latency_ms=round(total_latency_ms, 1),
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 6),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """Check system health including database and vector store connectivity."""
    components = {}

    # Check PostgreSQL
    try:
        import psycopg2
        conn = psycopg2.connect(settings.postgres_dsn)
        conn.close()
        components["postgresql"] = "healthy"
    except Exception:
        components["postgresql"] = "unavailable"

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=3)
        client.get_collections()
        components["qdrant"] = "healthy"
    except Exception:
        components["qdrant"] = "unavailable"

    # Check OpenAI API key
    components["openai"] = "configured" if (
        settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("sk-your")
    ) else "mock_mode"

    status = "healthy" if all(v != "error" for v in components.values()) else "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        environment=settings.ENVIRONMENT,
        components=components,
    )


@app.get(
    "/prompts",
    response_model=list[PromptVersionResponse],
    summary="List available prompt versions",
    tags=["System"],
)
async def list_prompts() -> list[PromptVersionResponse]:
    """List all available prompt engineering versions."""
    from app.core.prompts.registry import list_versions
    return [PromptVersionResponse(**v) for v in list_versions()]


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    tags=["System"],
)
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
