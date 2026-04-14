"""
PolicyPilot – API Middleware
-----------------------------
Request/response logging, latency tracking, and structured logging
for production observability.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("policypilot.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs every request with:
      • Unique request ID
      • Method + path
      • Response status code
      • Latency in milliseconds
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        start_time = time.perf_counter()

        # Log request
        log.info(
            "[%s] → %s %s",
            request_id,
            request.method,
            request.url.path,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            log.error(
                "[%s] ✗ %s %s | %.0fms | %s",
                request_id,
                request.method,
                request.url.path,
                latency_ms,
                str(exc),
            )
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Log response
        log.info(
            "[%s] ← %s %s | %d | %.0fms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )

        # Add headers for traceability
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = f"{latency_ms:.0f}"

        return response
