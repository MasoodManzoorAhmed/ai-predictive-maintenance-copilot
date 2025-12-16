# backend/api/main.py
"""
FastAPI entrypoint for AI Predictive Maintenance Copilot.

Phase 6 goals:
- Provide clean API surface for FD001–FD004 RUL inference
- Health check endpoint
- Central router registration
- Production-friendly app metadata

Enterprise add-on:
- Request logging middleware (CSV) with latency

NOTE:
- Model/scaler loading happens in services (model_loader.py) later.
- Keep this file minimal and stable.
"""

from __future__ import annotations

import csv
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers.copilot import router as copilot_router

# Routers
from backend.api.routers.fd001 import router as fd001_router
from backend.api.routers.fd002 import router as fd002_router
from backend.api.routers.fd003 import router as fd003_router
from backend.api.routers.fd004 import router as fd004_router
from backend.api.routers.single_engine import router as single_engine_router


# ----------------------------
# Request Logging (CSV)
# ----------------------------
def _append_request_log_row(row: dict) -> None:
    """
    Append one request log row to CSV.

    Env controls:
      REQUEST_LOG_ENABLED (default: "1")
      REQUEST_LOG_PATH    (default: "logs/request_log.csv")
    """
    enabled = os.getenv("REQUEST_LOG_ENABLED", "1").strip().lower() not in ("0", "false", "no")
    if not enabled:
        return

    log_path = Path(os.getenv("REQUEST_LOG_PATH", "logs/request_log.csv"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "ts_utc",
        "request_id",
        "method",
        "path",
        "status_code",
        "latency_ms",
        "client_ip",
        "user_agent",
        "content_length",
    ]

    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Predictive Maintenance Copilot API",
        version="0.1.0",
        description=(
            "Serves RUL prediction models (FD001–FD004) and (later) a GenAI RAG maintenance copilot."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — keep permissive for local dev; tighten in Phase 8 (cloud)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==================================
    # Middleware: log each request to CSV
    # ==================================
    @app.middleware("http")
    async def request_logger(request: Request, call_next):
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0

            client_ip = request.client.host if request.client else ""
            user_agent = request.headers.get("user-agent", "")
            content_length = request.headers.get("content-length", "")

            status_code = response.status_code if response is not None else ""

            row = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 3),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "content_length": content_length,
            }
            _append_request_log_row(row)

    # Routers
    app.include_router(fd001_router, prefix="/predict", tags=["FD001"])
    app.include_router(fd002_router, prefix="/predict", tags=["FD002"])
    app.include_router(fd003_router, prefix="/predict", tags=["FD003"])
    app.include_router(fd004_router, prefix="/predict", tags=["FD004"])

    app.include_router(single_engine_router, prefix="/single", tags=["Single Engine"])
    app.include_router(copilot_router, prefix="/copilot", tags=["Copilot"])

    @app.get("/health", tags=["Health"])
    def health_check():
        return {"status": "ok"}

    return app


app = create_app()
