# backend/api/main.py
"""
FastAPI entrypoint for AI Predictive Maintenance Copilot.

Goals:
- Clean API surface for FD001–FD004 RUL inference
- Health check endpoint
- Central router registration
- Production-friendly metadata
- Request logging middleware (CSV) with latency + request-id
- Always return JSON on unhandled errors

Notes:
- Model/scaler loading happens in service layers.
"""

from __future__ import annotations

import csv
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ----------------------------
# Load .env (local dev)
# ----------------------------
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# Routers
from backend.api.routers.copilot import router as copilot_router
from backend.api.routers.fd001 import router as fd001_router
from backend.api.routers.fd002 import router as fd002_router
from backend.api.routers.fd003 import router as fd003_router
from backend.api.routers.fd004 import router as fd004_router
from backend.api.routers.single_engine import router as single_engine_router


# ----------------------------
# Request Logging (CSV)
# ----------------------------
def _append_request_log_row(row: Dict[str, Any]) -> None:
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


def _parse_cors_origins() -> list[str]:
    """
    CORS_ALLOW_ORIGINS supports:
      - "*" (default)
      - "http://localhost:8501,http://127.0.0.1:8501"
    """
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or ["*"]


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Predictive Maintenance Copilot API",
        version="0.1.0",
        description="Serves RUL prediction models (FD001–FD004) and a GenAI RAG maintenance copilot.",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — permissive for local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==================================
    # Middleware: request-id + logging + JSON errors
    # ==================================
    @app.middleware("http")
    async def request_logger(request: Request, call_next):
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        response = None
        try:
            response = await call_next(request)
            return response

        except Exception as e:
            # Always return JSON for unhandled errors (better UX).
            # If you want debug in dev, set DEBUG_ERRORS=1
            debug = os.getenv("DEBUG_ERRORS", "0").strip().lower() in ("1", "true", "yes")

            payload: Dict[str, Any] = {
                "detail": "Internal Server Error",
                "request_id": request_id,
            }
            if debug:
                payload["error"] = str(e)

            response = JSONResponse(status_code=500, content=payload)
            return response

        finally:
            # Always attach request-id
            try:
                if response is not None:
                    response.headers["X-Request-ID"] = request_id
            except Exception:
                pass

            latency_ms = (time.perf_counter() - start) * 1000.0
            client_ip = request.client.host if request.client else ""
            user_agent = request.headers.get("user-agent", "")
            content_length = request.headers.get("content-length", "")
            status_code: Optional[int] = getattr(response, "status_code", None)

            row = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code if status_code is not None else "",
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
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
