"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from drowning_detector.core.config import settings
from drowning_detector.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown logic."""
    logger.info("Starting Drowning Detection API", version=settings.app_version)
    # TODO: Load ML models into app.state
    # TODO: Initialize Redis connection pool
    # TODO: Initialize database connection pool
    yield
    logger.info("Shutting down Drowning Detection API")
    # TODO: Cleanup resources


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Real-time drowning detection system API",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# ── Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ─────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "ok",
        "data": {
            "version": settings.app_version,
            "environment": settings.environment,
        },
        "error": None,
    }


@app.get("/ready", tags=["system"])
async def readiness_check() -> dict:
    """Readiness probe — checks that models and dependencies are loaded."""
    # TODO: Check model loaded, DB reachable, Redis reachable
    return {
        "status": "ok",
        "data": {"models_loaded": False, "db_connected": False, "redis_connected": False},
        "error": None,
    }


def start() -> None:
    """CLI entry point for `drowning-api` command."""
    import uvicorn

    uvicorn.run(
        "drowning_detector.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
