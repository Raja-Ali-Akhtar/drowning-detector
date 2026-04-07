"""Structured logging configuration using loguru.

Usage:
    from drowning_detector.core.logging import logger

    logger.info("Processing camera feed", camera_id="cam_01", fps=30)
    logger.warning("Low confidence detection", score=0.42)
    logger.error("Stream disconnected", exc_info=True)
"""

import sys
from pathlib import Path

from loguru import logger

from drowning_detector.core.config import PROJECT_ROOT, settings

# Remove default handler
logger.remove()

# ── Log format ───────────────────────────────────────────────────────────
_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

_JSON_LOG_FORMAT = "{message}"

# ── Console handler (always active) ─────────────────────────────────────
logger.add(
    sys.stderr,
    format=_LOG_FORMAT,
    level="DEBUG" if settings.debug else "INFO",
    colorize=True,
    backtrace=True,
    diagnose=settings.debug,
)

# ── File handler (rotate daily, keep 30 days) ───────────────────────────
_log_dir = PROJECT_ROOT / "logs"
_log_dir.mkdir(exist_ok=True)

logger.add(
    str(_log_dir / "app_{time:YYYY-MM-DD}.log"),
    format=_LOG_FORMAT,
    level="DEBUG",
    rotation="00:00",
    retention="30 days",
    compression="gz",
    enqueue=True,  # thread-safe
)

# ── JSON file handler for structured log aggregation ────────────────────
if settings.environment in ("staging", "production"):
    logger.add(
        str(_log_dir / "app_{time:YYYY-MM-DD}.json"),
        serialize=True,
        level="INFO",
        rotation="00:00",
        retention="30 days",
        compression="gz",
        enqueue=True,
    )

__all__ = ["logger"]
