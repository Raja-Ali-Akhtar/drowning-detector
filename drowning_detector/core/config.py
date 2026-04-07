"""Application configuration using pydantic-settings.

Loads from environment variables / .env file with validation and type safety.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root is two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────
    app_name: str = "Drowning Detection System"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = Field(default="development", pattern="^(development|staging|production)$")

    # ── Database ─────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/drowning_db"

    # ── Redis ────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── AWS / S3 ─────────────────────────────────────────
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    aws_s3_bucket: str = "drowning-detector-incidents"

    # ── Twilio (SMS alerts) ──────────────────────────────
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None
    alert_phone_numbers: list[str] = Field(default_factory=list)

    # ── Model paths ──────────────────────────────────────
    yolo_weights_path: Path = Path("models/detector/yolov8_pool.pt")
    model_weights_path: Path = Path("models/classifier/best.pt")

    # ── Detection thresholds ─────────────────────────────
    alert_confidence_threshold: float = 0.75
    consecutive_alert_windows: int = 2
    stillness_velocity_threshold: float = 0.005
    stillness_duration_seconds: float = 8.0

    # ── Inference ────────────────────────────────────────
    pose_sequence_length: int = 50  # 5 seconds at 10 FPS
    pose_fps: int = 10
    num_joints: int = 14
    yolo_input_size: int = 640
    yolo_confidence: float = 0.5

    # ── Monitoring ───────────────────────────────────────
    sentry_dsn: Optional[str] = None
    enable_prometheus: bool = True

    @field_validator("yolo_weights_path", "model_weights_path", mode="before")
    @classmethod
    def resolve_model_path(cls, v: str | Path) -> Path:
        """Resolve model paths relative to project root if not absolute."""
        p = Path(v)
        if not p.is_absolute():
            return PROJECT_ROOT / p
        return p


# Singleton — import this everywhere
settings = Settings()
