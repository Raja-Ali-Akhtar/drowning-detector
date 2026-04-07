"""Core module — config, logging, and constants."""

from drowning_detector.core.config import settings
from drowning_detector.core.constants import (
    ALERT_COOLDOWN_SECONDS,
    ALERT_THRESHOLD,
    CONSECUTIVE_WINDOWS,
    Joint,
    LABEL_DROWNING,
    LABEL_NAMES,
    LABEL_NORMAL,
    LABEL_TREADING,
    NUM_JOINTS,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    STILLNESS_FRAMES,
    STILLNESS_VELOCITY,
)

__all__ = [
    "settings",
    "Joint",
    "LABEL_NORMAL",
    "LABEL_DROWNING",
    "LABEL_TREADING",
    "LABEL_NAMES",
    "NUM_JOINTS",
    "SEQUENCE_LENGTH",
    "ALERT_THRESHOLD",
    "CONSECUTIVE_WINDOWS",
    "STILLNESS_VELOCITY",
    "STILLNESS_FRAMES",
    "ALERT_COOLDOWN_SECONDS",
    "RANDOM_SEED",
]
