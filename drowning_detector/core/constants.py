"""Project-wide constants.

Central place for magic numbers, joint indices, and label mappings so they are
never hard-coded elsewhere.
"""

from enum import IntEnum

# ── Label mapping ─────────────────────────────────────────────────────────
LABEL_NORMAL: int = 0
LABEL_DROWNING: int = 1
LABEL_TREADING: int = 2

LABEL_NAMES: dict[int, str] = {
    LABEL_NORMAL: "normal",
    LABEL_DROWNING: "drowning",
    LABEL_TREADING: "treading",
}


# ── Joint indices (14-joint skeleton) ─────────────────────────────────────
class Joint(IntEnum):
    NOSE = 0
    LEFT_SHOULDER = 1
    RIGHT_SHOULDER = 2
    LEFT_ELBOW = 3
    RIGHT_ELBOW = 4
    LEFT_WRIST = 5
    RIGHT_WRIST = 6
    LEFT_HIP = 7
    RIGHT_HIP = 8
    LEFT_KNEE = 9
    RIGHT_KNEE = 10
    LEFT_ANKLE = 11
    RIGHT_ANKLE = 12
    HEAD_CENTRE = 13


NUM_JOINTS: int = 14
JOINT_DIMS: int = 3  # (x, y, visibility)

# ── Pose sequence ─────────────────────────────────────────────────────────
SEQUENCE_LENGTH: int = 50  # 5 seconds × 10 FPS
POSE_FPS: int = 10

# ── Alert engine ──────────────────────────────────────────────────────────
ALERT_THRESHOLD: float = 0.75
CONSECUTIVE_WINDOWS: int = 2
STILLNESS_VELOCITY: float = 0.005
STILLNESS_FRAMES: int = 80  # 8 seconds at 10 FPS
ALERT_COOLDOWN_SECONDS: int = 60

# ── Random seeds ──────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
