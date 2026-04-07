"""Tests for project constants and configuration."""

from drowning_detector.core.constants import (
    ALERT_THRESHOLD,
    CONSECUTIVE_WINDOWS,
    Joint,
    LABEL_DROWNING,
    LABEL_NAMES,
    LABEL_NORMAL,
    LABEL_TREADING,
    NUM_JOINTS,
    SEQUENCE_LENGTH,
    STILLNESS_FRAMES,
    STILLNESS_VELOCITY,
)


class TestConstants:
    """Verify critical constants match project spec."""

    def test_label_values(self) -> None:
        assert LABEL_NORMAL == 0
        assert LABEL_DROWNING == 1
        assert LABEL_TREADING == 2

    def test_label_names_complete(self) -> None:
        assert len(LABEL_NAMES) == 3
        assert LABEL_NAMES[LABEL_DROWNING] == "drowning"

    def test_joint_count(self) -> None:
        assert NUM_JOINTS == 14
        assert len(Joint) == 14

    def test_joint_indices(self) -> None:
        assert Joint.NOSE == 0
        assert Joint.HEAD_CENTRE == 13

    def test_sequence_length(self) -> None:
        assert SEQUENCE_LENGTH == 50  # 5 sec × 10 FPS

    def test_alert_threshold(self) -> None:
        assert ALERT_THRESHOLD == 0.75
        assert CONSECUTIVE_WINDOWS == 2

    def test_stillness_thresholds(self) -> None:
        assert STILLNESS_VELOCITY == 0.005
        assert STILLNESS_FRAMES == 80  # 8 sec × 10 FPS
