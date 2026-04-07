"""Shared test fixtures."""

import numpy as np
import pytest
import torch

from drowning_detector.core.constants import JOINT_DIMS, NUM_JOINTS, SEQUENCE_LENGTH


@pytest.fixture
def sample_pose_sequence() -> np.ndarray:
    """Generate a random pose sequence for testing.

    Returns:
        np.ndarray of shape (50, 14, 3) — standard pose sequence.
    """
    np.random.seed(42)
    return np.random.rand(SEQUENCE_LENGTH, NUM_JOINTS, JOINT_DIMS).astype(np.float32)


@pytest.fixture
def sample_batch_tensor() -> torch.Tensor:
    """Generate a batch of LSTM input tensors.

    Returns:
        torch.Tensor of shape (4, 50, 47) — batch of 4 sequences.
    """
    torch.manual_seed(42)
    input_size = NUM_JOINTS * JOINT_DIMS + 5  # 47
    return torch.randn(4, SEQUENCE_LENGTH, input_size)


@pytest.fixture
def short_pose_sequence() -> np.ndarray:
    """Generate a short pose sequence (needs padding).

    Returns:
        np.ndarray of shape (20, 14, 3).
    """
    np.random.seed(42)
    return np.random.rand(20, NUM_JOINTS, JOINT_DIMS).astype(np.float32)
