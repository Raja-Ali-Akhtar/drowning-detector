"""LSTM-based drowning behaviour classifier.

Input:  (batch, T, 47) — 14 joints × 3 coords + 5 computed features
Output: (batch, 2)     — [normal, drowning] logits
"""

import torch
import torch.nn as nn

from drowning_detector.core.constants import JOINT_DIMS, NUM_JOINTS


class DrowningLSTM(nn.Module):
    """Bidirectional LSTM classifier for drowning detection from pose sequences."""

    COMPUTED_FEATURES = 5  # verticality, head_position, wrist_spread, mean_vis, mean_vel
    INPUT_SIZE = NUM_JOINTS * JOINT_DIMS + COMPUTED_FEATURES  # 14*3 + 5 = 47

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, T, input_size).

        Returns:
            Logits of shape (batch, num_classes).
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]  # take last timestep
        return self.classifier(last_hidden)
