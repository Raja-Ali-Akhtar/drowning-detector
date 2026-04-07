"""Tests for the DrowningLSTM model."""

import torch

from drowning_detector.models.classifier.model import DrowningLSTM


class TestDrowningLSTM:
    """Test suite for the LSTM classifier."""

    def test_model_output_shape(self, sample_batch_tensor: torch.Tensor) -> None:
        """Model output should be (batch_size, 2)."""
        model = DrowningLSTM()
        output = model(sample_batch_tensor)
        assert output.shape == (4, 2)

    def test_model_forward_no_error(self) -> None:
        """Forward pass should complete without errors."""
        model = DrowningLSTM()
        x = torch.randn(1, 50, 47)
        output = model(x)
        assert output is not None
        assert not torch.isnan(output).any()

    def test_model_default_input_size(self) -> None:
        """Default input size should be 47 (14*3 + 5)."""
        assert DrowningLSTM.INPUT_SIZE == 47

    def test_model_single_sequence(self) -> None:
        """Model should handle single sequence input."""
        model = DrowningLSTM()
        x = torch.randn(1, 50, 47)
        output = model(x)
        assert output.shape == (1, 2)

    def test_model_variable_sequence_length(self) -> None:
        """Model should handle different sequence lengths."""
        model = DrowningLSTM()
        for seq_len in [10, 30, 50, 100]:
            x = torch.randn(2, seq_len, 47)
            output = model(x)
            assert output.shape == (2, 2)
