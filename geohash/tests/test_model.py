"""Test neural network models."""

import pytest
import torch
import torch.nn as nn

from geohash.model import NextMagnitudeLSTM


class TestNextMagnitudeLSTM:
    """Test LSTM model."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=7,
            hidden_size=64,
            num_layers=1,
        )
        assert isinstance(model, nn.Module)

    def test_model_forward_pass(self):
        """Test forward pass produces output."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=7,
            hidden_size=64,
            num_layers=1,
        )

        batch_size, seq_len = 4, 10
        gh_ids = torch.randint(0, 100, (batch_size, seq_len))
        x_num = torch.randn(batch_size, seq_len, 7)
        lengths = torch.tensor([10, 9, 8, 7])

        output = model(gh_ids, x_num, lengths)

        assert output.shape == (batch_size, 1)

    def test_model_output_dtype(self):
        """Test output is float32."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=7,
            hidden_size=64,
        )

        gh_ids = torch.randint(0, 100, (2, 10))
        x_num = torch.randn(2, 10, 7)
        lengths = torch.tensor([10, 10])

        output = model(gh_ids, x_num, lengths)
        assert output.dtype == torch.float32

    def test_model_with_different_vocab_sizes(self):
        """Test model with different vocab sizes."""
        for vocab_size in [10, 100, 1000]:
            model = NextMagnitudeLSTM(
                vocab_size=vocab_size,
                embedding_dim=16,
                num_numeric=7,
                hidden_size=64,
            )
            gh_ids = torch.randint(0, vocab_size, (2, 10))
            x_num = torch.randn(2, 10, 7)
            lengths = torch.tensor([10, 10])

            output = model(gh_ids, x_num, lengths)
            assert output.shape == (2, 1)

    def test_model_with_padding(self):
        """Test model handles padded sequences."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=7,
            hidden_size=64,
        )

        # Batch with different lengths and padding
        gh_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]], dtype=torch.long)
        x_num = torch.randn(2, 5, 7)
        lengths = torch.tensor([3, 5])

        output = model(gh_ids, x_num, lengths)
        assert output.shape == (2, 1)

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=7,
            hidden_size=64,
        )

        gh_ids = torch.randint(0, 100, (2, 10))
        x_num = torch.randn(2, 10, 7, requires_grad=True)
        lengths = torch.tensor([10, 10])

        output = model(gh_ids, x_num, lengths)
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        assert model.embed.weight.grad is not None
        assert model.lstm.weight_ih_l0.grad is not None

    def test_model_multiple_layers(self):
        """Test model with multiple LSTM layers."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=7,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
        )

        gh_ids = torch.randint(0, 100, (2, 10))
        x_num = torch.randn(2, 10, 7)
        lengths = torch.tensor([10, 10])

        output = model(gh_ids, x_num, lengths)
        assert output.shape == (2, 1)

    def test_model_to_device(self):
        """Test model can be moved to device."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=7,
            hidden_size=64,
        )

        # Move to CPU explicitly
        model = model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"

    def test_model_eval_mode(self):
        """Test model eval mode (for dropout)."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=7,
            hidden_size=64,
            num_layers=2,
            dropout=0.5,
        )

        gh_ids = torch.randint(0, 100, (2, 10))
        x_num = torch.randn(2, 10, 7)
        lengths = torch.tensor([10, 10])

        model.eval()
        with torch.no_grad():
            output1 = model(gh_ids, x_num, lengths)
            output2 = model(gh_ids, x_num, lengths)

        # In eval mode with same input, output should be identical
        assert torch.allclose(output1, output2)
