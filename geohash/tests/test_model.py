"""Test neural network models."""

import pytest
import torch
import torch.nn as nn

from geohash.model import NextMagnitudeLSTM

_NUM_NUMERIC = 6  # 6 features after removing delta_lat / delta_lon


class TestNextMagnitudeLSTM:
    """Test LSTM model."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=_NUM_NUMERIC,
            hidden_size=64,
            num_layers=1,
        )
        assert isinstance(model, nn.Module)

    def test_model_has_input_norm(self):
        """Model must expose an input_norm LayerNorm layer."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=_NUM_NUMERIC,
            hidden_size=64,
        )
        assert hasattr(model, "input_norm"), "Model is missing input_norm attribute"
        assert isinstance(model.input_norm, nn.LayerNorm)
        assert model.input_norm.normalized_shape == (16 + _NUM_NUMERIC,)

    def test_input_norm_equalises_scale(self):
        """After input_norm the per-timestep feature vector should be near unit scale."""
        embed_dim, num_numeric = 16, _NUM_NUMERIC
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=embed_dim,
            num_numeric=num_numeric,
            hidden_size=64,
        )
        model.eval()

        # Simulate the scale mismatch: tiny embeddings, large numerics
        gh_ids = torch.zeros(2, 5, dtype=torch.long)   # all PAD → near-zero embeddings
        x_num = torch.randn(2, 5, num_numeric) * 10    # inflated numeric scale

        with torch.no_grad():
            gh_emb = model.geohash_encoder(gh_ids)
            x_cat = torch.cat([gh_emb, x_num], dim=-1)
            x_normed = model.input_norm(x_cat)

        # LayerNorm guarantees per-instance mean≈0, std≈1 across the feature dim
        mean = x_normed.mean(dim=-1)
        std = x_normed.std(dim=-1)
        assert mean.abs().max().item() < 1e-5
        assert (std - 1.0).abs().max().item() < 0.1

    def test_model_forward_pass(self):
        """Test forward pass produces output."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=_NUM_NUMERIC,
            hidden_size=64,
            num_layers=1,
        )

        batch_size, seq_len = 4, 10
        gh_ids = torch.randint(0, 100, (batch_size, seq_len))
        x_num = torch.randn(batch_size, seq_len, _NUM_NUMERIC)
        lengths = torch.tensor([10, 9, 8, 7])

        output = model(gh_ids, x_num, lengths)

        assert output.shape == (batch_size, 1)

    def test_model_output_dtype(self):
        """Test output is float32."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=_NUM_NUMERIC,
            hidden_size=64,
        )

        gh_ids = torch.randint(0, 100, (2, 10))
        x_num = torch.randn(2, 10, _NUM_NUMERIC)
        lengths = torch.tensor([10, 10])

        output = model(gh_ids, x_num, lengths)
        assert output.dtype == torch.float32

    def test_model_with_different_vocab_sizes(self):
        """Test model with different vocab sizes."""
        for vocab_size in [10, 100, 1000]:
            model = NextMagnitudeLSTM(
                vocab_size=vocab_size,
                embedding_dim=16,
                num_numeric=_NUM_NUMERIC,
                hidden_size=64,
            )
            gh_ids = torch.randint(0, vocab_size, (2, 10))
            x_num = torch.randn(2, 10, _NUM_NUMERIC)
            lengths = torch.tensor([10, 10])

            output = model(gh_ids, x_num, lengths)
            assert output.shape == (2, 1)

    def test_model_with_padding(self):
        """Test model handles padded sequences."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=_NUM_NUMERIC,
            hidden_size=64,
        )

        gh_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]], dtype=torch.long)
        x_num = torch.randn(2, 5, _NUM_NUMERIC)
        lengths = torch.tensor([3, 5])

        output = model(gh_ids, x_num, lengths)
        assert output.shape == (2, 1)

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=_NUM_NUMERIC,
            hidden_size=64,
        )

        gh_ids = torch.randint(0, 100, (2, 10))
        x_num = torch.randn(2, 10, _NUM_NUMERIC, requires_grad=True)
        lengths = torch.tensor([10, 10])

        output = model(gh_ids, x_num, lengths)
        loss = output.mean()
        loss.backward()

        assert model.geohash_encoder.embed.weight.grad is not None
        assert model.lstm.weight_ih_l0.grad is not None
        assert model.input_norm.weight.grad is not None

    def test_model_multiple_layers(self):
        """Test model with multiple LSTM layers."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=_NUM_NUMERIC,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
        )

        gh_ids = torch.randint(0, 100, (2, 10))
        x_num = torch.randn(2, 10, _NUM_NUMERIC)
        lengths = torch.tensor([10, 10])

        output = model(gh_ids, x_num, lengths)
        assert output.shape == (2, 1)

    def test_model_to_device(self):
        """Test model can be moved to device."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=_NUM_NUMERIC,
            hidden_size=64,
        )

        model = model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"

    def test_model_eval_mode(self):
        """Test model eval mode (for dropout)."""
        model = NextMagnitudeLSTM(
            vocab_size=100,
            embedding_dim=16,
            num_numeric=_NUM_NUMERIC,
            hidden_size=64,
            num_layers=2,
            dropout=0.5,
        )

        gh_ids = torch.randint(0, 100, (2, 10))
        x_num = torch.randn(2, 10, _NUM_NUMERIC)
        lengths = torch.tensor([10, 10])

        model.eval()
        with torch.no_grad():
            output1 = model(gh_ids, x_num, lengths)
            output2 = model(gh_ids, x_num, lengths)

        assert torch.allclose(output1, output2)

    def test_hierarchical_encoding_forward(self):
        model = NextMagnitudeLSTM(
            vocab_size=34,
            embedding_dim=8,
            num_numeric=_NUM_NUMERIC,
            hidden_size=16,
            encoding="hierarchical",
            geohash_precision=4,
        )
        gh_ids = torch.randint(0, 34, (2, 5, 4))
        x_num = torch.randn(2, 5, _NUM_NUMERIC)
        lengths = torch.tensor([5, 4])
        out = model(gh_ids, x_num, lengths)
        assert out.shape == (2, 1)

    def test_input_mode_numeric_only(self):
        model = NextMagnitudeLSTM(
            vocab_size=50,
            embedding_dim=8,
            num_numeric=_NUM_NUMERIC,
            hidden_size=16,
            input_mode="numeric_only",
        )
        gh_ids = torch.randint(0, 50, (2, 3))
        x_num = torch.randn(2, 3, _NUM_NUMERIC)
        out = model(gh_ids, x_num, torch.tensor([3, 2]))
        assert out.shape == (2, 1)
