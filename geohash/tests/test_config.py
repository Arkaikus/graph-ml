"""Test configuration validation."""

import pytest
from geohash.config import (
    RunConfig,
    USGSQueryParams,
    WindowConfig,
    GeohashConfig,
    ModelConfig,
    TrainingConfig,
)


class TestUSGSQueryParams:
    """Test USGS query parameter validation."""

    def test_valid_params(self):
        """Test valid parameter creation."""
        params = USGSQueryParams(
            min_latitude=32.0,
            max_latitude=42.0,
            min_longitude=-125.0,
            max_longitude=-114.0,
        )
        assert params.min_latitude == 32.0
        assert params.max_latitude == 42.0

    def test_latitude_order_validation(self):
        """Test that max_latitude >= min_latitude."""
        with pytest.raises(ValueError, match="max_latitude must be >= min_latitude"):
            USGSQueryParams(
                min_latitude=42.0,
                max_latitude=32.0,
                min_longitude=-125.0,
                max_longitude=-114.0,
            )

    def test_longitude_order_validation(self):
        """Test that max_longitude >= min_longitude."""
        with pytest.raises(ValueError, match="max_longitude must be >= min_longitude"):
            USGSQueryParams(
                min_latitude=32.0,
                max_latitude=42.0,
                min_longitude=-114.0,
                max_longitude=-125.0,
            )

    def test_latitude_bounds(self):
        """Test latitude is in [-90, 90]."""
        with pytest.raises(ValueError):
            USGSQueryParams(
                min_latitude=-100.0,
                max_latitude=42.0,
                min_longitude=-125.0,
                max_longitude=-114.0,
            )

    def test_longitude_bounds(self):
        """Test longitude is in [-180, 180]."""
        with pytest.raises(ValueError):
            USGSQueryParams(
                min_latitude=32.0,
                max_latitude=42.0,
                min_longitude=-200.0,
                max_longitude=-114.0,
            )


class TestWindowConfig:
    """Test window configuration validation."""

    def test_valid_window(self):
        """Test valid window config."""
        window = WindowConfig(min_len=5, max_len=30, stride=1)
        assert window.min_len == 5
        assert window.max_len == 30

    def test_max_len_validation(self):
        """Test that max_len >= min_len."""
        with pytest.raises(ValueError, match="max_len must be >= min_len"):
            WindowConfig(min_len=30, max_len=5, stride=1)


class TestModelConfig:
    """Test model configuration validation."""

    def test_valid_model(self):
        """Test valid model config."""
        model = ModelConfig(
            embedding_dim=16,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
        )
        assert model.embedding_dim == 16
        assert model.hidden_size == 64

    def test_positive_dimensions(self):
        """Test dimensions must be positive."""
        with pytest.raises(ValueError):
            ModelConfig(embedding_dim=0, hidden_size=64)


class TestTrainingConfig:
    """Test training configuration validation."""

    def test_valid_training(self):
        """Test valid training config."""
        training = TrainingConfig(
            batch_size=64,
            epochs=12,
            learning_rate=1e-3,
            seed=42,
        )
        assert training.batch_size == 64
        assert training.epochs == 12

    def test_positive_batch_size(self):
        """Test batch size must be positive."""
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)


class TestRunConfig:
    """Test complete run configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = RunConfig()
        assert config.training.batch_size == 64
        assert config.training.epochs == 12
        assert config.model.embedding_dim == 16

    def test_get_run_dir(self):
        """Test run directory generation."""
        from pathlib import Path
        config = RunConfig(experiment={"experiment_name": "test_run"})
        run_dir = config.get_run_dir()
        assert run_dir.parent == Path.home() / ".geohash-runs"
        assert "test_run" in run_dir.name
