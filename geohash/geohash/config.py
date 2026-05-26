"""Configuration models for geohash LSTM pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class USGSQueryParams(BaseModel):
    """USGS earthquake query parameters."""

    min_latitude: float = Field(-90.0, ge=-90.0, le=90.0, description="Minimum latitude")
    max_latitude: float = Field(90.0, ge=-90.0, le=90.0, description="Maximum latitude")
    min_longitude: float = Field(-180.0, ge=-180.0, le=180.0, description="Minimum longitude")
    max_longitude: float = Field(180.0, ge=-180.0, le=180.0, description="Maximum longitude")
    start_time: str = Field("2018-01-01", description="Start date (YYYY-MM-DD)")
    end_time: str = Field("2024-12-31", description="End date (YYYY-MM-DD)")
    min_magnitude: float = Field(1.0, ge=0.0, description="Minimum magnitude")
    order_by: Literal["time-asc", "time", "magnitude"] = Field("time-asc", description="Sort order")
    limit: int = Field(20000, gt=0, description="Maximum number of events")

    @field_validator("max_latitude")
    @classmethod
    def validate_latitude_order(cls, v: float, info):
        """Ensure max_latitude >= min_latitude."""
        if "min_latitude" in info.data and v < info.data["min_latitude"]:
            raise ValueError("max_latitude must be >= min_latitude")
        return v

    @field_validator("max_longitude")
    @classmethod
    def validate_longitude_order(cls, v: float, info):
        """Ensure max_longitude >= min_longitude."""
        if "min_longitude" in info.data and v < info.data["min_longitude"]:
            raise ValueError("max_longitude must be >= min_longitude")
        return v


class WindowConfig(BaseModel):
    """Sliding window configuration."""

    min_len: int = Field(5, gt=0, description="Minimum window length")
    max_len: int = Field(30, gt=0, description="Maximum window length")
    stride: int = Field(1, gt=0, description="Stride between windows")

    @field_validator("max_len")
    @classmethod
    def validate_max_len(cls, v: int, info):
        """Ensure max_len >= min_len."""
        if "min_len" in info.data and v < info.data["min_len"]:
            raise ValueError("max_len must be >= min_len")
        return v


class GeohashConfig(BaseModel):
    """Geohash encoding configuration."""

    precision: int = Field(4, gt=0, le=12, description="Geohash precision (1-12)")


class ModelConfig(BaseModel):
    """LSTM model architecture configuration."""

    embedding_dim: int = Field(16, gt=0, description="Embedding dimension")
    hidden_size: int = Field(64, gt=0, description="LSTM hidden size")
    num_layers: int = Field(1, gt=0, description="Number of LSTM layers")
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout rate")


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    batch_size: int = Field(64, gt=0, description="Batch size")
    epochs: int = Field(12, gt=0, description="Number of epochs")
    learning_rate: float = Field(1e-3, gt=0, description="Learning rate")
    train_split: float = Field(0.8, ge=0.0, le=1.0, description="Train/test split ratio")
    seed: int = Field(42, description="Random seed for reproducibility")
    device: Literal["cpu", "cuda"] = Field("cpu", description="Device: cpu or cuda")


class ExperimentConfig(BaseModel):
    """Experiment metadata and output configuration."""

    experiment_name: str = Field(
        default_factory=lambda: f"geohash_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description="Experiment identifier"
    )
    output_dir: Path = Field(
        default_factory=lambda: Path.home() / ".geohash-runs",
        description="Directory to save run results"
    )

    class Config:
        arbitrary_types_allowed = True


class RunConfig(BaseModel):
    """Complete configuration for a training run."""

    usgs: USGSQueryParams = Field(default_factory=USGSQueryParams)
    window: WindowConfig = Field(default_factory=WindowConfig)
    geohash: GeohashConfig = Field(default_factory=GeohashConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    class Config:
        arbitrary_types_allowed = True

    def get_run_dir(self) -> Path:
        """Get the directory for this run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.experiment.output_dir / f"{self.experiment.experiment_name}-{timestamp}"
        return run_dir
