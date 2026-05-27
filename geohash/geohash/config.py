"""Configuration models for geohash LSTM pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class USGSQueryParams(BaseModel):
    """USGS earthquake query parameters."""

    min_latitude: float = Field(-90.0, ge=-90.0, le=90.0, description="Minimum latitude")
    max_latitude: float = Field(90.0, ge=-90.0, le=90.0, description="Maximum latitude")
    min_longitude: float = Field(-180.0, ge=-180.0, le=180.0, description="Minimum longitude")
    max_longitude: float = Field(180.0, ge=-180.0, le=180.0, description="Maximum longitude")
    start_time: str = Field("1990-01-01", description="Start date (YYYY-MM-DD)")
    end_time: str = Field("2025-12-31", description="End date (YYYY-MM-DD)")
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

    mode: Literal["temporal", "spatial", "hybrid"] = Field(
        "temporal",
        description="Windowing strategy: temporal (time-ordered), spatial (geographically clustered), hybrid (both)",
    )

    min_len: int = Field(5, gt=0, description="Minimum window length")
    max_len: int = Field(30, gt=0, description="Maximum window length")
    stride: int = Field(1, gt=0, description="Stride between windows (temporal / hybrid mode)")

    spatial_radius_km: float = Field(
        50.0, gt=0, description="Spatial search radius in km (spatial / hybrid mode)"
    )
    temporal_window_days: float = Field(
        30.0, gt=0, description="Temporal lookback in days (spatial / hybrid mode)"
    )

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
    encoding: Literal["flat", "hierarchical"] = Field(
        "flat",
        description="flat: cell-level embedding; hierarchical: char-level sum pool",
    )


class ModelConfig(BaseModel):
    """LSTM model architecture configuration."""

    embedding_dim: int = Field(16, gt=0, description="Embedding dimension")
    hidden_size: int = Field(64, gt=0, description="LSTM hidden size")
    num_layers: int = Field(1, gt=0, description="Number of LSTM layers")
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout rate")
    input_mode: Literal["full", "numeric_only", "geohash_only"] = Field(
        "full",
        description="Ablation: full model, numeric-only, or geohash-only",
    )


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    batch_size: int = Field(64, gt=0, description="Batch size")
    epochs: int = Field(12, gt=0, description="Number of epochs")
    learning_rate: float = Field(1e-3, gt=0, description="Learning rate")
    seed: int = Field(42, description="Random seed for reproducibility")
    device: Literal["cpu", "cuda"] = Field("cpu", description="Device: cpu or cuda")

    split_strategy: Literal["temporal_event", "window_index"] = Field(
        "temporal_event",
        description="temporal_event (default) or window_index (legacy debug)",
    )
    train_ratio: float = Field(0.7, ge=0.0, le=1.0, description="Train event ratio")
    val_ratio: float = Field(0.1, ge=0.0, le=1.0, description="Validation event ratio")
    test_ratio: float = Field(0.2, ge=0.0, le=1.0, description="Test event ratio")
    train_split: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Legacy window-index split ratio (window_index strategy only)",
    )

    early_stopping_patience: int = Field(
        5, ge=0, description="Stop if val loss does not improve for N epochs (0 = disabled)"
    )
    lr_scheduler: Literal["none", "cosine", "plateau"] = Field(
        "plateau",
        description="Learning rate schedule: none, cosine annealing, or reduce-on-plateau",
    )
    lr_patience: int = Field(
        3, gt=0, description="Patience for ReduceLROnPlateau scheduler"
    )
    gradient_clip: float = Field(
        1.0, gt=0, description="Max norm for gradient clipping"
    )

    @model_validator(mode="after")
    def validate_split_ratios(self):
        """Ensure train/val/test ratios sum to 1.0 for temporal_event split."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if self.split_strategy == "temporal_event" and abs(total - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        return self


class ExperimentConfig(BaseModel):
    """Experiment metadata and output configuration."""

    experiment_name: str = Field(
        default_factory=lambda: f"geohash_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description="Experiment identifier",
    )
    output_dir: Path = Field(
        default_factory=lambda: Path.cwd() / ".geohash-runs",
        description="Directory to save run results",
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
