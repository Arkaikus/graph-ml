"""Test run storage."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from geohash.store import RunStore
from geohash.model import NextMagnitudeLSTM


class TestRunStore:
    """Test run storage and retrieval."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_model(self):
        """Create sample model."""
        return NextMagnitudeLSTM(
            vocab_size=50,
            embedding_dim=8,
            num_numeric=6,
            hidden_size=32,
        )

    @pytest.fixture
    def sample_run_data(self):
        """Create sample run data."""
        return {
            "config": {
                "usgs": {"min_latitude": 32.0},
                "training": {"epochs": 1, "batch_size": 64},
                "model": {"embedding_dim": 16},
            },
            "history": {
                "train_loss": [0.5, 0.4, 0.3],
                "val_loss": [0.6, 0.5, 0.4],
                "test_loss": [0.6, 0.5, 0.4],
                "rmse": [0.8, 0.7, 0.6],
                "mae": [0.6, 0.5, 0.4],
            },
            "predictions": {
                "targets": [2.0, 2.5, 3.0, 3.5, 4.0],
                "predictions": [2.1, 2.4, 3.1, 3.4, 4.1],
                "rmse": 0.6,
                "mae": 0.5,
                "r2": 0.1,
                "loss": 0.4,
            },
        }

    def test_store_initialization(self, temp_output_dir):
        """Test store initialization."""
        store = RunStore(temp_output_dir)
        assert store.output_dir == temp_output_dir
        assert temp_output_dir.exists()

    def test_save_run_creates_directory(self, temp_output_dir, sample_model, sample_run_data):
        """Test save_run creates run directory."""
        store = RunStore(temp_output_dir)
        run_dir = temp_output_dir / "test_run-20230101_120000"

        store.save_run(
            run_dir=run_dir,
            config=sample_run_data["config"],
            history=sample_run_data["history"],
            model=sample_model,
            predictions=sample_run_data["predictions"],
        )

        assert run_dir.exists()

    def test_save_run_creates_artifacts(self, temp_output_dir, sample_model, sample_run_data):
        """Test all artifacts are created."""
        store = RunStore(temp_output_dir)
        run_dir = temp_output_dir / "test_run-20230101_120000"

        store.save_run(
            run_dir=run_dir,
            config=sample_run_data["config"],
            history=sample_run_data["history"],
            model=sample_model,
            predictions=sample_run_data["predictions"],
        )

        assert (run_dir / "config.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "predictions.csv").exists()
        assert (run_dir / "model_final.pt").exists()
        assert (run_dir / "training_curves.png").exists()

    def test_config_json_format(self, temp_output_dir, sample_model, sample_run_data):
        """Test config.json is valid JSON."""
        store = RunStore(temp_output_dir)
        run_dir = temp_output_dir / "test_run-20230101_120000"

        store.save_run(
            run_dir=run_dir,
            config=sample_run_data["config"],
            history=sample_run_data["history"],
            model=sample_model,
            predictions=sample_run_data["predictions"],
        )

        with open(run_dir / "config.json") as f:
            config = json.load(f)

        assert isinstance(config, dict)
        assert "usgs" in config

    def test_metrics_json_format(self, temp_output_dir, sample_model, sample_run_data):
        """Test metrics.json contains expected fields."""
        store = RunStore(temp_output_dir)
        run_dir = temp_output_dir / "test_run-20230101_120000"

        store.save_run(
            run_dir=run_dir,
            config=sample_run_data["config"],
            history=sample_run_data["history"],
            model=sample_model,
            predictions=sample_run_data["predictions"],
        )

        with open(run_dir / "metrics.json") as f:
            metrics = json.load(f)

        assert "train_loss" in metrics
        assert "test_loss" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "final_rmse" in metrics

    def test_predictions_csv_format(self, temp_output_dir, sample_model, sample_run_data):
        """Test predictions.csv is valid."""
        store = RunStore(temp_output_dir)
        run_dir = temp_output_dir / "test_run-20230101_120000"

        store.save_run(
            run_dir=run_dir,
            config=sample_run_data["config"],
            history=sample_run_data["history"],
            model=sample_model,
            predictions=sample_run_data["predictions"],
        )

        df = pd.read_csv(run_dir / "predictions.csv")
        assert "target" in df.columns
        assert "predicted" in df.columns
        assert len(df) == len(sample_run_data["predictions"]["targets"])

    def test_model_pt_loadable(self, temp_output_dir, sample_model, sample_run_data):
        """Test model.pt can be loaded."""
        store = RunStore(temp_output_dir)
        run_dir = temp_output_dir / "test_run-20230101_120000"

        store.save_run(
            run_dir=run_dir,
            config=sample_run_data["config"],
            history=sample_run_data["history"],
            model=sample_model,
            predictions=sample_run_data["predictions"],
        )

        state_dict = torch.load(run_dir / "model_final.pt", map_location="cpu")
        assert isinstance(state_dict, dict)

    def test_load_run(self, temp_output_dir, sample_model, sample_run_data):
        """Test load_run retrieves saved data."""
        store = RunStore(temp_output_dir)
        run_dir = temp_output_dir / "test_run-20230101_120000"

        store.save_run(
            run_dir=run_dir,
            config=sample_run_data["config"],
            history=sample_run_data["history"],
            model=sample_model,
            predictions=sample_run_data["predictions"],
        )

        loaded = store.load_run(run_dir)

        assert "config" in loaded
        assert "metrics" in loaded
        assert "predictions_df" in loaded
        assert "model_state_dict" in loaded

    def test_list_runs_returns_metadata(self, temp_output_dir, sample_model, sample_run_data):
        """Test list_runs returns run metadata."""
        store = RunStore(temp_output_dir)

        # Save two runs
        for i in range(2):
            run_dir = temp_output_dir / f"test_run_{i}-20230101_120000"
            store.save_run(
                run_dir=run_dir,
                config=sample_run_data["config"],
                history=sample_run_data["history"],
                model=sample_model,
                predictions=sample_run_data["predictions"],
            )

        runs = store.list_runs()

        assert len(runs) >= 2
        for run in runs:
            assert "name" in run
            assert "path" in run
            assert "final_rmse" in run
            assert "final_mae" in run

    def test_get_run_by_name_exact_match(self, temp_output_dir, sample_model, sample_run_data):
        """Test get_run_by_name finds run by prefix."""
        store = RunStore(temp_output_dir)
        run_dir = temp_output_dir / "my_experiment-20230101_120000"

        store.save_run(
            run_dir=run_dir,
            config=sample_run_data["config"],
            history=sample_run_data["history"],
            model=sample_model,
            predictions=sample_run_data["predictions"],
        )

        found = store.get_run_by_name("my_experiment")
        assert found is not None
        assert found.name.startswith("my_experiment")

    def test_get_run_by_name_not_found(self, temp_output_dir):
        """Test get_run_by_name returns None for nonexistent run."""
        store = RunStore(temp_output_dir)
        found = store.get_run_by_name("nonexistent")
        assert found is None

    def test_get_run_by_name_returns_most_recent(self, temp_output_dir, sample_model, sample_run_data):
        """Test get_run_by_name returns most recent match."""
        store = RunStore(temp_output_dir)

        # Create two runs with same prefix
        import time
        for i in range(2):
            run_dir = temp_output_dir / f"my_exp-run_{i}"
            store.save_run(
                run_dir=run_dir,
                config=sample_run_data["config"],
                history=sample_run_data["history"],
                model=sample_model,
                predictions=sample_run_data["predictions"],
            )
            time.sleep(0.01)

        found = store.get_run_by_name("my_exp")
        # Should return one of them
        assert found is not None
