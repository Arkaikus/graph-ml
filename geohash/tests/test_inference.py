"""Test inference artifacts and predict helpers."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from geohash.inference.artifacts import (
    build_model_from_bundle,
    load_run_bundle,
    save_model_config_artifact,
    save_preprocess_artifact,
)
from geohash.model import NextMagnitudeLSTM


class TestInferenceArtifacts:
    @pytest.fixture
    def run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            stoi = {"<PAD>": 0, "<UNK>": 1, "abc1": 2}
            save_preprocess_artifact(
                run_dir=rd,
                stoi=stoi,
                numeric_mean=[0.0] * 6,
                numeric_std=[1.0] * 6,
                numeric_cols=["magnitude", "depth_km", "time_days", "delta_t_days", "delta_mag", "delta_distance_km"],
                geohash_precision=4,
                encoding="flat",
                input_mode="full",
            )
            save_model_config_artifact(
                rd,
                {
                    "vocab_size": 3,
                    "embedding_dim": 8,
                    "num_numeric": 6,
                    "hidden_size": 16,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "input_mode": "full",
                    "encoding": "flat",
                    "geohash_precision": 4,
                },
            )
            model = NextMagnitudeLSTM(vocab_size=3, embedding_dim=8, num_numeric=6, hidden_size=16)
            torch.save(model.state_dict(), rd / "model_final.pt")
            yield rd

    def test_load_run_bundle(self, run_dir):
        bundle = load_run_bundle(run_dir)
        assert "preprocess" in bundle
        assert "model_config" in bundle
        assert "model_state_dict" in bundle

    def test_build_model_from_bundle(self, run_dir):
        bundle = load_run_bundle(run_dir)
        model = build_model_from_bundle(bundle)
        model.load_state_dict(bundle["model_state_dict"])
        gh_ids = torch.tensor([[2, 2]], dtype=torch.long)
        x_num = torch.randn(1, 2, 6)
        lengths = torch.tensor([2])
        out = model(gh_ids, x_num, lengths)
        assert out.shape == (1, 1)

    def test_preprocess_json_roundtrip(self, run_dir):
        with open(run_dir / "preprocess.json") as f:
            data = json.load(f)
        assert data["encoding"] == "flat"
        assert len(data["numeric_mean"]) == 6
