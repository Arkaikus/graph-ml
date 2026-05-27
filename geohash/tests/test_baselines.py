"""Test baseline predictors."""

import torch

from geohash.training.baselines import beats_baseline, run_baselines


def _sample(mag_last, target):
    return {
        "gh_ids": torch.tensor([1, 2]),
        "x_num": torch.tensor([[2.0, 10.0, 0.0, 0.0, 0.0, 0.0], [mag_last, 10.0, 1.0, 0.0, 0.0, 0.0]]),
        "y": torch.tensor([target]),
    }


class TestBaselines:
    def test_run_baselines(self):
        train = [_sample(2.0, 2.5), _sample(2.5, 3.0)]
        test = [_sample(3.0, 3.2), _sample(2.8, 2.9)]
        baselines = run_baselines(train, test)
        assert "mean" in baselines
        assert "persistence" in baselines
        assert "linear_numeric" in baselines
        assert baselines["persistence"]["rmse"] >= 0

    def test_run_baselines_variable_length_windows(self):
        """Baselines must handle windows of different lengths."""
        train = [
            _sample(2.0, 2.5),
            {
                "gh_ids": torch.tensor([1, 2, 3, 4]),
                "x_num": torch.randn(4, 6),
                "y": torch.tensor([3.0]),
            },
        ]
        test = [
            {
                "gh_ids": torch.tensor([1, 2, 3]),
                "x_num": torch.randn(3, 6),
                "y": torch.tensor([3.2]),
            },
        ]
        baselines = run_baselines(train, test)
        assert baselines["linear_numeric"]["rmse"] >= 0

    def test_beats_baseline(self):
        model_m = {"rmse": 0.5}
        base_m = {"persistence": {"rmse": 0.8}, "mean": {"rmse": 0.9}}
        flags = beats_baseline(model_m, base_m)
        assert flags["persistence"] is True
        assert flags["mean"] is True
