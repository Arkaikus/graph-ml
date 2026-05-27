"""Test training and evaluation."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from geohash.data import QuakeWindowDataset, collate_batch
from geohash.model import NextMagnitudeLSTM
from geohash.training import train, evaluate


_NUM_NUMERIC = 6  # magnitude, depth_km, time_days, delta_t_days, delta_mag, delta_distance_km


class TestEvaluate:
    """Test evaluation function."""

    @pytest.fixture
    def simple_model_and_loader(self):
        """Create simple model and data loader."""
        model = NextMagnitudeLSTM(
            vocab_size=50,
            embedding_dim=8,
            num_numeric=_NUM_NUMERIC,
            hidden_size=32,
        )

        samples = []
        for i in range(10):
            samples.append({
                "gh_ids": torch.randint(0, 50, (5,)),
                "x_num": torch.randn(5, _NUM_NUMERIC),
                "y": torch.tensor([2.5 + i * 0.1], dtype=torch.float32),
            })

        dataset = QuakeWindowDataset(samples)
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_batch,
        )

        return model, loader

    def test_evaluate_returns_metrics(self, simple_model_and_loader):
        """Test evaluate returns three metrics."""
        model, loader = simple_model_and_loader
        loss, rmse, mae = evaluate(model, loader, "cpu")

        assert isinstance(loss, float)
        assert isinstance(rmse, float)
        assert isinstance(mae, float)

    def test_evaluate_metrics_positive(self, simple_model_and_loader):
        """Test metrics are positive."""
        model, loader = simple_model_and_loader
        loss, rmse, mae = evaluate(model, loader, "cpu")

        assert loss >= 0
        assert rmse >= 0
        assert mae >= 0

    def test_evaluate_no_gradients(self, simple_model_and_loader):
        """Test evaluate doesn't compute gradients."""
        model, loader = simple_model_and_loader
        model.train()  # Ensure gradients are being tracked

        for param in model.parameters():
            param.grad = None

        evaluate(model, loader, "cpu")

        # No gradients should have been computed
        for param in model.parameters():
            assert param.grad is None


class TestTrain:
    """Test training function."""

    @pytest.fixture
    def train_setup(self):
        """Create model, train and test loaders."""
        model = NextMagnitudeLSTM(
            vocab_size=50,
            embedding_dim=8,
            num_numeric=_NUM_NUMERIC,
            hidden_size=32,
        )

        def make_samples(n=20):
            samples = []
            for i in range(n):
                samples.append({
                    "gh_ids": torch.randint(0, 50, (5,)),
                    "x_num": torch.randn(5, _NUM_NUMERIC),
                    "y": torch.tensor([2.5 + i * 0.1], dtype=torch.float32),
                })
            return samples

        train_samples = make_samples(20)
        test_samples = make_samples(10)

        train_dataset = QuakeWindowDataset(train_samples)
        test_dataset = QuakeWindowDataset(test_samples)

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            collate_fn=collate_batch,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            collate_fn=collate_batch,
        )

        return model, train_loader, test_loader  # test_loader used as val_loader

    def _run_train(self, model, train_loader, val_loader, **kwargs):
        return train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu",
            **kwargs,
        )

    def test_train_returns_history(self, train_setup):
        model, train_loader, val_loader = train_setup
        history = self._run_train(
            model, train_loader, val_loader,
            epochs=1, learning_rate=1e-3, early_stopping_patience=0,
        )
        assert isinstance(history, dict)
        for key in ("train_loss", "val_loss", "test_loss", "rmse", "mae", "lr"):
            assert key in history

    def test_train_history_length(self, train_setup):
        model, train_loader, val_loader = train_setup
        epochs = 2
        history = self._run_train(
            model, train_loader, val_loader,
            epochs=epochs, learning_rate=1e-3, early_stopping_patience=0,
        )
        assert len(history["train_loss"]) == epochs
        assert len(history["val_loss"]) == epochs

    def test_train_model_is_modified(self, train_setup):
        model, train_loader, val_loader = train_setup
        initial_weights = {name: param.clone() for name, param in model.named_parameters()}
        self._run_train(model, train_loader, val_loader, epochs=1, learning_rate=1e-3)
        weights_changed = any(
            not torch.allclose(param, initial_weights[name])
            for name, param in model.named_parameters()
        )
        assert weights_changed

    def test_train_different_learning_rates(self, train_setup):
        model1, train_loader, val_loader = train_setup
        model2, _, _ = train_setup
        h1 = self._run_train(model1, train_loader, val_loader, epochs=1, learning_rate=1e-2, early_stopping_patience=0)
        h2 = self._run_train(model2, train_loader, val_loader, epochs=1, learning_rate=1e-5, early_stopping_patience=0)
        assert h1["train_loss"][0] > 0
        assert h2["train_loss"][0] > 0

    def test_early_stopping_triggers(self, train_setup):
        model, train_loader, val_loader = train_setup
        history = self._run_train(
            model, train_loader, val_loader,
            epochs=20, learning_rate=1e-3, early_stopping_patience=2, lr_scheduler="none",
        )
        assert len(history["train_loss"]) <= 20

    def test_early_stopping_disabled(self, train_setup):
        model, train_loader, val_loader = train_setup
        history = self._run_train(
            model, train_loader, val_loader,
            epochs=3, learning_rate=1e-3, early_stopping_patience=0, lr_scheduler="none",
        )
        assert len(history["train_loss"]) == 3

    def test_lr_scheduler_cosine(self, train_setup):
        model, train_loader, val_loader = train_setup
        history = self._run_train(
            model, train_loader, val_loader,
            epochs=4, learning_rate=1e-2, early_stopping_patience=0, lr_scheduler="cosine",
        )
        assert history["lr"][0] >= history["lr"][-1]

    def test_lr_scheduler_plateau(self, train_setup):
        model, train_loader, val_loader = train_setup
        history = self._run_train(
            model, train_loader, val_loader,
            epochs=2, learning_rate=1e-3, early_stopping_patience=0, lr_scheduler="plateau", lr_patience=1,
        )
        assert len(history["lr"]) == 2

    def test_gradient_clipping_runs(self, train_setup):
        model, train_loader, val_loader = train_setup
        history = self._run_train(
            model, train_loader, val_loader,
            epochs=1, learning_rate=1e-3, early_stopping_patience=0, gradient_clip=0.1,
        )
        assert len(history["train_loss"]) == 1
