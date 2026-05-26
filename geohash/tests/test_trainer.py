"""Test training and evaluation."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from geohash.data import QuakeWindowDataset, collate_batch
from geohash.model import NextMagnitudeLSTM
from geohash.training import train, evaluate


class TestEvaluate:
    """Test evaluation function."""

    @pytest.fixture
    def simple_model_and_loader(self):
        """Create simple model and data loader."""
        model = NextMagnitudeLSTM(
            vocab_size=50,
            embedding_dim=8,
            num_numeric=7,
            hidden_size=32,
        )

        # Create dummy samples
        samples = []
        for i in range(10):
            samples.append({
                "gh_ids": torch.randint(0, 50, (5,)),
                "x_num": torch.randn(5, 7),
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
            num_numeric=7,
            hidden_size=32,
        )

        # Create dummy samples
        def make_samples(n=20):
            samples = []
            for i in range(n):
                samples.append({
                    "gh_ids": torch.randint(0, 50, (5,)),
                    "x_num": torch.randn(5, 7),
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

        return model, train_loader, test_loader

    def test_train_returns_history(self, train_setup):
        """Test train returns history dict."""
        model, train_loader, test_loader = train_setup

        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device="cpu",
            epochs=1,
            learning_rate=1e-3,
        )

        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "test_loss" in history
        assert "rmse" in history
        assert "mae" in history

    def test_train_history_length(self, train_setup):
        """Test history has correct length."""
        model, train_loader, test_loader = train_setup
        epochs = 2

        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device="cpu",
            epochs=epochs,
            learning_rate=1e-3,
        )

        assert len(history["train_loss"]) == epochs
        assert len(history["test_loss"]) == epochs
        assert len(history["rmse"]) == epochs
        assert len(history["mae"]) == epochs

    def test_train_model_is_modified(self, train_setup):
        """Test that training modifies model weights."""
        model, train_loader, test_loader = train_setup

        # Store initial weights
        initial_weights = {
            name: param.clone() for name, param in model.named_parameters()
        }

        # Train
        train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device="cpu",
            epochs=1,
            learning_rate=1e-3,
        )

        # Check weights changed
        weights_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_weights[name]):
                weights_changed = True
                break

        assert weights_changed, "Model weights should have been updated during training"

    def test_train_different_learning_rates(self, train_setup):
        """Test training with different learning rates."""
        model1, train_loader, test_loader = train_setup
        model2, _, _ = train_setup

        history1 = train(
            model=model1,
            train_loader=train_loader,
            test_loader=test_loader,
            device="cpu",
            epochs=1,
            learning_rate=1e-2,
        )

        history2 = train(
            model=model2,
            train_loader=train_loader,
            test_loader=test_loader,
            device="cpu",
            epochs=1,
            learning_rate=1e-5,
        )

        # Higher learning rate should have different (usually larger) loss change
        # Just check that we get valid results
        assert history1["train_loss"][0] > 0
        assert history2["train_loss"][0] > 0
