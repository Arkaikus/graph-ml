"""Shared base for LSTM Ray Tune trainables."""

import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from data.data import EarthquakeData
from ray import tune
from ray.train import Checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class BaseLSTMTrainable(tune.Trainable, ABC):
    """Base trainable for LSTM tasks. Subclasses implement setup_data, setup_model, and optionally batch_metrics."""

    def setup(self, config: dict, qdata: EarthquakeData):
        """Parse config and delegate to subclass setup methods."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.trial_id)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.qdata = qdata
        self.config = config

        self.lookback = config.get("lookback")
        self.test_size = config.get("test_size")
        self.batch_size = config.get("batch_size", 32)
        self.hidden_size = config.get("hidden_size", 1)
        self.lstm_layers = config.get("lstm_layers", 2)
        self.learning_rate = config.get("lr", 0.001)
        self.max_epochs = config.get("max_epochs", 100)
        self.epoch = 0
        self.patience = 5
        self.best_loss = np.inf
        self.done = False

        self.setup_data()
        self.setup_loaders()
        self.setup_model()
        self.post_init()

    @abstractmethod
    def setup_data(self) -> None:
        """Prepare data. Must set self.x_train, self.x_test, self.y_train, self.y_test."""

    def setup_loaders(self) -> None:
        """Create DataLoaders from x_train, x_test, y_train, y_test (and x_val, y_val if present)."""
        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.test_dataset = TensorDataset(self.x_test, self.y_test)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # earthquake data is temporal
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        if hasattr(self, "x_val") and self.x_val is not None:
            self.val_dataset = TensorDataset(self.x_val, self.y_val)
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            self.val_loader = self.train_loader

    @abstractmethod
    def setup_model(self) -> None:
        """Create model, criterion, optimizer. Must set self.model, self.criterion, self.optimizer."""

    def post_init(self) -> None:
        """Override for subclass-specific initialization. Base sets up LR scheduler."""
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)

    def prepare_target_for_loss(self, target_batch: torch.Tensor) -> torch.Tensor:
        """Transform target for criterion. Override for classification (e.g. view(-1))."""
        return target_batch

    def batch_metrics(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        """Return per-batch metrics (e.g. correct, total for accuracy). Default: empty."""
        return {}

    def train_batch(self, input_batch: torch.Tensor, output_batch: torch.Tensor) -> tuple[float, dict]:
        """Single training step. Returns (loss, extra_metrics)."""
        self.optimizer.zero_grad()
        output = self.model(input_batch.to(self.device))
        target = self.prepare_target_for_loss(output_batch).to(self.device)
        loss = self.criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), self.batch_metrics(output, target)

    def train_epoch(self) -> dict:
        """One training pass. Aggregates loss and batch_metrics."""
        self.model.train()
        epoch_loss = 0.0
        agg_metrics: dict[str, float] = {}

        for input_batch, output_batch in self.train_loader:
            batch_loss, batch_metrics = self.train_batch(input_batch, output_batch)
            epoch_loss += batch_loss
            for k, v in batch_metrics.items():
                agg_metrics[k] = agg_metrics.get(k, 0) + v

        mean_loss = epoch_loss / len(self.train_loader)
        result = {"loss": epoch_loss, "mean_loss": mean_loss}
        if "correct" in agg_metrics and "total" in agg_metrics:
            result["train_accuracy"] = agg_metrics["correct"] / agg_metrics["total"]
        result.update(agg_metrics)
        return result

    def eval_batch(self, input_batch: torch.Tensor, output_batch: torch.Tensor) -> tuple[float, dict]:
        """Single eval step (no grad). Returns (loss, extra_metrics)."""
        output = self.model(input_batch.to(self.device))
        target = self.prepare_target_for_loss(output_batch).to(self.device)
        loss = self.criterion(output, target)
        return loss.item() * input_batch.size(0), self.batch_metrics(output, target)

    def eval(self, loader: DataLoader) -> dict:
        """Evaluate on loader. Aggregates loss and batch_metrics."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        agg_metrics: dict[str, float] = {}

        with torch.no_grad():
            for input_batch, output_batch in loader:
                batch_loss, batch_metrics = self.eval_batch(input_batch, output_batch)
                total_loss += batch_loss
                total_samples += input_batch.size(0)
                for k, v in batch_metrics.items():
                    agg_metrics[k] = agg_metrics.get(k, 0) + v

        mean_test_loss = total_loss / total_samples
        result = {"test_loss": total_loss, "mean_test_loss": mean_test_loss}
        if "correct" in agg_metrics and "total" in agg_metrics:
            result["accuracy"] = agg_metrics["correct"] / agg_metrics["total"]
        result.update(agg_metrics)
        return result

    def step(self) -> dict:
        """One training step. Returns combined metrics for Ray Tune."""
        self.epoch += 1
        epoch_metrics = self.train_epoch()
        val_metrics = self.eval(self.val_loader)
        eval_metrics = self.eval(self.test_loader)
        val_loss = val_metrics["mean_test_loss"]
        self.scheduler.step(val_loss)
        self.done = self.is_done(val_loss)
        metrics = {
            "checkpoint_dir_name": "",
            "patience": self.patience,
            "done": self.done,
            "val_loss": val_loss,
            **epoch_metrics,
            **eval_metrics,
        }
        logger.info("epoch metrics: %s val_metrics: %s eval_metrics: %s", epoch_metrics, val_metrics, eval_metrics)
        return metrics

    def is_done(self, val_loss: float) -> bool:
        """Early stopping and max epochs. Uses val loss when validation set exists."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience = min(self.patience + 1, 10)
        elif not self.done:
            self.patience -= 1
            if self.patience <= 0:
                self.logger.info("Early stopping")
                return True
        return self.epoch >= self.max_epochs - 1

    def save_checkpoint(self, checkpoint_dir: str) -> dict | None:
        """Save model and optimizer state."""
        self.logger.info("Saving model and optimizer to %s", checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        state = (self.model.state_dict(), self.optimizer.state_dict())
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Load model and optimizer state."""
        self.logger.info("Loading checkpoint %s", checkpoint)
        with checkpoint.as_directory() as loaded_checkpoint_dir:
            checkpoint_path = os.path.join(loaded_checkpoint_dir, "checkpoint.pth")
            model_state, optimizer_state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
