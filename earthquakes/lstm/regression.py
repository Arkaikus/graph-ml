import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.data import EarthquakeData
from ray import tune
from ray.air import Result
from ray.train import Checkpoint
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import MeanAbsolutePercentageError

from .model import LSTMModel
from .plot import plot_scatter, plot_timeseries

logger = logging.getLogger(__name__)


class RegressionTrainable(tune.Trainable):
    def setup(self, config: dict, qdata: EarthquakeData):
        """takes hyperparameter values in the config argument"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.trial_id)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.qdata = qdata

        lookback = config.get("lookback")
        test_size = config.get("test_size")
        batch_size = config["batch_size"]
        hidden_size = config.get("hidden_size", 1)
        lstm_layers = config.get("lstm_layers", 2)
        learning_rate = config["lr"]
        loss_type = config.get("loss_type", "mse")
        shuffle = config.get("shuffle", False)

        self.max_epochs = config.get("max_epochs", 100)
        self.epoch = 0

        assert lookback, "[lookback] cannot be None"

        sequences, targets = self.qdata.to_sequences(qdata.normalized_data, lookback)
        self.x_train, self.x_test, self.y_train, self.y_test = self.qdata.split(sequences, targets, test_size)

        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.test_dataset = TensorDataset(self.x_test, self.y_test)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)

        self.model = LSTMModel(lookback, len(qdata.targets), hidden_size, lstm_layers).to(self.device)
        loss_types = {"mse": nn.MSELoss, "huber": nn.HuberLoss, "mape": MeanAbsolutePercentageError, "mae": nn.L1Loss}
        loss_class = loss_types.get(loss_type, nn.MSELoss)
        self.criterion = loss_class().to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.patience = 5
        self.best_loss = np.inf
        self.done = False

    def stop(self, loss):
        """Handles early stopping"""
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = min(self.patience + 1, 10)
        elif not self.done:
            self.patience -= 1
            if self.patience <= 0:
                self.logger.info("Early stopping")
                return True

        return self.epoch >= self.max_epochs - 1

    def step(self):
        self.epoch += 1
        # Training phase
        self.model.train()
        epoch_loss = 0
        for input_batch, output_batch in self.train_loader:
            input_batch, output_batch = input_batch.to(self.device), output_batch[:, -1].to(self.device)

            self.optimizer.zero_grad()

            # forward pass
            output = self.model(input_batch)
            loss = self.criterion(output, output_batch)

            # backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # statistics
            batch_loss = loss.item()
            epoch_loss += batch_loss

        mean_loss = epoch_loss / len(self.train_loader)

        # Assuming test_loader is defined with your test dataset
        self.model.eval()
        test_loss = 0
        total_samples = 0

        with torch.no_grad():
            for input_batch, output_batch in self.test_loader:
                input_batch, output_batch = input_batch.to(self.device), output_batch[:, -1].to(self.device)
                output = self.model(input_batch)
                loss = self.criterion(output, output_batch)
                test_loss += loss.item() * input_batch.size(0)  # Accumulate loss
                total_samples += input_batch.size(0)

        mean_test_loss = test_loss / total_samples

        self.done = self.stop(epoch_loss)
        metrics = {
            "loss": epoch_loss,
            "mean_loss": mean_loss,
            "test_loss": test_loss,
            "mean_test_loss": mean_test_loss,
            "checkpoint_dir_name": "",
            "patience": self.patience,
            "done": self.done,
        }
        logger.info("Metrics: %s", metrics)
        return metrics

    def save_checkpoint(self, checkpoint_dir: str) -> torch.Dict | None:
        """saves checkpoint at the end of training"""
        self.logger.info("Saving model and optimizer to %s", checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pth")
        state = (self.model.state_dict(), self.optimizer.state_dict())
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint: Checkpoint):
        """loads checkpoint at the start of training  if checkpoint exists"""
        self.logger.info("Loading checkpoint %s", checkpoint)
        with checkpoint.as_directory() as loaded_checkpoint_dir:
            logger.info("Loading checkpoint %s", loaded_checkpoint_dir)
            checkpoint_path = os.path.join(loaded_checkpoint_dir, "checkpoint.pth")
            model_state, optimizer_state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)

    def forecast(self):
        self.model.eval()
        with torch.no_grad():
            train_output = [self.model(x.to(self.device)) for x, _ in self.train_loader]
            test_output = [self.model(x.to(self.device)) for x, _ in self.test_loader]
            train_output = torch.cat(train_output, dim=0)
            test_output = torch.cat(test_output, dim=0)

        return (
            (self.y_train[:, -1].numpy(), train_output.detach().cpu().numpy()),
            (self.y_test[:, -1].numpy(), test_output.detach().cpu().numpy()),
        )


def test_result(result: Result, qdata: EarthquakeData, metric, mode):
    logger.info("Loading testing from config")
    trainable_cls = tune.with_parameters(RegressionTrainable, qdata=qdata)
    trainable: RegressionTrainable = trainable_cls(config=result.config)
    best_checkpoint = result.get_best_checkpoint(metric, mode)
    trainable.load_checkpoint(best_checkpoint)

    print(result.path)
    print(result.metrics_dataframe)

    ((train_y, train_pred), (test_y, test_pred)) = trainable.forecast()

    def target_idx(y, pred, idx):
        return y[:, idx, None], pred[:, idx, None]

    save_to = Path.home() / "plots" / qdata.hash / Path(result.path).stem
    shutil.copytree(result.path, save_to, dirs_exist_ok=True)
    for idx, target in enumerate(trainable.qdata.targets):
        plot_scatter(*target_idx(train_y, train_pred, idx), save_to / f"{target}_train_scatter.png")
        plot_scatter(*target_idx(test_y, test_pred, idx), save_to / f"{target}_test_scatter.png")
        plot_timeseries(*target_idx(train_y, train_pred, idx), target, save_to / f"{target}_train_timeseries.png")
        plot_timeseries(*target_idx(test_y, test_pred, idx), target, save_to / f"{target}_test_timeseries.png")

    result.metrics_dataframe[["loss", "test_loss"]].plot(legend=True).get_figure().savefig(save_to / "loss.png")
