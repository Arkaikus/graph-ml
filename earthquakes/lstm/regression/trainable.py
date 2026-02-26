"""Regression trainable for LSTM (continuous target forecasting)."""

import logging
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from ray.air import Result
from torchmetrics.regression import MeanAbsolutePercentageError

from lstm.model import LSTMModel
from lstm.regression.plots import plot_scatter, plot_timeseries
from lstm.trainable.base import BaseLSTMTrainable

logger = logging.getLogger(__name__)


class RegressionTrainable(BaseLSTMTrainable):
    """Ray Tune trainable for LSTM regression (continuous target forecasting)."""

    def setup_data(self) -> None:
        assert self.lookback, "[lookback] cannot be None"
        sequences, targets = self.qdata.to_sequences(
            self.qdata.normalized_data, self.lookback
        )
        x_train, x_test, y_train, y_test = self.qdata.split(
            sequences, targets, self.test_size
        )
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train[:, -1]
        self.y_test = y_test[:, -1]

    def setup_model(self) -> None:
        num_features = len(self.qdata.features)
        self.model = LSTMModel(
            lookback=self.lookback,
            outputs=len(self.qdata.targets),
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            num_features=num_features,
        ).to(self.device)
        loss_types = {
            "mse": nn.MSELoss,
            "huber": nn.HuberLoss,
            "mape": MeanAbsolutePercentageError,
            "mae": nn.L1Loss,
        }
        loss_type = self.config.get("loss_type", "mse")
        loss_class = loss_types.get(loss_type, nn.MSELoss)
        self.criterion = loss_class().to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

    def forecast(self) -> tuple:
        """Return (y_train, train_pred, y_test, test_pred) as numpy arrays."""
        self.model.eval()
        with torch.no_grad():
            train_output = [
                self.model(x.to(self.device)) for x, _ in self.train_loader
            ]
            test_output = [
                self.model(x.to(self.device)) for x, _ in self.test_loader
            ]
            train_output = torch.cat(train_output, dim=0).detach().cpu().numpy()
            test_output = torch.cat(test_output, dim=0).detach().cpu().numpy()

        return (
            self.y_train.numpy(),
            train_output,
            self.y_test.numpy(),
            test_output,
        )

    def test_result(self, result: Result, metric: str, mode: str) -> None:
        logger.info("Loading testing checkpoint")
        best_checkpoint = result.get_best_checkpoint(metric, mode)
        self.load_checkpoint(best_checkpoint)

        print(result.path)
        print(result.metrics_dataframe)

        train_y, train_pred, test_y, test_pred = self.forecast()

        def target_idx(y, pred, idx):
            return y[:, idx : idx + 1], pred[:, idx : idx + 1]

        save_to = Path.cwd() / "plots" / self.qdata.hash / Path(result.path).stem
        shutil.copytree(result.path, save_to, dirs_exist_ok=True)
        for idx, target in enumerate(self.qdata.targets):
            plot_scatter(
                *target_idx(train_y, train_pred, idx),
                save_to / f"{target}_train_scatter.png",
            )
            plot_scatter(
                *target_idx(test_y, test_pred, idx),
                save_to / f"{target}_test_scatter.png",
            )
            plot_timeseries(
                *target_idx(train_y, train_pred, idx),
                target,
                save_to / f"{target}_train_timeseries.png",
            )
            plot_timeseries(
                *target_idx(test_y, test_pred, idx),
                target,
                save_to / f"{target}_test_timeseries.png",
            )

        result.metrics_dataframe[["loss", "test_loss"]].plot(
            legend=True
        ).get_figure().savefig(save_to / "loss.png")
