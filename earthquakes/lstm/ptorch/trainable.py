import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pathlib import Path
from ray import tune
from ray.air import Result
from ray.train import Checkpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sns.set_theme(style="darkgrid")


from data.data import EarthquakeData
from .model import LSTMModel

logger = logging.getLogger(__name__)


class LSTMTrainable(tune.Trainable):
    @classmethod
    def with_parameters(cls, config, qdata: EarthquakeData) -> "LSTMTrainable":
        return tune.with_parameters(cls, qdata=qdata)(config=config)

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

        self.max_epochs = config.get("max_epochs", 100)
        self.epoch = 0

        assert lookback, "[lookback] cannot be None"

        (
            # shapes: ((1-test_size)*N, F, Lookback), (test_size*N, Lookback, 1)
            (self.x_train, self.y_train),
            (self.x_test, self.y_test),
        ) = self.qdata.train_test_split(lookback, test_size, "standard")

        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.test_dataset = TensorDataset(self.x_test, self.y_test)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size)

        self.model = LSTMModel(lookback, hidden_size, lstm_layers).to(self.device)
        self.loss = nn.HuberLoss().to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.patience = 5
        self.best_loss = np.inf

    def step(self):
        done = self.epoch >= self.max_epochs - 1

        # Training phase
        self.model.train()
        epoch_loss = 0
        for input_batch, output_batch in self.train_loader:
            input_batch, output_batch = input_batch.to(self.device), output_batch.to(self.device)

            self.optimizer.zero_grad()

            # forward pass
            output = self.model(input_batch)
            loss = self.loss(output, output_batch[:, -1])

            # backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # statistics
            batch_loss = loss.item()
            epoch_loss += batch_loss

        mean_loss = epoch_loss / len(self.train_loader)
        logger.info("Epoch loss: %.3f Mean loss: %.3f Patience: %s", epoch_loss, mean_loss, self.patience)

        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.patience = min(self.patience + 1, 10)
        elif not done:
            self.patience -= 1
            if self.patience == 0:
                self.logger.info("Early stopping")
                done = True

        return {
            "loss": epoch_loss,
            "mean_loss": mean_loss,
            "checkpoint_dir_name": "",
            "done": done,
        }

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


def plot_scatter(original, forecast, file_path, name="scatter"):
    """
    Plots a scatter plot of the forecast against the original data with metrics
    this is useful for visualizing the performance of the model
    it uses seaborn to plot the scatter plot with a regression line
    """
    MSE = mean_squared_error(original, forecast)
    MAE = mean_absolute_error(original, forecast)
    R2 = r2_score(original, forecast)
    MAPE = mean_absolute_percentage_error(original, forecast)
    RMSE = np.sqrt(MSE)
    logger.info("Best trial R2 %s", R2)
    logger.info("Best trial MSE: %s", MSE)
    logger.info("Best trial RMSE: %s", RMSE)
    logger.info("Best trial MAE: %s", MAE)
    logger.info("Best trial MAPE: %s", MAPE)

    hstack = np.hstack((original, forecast))
    logger.info("Hstack %s", hstack.shape)

    g = sns.jointplot(
        x="Real",
        y="Forecast",
        data=pd.DataFrame(hstack, columns=["Real", "Forecast"]),
        kind="reg",
        truncate=False,
        color="m",
        height=7,
    )

    # Add metrics to the plot
    plt.figtext(
        0.15,
        0.70,
        f"R2: {R2:.2f}\nMSE: {MSE:.2f}\nMAE: {MAE:.2f}",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=12,
    )

    save_to = Path(file_path) / f"{name}.png"
    logger.info("Figure saved to %s", save_to)
    g.figure.savefig(save_to)
    plt.close(g.figure)


def plot_timeseries(original, forecast, target: str, file_path, name="timeseries"):
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.plot(original, label="Real")
    ax.plot(forecast, label="Forecast")
    ax.legend()
    plt.title("Test Data Real vs Forecast")
    plt.xlabel("Time")
    plt.ylabel(target.capitalize())

    save_to = Path(file_path) / f"{name}.png"
    logger.info("Figure saved to %s", save_to)
    fig.savefig(save_to)
    plt.close(fig)


def test_result(result: Result, qdata: EarthquakeData):
    logger.info("Loading testing from config")
    trainable = LSTMTrainable.with_parameters(result.config, qdata)
    best_checkpoint = result.get_best_checkpoint("loss", "min")
    trainable.load_checkpoint(best_checkpoint)

    print(result.metrics_dataframe)

    (
        (train_original, train_forecast),
        (test_original, test_forecast),
    ) = trainable.forecast()

    (target,) = trainable.qdata.targets
    plot_scatter(train_original, train_forecast, result.path, name="train_scatter")
    plot_scatter(test_original, test_forecast, result.path, name="test_scatter")
    plot_timeseries(train_original, train_forecast, target, result.path, name="train_timeseries")
    plot_timeseries(test_original, test_forecast, target, result.path, name="test_timeseries")
