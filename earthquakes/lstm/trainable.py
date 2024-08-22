import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ray import tune
from ray.air import Result
from ray.train import Checkpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn import MSELoss
from torch.optim import Adam
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

        batch_size = config["batch_size"]
        sequence_size = config.get("sequence_size")
        test_size = config.get("test_size")
        self.max_epochs = config.get("max_epochs", 100)
        self.epoch = 0

        assert sequence_size, "[sequence_size] cannot be None"

        # prepare train test split of data
        (
            (self.x_train, self.y_train),
            (self.x_val, self.y_val),
            (self.x_test, self.y_test),
        ) = qdata.train_test_split(sequence_size, test_size)

        train_dataset = TensorDataset(self.x_train, self.y_train)
        val_dataset = TensorDataset(self.x_val, self.y_val)
        test_dataset = TensorDataset(self.x_test, self.y_test)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        feature_size = self.x_train.shape[2]  # 0:batch, 1:lookback 2:features
        hidden_size = config.get("hidden_size", 1)
        lstm_layers = config.get("layers", 1)

        self.model = LSTMModel(feature_size, hidden_size, lstm_layers).to(self.device)
        self.loss = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=config["lr"])

    def step(self):
        """performs a training epoch"""
        epoch_loss, val_loss = 0, 0
        self.model.train()
        for inputs, outputs in self.train_dataloader:
            forecast = self.model(inputs.to(self.device))
            loss = self.loss(forecast, outputs.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        self.model.eval()
        with torch.no_grad():
            val_pred = []
            for val_input, val_output in self.val_dataloader:
                y_pred = self.model(val_input.to(self.device))
                val_loss += self.loss(y_pred, val_output.to(self.device)).item()
                val_pred.append(y_pred.detach().cpu())

            val_pred = torch.cat(val_pred, dim=0)
        self.epoch += 1
        mean_loss = epoch_loss / len(self.train_dataloader)
        val_r2 = r2_score(self.y_val.numpy(), val_pred.numpy())
        val_mse = mean_squared_error(self.y_val, val_pred).item()
        val_mae = mean_absolute_error(self.y_val, val_pred).item()
        return {
            "epoch": self.epoch,
            "loss": epoch_loss,
            "mean_loss": mean_loss,
            "val_loss": val_loss,
            "val_r2": val_r2,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "checkpoint_dir_name": "",
            "done": self.epoch >= self.max_epochs,
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

    def forecast(self, dataloader: DataLoader):
        """Runst the model against a dataloader"""
        self.model.eval()
        with torch.no_grad():
            y_preds = []
            for inputs, _ in dataloader:
                y_pred = self.model(inputs.to(self.device))
                y_preds.append(y_pred.detach().cpu())

            y_preds = torch.cat(y_preds, dim=0)

        return y_preds

    def get_metrics(self, y_true, y_pred):
        """Returns the metrics for the model"""
        test_loss = mean_squared_error(y_true, y_pred).item()
        test_mae = mean_absolute_error(y_true, y_pred).item()
        test_r2 = r2_score(y_true.numpy(), y_pred.numpy())
        test_rmse = np.sqrt(test_loss)
        return {
            "test_r2": test_r2,
            "test_loss": test_loss,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        }

    def eval_train(self):
        """performs a forecast on the test data"""
        y_preds = self.forecast(self.train_dataloader)
        metrics = self.get_metrics(self.y_train, y_preds)
        logger.info("Train metrics %s", metrics)
        return (self.y_train, y_preds, metrics)

    def eval_test(self):
        """performs a forecast on the test data"""
        y_preds = self.forecast(self.test_dataloader)
        metrics = self.get_metrics(self.y_test, y_preds)
        logger.info("Test metrics %s", metrics)
        return (self.y_test, y_preds, metrics)


def plot_scatter(original, forecast, metrics, file_path, name="scatter"):
    """
    Plots a scatter plot of the forecast against the original data with metrics
    this is useful for visualizing the performance of the model
    it uses seaborn to plot the scatter plot with a regression line
    """
    R2 = metrics["test_r2"]
    MSE = metrics["test_loss"]
    RMSE = metrics["test_rmse"]
    MAE = metrics["test_mae"]
    logger.info("Best trial R2 %s", R2)
    logger.info("Best trial MSE: %s", MSE)
    logger.info("Best trial RMSE: %s", RMSE)
    logger.info("Best trial MAE: %s", MAE)

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


def test_result(result: Result, qdata: EarthquakeData):
    logger.info("Loading testing from config")
    trainable = LSTMTrainable.with_parameters(result.config, qdata)
    best_checkpoint = result.get_best_checkpoint("loss", "min")
    trainable.load_checkpoint(best_checkpoint)

    print(result.metrics_dataframe)

    original, forecast, metrics = trainable.eval_train()
    plot_scatter(original, forecast, metrics, result.path, name="train")

    original, forecast, metrics = trainable.eval_test()
    plot_scatter(original, forecast, metrics, result.path, name="test")

    # create a
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.plot(original, label="Real")
    ax.plot(forecast, label="Forecast")
    ax.legend()
    plt.title("Test Data Real vs Forecast")
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    save_to = Path(result.path) / "forecast.png"
    logger.info("Figure saved to %s", save_to)
    fig.savefig(save_to)
    plt.close(fig)
