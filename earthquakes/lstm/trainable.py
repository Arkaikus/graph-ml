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
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.trial_id)
        """takes hyperparameter values in the config argument"""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        batch_size = config["batch_size"]
        sequence_size = config.get("sequence_size")
        test_size = config.get("test_size")
        self.epoch = 0

        assert sequence_size, "[sequence_size] cannot be None"

        # prepare train test split of data
        x_train, x_test, y_train, y_test = EarthquakeData.train_test_split(qdata, sequence_size, test_size)
        self.x_train = torch.Tensor(x_train).to(torch.float32)
        self.x_test = torch.Tensor(x_test).to(torch.float32)
        self.y_train = torch.Tensor(y_train).to(torch.float32)
        self.y_test = torch.Tensor(y_test).to(torch.float32)
        train_dataset = TensorDataset(self.x_train, self.y_train)
        test_dataset = TensorDataset(self.x_test, self.y_test)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        feature_size = x_train.shape[2]  # 0:batch, 1:seq 2:features
        hidden_size = config.get("hidden_size", 1)
        lstm_layers = config.get("layers", 1)

        self.model = LSTMModel(feature_size, hidden_size, lstm_layers, 1).to(self.device)
        self.loss = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=config["lr"])

    def step(self):
        epoch_loss = 0
        self.model.train()
        for inputs, outputs in self.train_dataloader:
            forecast = self.model(inputs.to(torch.float32).to(self.device))
            loss = self.loss(forecast, outputs.to(torch.float32).to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(self.train_dataloader)
        _, _, metrics = self.forecast()
        self.epoch += 1
        return {"epoch": self.epoch, "loss": epoch_loss, "mean_loss": mean_loss, "checkpoint_dir_name": "", **metrics}

    def save_checkpoint(self, checkpoint_dir: str) -> torch.Dict | None:
        self.logger.info("Saving model and optimizer to %s", checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pth")
        state = (self.model.state_dict(), self.optimizer.state_dict())
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint: Checkpoint):
        self.logger.info("Loading checkpoint %s", checkpoint)
        with checkpoint.as_directory() as loaded_checkpoint_dir:
            logger.info("Loading checkpoint %s", loaded_checkpoint_dir)
            checkpoint_path = os.path.join(loaded_checkpoint_dir, "checkpoint.pth")
            model_state, optimizer_state = torch.load(checkpoint_path)
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)

    def forecast(self):
        original = []
        forecast = []
        self.model.eval()
        with torch.no_grad():
            for inputs, outputs in self.test_dataloader:
                original.append(outputs.numpy())
                model_output = self.model(inputs.to(torch.float32).to(self.device))
                forecast.append(model_output.detach().cpu().numpy())

        original, forecast = np.vstack(original), np.vstack(forecast)
        test_loss = mean_squared_error(original, forecast)
        test_mae = mean_absolute_error(original, forecast)
        test_rmse = np.sqrt(test_loss)
        _original, _forecast = original.squeeze(), forecast.squeeze()
        test_r2 = r2_score(_original, _forecast)
        metrics = {
            "test_loss": test_loss,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "test_mae": test_mae,
        }
        logger.info("Test metrics %s", metrics)
        return (original, forecast, metrics)


def test_result(result: Result, qdata: EarthquakeData):
    logger.info("Loading testing from config")
    trainable = LSTMTrainable.with_parameters(result.config, qdata)
    best_checkpoint = result.get_best_checkpoint("loss", "min")
    trainable.load_checkpoint(best_checkpoint)

    print(result.metrics_dataframe)

    original, forecast, metrics = trainable.forecast()
    # target_scaler = scalers.get(target)
    # if target_scaler:
    #     original = target_scaler.inverse_transform(original)
    #     forecast = target_scaler.inverse_transform(forecast)

    R2 = metrics["test_r2"]
    MSE = metrics["test_loss"]
    RMSE = metrics["test_rmse"]
    MAE = metrics["test_mae"]

    logger.info("Best trial R2 %s", metrics["test_r2"])
    logger.info("Best trial MSE: %s", metrics["test_loss"])
    logger.info("Best trial RMSE: %s", metrics["test_rmse"])
    logger.info("Best trial MAE: %s", metrics["test_mae"])

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
        f"R2: {R2:.2f}\nRMSE: {RMSE:.2f}\nMAE: {MAE:.2f}",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=12,
    )

    save_to = Path(result.path) / "forecast.png"
    logger.info("Figure saved to %s", save_to)
    g.figure.savefig(save_to)
