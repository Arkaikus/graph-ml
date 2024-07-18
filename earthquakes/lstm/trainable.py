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
from torch.utils.data import DataLoader

sns.set_theme(style="darkgrid")


from .dataset import create_sequences
from .model import LSTMModel

logger = logging.getLogger(__name__)


class LSTMTrainable(tune.Trainable):
    @classmethod
    def with_parameters(cls, config, data, target) -> "LSTMTrainable":
        return tune.with_parameters(cls, data=data, target=target)(config=config)

    def setup(self, config: dict, data: pd.DataFrame, target: str):
        logging.basicConfig(level=logging.INFO)
        self.config = config
        self.logger = logging.getLogger(self.trial_id)
        """takes hyperparameter values in the config argument"""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        sequence_size = config.get("sequence_size")
        test_size = config.get("test_size")

        assert target, "[target] cannot be None"
        assert sequence_size, "[sequence_size] cannot be None"

        # prepare train test split of data
        self.train_dataset, self.test_dataset = create_sequences(data, target, sequence_size, test_size)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config["batch_size"], shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.config["batch_size"], shuffle=False)
        feature_size = data.shape[1]  # 0:seq 1:features
        hidden_size = config.get("hidden_size", 1)
        lstm_layers = config.get("layers", 1)

        self.model = LSTMModel(feature_size, hidden_size, lstm_layers, 1).to(self.device)
        self.loss = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=config["lr"])

    def step(self):
        epoch_loss = 0
        for inputs, outputs in self.train_dataloader:
            forecast = self.model(inputs.to(torch.float32).to(self.device))
            loss = self.loss(forecast, outputs.to(torch.float32).to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(self.train_dataloader)
        return {"loss": epoch_loss, "mean_loss": mean_loss, "checkpoint_dir_name": ""}

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

        logger.info("Loaded model %s", self.model)

        with torch.no_grad():
            for data in self.test_dataloader:
                inputs, outputs = data
                original.append(outputs)
                model_output = self.model(inputs.to(torch.float32).to(self.device))
                forecast.append(model_output.cpu())

        return np.vstack(original), np.vstack(forecast)


def test_result(result: Result, data: pd.DataFrame, scalers: dict, target: str):
    logger.info("Loading testing from config")
    trainable = LSTMTrainable.with_parameters(result.config, data, target)
    best_checkpoint = result.get_best_checkpoint("loss", "min")
    trainable.load_checkpoint(best_checkpoint)

    print(result.metrics_dataframe)

    original, forecast = trainable.forecast()
    # target_scaler = scalers.get(target)
    # if target_scaler:
    #     original = target_scaler.inverse_transform(original)
    #     forecast = target_scaler.inverse_transform(forecast)

    R2 = r2_score(original.squeeze(), forecast.squeeze())
    MSE = mean_squared_error(original.squeeze(), forecast.squeeze())
    MAE = mean_absolute_error(original.squeeze(), forecast.squeeze())
    logger.info("Best trial R2 %s", R2)
    logger.info("Best trial MSE: %s", MSE)
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

    save_to = Path(result.path) / "forecast.png"
    logger.info("Figure saved to %s", save_to)
    g.figure.savefig(save_to)
