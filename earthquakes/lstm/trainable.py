import logging
import os
import pandas as pd
import torch
from pathlib import Path
from ray import tune
from ray.air import Result
from ray.train import Checkpoint
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

from .dataset import MovingWindowDataset
from .model import LSTMModel

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(os.getcwd()) / "checkpoints"


class LSTMTrainable(tune.Trainable):
    def setup(self, config: dict):
        """takes hyperparameter values in the config argument"""
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        data: pd.DataFrame = config.get("data")  # Load your data here
        target: str = config.get("target")  # Set your target column name
        seq_length = config.get("window_size")
        dataset = MovingWindowDataset(data, target, seq_length, device)
        self.dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

        feature_size = data.shape[1]  # 0:seq 1:features
        hidden_size = config.get("hidden_size", 1)
        num_layers = config.get("num_layers", 1)

        self.model = LSTMModel(feature_size, hidden_size, num_layers, 1).to(device)
        self.loss = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=config["lr"])

    def step(self):

        epoch_loss = 0
        for inputs, targets in self.dataloader:
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(self.dataloader)
        return {"loss": epoch_loss, "mean_loss": mean_loss}

    def save_checkpoint(self, checkpoint_dir: str) -> torch.Dict | None:
        logger.info("Saving checkpoint to %s", checkpoint_dir)
        state = (self.model.state_dict(), self.optimizer.state_dict())
        torch.save(state, checkpoint_dir)

    def load_checkpoint(self, checkpoint: Checkpoint):
        with checkpoint.as_directory() as loaded_checkpoint_dir:
            logger.info("Loading checkpoint %s", loaded_checkpoint_dir)
            checkpoint_path = os.path.join(loaded_checkpoint_dir, "checkpoint.pth")
            model_state, optimizer_state = torch.load(checkpoint_path)
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)


def test_model(result: Result, test_data: pd.DataFrame, target: str):
    logger.info("Loading testing from config")
    seq_length = result.config["window_size"]
    batch_size = result.config["batch_size"]

    assert seq_length, "seq_length is empty"
    testset = MovingWindowDataset(test_data, target, seq_length)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    feature_size = test_data.shape[1]  # 0:seq 1:features
    hidden_size = result.config["hidden_size"]
    num_layers = result.config["num_layers"]
    assert hidden_size, "hiden_size is empty"
    assert num_layers, "num_layers is empty"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trained_model = LSTMModel(feature_size, hidden_size, num_layers, 1).to(device)

    checkpoint_path = Path(result.checkpoint.to_directory()) / "checkpoint.pth"
    model_state, optimizer_state = torch.load(checkpoint_path)
    trained_model.load_state_dict(model_state)

    forecast = []
    with torch.no_grad():
        for data in testloader:
            sequences, targets = data
            sequences, targets = sequences.to(device), targets.to(device)
            (output,) = trained_model(sequences)
            forecast.append(output)

    logger.info("Best trial testset MSE: %s", mean_squared_error(testset.targets, forecast))
