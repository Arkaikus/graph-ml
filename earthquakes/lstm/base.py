import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.data import EarthquakeData
from lstm.models.lstm_model import LSTMModel
from ray import tune
from ray.air import Result
from ray.train import Checkpoint
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import MeanAbsolutePercentageError

logger = logging.getLogger(__name__)


class BaseTrainable(tune.Trainable):
    def setup(self, config: dict, qdata: EarthquakeData):
        """takes hyperparameter values in the config argument"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.trial_id)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.qdata = qdata
        self.config = config

        self.lookback = config.get("lookback")
        self.test_size = config.get("test_size")
        self.batch_size = config["batch_size"]
        self.hidden_size = config.get("hidden_size", 1)
        self.lstm_layers = config.get("lstm_layers", 2)
        self.learning_rate = config["lr"]
        self.loss_type = config.get("loss_type", "mse")
        self.shuffle = config.get("shuffle", False)
        self.max_epochs = config.get("max_epochs", 100)
        self.epoch = 0
        self.patience = 5
        self.best_loss = np.inf
        self.done = False

        self.setup_loaders()
        self.setup_model()
        self.post_init()

    def post_init(self):
        pass

    def setup_data(self):
        assert self.lookback, "[lookback] cannot be None"
        sequences, targets = self.qdata.to_sequences(self.qdata.normalized_data, self.lookback)
        return self.qdata.split(sequences, targets, self.test_size)

    def setup_loaders(self):
        self.x_train, self.x_test, self.y_train, self.y_test = self.setup_data()
        self.train_dataset = TensorDataset(self.x_train, self.y_train[:, -1])
        self.test_dataset = TensorDataset(self.x_test, self.y_test[:, -1])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_model(self):
        return LSTMModel(
            self.lookback,
            len(self.qdata.targets),
            self.hidden_size,
            self.lstm_layers,
        ).to(self.device)

    def setup_model(self):
        self.model = self.get_model()
        loss_types = {"mse": nn.MSELoss, "huber": nn.HuberLoss, "mape": MeanAbsolutePercentageError, "mae": nn.L1Loss}
        loss_class = loss_types.get(self.loss_type, nn.MSELoss)
        self.criterion = loss_class().to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

    def is_done(self, epoch_loss):
        """Handles early stopping"""
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.patience = min(self.patience + 1, 10)
        elif not self.done:
            self.patience -= 1
            if self.patience <= 0:
                self.logger.info("Early stopping")
                return True

        return self.epoch >= self.max_epochs - 1

    def train_batch(self, input_batch, output_batch):
        """takes input and output batches and returns the output and loss"""
        self.optimizer.zero_grad()

        # forward pass
        output = self.model(input_batch.to(self.device))
        loss = self.criterion(output, output_batch.to(self.device))

        # backward pass and optimize
        loss.backward()
        self.optimizer.step()

        return output, loss.item()

    def train_epoch(self):
        """goes through all batches once, returns the loss in a metric dictionary"""
        self.model.train()
        epoch_loss = 0
        for input_batch, output_batch in self.train_loader:
            _, batch_loss = self.train_batch(input_batch, output_batch)
            epoch_loss += batch_loss

        mean_loss = epoch_loss / len(self.train_loader)
        return {
            "loss": epoch_loss,
            "mean_loss": mean_loss,
        }

    def test_batch(self, input_batch, output_batch):
        input_batch, output_batch = input_batch.to(self.device), output_batch.to(self.device)
        output = self.model(input_batch)
        loss = self.criterion(output, output_batch)
        return output, loss.item()

    def eval(self, loader: DataLoader):
        self.model.eval()
        test_loss = 0
        total_samples = 0

        with torch.no_grad():
            for input_batch, output_batch in loader:
                _, loss = self.test_batch(input_batch, output_batch)
                test_loss += loss * input_batch.size(0)  # Accumulate loss
                total_samples += input_batch.size(0)

        mean_test_loss = test_loss / total_samples
        return {
            "test_loss": test_loss,
            "mean_test_loss": mean_test_loss,
        }

    def step(self):
        self.epoch += 1
        # Training phase
        epoch_metrics = self.train_epoch()
        eval_metrics = self.eval(self.test_loader)
        self.done = self.is_done(epoch_metrics["mean_loss"])
        metrics = {
            "checkpoint_dir_name": "",
            "patience": self.patience,
            "done": self.done,
            **epoch_metrics,
            **eval_metrics,
        }
        logger.info("epoch metrics: %s eval_metrics: %s", epoch_metrics, eval_metrics)
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
        raise NotImplementedError

    @classmethod
    def test_result(cls, result: Result, metric, mode):
        raise NotImplementedError
