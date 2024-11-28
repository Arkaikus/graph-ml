import logging
import os
import pdb
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.air import Result
from ray.train import Checkpoint
from torch.utils.data import DataLoader, TensorDataset

from data.data import EarthquakeData
from lstm.model import LSTMModel
from lstm.plot import plot_confusion_matrix, plot_roc_auc

logger = logging.getLogger(__name__)

import torch.nn as nn


class ClassificationTrainable(tune.Trainable):
    def setup(self, config: dict, qdata: EarthquakeData):
        """takes hyperparameter values in the config argument"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.trial_id)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.qdata = qdata
        self.lookback = config.get("lookback")
        self.test_size = config.get("test_size")
        self.batch_size = config.get("batch_size", 32)
        self.hidden_size = config.get("hidden_size", 1)
        self.lstm_layers = config.get("lstm_layers", 2)
        self.learning_rate = config.get("lr", 0.001)
        self.max_epochs = config.get("max_epochs", 100)
        self.quantiles = config.get("quantiles", 4)
        self.epoch = 0
        self.patience = 5
        self.best_loss = np.inf
        self.done = False

        assert self.lookback, "[lookback] cannot be None"

        one_hot, binned = qdata.categorical(self.quantiles)
        features = list(one_hot.columns)
        (target,) = qdata.targets
        one_hot["target"] = binned[f"{target}_binned"]
        sequences, targets = qdata.to_sequences(one_hot, self.lookback, features=features, targets=["target"])
        x_train, x_test, y_train, y_test = qdata.split(
            sequences,
            targets[:, -1],
            test_size=self.test_size,
            shuffle=True,
            stratify=one_hot["target"][self.lookback :],
        )

        self.train_dataset = TensorDataset(x_train, y_train.to(torch.long))
        self.test_dataset = TensorDataset(x_test, y_test.to(torch.long))
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        self.model = LSTMModel(self.lookback, self.quantiles, self.hidden_size, self.lstm_layers).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        total_samples = 0
        correct_predictions = 0
        for input_batch, output_batch in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(input_batch.to(self.device))
            output_batch = output_batch.to(self.device).view(-1)
            loss = self.criterion(output, output_batch.to(self.device))
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total_samples += input_batch.size(0)
            correct_predictions += (predicted.detach().cpu() == output_batch.detach().cpu()).sum().item()
            epoch_loss += loss.item()
        mean_loss = epoch_loss / len(self.train_loader)
        accuracy = correct_predictions / float(total_samples)
        return {"loss": epoch_loss, "mean_loss": mean_loss, "train_accuracy": accuracy}

    def eval_loader(self, loader: DataLoader):
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in loader:
                outputs = self.model(x_batch.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                all_preds.append(predicted.cpu())
                all_labels.append(y_batch.cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        return all_preds, all_labels

    def eval(self, loader: DataLoader):
        test_loss = 0
        total_samples = 0
        correct_predictions = 0
        self.model.eval()
        with torch.no_grad():
            for input_batch, output_batch in loader:
                input_batch, output_batch = input_batch.to(self.device), output_batch.to(self.device).view(-1)
                output = self.model(input_batch)
                loss = self.criterion(output, output_batch)
                test_loss += loss.item() * input_batch.size(0)
                total_samples += input_batch.size(0)
                _, predicted = torch.max(output.data, 1)
                correct_predictions += (predicted.cpu() == output_batch.cpu()).sum().item()
        mean_test_loss = test_loss / total_samples
        accuracy = correct_predictions / float(total_samples)
        return {"test_loss": test_loss, "mean_test_loss": mean_test_loss, "accuracy": accuracy}

    def step(self):
        self.epoch += 1
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

    def is_done(self, epoch_loss):
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.patience = min(self.patience + 1, 10)
        elif not self.done:
            self.patience -= 1
            if self.patience <= 0:
                self.logger.info("Early stopping")
                return True
        return self.epoch >= self.max_epochs - 1

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
            # model_state, optimizer_state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model_state, optimizer_state = torch.load(checkpoint_path)
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)

    def test_result(self, result: Result, metric, mode):
        logger.info("Loading testing from config")
        best_checkpoint = result.get_best_checkpoint(metric, mode)
        self.load_checkpoint(best_checkpoint)

        print(result.path)
        print(result.metrics_dataframe)

        train_pred, train_y = self.eval_loader(self.train_loader)
        test_pred, test_y = self.eval_loader(self.test_loader)

        save_to = Path.cwd() / "plots" / self.qdata.hash / Path(result.path).stem
        shutil.copytree(result.path, save_to, dirs_exist_ok=True)
        plot_confusion_matrix(train_y, train_pred, save_to / "train_confusion.png")
        plot_confusion_matrix(test_y, test_pred, save_to / "test_confusion.png")
        plot_roc_auc(train_y, train_pred, self.quantiles, save_to / "roc_auc_train.png")
        plot_roc_auc(test_y, test_pred, self.quantiles, save_to / "roc_auc.png")
