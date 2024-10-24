import logging
import os
import numpy as np
import torch
import shutil
from pathlib import Path
from data.data import EarthquakeData
from ray import tune
from ray.air import Result
from ray.train import Checkpoint
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import Accuracy

from lstm.base import BaseTrainable
from lstm.model import LSTMModel
from lstm.plot import plot_confusion_matrix, plot_timeseries

import torch.nn as nn
import torch.optim as optim


logger = logging.getLogger(__name__)


class ClassificationTrainable(BaseTrainable):
    def setup_data(self):
        self.quantiles = self.config.get("quantiles", 4)
        assert self.lookback, "[lookback] cannot be None"
        one_hot, binned = self.qdata.categorical(self.quantiles)
        features = list(one_hot.columns)
        (target,) = self.qdata.targets
        one_hot["target"] = binned[f"{target}_binned"]
        sequences, targets = self.qdata.to_sequences(one_hot, self.lookback, features=features, targets=["target"])
        return self.qdata.split(sequences, targets[:, -1], self.test_size)

    def setup_loaders(self):
        self.x_train, self.x_test, self.y_train, self.y_test = self.setup_data()
        self.train_dataset = TensorDataset(self.x_train, self.y_train.to(torch.long))
        self.test_dataset = TensorDataset(self.x_test, self.y_test.to(torch.long))
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_model(self):
        return LSTMModel(
            self.lookback,
            self.quantiles,
            self.hidden_size,
            self.lstm_layers,
        ).to(self.device)

    def setup_model(self):
        self.model = self.get_model()
        # loss_types = {"cross_entropy": nn.CrossEntropyLoss}
        # loss_class = loss_types.get(self.loss_type, nn.CrossEntropyLoss)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

    def train_batch(self, input_batch, output_batch):
        return super().train_batch(input_batch, output_batch.flatten())

    def test_batch(self, input_batch, output_batch):
        input_batch, output_batch = input_batch.to(self.device), output_batch.to(self.device)
        output = self.model(input_batch)
        loss = self.criterion(output, output_batch.flatten())
        test_loss = loss.item() * input_batch.size(0)
        total_samples = input_batch.size(0)
        _, predicted = torch.max(output, 1)
        return test_loss, total_samples, (predicted == output_batch).sum().item()

    def eval(self, loader: DataLoader):
        self.model.eval()
        test_loss = 0
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for input_batch, output_batch in loader:
                b_loss, b_samples, b_corect = self.test_batch(input_batch, output_batch)
                test_loss += b_loss
                total_samples += b_samples
                correct_predictions += b_corect

        mean_test_loss = test_loss / total_samples
        accuracy = correct_predictions / total_samples
        return {
            "test_loss": test_loss,
            "mean_test_loss": mean_test_loss,
            "accuracy": accuracy * 100,
        }

    def pred_loader(self, loader: DataLoader):
        all_preds = []
        all_labels = []
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = self.model(X_batch.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        return all_preds, all_labels

    def test_result(self, result: Result, metric, mode):
        logger.info("Loading testing from config")
        best_checkpoint = result.get_best_checkpoint(metric, mode)
        self.load_checkpoint(best_checkpoint)

        print(result.path)
        print(result.metrics_dataframe)

        # Compute confusion matrix
        train_pred, train_y = self.pred_loader(self.train_loader)
        test_pred, test_y = self.pred_loader(self.test_loader)

        def target_idx(y, pred, idx):
            return y[:, idx], pred[:, idx]

        save_to = Path.home() / "plots" / self.qdata.hash / Path(result.path).stem
        shutil.copytree(result.path, save_to, dirs_exist_ok=True)
        # for idx, target in enumerate(self.qdata.targets):
        plot_confusion_matrix(train_y, train_pred, save_to / "train_confusion.png")
        plot_confusion_matrix(test_y, test_pred, save_to / "test_confusion.png")

        result.metrics_dataframe[["loss", "test_loss", "accuracy"]].plot(legend=True).get_figure().savefig(
            save_to / "metrics.png"
        )
