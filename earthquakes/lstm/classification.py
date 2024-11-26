import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.data import EarthquakeData
from lstm.base import BaseTrainable
from lstm.models.lstm_model import LSTMModel
from lstm.plot import plot_confusion_matrix, plot_roc_auc
from ray import tune
from ray.air import Result
from ray.train import Checkpoint
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import Accuracy

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
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_batch(self, input_batch, output_batch):
        return super().train_batch(input_batch, output_batch.flatten())

    def train_epoch(self):
        """goes through all batches once, returns the loss in a metric dictionary"""
        self.model.train()
        epoch_loss = 0
        total_samples = 0
        correct_predictions = 0
        for input_batch, output_batch in self.train_loader:
            output, batch_loss = self.train_batch(input_batch, output_batch)
            _, predicted = torch.max(output, 1)
            total_samples += input_batch.size(0)
            correct_predictions += (predicted.detach().cpu() == output_batch.detach().view(-1).cpu()).sum().item()
            epoch_loss += batch_loss

        mean_loss = epoch_loss / len(self.train_loader)
        accuracy = correct_predictions / float(total_samples)
        return {
            "loss": epoch_loss,
            "mean_loss": mean_loss,
            "train_accuracy": accuracy,
        }

    def test_batch(self, input_batch, output_batch):
        output, loss = super().test_batch(input_batch, output_batch.flatten())
        test_loss = loss * input_batch.size(0)
        total_samples = input_batch.size(0)
        _, predicted = torch.max(output.data, 1)
        correct_predictions = (predicted.cpu() == output_batch.view(-1).cpu()).sum().item()
        return test_loss, total_samples, correct_predictions

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
        accuracy = correct_predictions / float(total_samples)
        return {
            "test_loss": test_loss,
            "mean_test_loss": mean_test_loss,
            "accuracy": accuracy,
        }

    def eval_loader(self, loader: DataLoader):
        all_preds = []
        all_labels = []

        self.model.eval()  # Set self.model to evaluation mode
        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = self.model(X_batch.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        return np.stack(all_preds), np.concatenate(all_labels)

    def test_result(self, result: Result, metric, mode):
        logger.info("Loading testing from config")
        best_checkpoint = result.get_best_checkpoint(metric, mode)
        self.load_checkpoint(best_checkpoint)

        print(result.path)
        print(result.metrics_dataframe)

        # Compute confusion matrix
        train_pred, train_y = self.eval_loader(self.train_loader)
        test_pred, test_y = self.eval_loader(self.test_loader)

        save_to = Path.home() / "plots" / self.qdata.hash / Path(result.path).stem
        shutil.copytree(result.path, save_to, dirs_exist_ok=True)
        # for idx, target in enumerate(self.qdata.targets):
        plot_confusion_matrix(train_y, train_pred, save_to / "train_confusion.png")
        plot_confusion_matrix(test_y, test_pred, save_to / "test_confusion.png")

        save_metrics = save_to / "metrics.png"
        result.metrics_dataframe[["loss", "test_loss", "accuracy"]].plot(legend=True).get_figure().savefig(save_metrics)

        plot_roc_auc(train_y, train_pred, self.quantiles, save_to / "roc_auc_train.png")
        plot_roc_auc(test_y, test_pred, self.quantiles, save_to / "roc_auc.png")
