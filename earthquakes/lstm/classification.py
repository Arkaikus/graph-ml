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
    def post_init(self):
        self.quantiles = self.config.get("quantiles", 4)

    def setup_data(self):
        assert self.lookback, "[lookback] cannot be None"
        one_hot, binned = self.qdata.categorical(self.quantiles)
        features = list(one_hot.columns)
        (target,) = self.qdata.targets
        one_hot["target"] = binned[f"{target}_binned"]
        sequences, targets = self.qdata.to_sequences(one_hot, self.lookback, features=features, targets=["target"])
        return self.qdata.split(sequences, targets, self.test_size)

    def setup_model(self):
        self.model = self.get_model()
        loss_types = {"cross_entropy": nn.CrossEntropyLoss, "nll": nn.NLLLoss}
        loss_class = loss_types.get(self.loss_type, nn.CrossEntropyLoss)
        self.criterion = loss_class().to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

    def eval(self):
        self.model.eval()
        test_loss = 0
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for input_batch, output_batch in self.test_loader:
                input_batch, output_batch = input_batch.to(self.device), output_batch.to(self.device)
                output = self.model(input_batch)
                loss = self.criterion(output, output_batch)
                test_loss += loss.item() * input_batch.size(0)
                total_samples += input_batch.size(0)
                _, predicted = torch.max(output, 1)
                correct_predictions += (predicted == output_batch).sum().item()

        mean_test_loss = test_loss / total_samples
        accuracy = correct_predictions / total_samples
        return {
            "test_loss": test_loss,
            "mean_test_loss": mean_test_loss,
            "accuracy": accuracy,
        }

    def forecast(self):
        self.model.eval()
        with torch.no_grad():
            train_output = torch.cat([self.model(ib.to(self.device)).cpu() for ib, _ in self.train_loader])
            test_output = torch.cat([self.model(ib.to(self.device)).cpu() for ib, _ in self.test_loader])

        return (
            (self.y_train[:, -1].numpy(), train_output.detach().cpu().numpy()),
            (self.y_test[:, -1].numpy(), test_output.detach().cpu().numpy()),
        )

    @staticmethod
    def test_result(result: Result, qdata: EarthquakeData, metric, mode):
        logger.info("Loading testing from config")
        trainable_cls = tune.with_parameters(ClassificationTrainable, qdata=qdata)
        trainable: ClassificationTrainable = trainable_cls(config=result.config)
        best_checkpoint = result.get_best_checkpoint(metric, mode)
        trainable.load_checkpoint(best_checkpoint)

        print(result.path)
        print(result.metrics_dataframe)

        ((train_y, train_pred), (test_y, test_pred)) = trainable.forecast()

        def target_idx(y, pred, idx):
            return y[:, idx], pred[:, idx]

        save_to = Path.home() / "plots" / qdata.hash / Path(result.path).stem
        shutil.copytree(result.path, save_to, dirs_exist_ok=True)
        for idx, target in enumerate(trainable.qdata.targets):
            plot_confusion_matrix(*target_idx(train_y, train_pred, idx), save_to / f"{target}_train_confusion.png")
            plot_confusion_matrix(*target_idx(test_y, test_pred, idx), save_to / f"{target}_test_confusion.png")

        result.metrics_dataframe[["loss", "test_loss", "accuracy"]].plot(legend=True).get_figure().savefig(
            save_to / "metrics.png"
        )
