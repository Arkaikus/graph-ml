"""Classification trainable for LSTM (binned target prediction)."""

import json
import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from lstm.classification.losses import FocalLoss, LabelSmoothingCrossEntropy
from lstm.classification.plots import plot_confusion_matrix, plot_roc_auc
from lstm.model import LSTMModel
from lstm.trainable.base import BaseLSTMTrainable
from ray.air import Result
from sklearn import metrics
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ClassificationTrainable(BaseLSTMTrainable):
    """Ray Tune trainable for LSTM classification (quantile-binned targets)."""

    def setup_data(self) -> None:
        assert self.lookback, "[lookback] cannot be None"
        self.quantiles = self.config.get("quantiles", 4)
        network_features = self.config.get("network_features", [])
        keep_node = bool(network_features)  # retain node for graph features when networkx enabled
        self.one_hot, self.binned = self.qdata.categorical(self.quantiles, keep_node=keep_node)
        features = list(self.one_hot.columns)
        (self.target,) = self.qdata.targets
        self.one_hot["target"] = self.binned[f"{self.target}_binned"]

        sequences, targets = self.qdata.to_sequences(
            self.one_hot,
            self.lookback,
            features=features,
            targets=["target"],
            network_features=network_features,
            network_lookback=self.config.get("network_lookback", 5),
        )
        x_train, x_test, y_train, y_test, x_val, y_val = self.qdata.split(
            sequences,
            targets[:, -1],
            test_size=self.test_size,
            shuffle=False,
            temporal=True,
            val_ratio=self.config.get("val_ratio", 0.15),
        )
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train.to(torch.long)
        self.y_test = y_test.to(torch.long)
        self.y_val = y_val.to(torch.long)

    def setup_model(self) -> None:
        # Use actual sequence dimension to avoid mismatch with to_sequences output
        # (len(one_hot.columns) can diverge due to target column, network features, etc.)
        num_features = self.x_train.size(1)
        dropout = self.config.get("dropout", 0.0)
        use_attention = self.config.get("use_attention", False)
        self.model = LSTMModel(
            lookback=self.lookback,
            outputs=self.quantiles,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            num_features=num_features,
            dropout=dropout,
            use_attention=use_attention,
        ).to(self.device)
        loss_type = self.config.get("loss_type", "cross_entropy")
        loss_map = {
            "cross_entropy": nn.CrossEntropyLoss(),
            "focal": FocalLoss(gamma=2.0),
            "label_smoothing": LabelSmoothingCrossEntropy(smoothing=0.1),
        }
        self.criterion = loss_map.get(loss_type, nn.CrossEntropyLoss()).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def prepare_target_for_loss(self, target_batch: torch.Tensor) -> torch.Tensor:
        return target_batch.view(-1)

    def batch_metrics(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        _, predicted = torch.max(output.data, 1)
        target_flat = target.view(-1)
        correct = (predicted.detach().cpu() == target_flat.detach().cpu()).sum().item()
        return {"correct": correct, "total": target.size(0)}

    def eval_loader(self, loader: DataLoader) -> tuple:
        """Return (predictions, labels) for plotting."""
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
        all_labels = torch.cat(all_labels).view(-1).numpy()
        return all_preds, all_labels

    def plot(self, save_to: Path) -> None:
        train_pred, train_y = self.eval_loader(self.train_loader)
        test_pred, test_y = self.eval_loader(self.test_loader)

        plot_confusion_matrix(train_y, train_pred, save_to / "train_confusion.png")
        plot_confusion_matrix(test_y, test_pred, save_to / "test_confusion.png")
        plot_roc_auc(train_y, train_pred, self.quantiles, save_to / "roc_auc_train.png")
        plot_roc_auc(test_y, test_pred, self.quantiles, save_to / "roc_auc.png")

        self.binned[f"{self.target}_binned"].plot(kind="hist", title="Target binned", sharex=True)
        plt.gcf().savefig(save_to / f"{self.target}_binned.png")

        json.dump(
            {
                "accuracy": float(metrics.accuracy_score(test_y, test_pred)),
                "roc_auc_score": float(metrics.roc_auc_score(test_y, test_pred)),
                "precision_score": float(metrics.precision_score(test_y, test_pred)),
                "recall_score": float(metrics.recall_score(test_y, test_pred)),
                "f1_score": float(metrics.f1_score(test_y, test_pred)),
            },
            open(save_to / "metrics.json", "w"),
        )

    def test_result(self, result: Result, metric: str, mode: str) -> None:
        logger.info("Loading testing checkpoint")
        best_checkpoint = result.get_best_checkpoint(metric, mode)
        self.load_checkpoint(best_checkpoint)

        print(result.path)
        print(result.metrics_dataframe)

        save_to = Path.cwd() / "plots" / self.qdata.hash / Path(result.path).stem
        shutil.copytree(result.path, save_to, dirs_exist_ok=True)
        self.plot(save_to)
