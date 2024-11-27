import logging
import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from data.store import Store
from gensim.models import Word2Vec
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import ASHAScheduler
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch.utils.data import DataLoader, TensorDataset

from .edge_splitter import EdgeSplitter
from .model import SimpleNN

logger = logging.getLogger(__name__)


def train_embedding(graph, name, **n2v_kwargs) -> Word2Vec:
    """trains and saves to disk a word2vec trained model"""
    store = Store("embeddings")
    data = store.load(name)
    if not data:
        data = Node2Vec(graph, **n2v_kwargs).fit()
        store.save(data, name)

    return data


class GNNTrainable(Trainable):
    def setup(self, config):
        input_dim = config["input_dim"]
        hidden_dim1 = config["hidden_dim1"]
        hidden_dim2 = config["hidden_dim2"]
        self.model = SimpleNN(input_dim, hidden_dim1, hidden_dim2)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])

        dataset = TensorDataset(
            torch.tensor(config["train_embeddings"], dtype=torch.float32),
            torch.tensor(config["train_labels"], dtype=torch.float32),
        )
        self.dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        self.test_embeddings = torch.tensor(self.config["test_embeddings"], dtype=torch.float32)
        self.test_labels = torch.tensor(self.config["test_labels"], dtype=torch.float32)

    def step(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for batch_embeddings, batch_labels in self.dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(batch_embeddings).squeeze()
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)

        train_accuracy = correct_predictions / total_predictions

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.test_embeddings).squeeze()
            predicted = (outputs > 0.5).float()
            correct_predictions = (predicted == self.test_labels).sum().item()
            total_predictions = self.test_labels.size(0)
            test_accuracy = correct_predictions / total_predictions

        return {
            "loss": total_loss / len(self.dataloader),
            "train_accuracy": train_accuracy,
            "accuracy": test_accuracy,
        }

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pth")
        state = (self.model.state_dict(), self.optimizer.state_dict())
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint):
        with checkpoint.as_directory() as loaded_checkpoint_dir:
            logger.info("Loading checkpoint %s", loaded_checkpoint_dir)
            checkpoint_path = os.path.join(loaded_checkpoint_dir, "checkpoint.pth")
            model_state, optimizer_state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)


def tune_neural_network(
    embeddings: tuple, labels: tuple, samples, epochs=20, experiment_path=None, save_to="tuning_results.csv"
):
    """trains and returns the best neural network model using hyperparameter tuning"""
    train_vectors, test_vectors = embeddings
    train_labels, test_labels = labels
    input_dim = train_vectors.shape[1]
    config = {
        "input_dim": input_dim,
        "hidden_dim1": tune.choice([64, 128, 256]),
        "hidden_dim2": tune.choice([32, 64, 128]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "train_embeddings": train_vectors,
        "test_embeddings": test_vectors,
        "train_labels": train_labels,
        "test_labels": test_labels,
    }

    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=epochs, grace_period=1, reduction_factor=2)
    trainable = tune.with_resources(GNNTrainable, resources={"cpu": 8, "gpu": 1})
    if experiment_path:
        tuner = tune.Tuner.restore(
            path=experiment_path.absolute().as_posix(),
            trainable=trainable,  # Replace with your actual trainable class
            resume_unfinished=True,  # Continue unfinished trials
            resume_errored=True,  # Resume errored trials
            param_space=config,
        )
    else:
        tuner = tune.Tuner(
            trainable,
            param_space=config,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=samples,
                max_concurrent_trials=None,
            ),
        )

    analysis = tuner.fit()
    result = analysis.get_best_result("accuracy", "max")
    trainable = GNNTrainable(result.config)
    best_checkpoint = result.get_best_checkpoint("accuracy", "max")
    trainable.load_checkpoint(best_checkpoint)

    result_df = analysis.get_dataframe()
    drop_columns = [c for c in result_df.columns if c.lower().endswith(("embeddings", "labels"))]
    drop_columns.extend(
        [
            "done",
            "training_iteration",
            "trial_id",
            "date",
            "timestamp",
            "time_this_iter_s",
            "time_total_s",
            "pid",
            "hostname",
            "node_ip",
            "time_since_restore",
            "iterations_since_restore",
            "logdir",
        ]
    )

    result_df = analysis.get_dataframe().drop(columns=drop_columns).sort_values(by="accuracy", ascending=False)
    result_df.columns = [
        c.replace("_", " ").replace("Config/", "").replace("config/", "").title() for c in result_df.columns
    ]
    result_df.to_csv(save_to, index=False)
    result_df.to_latex(str(save_to).replace(".csv", ".tex"), index=False)

    return trainable.model


def tune_link_prediction(
    file: str,
    test_size,
    samples,
    experiment_path=None,
    embedder_class=HadamardEmbedder,
    **n2v_kwargs,
):
    logger.info("Link prediction with neural networks")
    file_path = Path(file)
    assert file_path.exists()
    edge_list = pd.read_csv(file_path, dtype=str)
    assert "source" in edge_list.columns
    assert "target" in edge_list.columns

    logger.info("Splitting data into train and test with EdgeSplitter")
    graph: nx.DiGraph = nx.from_pandas_edgelist(edge_list, create_using=nx.DiGraph)

    logger.info("Training/loading node2vec model")
    n2v_params = ",".join(f"{k}:{v}" for k, v in n2v_kwargs.items())
    n2v_name = file_path.stem + n2v_params
    n2v_train = train_embedding(graph, n2v_name, **n2v_kwargs)

    test_splitter = EdgeSplitter()
    train_samples, train_labels, test_samples, test_labels = test_splitter.train_test_split(graph, test_size=test_size)

    logger.info("Querying %s vectors", embedder_class.__name__)
    embeddings = embedder_class(keyed_vectors=n2v_train.wv)
    train_vectors = np.array([embeddings[u, v] for u, v in train_samples])
    test_vectors = np.array([embeddings[u, v] for u, v in test_samples])

    logger.info("Training Neural Network")
    rfn = f"{file_path.stem}_{embedder_class.__name__}_test:{test_size}_{n2v_params}"
    save_to = Path.home() / "earthquakes" / "plots" / file_path.stem
    save_to.mkdir(exist_ok=True, parents=True)
    model = tune_neural_network(
        (train_vectors, test_vectors),
        (train_labels, test_labels),
        samples,
        experiment_path=experiment_path,
        save_to=save_to / f"{rfn}.csv",
    )

    model: nn.Module
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_vectors, dtype=torch.float32)
        y_pred = model(test_tensor).squeeze().numpy()
        y_pred = (y_pred > 0.5).astype(int)

    # Plot ROC AUC
    fpr, tpr, _ = roc_curve(test_labels, y_pred)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % metrics.roc_auc_score(test_labels, y_pred),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.gcf().savefig(save_to / f"roc_auc_{rfn}.png")

    # Plot Confusion Matrix
    cm = confusion_matrix(test_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # plt.show()
    plt.gcf().savefig(save_to / f"confusion_matrix_{rfn}.png")
    roc_auc_score = metrics.roc_auc_score(test_labels, y_pred)
    precision_score = metrics.precision_score(test_labels, y_pred)
    recall_score = metrics.recall_score(test_labels, y_pred)
    f1_score = metrics.f1_score(test_labels, y_pred)

    # Calculate True Positive Rate, True Negative Rate, and Overall Accuracy
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)  # True Positive Rate
    tnr = tn / (tn + fp)  # True Negative Rate
    accuracy = (tp + tn) / (tp + tn + fp + fn)  # Overall Accuracy

    # Plot Bar Chart
    plt.figure()
    metrics_names = ["True Positive Rate", "True Negative Rate", "Accuracy"]
    metrics_values = [tpr, tnr, accuracy]
    bars = plt.bar(metrics_names, metrics_values, color=["blue", "green", "red"])
    plt.ylim([0, 1])
    plt.ylabel("Rate")
    plt.title("Performance Metrics")

    # Add percentage value on top of each bar
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2%}", ha="center", va="bottom")

    plt.gcf().savefig(save_to / f"performance_metrics_{rfn}.png")

    json.dump(
        {
            "accuracy": accuracy,
            "roc_auc_score": roc_auc_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
        },
        open(save_to / f"metrics_{rfn}.json", "w"),
    )

    return (roc_auc_score, precision_score, recall_score, f1_score)
