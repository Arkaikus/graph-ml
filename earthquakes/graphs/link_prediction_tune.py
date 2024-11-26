import logging
import os
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


class TrainSimpleNN(Trainable):
    def setup(self, config):
        input_dim = config["input_dim"]
        hidden_dim1 = config["hidden_dim1"]
        hidden_dim2 = config["hidden_dim2"]
        self.model = SimpleNN(input_dim, hidden_dim1, hidden_dim2)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])

        dataset = TensorDataset(
            torch.tensor(config["embeddings"], dtype=torch.float32),
            torch.tensor(config["labels"], dtype=torch.float32),
        )
        self.dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

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

        accuracy = correct_predictions / total_predictions
        return {"loss": total_loss / len(self.dataloader), "accuracy": accuracy}

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


def tune_neural_network(embeddings, labels, samples, epochs=20, save_to="tuning_results.csv"):
    """trains and returns the best neural network model using hyperparameter tuning"""
    input_dim = embeddings.shape[1]
    config = {
        "input_dim": input_dim,
        "hidden_dim1": tune.choice([64, 128, 256]),
        "hidden_dim2": tune.choice([32, 64, 128]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "embeddings": embeddings,
        "labels": labels,
    }

    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(TrainSimpleNN, resources={"cpu": 8, "gpu": 1}),
        param_space=config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=samples,
            max_concurrent_trials=None,
        ),
    )
    analysis = tuner.fit()

    result = analysis.get_best_result("loss", "min")
    trainable = TrainSimpleNN(result.config)
    best_checkpoint = result.get_best_checkpoint("loss", "min")
    trainable.load_checkpoint(best_checkpoint)

    result_df = analysis.get_dataframe()
    drop_columns = [c for c in result_df.columns if c.lower().endswith(("embeddings", "labels"))]

    result_df = analysis.get_dataframe().drop(columns=drop_columns)
    result_df.to_csv(save_to, index=False)
    result_df.to_latex(str(save_to).replace(".csv", ".tex"), index=False)

    return trainable.model


def tune_link_prediction(file: str, test_size, samples, embedder_class=HadamardEmbedder, **n2v_kwargs):
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
    save_to = Path.home() / "plots" / file_path.stem
    save_to.mkdir(exist_ok=True, parents=True)
    model = tune_neural_network(train_vectors, train_labels, samples, save_to=save_to / f"{rfn}.csv")

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

    return (
        metrics.roc_auc_score(test_labels, y_pred),
        metrics.precision_score(test_labels, y_pred),
        metrics.recall_score(test_labels, y_pred),
        metrics.f1_score(test_labels, y_pred),
    )
