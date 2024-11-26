import logging
from pathlib import Path

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
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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


def train_neural_network(embeddings, labels, epochs=10, batch_size=32, learning_rate=0.001):
    """trains and saves to disk a neural network model"""
    input_dim = embeddings.shape[1]
    model = SimpleNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(epochs)):
        model.train()
        for batch_embeddings, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_embeddings).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model


def run_link_prediction(file: str, test_size, embedder_class=HadamardEmbedder, **n2v_kwargs):
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
    model = train_neural_network(train_vectors, train_labels)

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_vectors, dtype=torch.float32)
        y_pred = model(test_tensor).squeeze().numpy()
        y_pred = (y_pred > 0.5).astype(int)

    return (
        metrics.roc_auc_score(test_labels, y_pred),
        metrics.precision_score(test_labels, y_pred),
        metrics.recall_score(test_labels, y_pred),
        metrics.f1_score(test_labels, y_pred),
    )
