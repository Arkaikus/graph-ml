import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from processing.store import Store

from .edge_splitter import EdgeSplitter

logger = logging.getLogger(__name__)


def train_embedding(graph, name, **n2v_kwargs) -> Word2Vec:
    """trains and saves to disk a word2vec trained model"""
    store = Store("embeddings")
    data = store.load(name)
    if not data:
        data = Node2Vec(graph, **n2v_kwargs).fit()
        store.save(data, name)

    return data


def train_randomforest(embeddings, labels, name) -> RandomForestClassifier:
    """trains and saves to disk a RandomForestClassifier"""
    store = Store("randomforest")
    data = store.load(name)
    if not data:
        rf = RandomForestClassifier(n_estimators=1000, verbose=1)
        data = rf.fit(embeddings, labels)
        store.save(data, name)

    return data


def run_link_prediction(file: str, test_size, embedder_class=HadamardEmbedder, **n2v_kwargs):
    """
    Implementation of Graph Machine Learning Book Chapter 05 Link Prediction algorithm
    this algorithm takes an edge list, creates a graph, and splits the graph edges nto training and test sets,
    then trains a Node2Vec model to embed the pair of nodes with a 0/1 label if the nodes are connected
    finally it trains a RandomForestClassifier with the embedded nodes and the 0/1 labels

    Note: EdgeSplitter.train_test_split returns the reduced graph, %p sampled nodes and %p sampled labels
    """
    file_path = Path(file)
    assert file_path.exists()
    edge_list = pd.read_csv(file_path, dtype=str)
    assert "source" in edge_list.columns
    assert "target" in edge_list.columns

    logger.info("Splitting data into train and test with EdgeSplitter")
    graph: nx.DiGraph = nx.from_pandas_edgelist(edge_list, create_using=nx.DiGraph)

    logger.info("Training/loading node2vec model")
    # .fit() will walk the graph, then run index2words from the walks
    # finally it will fit a word2vec model, this will generate
    # a vector associated to each node
    n2v_params = ",".join(f"{k}:{v}" for k, v in n2v_kwargs.items())
    n2v_name = file_path.stem + n2v_params
    n2v_train = train_embedding(graph, n2v_name, **n2v_kwargs)

    test_splitter = EdgeSplitter()
    # takes nodes and edges way from original graph, an creates test dataset
    # test_samples is a list of pairs, and test_labels is a list of 0/1 values if a pair is connected
    (
        train_samples,  # big chunk of graph edges, with random generated negative edges
        train_labels,  # train 0/1 labels where 1 cases are TP, and 0 cases are TN
        test_samples,  # small chunk of graph edges, with random generated negative edges
        test_labels,  # test 0/1 labels where 1 are TP, and 0 are TN
    ) = test_splitter.train_test_split(graph, test_size=test_size)

    print("Train dataset", train_samples.shape, train_labels.shape)
    print("Test dataset", test_samples.shape, test_labels.shape)

    logger.info("Querying %s vectors", embedder_class.__name__)
    embeddings = embedder_class(keyed_vectors=n2v_train.wv)
    train_vectors = np.array([embeddings[u, v] for u, v in train_samples])
    test_vectors = np.array([embeddings[u, v] for u, v in test_samples])

    logger.info("Training RandomForestClassifier")
    rfn = file_path.stem + embedder_class.__name__ + f"_test:{test_size}" + n2v_params
    rf = train_randomforest(train_vectors, train_labels, rfn)
    y_pred = rf.predict(test_vectors)

    return (
        metrics.roc_auc_score(test_labels, y_pred),
        metrics.precision_score(test_labels, y_pred),
        metrics.recall_score(test_labels, y_pred),
        metrics.f1_score(test_labels, y_pred),
    )
