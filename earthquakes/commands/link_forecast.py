import logging
from pathlib import Path

import click
import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from processing.edge_splitter import EdgeSplitter

logger = logging.getLogger(__name__)


@click.command()
@click.option("-f", "--file", type=str, help="Catalog csv file to turn into edge list")
def link_forecast(file):

    file_path = Path(file)
    assert file_path.exists()

    logger.info("Reading edge list at %s", file)
    edge_list = pd.read_csv(file_path)
    assert "source" in edge_list.columns
    assert "target" in edge_list.columns

    logger.info("Splitting data into train and test with EdgeSplitter")
    graph = nx.from_pandas_edgelist(edge_list)
    test_splitter = EdgeSplitter(graph)
    graph_test, samples_test, labels_test = test_splitter.train_test_split(p=0.1, method="global")
    train_splitter = EdgeSplitter(graph_test, graph)
    graph_train, samples_train, labels_train = train_splitter.train_test_split(p=0.1, method="global")

    logger.info("Training node2vec model")
    node2vec = Node2Vec(graph_train)
    model = node2vec.fit()
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
    train_embeddings = [edges_embs[str(x[0]), str(x[1])] for x in samples_train]
    test_embeddings = [edges_embs[str(x[0]), str(x[1])] for x in samples_test]

    logger.info("Training RandomForestClassifier")
    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(train_embeddings, labels_train)

    y_pred = rf.predict(test_embeddings)
    print("Precision:", metrics.precision_score(labels_test, y_pred))
    print("Recall:", metrics.recall_score(labels_test, y_pred))
    print("F1-Score:", metrics.f1_score(labels_test, y_pred))
