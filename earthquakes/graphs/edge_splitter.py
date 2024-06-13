# Taken and adapted from https://github.com/stellargraph/stellargraph/blob/develop/stellargraph/data/edge_splitter.py

import datetime
import warnings
from math import isclose

import networkx as nx
import numpy as np
import pandas as pd


class EdgeSplitter:
    """
    Class for generating training and test data for link prediction in graphs.

    The class requires as input a graph (in networkx format) and a percentage as a function of the total number of edges
    in the given graph of the number of positive and negative edges to sample. For heterogeneous graphs, the caller
    can also specify the type of edge and an edge property to split on. In the latter case, only a date property
    can be used and it must be in the format ``dd/mm/yyyy``. A date to be used as a threshold value such that only
    edges that have date after the threshold must be given. This effects only the sampling of positive edges.

    Negative edges are sampled at random by (for 'global' method) selecting two nodes in the graph and
    then checking if these edges are connected or not. If not, the pair of nodes is considered a negative sample.
    Otherwise, it is discarded and the process repeats. Alternatively, negative edges are sampled (for 'local' method)
    using DFS search at a distance from the source node (selected at random from all nodes in the graph)
    sampled according to a given set of probabilities.

    Positive edges can be sampled so that when they are subsequently removed from the graph, the reduced graph is either
    guaranteed, or not guaranteed, to remain connected. In the former case, graph connectivity is maintained by first
    calculating the minimum spanning tree. The edges that belong to the minimum spanning tree are protected from
    removal, and therefore cannot be sampled for the training set. The edges that do not belong to the minimum spanning
    tree are then sampled uniformly at random, until the required number of positive edges have been sampled for the
    training set. In the latter case, when connectedness of the reduced graph is not guaranteed, positive edges are
    sampled uniformly at random from all the edges in the graph, regardless of whether they belong to the spanning tree
    (which is not calculated in this case).

    Args:
        g (networkx object): The graph to sample edges from.
        g_master (networkx object): The graph representing the original dataset and a superset of the
            graph g. If it is not None, then when positive and negative edges are sampled, care is taken to make sure
            that a true positive edge is not sampled as a negative edge.

    """

    def __init__(self):
        self.positive_edges_ids = None
        self.positive_edges_labels = None
        self.negative_edges_ids = None
        self.negative_edges_labels = None
        self._random = np.random.RandomState(seed=None)

    def ttt_global(self, g: nx.Graph, g_master: nx.Graph, p):
        """
        Method for edge splitting applied to homogeneous graphs.

        Args:
            p (float): Percent of edges to be returned. It is calculated as a function of the total number of edges
             in the original graph. If the graph is heterogeneous, the percentage is calculated
             as a function of the total number of edges that satisfy the edge_label, edge_attribute_label and
             edge_attribute_threshold values given.
            method (string): Should be 'global' or 'local'. Specifies the method for selecting negative examples.

        Returns:
            2 numpy arrays, the first Nx2 holding the node ids for the edges and the second Nx1 holding the edge
        labels, 0 for negative and 1 for positive example.

        """

        # Sample the positive examples
        reduced_graph, positive_edges = self.reduce_graph(g, p=p)
        df = pd.DataFrame(positive_edges)
        self.positive_edges_ids = np.array(df.iloc[:, 0:2])
        self.positive_edges_labels = np.array(df.iloc[:, 2])

        # method global
        negative_edges = self.sample_negative_global(g, g_master, p=p, limit_samples=len(positive_edges))

        df = pd.DataFrame(negative_edges)
        self.negative_edges_ids = np.array(df.iloc[:, 0:2])
        self.negative_edges_labels = np.array(df.iloc[:, 2])

        if len(self.positive_edges_ids) == 0:
            raise Exception("Could not sample any positive edges")
        if len(self.negative_edges_ids) == 0:
            raise Exception("Could not sample any negative edges")

        edge_samples = np.vstack((self.positive_edges_ids, self.negative_edges_ids))
        label_samples = np.hstack((self.positive_edges_labels, self.negative_edges_labels))
        print(
            "** Sampled {} positive and {} negative edges. **".format(
                len(self.positive_edges_ids), len(self.negative_edges_ids)
            )
        )

        return reduced_graph, edge_samples, label_samples

    def train_test_split(self, graph: nx.Graph, test_size=0.5, seed=None):
        """
        Generates positive and negative edges and a graph that has the same nodes as the original but the positive
        edges removed. It can be used to generate data from homogeneous graphs.

        Args:
            p (float): Percent of edges to be returned. It is calculated as a function of the total number of edges
             in the original graph.
            method (default:global): How negative edges are sampled. If 'global', then nodes are selected at random.
            seed (int, optional): seed for random number generator, positive int or 0

        Returns:
            The reduced graph (positive edges removed) and the edge data as 2 numpy arrays, the first array of
            dimensionality N × 2 (where N is the number of edges) holding the node ids for the edges and the second of
            dimensionality N × 1 holding the edge labels, 0 for negative and 1 for positive examples. The graph
            matches the input graph passed to the :class:`.EdgeSplitter` constructor: the returned graph is a subgraph
            graph was one.
        """
        assert 0 <= test_size <= 1, "The value of p must be in the interval (0,1)"
        if seed is not None:
            assert isinstance(seed, int)
            assert seed > 0
            self._random = np.random.RandomState(seed=seed)

        reduced_graph, test_edge_data, test_edge_labels = self.ttt_global(graph, None, p=test_size)  # take p
        _, train_edge_data, train_edge_labels = self.ttt_global(reduced_graph, graph, p=1)  # take the rest
        return train_edge_data, train_edge_labels, test_edge_data, test_edge_labels

    def sample_negative_global(self, graph: nx.Graph, graph_master, p=0.5, limit_samples=None):
        """
        This method samples uniformly at random nodes from the graph and, if they don't have an edge in the graph,
        it records the pair as a negative edge.

        Args:
            p: (float) factor that multiplies the number of edges in the graph and determines the number of negative
            edges to be sampled.
            limit_samples: (int, optional) it limits the maximum number of samples to the given number, if not None

        Returns:
            (list) A list of 2-tuples that are pairs of node IDs that don't have an edge between them in the graph.
        """
        num_edges_to_sample = int(graph.number_of_edges() * p)

        if limit_samples is not None:
            if num_edges_to_sample > limit_samples:
                num_edges_to_sample = limit_samples

        if graph_master is None:
            edges = list(graph.edges())
        else:
            edges = list(graph_master.edges())

        # to speed up lookup of edges in edges list, create a set the values stored are the concatenation of
        # the source and target node ids.
        edges_set = set(edges)
        edges_set.update({(v, u) for u, v in edges})
        sampled_edges_set = set()

        start_nodes = list(graph.nodes(data=False))
        end_nodes = list(graph.nodes(data=False))

        count = 0
        sampled_edges = []

        num_iter = int(np.ceil(num_edges_to_sample / (1.0 * len(start_nodes)))) + 1
        for _ in np.arange(0, num_iter):
            self._random.shuffle(start_nodes)
            self._random.shuffle(end_nodes)
            for u, v in zip(start_nodes, end_nodes):
                if (u != v) and ((u, v) not in edges_set) and ((u, v) not in sampled_edges_set):
                    sampled_edges.append((u, v, 0))  # the last entry is the class label
                    sampled_edges_set.update({(u, v), (v, u)})  # test for bi-directional edges
                    count += 1
                if count == num_edges_to_sample:
                    return sampled_edges

        if len(sampled_edges) != num_edges_to_sample:
            raise ValueError(
                "Unable to sample {} negative edges. Consider using smaller value for p.".format(num_edges_to_sample)
            )

    def reduce_graph(self, graph, minedges=None, p=0.5):
        """
        Reduces the graph self.g_train by a factor p by removing existing edges not on minedges list such that
        the reduced tree remains connected. Edge type is ignored and all edges are treated equally.

        Args:
            minedges (list): Minimum spanning tree edges that cannot be removed.
            p (float): Factor by which to reduce the size of the graph.

        Returns:
            (list) Returns the list of edges removed from self.g_train (also modifies self.g_train by removing the
            said edges)
        """
        # copy the original graph and start over in case this is not the first time
        # reduce_graph has been called.
        _graph = graph.copy()
        minedges = minedges or set()

        # For multigraphs we should probably use keys
        use_keys_in_edges = _graph.is_multigraph()

        # For NX 1.x/2.x compatibilty we need to match length of minedges
        if len(minedges) > 0:
            use_keys_in_edges = len(next(iter(minedges))) == 3

        if use_keys_in_edges:
            all_edges = list(_graph.edges(keys=True))
        else:
            all_edges = list(_graph.edges())

        num_edges_to_remove = int(_graph.number_of_edges() * p)

        if num_edges_to_remove > (_graph.number_of_edges() - len(minedges)):
            raise ValueError(
                "Not enough positive edges to sample after reserving {} number of edges for maintaining graph connectivity. Consider setting keep_connected=False.".format(
                    len(minedges)
                )
            )

        # shuffle the edges
        self._random.shuffle(all_edges)
        # iterate over the list of edges and for each edge if the edge is not in minedges, remove it from the graph
        # until num_edges_to_remove edges have been removed and the graph reduced to p of its original size
        count = 0
        removed_edges = []
        for edge in all_edges:
            if edge not in minedges:
                removed_edges.append((edge[0], edge[1], 1))  # the last entry is the label
                _graph.remove_edge(*edge)

                count += 1
            if count == num_edges_to_remove:
                return _graph, removed_edges
