"""Tests for EdgeSplitter."""

import networkx as nx
import numpy as np
import pytest

from graphs.edge_splitter import EdgeSplitter


def test_train_test_split_seed_zero_allowed():
    """Seed 0 must be accepted (non-negative)."""
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 2)])
    splitter = EdgeSplitter()
    train_s, train_l, test_s, test_l = splitter.train_test_split(g, test_size=0.5, seed=0)
    assert len(train_s) + len(test_s) >= 0
    assert train_s.dtype in (np.int64, np.int32) or train_s.shape[1] == 2


def test_train_test_split_seed_negative_rejected():
    splitter = EdgeSplitter()
    with pytest.raises(ValueError, match="non-negative"):
        splitter.train_test_split(nx.DiGraph([(0, 1)]), test_size=0.5, seed=-1)


def test_train_test_split_reproducible():
    g = nx.DiGraph()
    g.add_edges_from([(i, i + 1) for i in range(10)] + [(10, 0)])
    s1 = EdgeSplitter()
    s2 = EdgeSplitter()
    out1 = s1.train_test_split(g, test_size=0.3, seed=42)
    out2 = s2.train_test_split(g, test_size=0.3, seed=42)
    np.testing.assert_array_equal(out1[0], out2[0])
    np.testing.assert_array_equal(out1[1], out2[1])
