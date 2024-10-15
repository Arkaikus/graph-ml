import networkx as nx
import pandas as pd
from functools import partial


def nodes2graph(nodes, max_nodes, max_prev=5) -> nx.Graph:
    """
    Transforms a list of sequential node ids to networkx multigraph/multidigraph
    node_a -> node_b -> node_c

    it'll also link max_prev previous nodes
    [node_a, ..., node_n-1] -> node_n

    Parameters
    ----------
    nodes : numpy.ndarray
    number_of_nodes : int
    directed : boolean, default=True
    """
    graph = nx.Graph()

    graph.add_nodes_from(range(max_nodes))
    for i in range(nodes.shape[0] - 1):
        end = i + 1
        start = max(end - max_prev, 0)
        nodes_from = nodes[start:end]
        node_to = nodes[end]
        for node in nodes_from:
            if graph.has_edge(node, node_to):
                graph[node][node_to]["weight"] += 1
            else:
                graph.add_edge(node, node_to, weight=1)

    return graph


properties = {
    "degree_centrality": nx.degree_centrality,
    "clustering": partial(nx.clustering, weight="weight"),
    "betweenness_centrality": partial(nx.betweenness_centrality, weight="weight"),
    "closeness_centrality": nx.closeness_centrality,
    "pagerank": partial(nx.pagerank, weight="weight"),
}


def networkx_property(graph: nx.Graph, property: str) -> pd.DataFrame:
    """
    Computes a network property for each node in the graph from a list of properties
    available properties are:
    - degree_centrality
    - clustering
    - betweenness_centrality
    - closeness_centrality
    - pagerank
    
    returns a dataframe with node index and property value per node
    """
    assert isinstance(graph, nx.Graph)
    assert min(graph.nodes) == 0, "Graph node must start at 0"
    if property not in properties:
        return None

    G = graph.copy()
    # Remove isolated nodes
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    property_fn = properties[property]
    prop = property_fn(G)

    # Default excluded nodes to zero
    for isolate in isolates:
        prop[isolate] = 0

    df = pd.DataFrame(pd.Series(prop), columns=[property])
    df.index.names = ["node"]
    df.index += 1
    return df.reset_index()
