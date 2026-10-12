import networkx as nx

from project.task4 import ms_bfs_based_rpq


def test_ms_bfs_keeps_sources_separate():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 2, label="a")
    graph.add_edge(1, 2, label="b")

    assert ms_bfs_based_rpq("a", graph, {0, 1}, {2}) == {(0, 2)}


def test_ms_bfs_handles_cycles_parallel_edges_and_epsilon():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="x")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 1, label="b")

    assert ms_bfs_based_rpq("a b*", graph, {0, 3}, {1, 2, 3}) == {
        (0, 1),
        (0, 2),
    }
    assert ms_bfs_based_rpq("a*", graph, {0, 3}, {0, 1, 3}) == {
        (0, 0),
        (0, 1),
        (3, 3),
    }


def test_ms_bfs_treats_empty_filters_as_all_nodes():
    graph = nx.MultiDiGraph([(10, 20, {"label": "a"})])

    assert ms_bfs_based_rpq("a", graph, set(), set()) == {(10, 20)}
