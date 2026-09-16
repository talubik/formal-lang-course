from pathlib import Path

import cfpq_data
import networkx as nx
import pydot
import pytest

from project.task1 import GraphInfo, create_two_cycles_graph, get_graph_info


def test_get_graph_info(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="b")
    graph.add_edge(2, 3, label="a")
    graph_path = tmp_path / "graph.csv"

    monkeypatch.setattr(cfpq_data, "download", lambda name: graph_path)
    monkeypatch.setattr(cfpq_data, "graph_from_csv", lambda path: graph)

    assert get_graph_info("example") == GraphInfo(4, 3, {"a", "b"})


def test_create_two_cycles_graph(tmp_path: Path):
    output_path = tmp_path / "two_cycles.dot"

    result = create_two_cycles_graph(2, 3, ("left", "right"), output_path)
    graph = pydot.graph_from_dot_file(output_path)[0]

    assert result == output_path.resolve()
    assert len(graph.get_nodes()) == 6
    assert len(graph.get_edges()) == 7
    assert {edge.get_attributes()["label"] for edge in graph.get_edges()} == {
        "left",
        "right",
    }


@pytest.mark.parametrize("first_size, second_size", [(0, 1), (1, 0), (-1, 2)])
def test_create_two_cycles_graph_rejects_non_positive_sizes(
    first_size: int, second_size: int, tmp_path: Path
):
    with pytest.raises(ValueError, match="positive"):
        create_two_cycles_graph(first_size, second_size, ("a", "b"), tmp_path / "x.dot")
