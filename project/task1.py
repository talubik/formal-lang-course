"""Utilities for loading and generating labelled graphs."""

from pathlib import Path
from typing import Any, NamedTuple

import cfpq_data
from networkx.drawing.nx_pydot import write_dot


class GraphInfo(NamedTuple):
    """Basic information about a graph from the CFPQ_Data dataset."""

    vertices: int
    edges: int
    labels: set[Any]


def get_graph_info(graph_name: str) -> GraphInfo:
    """Load a dataset graph by name and return its basic properties.

    Parameters
    ----------
    graph_name:
        A graph name from the CFPQ_Data dataset.

    Returns
    -------
    GraphInfo
        The number of vertices, the number of edges, and the set of edge
        labels, in that order.
    """

    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)
    labels = {data["label"] for _, _, data in graph.edges(data=True)}

    return GraphInfo(graph.number_of_nodes(), graph.number_of_edges(), labels)


def create_two_cycles_graph(
    first_cycle_size: int,
    second_cycle_size: int,
    labels: tuple[str, str],
    output_path: str | Path,
) -> Path:
    if first_cycle_size <= 0 or second_cycle_size <= 0:
        raise ValueError("Cycle sizes must be positive")

    graph = cfpq_data.labeled_two_cycles_graph(
        first_cycle_size,
        second_cycle_size,
        labels=labels,
    )
    destination = Path(output_path)
    write_dot(graph, destination)

    return destination.resolve()
