"""Multiple-source matrix BFS for regular path queries."""

import numpy as np
from networkx import MultiDiGraph
from pyformlang.finite_automaton import State
from scipy.sparse import csr_matrix

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA, intersect_automata


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Evaluate an RPQ using multiple-source BFS over the product automaton.

    Every row of ``reachable`` belongs to one graph source, so paths reached
    from different sources are never mixed. A BFS step for all sources is a
    Boolean sparse matrix multiplication by the product adjacency matrix.
    """

    graph_automaton = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_automaton = AdjacencyMatrixFA(regex_to_dfa(regex))
    product = intersect_automata(graph_automaton, regex_automaton)

    graph_start_states = tuple(graph_automaton.start_states)
    regex_start_states = tuple(regex_automaton.start_states)
    if not graph_start_states or not regex_start_states:
        return set()

    frontier = _initial_frontier(
        graph_start_states,
        regex_start_states,
        regex_automaton.num_states,
        product.num_states,
    )
    reachable = frontier.copy()
    adjacency = _combined_adjacency(product)

    while frontier.nnz:
        candidates = (frontier @ adjacency).astype(bool)
        candidates.eliminate_zeros()
        already_reached = candidates.multiply(reachable)
        frontier = (
            candidates.astype(np.int8) - already_reached.astype(np.int8)
        ).astype(bool)
        frontier.eliminate_zeros()
        reachable = reachable.maximum(frontier)

    answer: set[tuple[int, int]] = set()
    for row, graph_start_index in enumerate(graph_start_states):
        source = _state_value(graph_automaton.states[graph_start_index])
        for product_index in reachable.getrow(row).indices:
            if product_index in product.final_states:
                graph_final_state, _ = product.states[product_index]
                answer.add((source, _state_value(graph_final_state)))

    return answer


def _initial_frontier(
    graph_start_states: tuple[int, ...],
    regex_start_states: tuple[int, ...],
    regex_state_count: int,
    product_state_count: int,
) -> csr_matrix:
    """Create one initial product-state vector per graph source."""

    rows: list[int] = []
    columns: list[int] = []
    for row, graph_state in enumerate(graph_start_states):
        for regex_state in regex_start_states:
            rows.append(row)
            columns.append(graph_state * regex_state_count + regex_state)

    return csr_matrix(
        (
            np.ones(len(rows), dtype=bool),
            (rows, columns),
        ),
        shape=(len(graph_start_states), product_state_count),
        dtype=bool,
    )


def _combined_adjacency(automaton: AdjacencyMatrixFA) -> csr_matrix:
    """Combine labelled transition matrices into one Boolean adjacency matrix."""

    adjacency = csr_matrix(
        (automaton.num_states, automaton.num_states),
        dtype=bool,
    )
    for matrix in automaton.matrices.values():
        adjacency = adjacency.maximum(matrix)
    return adjacency


def _state_value(state: object) -> object:
    """Unwrap pyformlang states while retaining arbitrary graph node values."""

    return state.value if isinstance(state, State) else state
