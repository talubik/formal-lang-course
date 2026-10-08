"""Sparse-matrix algorithms for regular path queries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from scipy.sparse import csr_matrix, eye, kron

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    """A finite automaton represented by one sparse matrix per symbol.

    Rows are source states and columns are destination states.  Keeping the
    individual Boolean matrices (instead of one labelled matrix) also makes
    automata intersection a Kronecker product for every common symbol.
    """

    def __init__(self, automaton: NondeterministicFiniteAutomaton) -> None:
        self.states = tuple(automaton.states)
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        self.num_states = len(self.states)
        self.start_states = {
            self.state_to_index[state] for state in automaton.start_states
        }
        self.final_states = {
            self.state_to_index[state] for state in automaton.final_states
        }

        edges_by_symbol: dict[Symbol, tuple[list[int], list[int]]] = {}
        for source, symbol, target in automaton._transition_function.get_edges():
            rows, columns = edges_by_symbol.setdefault(symbol, ([], []))
            rows.append(self.state_to_index[source])
            columns.append(self.state_to_index[target])

        shape = (self.num_states, self.num_states)
        self.matrices = {
            symbol: csr_matrix(
                (
                    np.ones(len(rows), dtype=bool),
                    (rows, columns),
                ),
                shape=shape,
                dtype=bool,
            )
            for symbol, (rows, columns) in edges_by_symbol.items()
        }
        # Descriptive aliases are convenient for clients and preserve the
        # conventional name used in solutions to this assignment.
        self.adjacency_matrices = self.matrices
        self.bool_matrices = self.matrices
        self._closure: csr_matrix | None = None

    @classmethod
    def _from_components(
        cls,
        states: tuple[object, ...],
        start_states: set[int],
        final_states: set[int],
        matrices: dict[Symbol, csr_matrix],
    ) -> AdjacencyMatrixFA:
        """Construct a matrix automaton without converting it through an NFA."""

        result = cls.__new__(cls)
        result.states = states
        result.state_to_index = {state: index for index, state in enumerate(states)}
        result.num_states = len(states)
        result.start_states = start_states
        result.final_states = final_states
        result.matrices = matrices
        result.adjacency_matrices = matrices
        result.bool_matrices = matrices
        result._closure = None
        return result

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Return whether at least one run accepts ``word``."""

        current_states = self._indicator_vector(self.start_states)
        for raw_symbol in word:
            symbol = (
                raw_symbol if isinstance(raw_symbol, Symbol) else Symbol(raw_symbol)
            )
            matrix = self.matrices.get(symbol)
            if matrix is None or current_states.nnz == 0:
                return False

            current_states = (current_states @ matrix).astype(bool)
            current_states.eliminate_zeros()

        return bool(current_states[:, list(self.final_states)].nnz)

    def _indicator_vector(self, states: Iterable[int]) -> csr_matrix:
        """Create a Boolean row vector marking the supplied state indices."""

        indices = np.fromiter(states, dtype=int)
        return csr_matrix(
            (
                np.ones(indices.size, dtype=bool),
                (np.zeros(indices.size, dtype=int), indices),
            ),
            shape=(1, self.num_states),
            dtype=bool,
        )

    def _transitive_closure(self) -> csr_matrix:
        """Compute reflexive transitive closure by Boolean matrix squaring."""

        if self._closure is not None:
            return self._closure

        closure = eye(self.num_states, format="csr", dtype=bool)
        for matrix in self.matrices.values():
            closure = closure.maximum(matrix)

        while True:
            expanded = (closure @ closure).astype(bool)
            expanded.eliminate_zeros()
            next_closure = closure.maximum(expanded)
            if next_closure.nnz == closure.nnz:
                self._closure = next_closure
                return self._closure
            closure = next_closure

    def _reachable_from(self, starts: Iterable[int]) -> set[int]:
        """Find reachable states using a Boolean vector-matrix product."""

        reachable = self._indicator_vector(starts) @ self._transitive_closure()
        return {int(state) for state in reachable.indices}

    def is_empty(self) -> bool:
        """Return whether the language of the automaton is empty."""

        return not bool(self._reachable_from(self.start_states) & self.final_states)


def intersect_automata(
    automaton1: AdjacencyMatrixFA,
    automaton2: AdjacencyMatrixFA,
) -> AdjacencyMatrixFA:
    """Build the language intersection using sparse Kronecker products."""

    states = tuple(
        (state1, state2) for state1 in automaton1.states for state2 in automaton2.states
    )
    width = automaton2.num_states
    start_states = {
        state1 * width + state2
        for state1 in automaton1.start_states
        for state2 in automaton2.start_states
    }
    final_states = {
        state1 * width + state2
        for state1 in automaton1.final_states
        for state2 in automaton2.final_states
    }
    matrices = {
        symbol: kron(
            automaton1.matrices[symbol],
            automaton2.matrices[symbol],
            format="csr",
        ).astype(bool)
        for symbol in automaton1.matrices.keys() & automaton2.matrices.keys()
    }

    return AdjacencyMatrixFA._from_components(
        states, start_states, final_states, matrices
    )


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Evaluate an all-pairs regular path query by automata intersection."""

    graph_automaton = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_automaton = AdjacencyMatrixFA(regex_to_dfa(regex))
    product = intersect_automata(graph_automaton, regex_automaton)

    answer: set[tuple[int, int]] = set()
    for start_index in product.start_states:
        graph_start, _ = product.states[start_index]
        reachable_finals = product._reachable_from({start_index}) & product.final_states
        for final_index in reachable_finals:
            graph_final, _ = product.states[final_index]
            answer.add((_state_value(graph_start), _state_value(graph_final)))
    return answer


def _state_value(state: object) -> object:
    """Unwrap pyformlang states while retaining arbitrary graph node values."""

    return state.value if isinstance(state, State) else state
