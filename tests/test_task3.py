import networkx as nx
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol

from project.task2 import regex_to_dfa
from project.task3 import AdjacencyMatrixFA, intersect_automata, tensor_based_rpq


def test_matrix_nfa_accepts_nondeterministic_paths_and_symbol_objects():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_start_state(0)
    nfa.add_final_state(2)
    nfa.add_transition(0, "a", 1)
    nfa.add_transition(0, "a", 2)
    nfa.add_transition(1, "b", 2)
    automaton = AdjacencyMatrixFA(nfa)

    assert automaton.accepts(["a"])
    assert automaton.accepts([Symbol("a"), Symbol("b")])
    assert not automaton.accepts([])
    assert not automaton.accepts(["missing"])


def test_matrix_automaton_emptiness_accounts_for_zero_length_paths():
    accepts_empty = NondeterministicFiniteAutomaton()
    accepts_empty.add_start_state("start")
    accepts_empty.add_final_state("start")

    disconnected = NondeterministicFiniteAutomaton()
    disconnected.add_start_state(0)
    disconnected.add_final_state(2)
    disconnected.add_transition(0, "a", 1)

    assert not AdjacencyMatrixFA(accepts_empty).is_empty()
    assert AdjacencyMatrixFA(disconnected).is_empty()


def test_intersection_accepts_exactly_words_common_to_both_languages():
    left = AdjacencyMatrixFA(regex_to_dfa("a (b | c)*"))
    right = AdjacencyMatrixFA(regex_to_dfa("a b*"))
    intersection = intersect_automata(left, right)

    assert intersection.accepts(["a"])
    assert intersection.accepts(["a", "b", "b"])
    assert not intersection.accepts(["a", "c"])
    assert not intersection.accepts(["b"])


def test_tensor_rpq_respects_node_filters_parallel_edges_and_epsilon():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="x")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 1, label="b")

    assert tensor_based_rpq("a b*", graph, {0, 3}, {1, 2, 3}) == {
        (0, 1),
        (0, 2),
    }
    assert tensor_based_rpq("a*", graph, {0, 3}, {0, 1, 3}) == {
        (0, 0),
        (0, 1),
        (3, 3),
    }


def test_tensor_rpq_treats_empty_filters_as_all_graph_nodes():
    graph = nx.MultiDiGraph([(10, 20, {"label": "a"})])

    assert tensor_based_rpq("a", graph, set(), set()) == {(10, 20)}
