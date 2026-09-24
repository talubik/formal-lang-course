import networkx as nx
from pyformlang.finite_automaton import State

from project.task2 import graph_to_nfa, regex_to_dfa


def test_regex_to_dfa_returns_minimal_automaton():
    dfa = regex_to_dfa("a (b | c)*")

    assert dfa.is_deterministic()
    assert len(dfa.states) == len(dfa.minimize().states)
    assert dfa.accepts(["a"])
    assert dfa.accepts(["a", "b", "c"])
    assert not dfa.accepts(["b"])


def test_graph_to_nfa_uses_selected_states_and_all_parallel_edges():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="b")

    nfa = graph_to_nfa(graph, {0}, {1})

    assert nfa.states == {State(0), State(1), State(2)}
    assert nfa.start_states == {State(0)}
    assert nfa.final_states == {State(1)}
    assert nfa.accepts(["a"])
    assert nfa.accepts(["b"])
    assert not nfa.accepts([])


def test_graph_to_nfa_treats_empty_state_sets_as_all_nodes():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1])

    nfa = graph_to_nfa(graph, set(), set())

    assert nfa.start_states == {State(0), State(1)}
    assert nfa.final_states == {State(0), State(1)}
    assert nfa.accepts([])
