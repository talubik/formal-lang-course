"""Conversions from regular expressions and labelled graphs to automata."""

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Build a minimal deterministic finite automaton for ``regex``.

    The expression is parsed according to the :mod:`pyformlang` regular
    expression syntax.  Parsing errors are intentionally propagated to the
    caller instead of being hidden behind a partially constructed automaton.
    """

    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_states: set[int],
    final_states: set[int],
) -> NondeterministicFiniteAutomaton:
    """Convert a labelled directed multigraph to an NFA.

    Graph vertices become automaton states and every edge ``u -> v`` becomes
    a transition from ``u`` to ``v`` labelled with the edge's ``label``
    attribute.  An empty set of start or final states means all graph
    vertices, as required by the assignment.

    Neither the graph nor the supplied sets are modified.
    """

    nodes = set(graph.nodes)
    actual_start_states = nodes if not start_states else set(start_states)
    actual_final_states = nodes if not final_states else set(final_states)

    # Passing the complete state set to the constructor preserves isolated
    # vertices which are neither explicitly initial nor final.
    nfa = NondeterministicFiniteAutomaton(states=nodes)

    for state in actual_start_states:
        nfa.add_start_state(state)
    for state in actual_final_states:
        nfa.add_final_state(state)
    for source, target, data in graph.edges(data=True):
        nfa.add_transition(source, data["label"], target)

    return nfa
