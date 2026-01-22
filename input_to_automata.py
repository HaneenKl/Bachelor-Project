from graphviz import Digraph

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang import regular_expression
from pyformlang.finite_automaton import State

import re


def plot_min_dfa(min_dfa):
    dot = Digraph()
    dot.attr(rankdir="LR")

    for s in min_dfa.states:
        shape = "doublecircle" if s in min_dfa.final_states else "circle"
        dot.node(str(s), shape=shape)

    dot.node("start", shape="point")
    dot.edge("start", str(min_dfa.start_state))

    for src_state, outgoing in min_dfa._transition_function._transitions.items():
        for symbol, dst_state in outgoing.items():
            dot.edge(str(src_state), str(dst_state), label=str(symbol))

    return dot.pipe(format="svg").decode("utf-8")


def complete_dfa(dfa):
    alphabet = set(dfa._input_symbols)
    phi = State("φ")
    need_phi = False
    for q in list(dfa.states):
        for a in alphabet:
            if len(dfa._transition_function(q, a)) == 0:
                need_phi = True
                break

    if not need_phi:
        return dfa

    for q in list(dfa.states):
        for a in alphabet:
            if len(dfa._transition_function(q, a)) == 0:
                dfa.add_transition(q, a, phi)

    for a in alphabet:
        dfa.add_transition(phi, a, phi)

    return dfa


def parse_equations(text):
    """
    Example: converts text like:
         A = aA + bB + ε
         B = aA
    into:
         {"A": ["aA","bB","ε"], "B": ["aA"]}
    """
    eq = {}
    for line in text.split("\n"):
        if "=" not in line:
            continue
        left, right = line.split("=")
        var = left.strip()
        terms = [t.strip() for t in right.split("+")]
        eq[var] = terms
    return eq


#############--- user input as arden equations ---##############
def arden_to_pyformlang_min_dfa(equations):
    eq_dict = parse_equations(equations)

    start_state = list(eq_dict.keys())[0]
    accepting_states = set()
    transitions = {}

    # build transitions and accepting states
    for var, terms in eq_dict.items():
        # find all single-letter terminal terms in this var
        terminal_symbols = [t for t in terms if len(t) == 1]

        for term in terms:
            if term in ("ε", "epsilon"):
                accepting_states.add(var)
                continue
            elif len(term) >= 2:
                symbol, dest = term[0:len(term) - 1], term[-1]
                transitions[(var, symbol)] = dest
                # if symbol appears as a terminal term, dest must be accepting
                if symbol in terminal_symbols:
                    accepting_states.add(dest)

    min_dfa = automaton_to_pyformlang_min_dfa(start_state, accepting_states, transitions)

    return min_dfa


def automaton_to_pyformlang_min_dfa(start_sate, accepting_states, transitions):
    nfa = NondeterministicFiniteAutomaton()
    for (src, sym), dst in transitions.items():
        nfa.add_transition(src, sym, dst)

    nfa.add_start_state(start_sate)
    for q in accepting_states:
        nfa.add_final_state(q)

    dfa = nfa.to_deterministic()
    min_dfa = dfa.minimize()
    min_dfa = complete_dfa(min_dfa)

    return min_dfa


#############--- user input as regex ---##############

def regex_to_pyformlang_min_dfa(regex):
    regex_parse = regular_expression.Regex(regex)
    nfa = regex_parse.to_epsilon_nfa()
    dfa = nfa.to_deterministic()
    min_dfa = dfa.minimize()
    min_dfa = complete_dfa(min_dfa)
    return min_dfa


def add_spacing_to_regex(s):
    """
    Insert spaces between letters.
    Example: 'abc' -> 'a b c'
    """
    return re.sub(r'([a-zA-Z])(?=[a-zA-Z])', r'\1 ', s)
