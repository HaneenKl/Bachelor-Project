from automata_toolkit.regex_to_nfa import regex_to_nfa
from automata_toolkit.dfa_to_efficient_dfa import dfa_to_efficient_dfa
from automata_toolkit.nfa_to_dfa import nfa_to_dfa

from libsemigroups_pybind11.transf import Transf
from libsemigroups_pybind11 import FroidurePin



def transf_for_letter(dfa, letter, state_index):
    images = []  # this will become the list [δ(q, letter) for all q]

    for q in dfa.states:
        # compute the target state after reading `letter`
        target_state = dfa.transition(q, letter)

        # convert target state to its integer index
        target_index = state_index[target_state]

        # append to the list
        images.append(target_index)

    # build the transformation from the list
    return Transf(images)

def regex_to_semigroup(regex: str):
    nfa = regex_to_nfa(regex)
    dfa = nfa_to_dfa(nfa)
    min_dfa = dfa_to_efficient_dfa(dfa)
    print(min_dfa)
    print(list(min_dfa['reachable_states']))

    states = list(min_dfa['reachable_states'])
    state_index = {}
    for i, q in enumerate(states):
        state_index[q] = i


    print(state_index)

    alphabet = min_dfa['alphabets']
    print("alphabet:", alphabet)

    generators = []

    for a in alphabet:
        images = []

        for q in states:
            target_state = min_dfa['transition_function'][q][a]
            target_index = state_index[target_state]
            images.append(target_index)

        t = Transf(images)
        generators.append(t)

        print(f"{a} → {images}")



    S = FroidurePin(generators)

    print("Syntactic monoid size:", S.size())
    for x in S:
        print(list(x))


    return min_dfa

regx = "ab*"
regex_to_semigroup(regx)