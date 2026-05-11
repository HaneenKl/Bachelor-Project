from pyformlang import regular_expression
# way faster than automat-toolkit library





def regex_to_automathon():
    regex = regular_expression.Regex("a b b* c*")
    nfa = regex.to_epsilon_nfa()
    dfa = nfa.to_deterministic()
    min_dfa = dfa.minimize()
    print("NFA states:", nfa.states)
    print("DFA states:", dfa.states)
    print("Minimized DFA states:", min_dfa.states)
    print(min_dfa._input_symbols)
    states = min_dfa.states.split(", ")
    print("States list:", states)
    print(min_dfa.is_deterministic())


regex_to_automathon()
