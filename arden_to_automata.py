from graphviz import Digraph

"""
Input are Equations in Arden's form
    return is an equivalent Finite Automaton
"""


class Automaton:
    def __init__(self, states, start, accepting, transitions):
        self.states = set(states)
        self.start = start
        self.accepting = set(accepting)
        self.transitions = transitions  # dict: (src, symbol) -> dst

    def add_state(self, state, accepting=False):
        self.states.add(state)
        if accepting:
            self.accepting.add(state)

    def add_transition(self, src, symbol, dst):
        self.transitions[(src, symbol)] = dst

    def plot(self, filename="automaton"):
        dot = Digraph()
        dot.attr(rankdir="LR")

        # Draw states
        for s in self.states:
            shape = "doublecircle" if s in self.accepting else "circle"
            dot.node(s, shape=shape)

        # Draw start arrow
        dot.node("start", shape="point")
        dot.edge("start", self.start)

        # Draw transitions
        for (src, sym), dst in self.transitions.items():
            dot.edge(src, dst, label=sym)

        dot.render(filename, format="png", view=True)

        return dot


def arden_to_automata(equations: dict[str, list[str]]):
    # inistialize automaton components
    states = set(equations.keys())
    start_state = list(equations.keys())[0]
    accepting_states = set()
    transitions = {}

    # build transitions and accepting states
    # epsilon gives acceptance
    for var, terms in equations.items():
        if "ε" in terms or "epsilon" in terms:
            accepting_states.add(var)

    # terminal symbols extend acceptance to destinations
    for var, terms in equations.items():

        # find all single-letter terminal terms in this var
        terminal_symbols = [t for t in terms if len(t) == 1]

        for term in terms:
            if len(term) == 2:  # symbol + destination
                symbol, dest = term[0], term[1]
                # if symbol appears as a terminal term, dest must be accepting
                if symbol in terminal_symbols:
                    accepting_states.add(dest)

    #  build transitions
    for var, terms in equations.items():
        for term in terms:
            if term in ("ε", "epsilon"):
                continue
            elif len(term) == 2:
                symbol = term[0]
                dest = term[1]
                transitions[(var, symbol)] = dest

    return Automaton(states, start_state, accepting_states, transitions)


# Example usage:
equations = {
    'A': ['aA', 'bB', 'b'],
    'B': ['aA', 'bB']
}
automaton = arden_to_automata(equations)
print(automaton.transitions)
print(automaton.start)
print(automaton.accepting)
automaton.plot("arden_automaton")
