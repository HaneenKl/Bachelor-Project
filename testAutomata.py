
from graphviz import Digraph






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

        return dot  # optional, in case you want to inspect the DOT source




A = Automaton( states={'q0', 'q1', 'q2'},
               start='q0',
               accepting={'q2'},
               transitions={
                   ('q0', 'a'): 'q1',
                   ('q1', 'c'): 'q1',
                   ('q1', 'b'): 'q2',
                   ('q2', 'a'): 'q2'
               })


A.plot("example_automaton")


