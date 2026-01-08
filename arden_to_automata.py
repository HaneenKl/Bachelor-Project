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

    def plot(self):
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

        # Return SVG (as UTF-8 string)
        return dot.pipe(format="svg").decode("utf-8")


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


def arden_to_automata(equations: dict[str, list[str]]):
    # inistialize automaton components
    states = set(equations.keys())
    start_state = list(equations.keys())[0]
    accepting_states = set()
    transitions = {}

    # build transitions and accepting states
    for var, terms in equations.items():
        # epsilon gives acceptance
        if "ε" in terms or "epsilon" in terms:
            accepting_states.add(var)

        # terminal symbols extend acceptance to destinations
        # find all single-letter terminal terms in this var
        terminal_symbols = [t for t in terms if len(t) == 1]

        for term in terms:
            if term in ("ε", "epsilon"):
                continue
            elif len(term) == 2:  # symbol + destination
                symbol, dest = term[0], term[1]
                transitions[(var, symbol)] = dest
                # if symbol appears as a terminal term, dest must be accepting
                if symbol in terminal_symbols:
                    accepting_states.add(dest)

    return Automaton(states, start_state, accepting_states, transitions)


"""
# test code
# test parsing equations to dict
arden_eq_text = "A = aA + bB + ε\nB = aA + bA"
parsed_eq = parse_equations(arden_eq_text)
dict_eq = arden_to_automata(parsed_eq)
dict_eq.plot()
"""

"""
# Example usage:
equations = {
    'A': ['aA', 'bB', 'b'],
    'B': ['aA', 'bB']
}
automaton = arden_to_automata(equations)
print(automaton.transitions)
print(automaton.start)
print(automaton.accepting)
automaton.plot("arden_automaton")"""
