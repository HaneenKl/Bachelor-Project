from libsemigroups_pybind11 import FroidurePin
from graphviz import Digraph

from semi_group import build_letter_generators, shortest_representatives




def cayley_graph_svg(wg, node_labels, alphabet):
    dot = Digraph(format="svg")
    dot.attr(rankdir="LR")

    dot.attr("node", shape="circle", width="0.6", fixedsize="true")
    dot.attr("edge", fontsize="12")

    # nodes
    for i in range(wg.number_of_nodes()):
        dot.node(str(i), label=node_labels.get(i, str(i)))

    # edges
    for s in wg.nodes():
        for label, target in wg.labels_and_targets(s):
            if target is None:
                continue
            dot.edge(str(s), str(target), label=str(alphabet[label]))

    return dot.pipe(format="svg").decode("utf-8")


def froidure_pin_alg(min_dfa):
    states, alphabet, letter_transf = build_letter_generators(min_dfa)

    gens = [letter_transf[a] for a in alphabet]
    transf_monoid = FroidurePin(gens)
    reps = shortest_representatives(alphabet, letter_transf, len(states))

    # node labels: show transformation + representative word
    node_labels = {}
    for i, x in enumerate(list(transf_monoid)):
        k = tuple(list(x))
        w = reps.get(k, "?")
        # show epsilon nicely
        w_display = "ε" if w == "" else w
        node_labels[i] = w_display
    return transf_monoid, node_labels


def right_cayley_graph_svg(min_dfa):
    transf_monoid, node_labels = froidure_pin_alg(min_dfa)
    wg = transf_monoid.right_cayley_graph()
    return cayley_graph_svg(wg, node_labels, sorted(min_dfa._input_symbols, key=str))


def left_cayley_graph_svg(min_dfa):
    transf_monoid, node_labels = froidure_pin_alg(min_dfa)
    wg = transf_monoid.left_cayley_graph()
    return cayley_graph_svg(wg, node_labels, sorted(min_dfa._input_symbols, key=str))