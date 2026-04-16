from graphviz import Digraph

from semigroup import _build_fp_and_reps




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
    # node labels: show transformation + representative word
    reps, _fp, _alphabet = _build_fp_and_reps(min_dfa)

    node_labels = {
        i: ("ε" if w == "" else w)
        for i, w in enumerate(reps.values())
    }
    return _alphabet, _fp, node_labels


def right_cayley_graph_svg(min_dfa):
    alphabet, fp, node_labels = froidure_pin_alg(min_dfa)
    wg = fp.right_cayley_graph()
    return cayley_graph_svg(wg, node_labels, alphabet)


def left_cayley_graph_svg(min_dfa):
    alphabet, fp, node_labels = froidure_pin_alg(min_dfa)
    wg = fp.left_cayley_graph()
    return cayley_graph_svg(wg, node_labels, alphabet)