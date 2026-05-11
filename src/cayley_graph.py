from semigroup import build_fp_and_reps, build_stable_cayley_adj
from graphviz import Digraph


# =============================================================================
# PLOT CAYLEY GRAPHS IN SVG
# =============================================================================
def cayley_graph_svg(node_ids, node_labels, edges):
    dot = Digraph(format="svg")
    dot.attr(rankdir="LR")

    dot.attr("node", shape="circle", width="0.6", fixedsize="true")
    dot.attr("edge", fontsize="12")

    # nodes
    for i in node_ids:
        dot.node(str(i), label=node_labels.get(i, str(i)))

    # edges
    for src, dst, label in edges:
        dot.edge(str(src), str(dst), label=str(label))

    return dot.pipe(format="svg").decode("utf-8")


# =============================================================================
# SEMIGROUP CAYLEY GRAPHS
# =============================================================================
def extract_semigroup_cayley(min_dfa, side):
    """
    Build the Cayley graph of the syntactic semigroup using right_/left_cayley_graph() of
    libsemigroups_pybind11 library.
     """
    reps, fp, alphabet = build_fp_and_reps(min_dfa)
    node_ids = list(range(fp.size()))
    elements = list(reps.keys())
    words = list(reps.values())

    node_labels = {
        i: ("ε" if w == "" else w)
        for i, w in enumerate(words)
    }

    if side == "right":
        wg = fp.right_cayley_graph()
    else:
        wg = fp.left_cayley_graph()

    edges = []
    for node in wg.nodes():
        for label, dst in wg.labels_and_targets(node):
            if dst is None:
                continue
            edges.append((node, dst, str(alphabet[label])))

    return node_ids, node_labels, edges, elements, alphabet, reps


def right_cayley_graph_svg_semigroup(min_dfa):
    node_ids, node_labels, edges, elements, alphabet, reps = extract_semigroup_cayley(min_dfa, side="right")
    return cayley_graph_svg(node_ids, node_labels, edges)


def left_cayley_graph_svg_semigroup(min_dfa):
    node_ids, node_labels, edges, elements, alphabet, reps = extract_semigroup_cayley(min_dfa, side="left")
    return cayley_graph_svg(node_ids, node_labels, edges)


# =============================================================================
# MONOID CAYLEY GRAPHS
# =============================================================================
def add_identity_node(node_ids, node_labels, edges, elements, alphabet):
    """
    Add an identity node to the Cayley graph of the initial Cayley graph of the semigroup if
    the identity element is not already present in the semigroup.
    The identity node will have edges to all nodes corresponding to letters in the alphabet.
    """
    n_states = len(elements[0])
    identity = tuple(range(n_states))
    if identity in elements:
        return node_ids, node_labels, edges

    new_node_ids = ["epsilon"] + node_ids
    new_node_labels = node_labels
    new_node_labels["epsilon"] = "ε"

    letter_to_node = {}
    for idx in node_ids:
        w = node_labels[idx]
        if w != "ε" and len(w) == 1:
            letter_to_node[w] = idx

    new_edges = list(edges)
    for a in alphabet:
        a_str = str(a)
        if a_str in letter_to_node:
            new_edges.append(("epsilon", letter_to_node[a_str], a_str))

    return new_node_ids, new_node_labels, new_edges


def right_cayley_graph_svg_monoid(min_dfa):
    node_ids, node_labels, edges, elements, alphabet, reps = extract_semigroup_cayley(min_dfa, side="right")
    node_ids, node_labels, edges = add_identity_node(node_ids, node_labels, edges, elements, alphabet)
    return cayley_graph_svg(node_ids, node_labels, edges)


def left_cayley_graph_svg_monoid(min_dfa):
    node_ids, node_labels, edges, elements, alphabet, reps = extract_semigroup_cayley(min_dfa, side="left")
    node_ids, node_labels, edges = add_identity_node(node_ids, node_labels, edges, elements, alphabet)
    return cayley_graph_svg(node_ids, node_labels, edges)


# =============================================================================
# STABLE SEMIGROUP CAYLEY GRAPHS
# =============================================================================
def build_stable_cayley(min_dfa, side):
    reps_stable, elems, _index_of, gens, right_adj, left_adj = build_stable_cayley_adj(min_dfa)

    node_ids = list(range(len(elems)))
    node_labels = {i: ("ε" if reps_stable[e] == "" else reps_stable[e]) for i, e in enumerate(elems)}

    adj = right_adj if side == "right" else left_adj
    edges = []

    for src, outgoing in adj.items():
        for g_idx, dst in outgoing:
            label = reps_stable[gens[g_idx]]
            if label == "":
                label = "ε"
            edges.append((src, dst, str(label)))
    return node_ids, node_labels, edges


def right_cayley_graph_svg_stable(min_dfa):
    node_ids, node_labels, edges = build_stable_cayley(min_dfa, side="right")
    return cayley_graph_svg(node_ids, node_labels, edges)


def left_cayley_graph_svg_stable(min_dfa):
    node_ids, node_labels, edges = build_stable_cayley(min_dfa, side="left")
    return cayley_graph_svg(node_ids, node_labels, edges)
