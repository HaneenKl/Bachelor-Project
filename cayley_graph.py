from semigroup import *



############ Cayley graph visualization for semigroups ############
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



def extract_semigroup_cayley(min_dfa, side):
    # node labels: show transformation + representative word
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


############### Cayley graph visualization for monoids and stable semigroups ############
def add_identity_node(node_ids, node_labels, edges, elements, alphabet):
    n_states = len(elements[0])
    identity = tuple(range(n_states))
    if identity in elements:
        return node_ids, node_labels, edges

    new_node_ids = ["epsilon"] + node_ids
    new_node_labels = node_labels
    new_node_labels["epsilon"] = "ε"

    # map each single-letter word to its node id in the semigroup
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



###############--- monoid ---##############

def right_cayley_graph_svg_monoid(min_dfa):
    node_ids, node_labels, edges, elements, alphabet, reps = extract_semigroup_cayley(min_dfa, side="right")
    node_ids, node_labels, edges = add_identity_node(node_ids, node_labels, edges, elements, alphabet)
    return cayley_graph_svg(node_ids, node_labels, edges)


def left_cayley_graph_svg_monoid(min_dfa):
    node_ids, node_labels, edges, elements, alphabet, reps = extract_semigroup_cayley(min_dfa, side="left")
    node_ids, node_labels, edges = add_identity_node(node_ids, node_labels, edges, elements, alphabet)
    return cayley_graph_svg(node_ids, node_labels, edges)


############--- stable semigroup ---##############
def restrict_to_stable_elements(node_ids, node_labels, edges, elements, reps):
    stable = compute_stable_subsemigroup(reps)
    stable_ids = {idx for idx, elem in enumerate(elements) if elem in stable}

    new_node_ids = [i for i in node_ids if i in stable_ids]
    new_node_labels = {i: node_labels[i] for i in new_node_ids}
    new_edges = [(src, dst, label) for src, dst, label in edges if src in stable_ids and dst in stable_ids]
    return new_node_ids, new_node_labels, new_edges


def right_cayley_graph_svg_stable(min_dfa):
    node_ids, node_labels, edges, elements, alphabet, reps = extract_semigroup_cayley(min_dfa, side="right")
    node_ids, node_labels, edges = restrict_to_stable_elements(node_ids, node_labels, edges, elements, reps)
    return cayley_graph_svg(node_ids, node_labels, edges)



def left_cayley_graph_svg_stable(min_dfa):
    node_ids, node_labels, edges, elements, alphabet, reps = extract_semigroup_cayley(min_dfa, side="left")
    node_ids, node_labels, edges = restrict_to_stable_elements(node_ids, node_labels, edges, elements, reps)
    return cayley_graph_svg(node_ids, node_labels, edges)
