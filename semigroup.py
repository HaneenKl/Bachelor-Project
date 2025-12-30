from libsemigroups_pybind11 import FroidurePin
from libsemigroups_pybind11.transf import Transf
import subprocess



def wg_to_dot(wg, S):
    # Map node indices -> actual transformations
    elements = list(S)
    node_label = {i: str(list(elements[i])) for i in range(len(elements))}

    # ---- build DOT by hand ----
    lines = ["digraph Cayley {", "  rankdir=LR;", "  node [shape=circle];"]

    # nodes
    for i in range(wg.number_of_nodes()):
        lines.append(f'  {i} [label="{node_label[i]}"];')

    # edges
    for s in wg.nodes():
        for label, target in wg.labels_and_targets(s):
            if target is not None:
                lines.append(f'  {s} -> {target} [label="{label}"];')

    lines.append("}")

    dott = "\n".join(lines)

    return dott

def create_semigroup():
    # transformations on {0,1}
    gens = [
        Transf([1, 0]),  # swap
        Transf([0, 0])  # collapse
    ]

    S = FroidurePin(gens)
    wg = S.right_cayley_graph()

    dot = wg_to_dot(wg, S)

    # write DOT
    with open("semigroup.dot", "w") as f:
        f.write(dot)

    # DOT -> SVG
    subprocess.run(
        ["dot", "-Tsvg", "semigroup.dot", "-o", "semigroup.svg"],
        check=True
    )







