from automata_toolkit.regex_to_nfa import regex_to_nfa
from automata_toolkit.dfa_to_efficient_dfa import dfa_to_efficient_dfa
from automata_toolkit.nfa_to_dfa import nfa_to_dfa

from libsemigroups_pybind11.transf import Transf
from libsemigroups_pybind11 import FroidurePin

from collections import deque
import subprocess



def build_letter_generators(min_dfa):

    # stable order: start state first, then the rest sorted by str
    start = min_dfa["initial_state"]
    rest = [q for q in min_dfa["reachable_states"] if q != start]
    rest = sorted(rest, key=str)
    states = [start] + rest

    state_index = {q: i for i, q in enumerate(states)}
    alphabet = list(min_dfa["alphabets"])
    alphabet = sorted(alphabet)  # stable order

    alphabet = list(min_dfa['alphabets'])
    alphabet.sort()



    # build transformations for each letter
    letter_transf = {}
    for a in alphabet:
        images = []
        for q in states:
            tgt = min_dfa["transition_function"][q][a]
            images.append(state_index[tgt])
        letter_transf[a] = Transf(images)





    return states, alphabet, letter_transf
def shortest_representatives(alphabet, letter_transf, n_states, max_len=30):
    """
    Returns dict: transformation_tuple -> shortest word (string)
    """
    id_t = Transf(list(range(n_states)))
    rep = {tuple(list(id_t)): ""}

    q = deque([id_t])
    while q:
        t = q.popleft()
        w = rep[tuple(list(t))]
        if len(w) >= max_len:
            continue

        for a in alphabet:
            # word concat on the right corresponds to multiplying by generator on the right
            t2 = t * letter_transf[a]
            k2 = tuple(list(t2))
            if k2 not in rep:
                rep[k2] = w + a
                q.append(t2)

    return rep

def wordgraph_to_dot(wg, node_labels):
    lines = ["digraph Cayley {", "  rankdir=LR;", "  node [shape=box];"]

    # nodes
    for i in range(wg.number_of_nodes()):
        lab = node_labels.get(i, str(i)).replace('"', '\\"')
        lines.append(f'  {i} [label="{lab}"];')

    # edges
    for s in wg.nodes():
        for label, target in wg.labels_and_targets(s):
            if target is None:
                continue
            lines.append(f'  {s} -> {target} [label="{label}"];')

    lines.append("}")
    return "\n".join(lines)

def visualize_syntactic_monoid(min_dfa, out_prefix="syntactic"):
    states, alphabet, letter_transf = build_letter_generators(min_dfa)

    gens = [letter_transf[a] for a in alphabet]
    S = FroidurePin(gens)

    # enumerate elements in the order FroidurePin uses
    elements = list(S)

    J = greens_relations = S.greens_relations()

    # compute shortest word reps
    reps = shortest_representatives(alphabet, letter_transf, n_states=len(states), max_len=50)

    # node labels: show transformation + representative word
    node_labels = {}
    for i, x in enumerate(elements):
        k = tuple(list(x))
        w = reps.get(k, "?")
        # show epsilon nicely
        w_display = "ε" if w == "" else w
        node_labels[i] = f"{list(x)}\\nrep: {w_display}"

    # build Cayley graph
    wg = S.right_cayley_graph()

    # NOTE: in your version WordGraph has no dot export, so we build DOT manually
    dot = wordgraph_to_dot(wg, node_labels)

    with open(f"{out_prefix}.dot", "w") as f:
        f.write(dot)

    subprocess.run(
        ["dot", "-Tsvg", f"{out_prefix}.dot", "-o", f"{out_prefix}.svg"],
        check=True
    )

    print(f"Wrote {out_prefix}.svg")
    return S

# --- usage ---
min_dfa = dfa_to_efficient_dfa(nfa_to_dfa(regex_to_nfa("ab*")))
visualize_syntactic_monoid(min_dfa, out_prefix="abstar_monoid")