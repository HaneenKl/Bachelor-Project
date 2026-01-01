from pyformlang import regular_expression

from libsemigroups_pybind11.transf import Transf
from libsemigroups_pybind11 import Konieczny

from collections import deque

from pyformlang.finite_automaton import State
from collections import defaultdict
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

from graphviz import Digraph


from arden_to_automata import Automaton




def automaton_to_pyformlang_min_dfa(auto: Automaton):
    nfa = NondeterministicFiniteAutomaton()

    # Transitions (erzeugen implizit Zustände)
    for (src, sym), dst in auto.transitions.items():
        nfa.add_transition(src, sym, dst)

    # Start state
    nfa.add_start_state(auto.start)

    # Accepting states
    for q in auto.accepting:
        nfa.add_final_state(q)

    dfa = nfa.to_deterministic()
    min_dfa = dfa.minimize()

    min_dfa = complete_dfa(min_dfa)

    return min_dfa


def regex_to_pyformlang_min_dfa(regex):
    regex_parse = regular_expression.Regex(regex)
    nfa = regex_parse.to_epsilon_nfa()
    dfa = nfa.to_deterministic()
    min_dfa = dfa.minimize()
    min_dfa = complete_dfa(min_dfa)

    return min_dfa


def complete_dfa(dfa):
    alphabet = set(dfa._input_symbols)
    phi = State("φ")

    # Prüfen, ob φ überhaupt nötig ist
    need_phi = False
    for q in dfa.states:
        for a in alphabet:
            if len(dfa._transition_function(q, a)) == 0:
                need_phi = True
                break

    if not need_phi:
        return dfa  # DFA ist bereits vollständig

    # fehlende Übergänge → φ
    for q in list(dfa.states):
        for a in alphabet:
            if len(dfa._transition_function(q, a)) == 0:
                dfa.add_transition(q, a, phi)

    # φ schleift auf sich selbst
    for a in alphabet:
        dfa.add_transition(phi, a, phi)

    return dfa


def build_letter_generators(min_dfa):
    # stable order: start state first, then the rest sorted by str
    start = min_dfa.start_state
    rest = [q for q in min_dfa.states if q != start]
    rest.sort(key=str)
    states = [start] + rest

    state_index = {q: i for i, q in enumerate(states)}

    alphabet = sorted(min_dfa._input_symbols, key=str)
    print("alphabet:", alphabet)

    # build transformations for each letter
    letter_transf = {}
    for a in alphabet:
        images = []
        for q in states:
            targets = min_dfa._transition_function(q, a)
            print(a, q, targets)

            tgt = targets[0]
            images.append(state_index[tgt])

        letter_transf[a] = Transf(images)

    return states, alphabet, letter_transf


def shortest_representatives(alphabet, letter_transf, n_states, max_len=30):
    """
    Returns dict: transformation_tuple -> shortest word (string)
    """
    id_t = Transf(list(range(n_states)))
    rep = {tuple(id_t[i] for i in range(n_states)): ""}

    q = deque([id_t])
    while q:
        t = q.popleft()
        t_key = tuple(t[i] for i in range(n_states))
        w = rep[t_key]

        if len(w) >= max_len:
            continue

        for a in alphabet:
            # word concat on the right corresponds to multiplying by generator on the right
            t2 = t * letter_transf[a]
            k2 = tuple(list(t2))
            if k2 not in rep:
                rep[k2] = w + str(a)
                q.append(t2)

    return rep


def kernel(t):
    blocks = {}
    for i, v in enumerate(t):
        blocks.setdefault(v, set()).add(i)
    return frozenset(frozenset(b) for b in blocks.values())


def image(t):
    return frozenset(t)

def d_key_for(ker, img):
    # rank + sorted kernel block sizes: stable D-like signature for transformations
    return (len(img), tuple(sorted(len(b) for b in ker)))

def build_eggboxes_from_H(H_classes):
    """
    Returns list of eggboxes:
      eggbox = {
        "dkey": ...,
        "rows": [ker1, ker2, ...],
        "cols": [img1, img2, ...],
        "cell": dict[(ri,ci)] -> H_key (ker,img)
      }
    """
    groups = defaultdict(list)
    for H_key in H_classes.keys():
        ker, img = H_key
        groups[d_key_for(ker, img)].append(H_key)

    eggboxes = []
    for dkey, hkeys in groups.items():
        kernels = sorted({ker for (ker, img) in hkeys}, key=lambda k: (len(k), sorted(map(len, k))))
        images  = sorted({img for (ker, img) in hkeys}, key=lambda s: (len(s), sorted(s)))

        ker_index = {k:i for i,k in enumerate(kernels)}
        img_index = {s:j for j,s in enumerate(images)}

        cell = {}
        for (ker, img) in hkeys:
            cell[(ker_index[ker], img_index[img])] = (ker, img)

        eggboxes.append({"dkey": dkey, "rows": kernels, "cols": images, "cell": cell})

    # nice ordering: higher rank first
    eggboxes.sort(key=lambda b: (-b["dkey"][0], b["dkey"][1]))
    return eggboxes

def plot_eggbox_svg(eggboxes, H_rep):
    dot = Digraph(format="svg")
    dot.attr(rankdir="TB")
    dot.attr("node", shape="plaintext")

    for i, box in enumerate(eggboxes):
        rows = len(box["rows"])
        cols = len(box["cols"])

        table = []
        table.append('<table border="1" cellborder="1" cellspacing="0">')

        # ---- Header row (D-class label) ----
        table.append(
            f'<tr><td colspan="{cols}" bgcolor="lightgray">'
            f'<b>D-class {i}</b>'
            f'</td></tr>'
        )

        # ---- Eggbox cells ----
        for r in range(rows):
            table.append("<tr>")
            for c in range(cols):
                hkey = box["cell"].get((r, c))
                if hkey is None:
                    table.append("<td></td>")
                else:
                    w = H_rep[hkey]
                    w_disp = "ε" if w == "" else w
                    table.append(f"<td align='center'><b>{w_disp}</b></td>")
            table.append("</tr>")

        table.append("</table>")

        dot.node(f"D{i}", label="<" + "".join(table) + ">")

    # Return SVG as UTF-8 string (exactly like your automaton plot)
    return dot.pipe(format="svg").decode("utf-8")



def visualize_syntactic_monoid(min_dfa):

    states, alphabet, letter_transf = build_letter_generators(min_dfa)

    print("HALO")
    gens = [letter_transf[a] for a in alphabet]
    print(gens)
    print("HALLOOO")
    K = Konieczny(gens)
    d_classes = list(K.D_classes())

    for i, d in enumerate(d_classes):
        print(f"D-class {d}:")

    # compute shortest word reps
    reps = shortest_representatives(alphabet, letter_transf, n_states=len(states), max_len=50)


    H_classes = defaultdict(list)

    for t, word in reps.items():
        key = (kernel(t), image(t))
        H_classes[key].append((t, word))

    H_rep = {}

    for key, elems in H_classes.items():
        # elems = [(t, word), ...]
        rep_word = min(elems, key=lambda x: len(x[1]))[1]
        H_rep[key] = rep_word


    print(H_rep)

    H_rep_filtered = {
        key: word
        for key, word in H_rep.items()
        if K.contains(Transf(list(next(iter(key[1])) for _ in range(len(states)))))
    }

    print(H_rep_filtered)

    for (ker, img), word in H_rep.items():
        w = "ε" if word == "" else word
        print(f"H-class represented by {w}")
        print(f"  kernel: {sorted(map(sorted, ker))}")
        print(f"  image:  {sorted(img)}")
        print()

    eggboxes = build_eggboxes_from_H(H_classes)
    svg = plot_eggbox_svg(eggboxes, H_rep)
    return svg


# --- usage ---

#min_dfaa= regex_to_pyformlang_min_dfa("abb*c*")
#visualize_syntactic_monoid(min_dfaa)
