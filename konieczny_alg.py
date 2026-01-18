from pyformlang import regular_expression

from libsemigroups_pybind11.transf import Transf
from libsemigroups_pybind11 import Konieczny
from libsemigroups_pybind11 import FroidurePin

import subprocess
from collections import deque

from pyformlang.finite_automaton import State
from collections import defaultdict
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

from graphviz import Digraph


from arden_to_automata import Automaton

import re

import itertools






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


def shortest_representatives(alphabet, letter_transf, n_states):
    """
    Returns dict: transformation_tuple -> shortest word (string)
    """
    id_t = Transf(list(range(n_states)))
    print("id_t", id_t)
    rep = {tuple(id_t[i] for i in range(n_states)): ""}
    print("epiis", rep)

    q = deque([id_t])
    while q:
        t = q.popleft()
        t_key = tuple(t[i] for i in range(n_states))
        w = rep[t_key]


        for a in alphabet:
            print("ttt", t)
            print(letter_transf[a])
            # word concat on the right corresponds to multiplying by generator on the right
            t2 = t * letter_transf[a]
            print("t22",t2)
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
        print("ker",ker)
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


def compute_syntactic_semigroup(min_dfa):
    states, alphabet, letter_transf = build_letter_generators(min_dfa)

    reps = shortest_representatives(
        alphabet,
        letter_transf,
        n_states=len(states))

    elements = [list(t) for t in reps.keys()]
    print("elemnttss",elements)

    return elements, reps


def visualize_syntactic_monoid(min_dfa):
    elements, reps = compute_syntactic_semigroup(min_dfa)

    #gens = [letter_transf[a] for a in alphabet]
    #print(gens)

   # K = Konieczny(gens)

    H_classes = defaultdict(list)

    for t, word in reps.items():
        key = (kernel(t), image(t))
        H_classes[key].append((t, word))


    print("H classes:", H_classes)

    H_rep = {}

    for key, elems in H_classes.items():
        words = [
            "ε" if word == "" else word
            for (_, word) in elems
        ]

        # ε immer zuerst, danach nach Länge und lexikographisch
        words.sort(key=lambda w: (w != "ε", len(w), w))

        H_rep[key] = words


    print("Hreppp",H_rep)


    for (ker, img), words in H_rep.items():
        print("  words:", ", ".join(words))
        print(f"  kernel: {sorted(map(sorted, ker))}")
        print(f"  image:  {sorted(img)}")
        print()

    eggboxes = build_eggboxes_from_H(H_classes)
    svg = plot_eggbox_svg(eggboxes, H_rep)
    return svg

def mul(f, g):
    return [g[f[q]] for q in range(len(f))]


def parse_side(side):
    """
    Parses something like:
      'x^3 y z^2'
    into:
      ['x','x','x','y','z','z']
    """
    tokens = side.strip().split()
    word = []

    for tok in tokens:
        m = re.fullmatch(r'([a-zA-Z]+)(\^(\d+))?', tok)
        if not m:
            raise ValueError(f"Invalid token: {tok}")

        var = m.group(1)
        power = int(m.group(3)) if m.group(3) else 1

        word.extend([var] * power)

    return word


def parse_equation(eq):
    if "=" not in eq:
        raise ValueError("Equation must contain '='")

    left, right = eq.split("=")
    return parse_side(left), parse_side(right)

def eval_word(word, assignment):
    value = assignment[word[0]]
    for v in word[1:]:
        value = mul(value, assignment[v])
    return value

def check_equation(left, right, elements):
    variables = sorted(set(left + right))

    for values in itertools.product(elements, repeat=len(variables)):
        assignment = dict(zip(variables, values))

        if eval_word(left, assignment) != eval_word(right, assignment):
            return False, assignment

    return True, None



def check_equation_sat(elements, reps, equation):
        """
        Returns:
          {
            "holds": bool,
            "counterexample": dict[str, str] | None
          }
        """
        left, right = parse_equation(equation)
        ok, ce = check_equation(left, right, elements)

        if ok:
            return {
                "holds": True,
                "counterexample": None
            }

        # build readable counterexample
        counterexample = {}
        for var, elem in ce.items():
            word = reps[tuple(elem)]
            counterexample[var] = "ε" if word == "" else word

        return {
            "holds": False,
            "counterexample": counterexample
        }

def cayley_graph_svg(wg, node_labels, alphabet, title=None):
    dot = Digraph(format="svg")
    dot.attr(rankdir="LR")

    dot.attr("node", shape="circle", width="0.6", fixedsize="true")
    dot.attr("edge", fontsize="12")

    if title is not None:
        dot.attr(label=title, labelloc="t", fontsize="16")

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
    S = FroidurePin(gens)
    reps = shortest_representatives(alphabet, letter_transf, n_states=len(states))

    # node labels: show transformation + representative word
    node_labels = {}
    for i, x in enumerate(list(S)):
        k = tuple(list(x))
        w = reps.get(k, "?")
        # show epsilon nicely
        w_display = "ε" if w == "" else w
        node_labels[i] = w_display
    return S, node_labels

def right_cayley_graph_svg(min_dfa):
    S, node_labels = froidure_pin_alg(min_dfa)

    # build Cayley graph
    wg = S.right_cayley_graph()

    return cayley_graph_svg(
        wg, node_labels, sorted(min_dfa._input_symbols, key=str), title="Right Cayley Graph"
    )


def left_cayley_graph_svg(min_dfa):
    S, node_labels = froidure_pin_alg(min_dfa)

    # build Cayley graph
    wg = S.left_cayley_graph()

    return cayley_graph_svg(
        wg, node_labels, sorted(min_dfa._input_symbols, key=str), title="Left Cayley Graph"
    )


def add_spacing(s):
    """
    Insert spaces between consecutive letters.
    Example: 'abc' -> 'a b c'
    """
    return re.sub(r'([a-zA-Z])(?=[a-zA-Z])', r'\1 ', s)

# --- usage ---

min_dfaa= regex_to_pyformlang_min_dfa("(a a)*")
#visualize_syntactic_monoid(min_dfaa)
right_cayley_graph_svg(min_dfaa)
left_cayley_graph_svg(min_dfaa)