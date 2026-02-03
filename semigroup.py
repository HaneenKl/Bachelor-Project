from libsemigroups_pybind11.transf import Transf
from collections import deque

from collections import defaultdict

from graphviz import Digraph

import re

import itertools


############--- Syntactic semigroup computation and visualization ---##############
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
    Returns dict: transformation_tuple -> the shortest word (string)
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
            print("t22", t2)
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
    return len(img), tuple(sorted(len(b) for b in ker))


def build_eggboxes_from_h(h_classes):
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
    for H_key in h_classes.keys():
        ker, img = H_key
        print("ker", ker)
        groups[d_key_for(ker, img)].append(H_key)

    eggboxes = []
    for dkey, hkeys in groups.items():
        kernels = sorted({ker for (ker, img) in hkeys}, key=lambda k: (len(k), sorted(map(len, k))))
        images = sorted({img for (ker, img) in hkeys}, key=lambda s: (len(s), sorted(s)))

        ker_index = {k: i for i, k in enumerate(kernels)}
        img_index = {s: j for j, s in enumerate(images)}

        cell = {}
        for (ker, img) in hkeys:
            cell[(ker_index[ker], img_index[img])] = (ker, img)

        eggboxes.append({"dkey": dkey, "rows": kernels, "cols": images, "cell": cell})

    # nice ordering: higher rank first
    eggboxes.sort(key=lambda b: (-b["dkey"][0], b["dkey"][1]))
    return eggboxes


def plot_eggbox_svg(eggboxes, h_rep):
    dot = Digraph(format="svg")
    dot.attr(rankdir="TB")
    dot.attr("node", shape="plaintext")

    for i, box in enumerate(eggboxes):
        rows = len(box["rows"])
        cols = len(box["cols"])

        table = ['<table border="1" cellborder="1" cellspacing="0">', f'<tr><td colspan="{cols}" bgcolor="lightgray">'
                                                                      f'<b>{i} D-class</b>'
                                                                      f'</td></tr>']

        # ---- Header row (D-class label) ----

        # ---- Eggbox cells ----
        for r in range(rows):
            table.append("<tr>")
            for c in range(cols):
                hkey = box["cell"].get((r, c))
                if hkey is None:
                    table.append("<td></td>")
                else:
                    w = h_rep[hkey]
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
    print("elemnttss", elements)

    return elements, reps


def visualize_syntactic_monoid(min_dfa):
    elements, reps = compute_syntactic_semigroup(min_dfa)

    h_classes = defaultdict(list)

    for t, word in reps.items():
        key = (kernel(t), image(t))
        h_classes[key].append((t, word))

    print("H classes:", h_classes)

    h_rep = {}

    for key, elems in h_classes.items():
        words = [
            "ε" if word == "" else word
            for (_, word) in elems
        ]

        words.sort(key=lambda w: (w != "ε", len(w), w))

        h_rep[key] = words

    print("Hreppp", h_rep)

    for (ker, img), words in h_rep.items():
        print("  words:", ", ".join(words))
        print(f"  kernel: {sorted(map(sorted, ker))}")
        print(f"  image:  {sorted(img)}")
        print()

    eggboxes = build_eggboxes_from_h(h_classes)
    svg = plot_eggbox_svg(eggboxes, h_rep)
    return svg


############--- Equation checking ---##############
def mul(f, g):
    return [g[f[q]] for q in range(len(f))]




import re

def tokenize(s):
    return re.findall(r'\(|\)|\^|=|w|\d+|[a-zA-Z]+', s)

def parse_equation_to_ast(s):
    tokens = tokenize(s)
    i = 0

    def parse_expr():
        nonlocal i
        node = parse_factor()
        while i < len(tokens) and tokens[i] not in [')', '=']:
            node = ("concat", node, parse_factor())
        return node

    def parse_factor():
        nonlocal i

        if tokens[i] == '(':
            i += 1
            node = parse_expr()
            if tokens[i] != ')':
                raise ValueError("Missing ')'")
            i += 1
        else:
            node = tokens[i]
            i += 1

        # optional power
        if i < len(tokens) and tokens[i] == '^':
            i += 1
            if tokens[i] == 'w':
                node = ("pow", node, "omega")
            else:
                node = ("pow", node, int(tokens[i]))
            i += 1

        return node

    left = parse_expr()

    if tokens[i] != '=':
        raise ValueError("Expected '=' in equation")
    i += 1

    right = parse_expr()

    if i != len(tokens):
        raise ValueError("Unexpected tokens at end")

    return left, right


def repeat(x, n):
    r = x
    for _ in range(n - 1):
        r = mul(r, x)
    return r


def eval_expr(expr, assignment, omega):
    if isinstance(expr, str):  # variable
        return assignment[expr]

    kind = expr[0]

    if kind == "concat":
        return mul(
            eval_expr(expr[1], assignment, omega),
            eval_expr(expr[2], assignment, omega)
        )

    if kind == "pow":
        base = eval_expr(expr[1], assignment, omega)
        exp = expr[2]

        if exp == "omega":
            return omega(base)
        else:
            return repeat(base, exp)


def omega(x):
    """
    Compute x^ω (the idempotent in <x>) for a finite semigroup element x
    """
    seen = {}
    seq = []

    cur = x
    while tuple(cur) not in seen:
        seen[tuple(cur)] = len(seq)
        seq.append(cur)
        cur = mul(cur, x)

    # cycle starts here
    cycle_start = seen[tuple(cur)]
    cycle = seq[cycle_start:]

    # find idempotent in the cycle
    for e in cycle:
        if mul(e, e) == e:
            return e

    raise RuntimeError("No idempotent found (should be impossible in finite semigroup)")



def vars_in(expr):
    if isinstance(expr, str):
        return {expr}

    kind = expr[0]

    if kind == "concat":
        return vars_in(expr[1]) | vars_in(expr[2])

    if kind == "pow":
        return vars_in(expr[1])

    raise ValueError(f"Unknown expression node: {expr}")



def check_equation_ast(left, right, elements, omega):
    # 1. collect all variables appearing anywhere in the ASTs
    variables = sorted(vars_in(left) | vars_in(right))

    # 2. try all assignments of variables to elements
    for values in itertools.product(elements, repeat=len(variables)):
        assignment = dict(zip(variables, values))

        # 3. evaluate both sides under this assignment
        lhs_val = eval_expr(left, assignment, omega)
        rhs_val = eval_expr(right, assignment, omega)

        # 4. if they differ, we found a counterexample
        if lhs_val != rhs_val:
            return False, assignment

    # 5. all assignments satisfied the equation
    return True, None



def check_equation_sat(elements, reps, equation):
    """
    Returns:
      {
        "holds": bool,
        "counterexample": dict[str, str] | None
      }
    """

    # 1. parse user equation into ASTs
    left, right = parse_equation_to_ast(equation)

    # 2. check equation semantically
    ok, assignment = check_equation_ast(left, right, elements, omega)

    if ok:
        return {
            "holds": True,
            "counterexample": None
        }

    # 3. build readable counterexample (VARIABLE → WORD)
    counterexample = {}
    for var, elem in assignment.items():
        word = reps[tuple(elem)]
        counterexample[var] = "ε" if word == "" else word

    return {
        "holds": False,
        "counterexample": counterexample
    }


def check_equations_batch(elements, reps, equations_text):
    """
    Check multiple equations given as a multiline string.

    Parameters
    ----------
    elements : list
        Semigroup elements (transformations)
    reps : dict
        Maps tuple(element) -> representative word
    equations_text : str
        Multiline user input, one equation per line

    Returns
    -------
    list of dicts, one per equation:
      {
        "equation": str,
        "holds": bool,
        "counterexample": dict[str, str] | None
      }
    """

    results = []

    # split lines, ignore empty ones
    lines = [
        line.strip()
        for line in equations_text.splitlines()
        if line.strip()
    ]

    for eq in lines:
        try:
            res = check_equation_sat(elements, reps, eq)
            results.append({
                "equation": eq,
                "holds": res["holds"],
                "counterexample": res["counterexample"]
            })
        except Exception as e:
            # syntax or parsing error
            results.append({
                "equation": eq,
                "holds": False,
                "counterexample": None,
                "error": str(e)
            })

    return results
