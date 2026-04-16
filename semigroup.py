from libsemigroups_pybind11.transf import Transf
from libsemigroups_pybind11 import FroidurePin
from collections import defaultdict
from graphviz import Digraph
import re
import itertools


############--- Syntactic semigroup computation and visualization ---##############
def build_letter_generators(min_dfa):
    """Returns alphabet, and dict letter -> letter transformation."""
    states = min_dfa.states

    state_index = {q: i for i, q in enumerate(states)}

    alphabet = sorted(min_dfa._input_symbols, key=str)

    # build transformations for each letter
    letter_transf = {}
    for a in alphabet:
        images = []
        for q in states:
            targets = min_dfa._transition_function(q, a)

            tgt = targets[0]
            images.append(state_index[tgt])

        letter_transf[a] = Transf(images)

    return alphabet, letter_transf


def build_fp_and_reps(min_dfa):
    alphabet, letter_transf = build_letter_generators(min_dfa)
    gens = [letter_transf[a] for a in alphabet]
    fp = FroidurePin(gens)
    fp.run()

    def factorize(j):
        factors_list = []
        while fp.length(j) > 1:
            factors_list.append(fp.final_letter(j))
            j = fp.prefix(j)
        # j is a generator
        factors_list.append(fp.final_letter(j))
        factors_list.reverse()
        return factors_list

    reps = {}
    for i, x in enumerate(fp):
        elem = tuple(list(x))
        factors = factorize(i)
        word = "".join(str(alphabet[g]) for g in factors)
        reps[elem] = word

    return reps, fp, alphabet

def compute_syntactic_semigroup(min_dfa):
    reps, _fp, _alphabet = build_fp_and_reps(min_dfa)
    return reps

def compute_syntactic_monoid(min_dfa):
    reps = compute_syntactic_semigroup(min_dfa)

    # add identity to reps if not already present
    n_states = len(next(iter(reps.keys())))
    identity = tuple(range(n_states))
    if identity not in reps:
        reps[identity] = ""

    return reps

############--- Green's relations from Cayley graphs ---##############
def kosaraju_sccs(n, adj):
    """Returns list of SCCs as sets of node indices."""
    visited = set()
    order = []

    def dfs1(a):
        stack = [(a, iter(adj.get(a, [])))]
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    visited.add(child)
                    stack.append((child, iter(adj.get(child, []))))
            except StopIteration:
                order.append(node)
                stack.pop()

    for v in range(n):
        if v not in visited:
            visited.add(v)
            dfs1(v)

    # reverse graph
    radj = defaultdict(set)
    for u, vs in adj.items():
        for v in vs:
            radj[v].add(u)

    visited2 = set()
    sccs = []

    def dfs2(b):
        second_dfs_scc = set()
        stack = [b]
        while stack:
            node = stack.pop()
            if node in visited2:
                continue
            visited2.add(node)
            second_dfs_scc.add(node)
            for nb in radj.get(node, []):
                if nb not in visited2:
                    stack.append(nb)
        return second_dfs_scc

    for v in reversed(order):
        if v not in visited2:
            scc = dfs2(v)
            sccs.append(scc)

    return sccs


def build_adj(wg):
    adj = defaultdict(set)
    for s in wg.nodes():
        for _label, target in wg.labels_and_targets(s):
            if target is not None:
                adj[s].add(target)
    return adj


def compute_green_classes_semigroup(min_dfa):
    """
    Returns:
      fp           : FroidurePin enumeration
      reps         : dict transf tuple -> the shortest representative word
      r_class      : dict node_index -> r_class_id  (from right Cayley graph SCCs)
      l_class      : dict node_index -> l_class_id  (from left Cayley graph SCCs)
      d_class      : dict node_index -> d_class_id
    """

    reps, fp, _alphabet = build_fp_and_reps(min_dfa)

    n = fp.size()

    # R-classes = SCCs of right Cayley graph
    rcg = fp.right_cayley_graph()
    r_adj = build_adj(rcg)
    r_sccs = kosaraju_sccs(n, r_adj)

    # L-classes = SCCs of left Cayley graph
    lcg = fp.left_cayley_graph()
    l_adj = build_adj(lcg)
    l_sccs = kosaraju_sccs(n, l_adj)

    node_to_r = {}
    for r_id, scc in enumerate(r_sccs):
        for node in scc:
            node_to_r[node] = r_id

    node_to_l = {}
    for l_id, scc in enumerate(l_sccs):
        for node in scc:
            node_to_l[node] = l_id

    # D-classes: union-find over R-classes that share an L-class
    parent = list(range(len(r_sccs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # group R-class ids by L-class
    l_to_r_classes = defaultdict(set)
    for node in range(n):
        l_to_r_classes[node_to_l[node]].add(node_to_r[node])

    for r_ids in l_to_r_classes.values():
        r_list = list(r_ids)
        for i in range(1, len(r_list)):
            union(r_list[0], r_list[i])

    node_to_d = {node: find(node_to_r[node]) for node in range(n)}

    return fp, reps, node_to_r, node_to_l, node_to_d

def compute_green_classes_monoid(min_dfa):
    _fp, reps, node_to_r, node_to_l, node_to_d = compute_green_classes_semigroup(min_dfa)

    fp_elems = list(reps.keys())
    n_states = len(fp_elems[0])
    identity = tuple(range(n_states))

    # only add identity if the semigroup doesn't already contain it
    if identity not in set(fp_elems):
        n = len(fp_elems)
        fp_elems.append(identity)
        # update reps so the egg-box can label the identity cell as ε
        reps[identity] = ""

        existing = set(node_to_r.values()) | set(node_to_l.values()) | set(node_to_d.values())
        new_id = max(existing) + 1 if existing else 0

        node_to_r[n] = new_id
        node_to_l[n] = new_id
        node_to_d[n] = new_id

    return _fp, reps, node_to_r, node_to_l, node_to_d


############--- Egg-box diagram ---##############
def build_eggbox_svg(fp, reps, node_to_r, node_to_l, node_to_d):
    fp_elems = list(reps.keys())
    n = len(fp_elems)

    def word(t):
        return reps.get(t, "?")

    # group nodes by D-class
    d_groups = defaultdict(list)
    for i in range(n):
        d_groups[node_to_d[i]].append(i)

    # sort D-classes by rank (size of image) descending
    def rank(j):
        return len(set(fp_elems[j]))

    d_groups_sorted = sorted(d_groups.values(), key=lambda d_nodes: -max(rank(j) for j in d_nodes))

    eggboxes = []
    for nodes in d_groups_sorted:
        # collect R and L class ids within this D-class
        r_ids = sorted({node_to_r[i] for i in nodes})
        l_ids = sorted({node_to_l[i] for i in nodes})

        r_index = {r: j for j, r in enumerate(r_ids)}
        l_index = {l: j for j, l in enumerate(l_ids)}

        # H-class cells: (l_row, r_col) -> list of words
        cells = defaultdict(list)
        for i in nodes:
            r = r_index[node_to_r[i]]
            l = l_index[node_to_l[i]]
            w = word(fp_elems[i])
            cells[(r, l)].append("ε" if w == "" else w)

        # sort words within each H-class
        for key in cells:
            cells[key].sort(key=lambda wrd: (wrd != "ε", len(wrd), wrd))

        eggboxes.append({
            "n_rows": len(r_ids),
            "n_cols": len(l_ids),
            "cells": cells,
        })

    return plot_eggbox_svg(eggboxes)

def visualize_syntactic_semigroup(min_dfa):
    fp, reps, node_to_r, node_to_l, node_to_d = compute_green_classes_semigroup(min_dfa)
    return build_eggbox_svg(fp, reps, node_to_r, node_to_l, node_to_d)


def visualize_syntactic_monoid(min_dfa):
    fp, reps, node_to_r, node_to_l, node_to_d = compute_green_classes_monoid(min_dfa)
    return build_eggbox_svg(fp, reps, node_to_r, node_to_l, node_to_d)

def plot_eggbox_svg(eggboxes):
    dot = Digraph(format="svg")
    dot.attr(rankdir="TB")
    dot.attr("node", shape="plaintext")

    for i, box in enumerate(eggboxes):
        n_rows = box["n_rows"]
        n_cols = box["n_cols"]
        cells = box["cells"]

        table = ['<table border="1" cellborder="1" cellspacing="0">',]

        for r in range(n_rows):
            table.append("<tr>")
            for c in range(n_cols):
                words = cells.get((r, c), [])
                if not words:
                    table.append("<td width='100'> </td>")
                else:
                    display = ", ".join(words)
                    table.append(f"<td width='100' align='center'><b>{display}</b></td>")
            table.append("</tr>")

        table.append("</table>")
        dot.node(f"D{i}", label="<" + "".join(table) + ">")
    for i in range(len(eggboxes) - 1):
        dot.edge(f"D{i}", f"D{i+1}", style="invis")

    return dot.pipe(format="svg").decode("utf-8")


############--- Latex multiplication table ---##############
def mul(f, g):
    return tuple(g[f[q]] for q in range(len(f)))

def sort_elements(reps):
    elements = list(reps.keys())

    # find sink element
    sink = None
    for z in elements:
        if all(mul(z, x) == z and mul(x, z) == z for x in elements):
            sink = z
            break

    def sort_key(e):
        if reps[e] == "":
            return 0, ""
        if e == sink:
            return 2, ""
        return 1, reps[e]

    return sorted(elements, key=sort_key)

def build_multiplication_table(reps):
    elements = sort_elements(reps)
    table = []
    for f in elements:
        row = []
        for g in elements:
            fg = mul(f, g)
            row.append(reps[fg])
        table.append(row)
    return table, elements


def reps_to_latex(reps):
    table, elements = build_multiplication_table(reps)

    def label(w):
        return r"\varepsilon" if w == "" else w

    labels = [label(reps[e]) for e in elements]
    n = len(labels)

    col_spec = "c|" + "c" * n

    lines = [r"\[", r"\begin{array}{" + col_spec + "}", r"\cdot & " + " & ".join(labels) + r" \\", r"\hline"]

    for i, row in enumerate(table):
        cells = [label(v) for v in row]
        lines.append(labels[i] + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\end{array}" + r"\]")
    return "\n".join(lines)


def multiplication_table_to_latex_semigroup(min_dfa):
    reps = compute_syntactic_semigroup(min_dfa)
    return reps_to_latex(reps)


def multiplication_table_to_latex_monoid(min_dfa):
    reps = compute_syntactic_monoid(min_dfa)
    return reps_to_latex(reps)


############--- Equation checking ---##############
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


def eval_expr(expr, assignment, omega_f):
    if isinstance(expr, str):  # variable
        return assignment[expr]

    kind = expr[0]

    if kind == "concat":
        return mul(
            eval_expr(expr[1], assignment, omega_f),
            eval_expr(expr[2], assignment, omega_f)
        )

    if kind == "pow":
        base = eval_expr(expr[1], assignment, omega_f)
        exp = expr[2]

        if exp == "omega":
            return omega_f(base)
        else:
            return repeat(base, exp)
    return None


def omega(x):
    """
    Compute x^w (the idempotent) for a semigroup element x
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

    raise RuntimeError("No idempotent found")


def vars_in(expr):
    if isinstance(expr, str):
        return {expr}

    kind = expr[0]

    if kind == "concat":
        return vars_in(expr[1]) | vars_in(expr[2])

    if kind == "pow":
        return vars_in(expr[1])

    raise ValueError(f"Unknown expression node: {expr}")


def check_equation_ast(left, right, elements, omega_f):
    # 1. collect all variables appearing anywhere in the ASTs
    variables = sorted(vars_in(left) | vars_in(right))

    # 2. try all assignments of variables to elements
    for values in itertools.product(elements, repeat=len(variables)):
        assignment = dict(zip(variables, values))

        # 3. evaluate both sides under this assignment
        lhs_val = eval_expr(left, assignment, omega_f)
        rhs_val = eval_expr(right, assignment, omega_f)

        # 4. if they differ, we found a counterexample
        if lhs_val != rhs_val:
            return False, assignment

    # 5. all assignments satisfied the equation
    return True, None


def check_equation_sat(reps, equation):
    # 1. parse user equation into ASTs
    left, right = parse_equation_to_ast(equation)

    elements = list(reps.keys())

    # 2. check equation semantically
    ok, assignment = check_equation_ast(left, right, elements, omega)

    if ok: return {"holds": True, "counterexample": None}

    # 3. build readable counterexample (VARIABLE → WORD)
    counterexample = {}
    for var, elem in assignment.items():
        word = reps[elem]
        counterexample[var] = "ε" if word == "" else word

    return {"holds": False, "counterexample": counterexample}


def check_equations_with_reps(reps, equations_text):
    results = []

    # split lines, ignore empty ones
    lines = [
        line.strip()
        for line in equations_text.splitlines()
        if line.strip()
    ]

    for eq in lines:
        try:
            res = check_equation_sat(reps, eq)
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


def check_equations_batch_semigroup(min_dfa, equations_text):
    reps = compute_syntactic_semigroup(min_dfa)
    return check_equations_with_reps(reps, equations_text)


def check_equations_batch_monoid(min_dfa, equations_text):
    reps = compute_syntactic_monoid(min_dfa)
    return check_equations_with_reps(reps, equations_text)