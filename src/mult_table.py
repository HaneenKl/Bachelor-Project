from semigroup import mul, compute_syntactic_semigroup, compute_syntactic_monoid, compute_stable_semigroup



#=================================================================
# BUILD LATEX MULTIPLICATION TABLE
#=================================================================
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


# ==================================================================
# BUILD LATEX MULTIPLICATION TABLE OF SYNTACTIC SEMIGROUP
# ==================================================================
def multiplication_table_to_latex_semigroup(min_dfa):
    reps, _alphabet = compute_syntactic_semigroup(min_dfa)
    fp_elements = list(reps.keys())
    n_states = len(fp_elements[0])
    identity = tuple(range(n_states))
    if identity in set(fp_elements):
        reps[identity] = ""

    return reps_to_latex(reps)


# ==================================================================
# BUILD LATEX MULTIPLICATION TABLE OF SYNTACTIC MONOID
# ==================================================================
def multiplication_table_to_latex_monoid(min_dfa):
    reps = compute_syntactic_monoid(min_dfa)
    fp_elements = list(reps.keys())
    n_states = len(fp_elements[0])
    identity = tuple(range(n_states))
    reps[identity] = ""

    return reps_to_latex(reps)


# ==================================================================
# BUILD LATEX MULTIPLICATION TABLE OF STABLE SEMIGROUP
# ==================================================================
def multiplication_table_to_latex_stable_semigroup(min_dfa):
    reps, alphabet = compute_syntactic_semigroup(min_dfa)
    stable = compute_stable_semigroup(reps)
    reps_stable = {e: w for e, w in reps.items() if e in stable}
    fp_elements = list(reps_stable.keys())
    if fp_elements:
        n_states = len(fp_elements[0])
        identity = tuple(range(n_states))
        if identity in set(fp_elements):
            reps_stable[identity] = ""

    return reps_to_latex(reps_stable)
