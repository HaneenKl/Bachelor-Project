import re
import itertools
from semigroup import compute_syntactic_semigroup, compute_syntactic_monoid, compute_stable_semigroup, mul


# ==============================================================================
# EQUATION PARSING AND EVALUATION
# ==============================================================================
def tokenize(s):
    return re.findall(r'\(|\)|\^|=|-|w|\d+|[a-zA-Z]+', s)


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

        # optional power: ^w, ^(w-1), or ^<integer>
        if i < len(tokens) and tokens[i] == '^':
            i += 1
            if tokens[i] == '(':
                # expect (w-1)
                i += 1  # consume '('
                if tokens[i] != 'w':
                    raise ValueError("Expected 'w' inside '^(...)'")
                i += 1  # consume 'w'
                if tokens[i] != '-':
                    raise ValueError("Expected '-' after 'w' inside '^(...)'")
                i += 1  # consume '-'
                if tokens[i] != '1':
                    raise ValueError(
                        "Only '^(w-1)' is supported, not '^(w-{})'".format(tokens[i])
                    )
                i += 1  # consume '1'
                if tokens[i] != ')':
                    raise ValueError("Expected ')' to close '^(...)'")
                i += 1  # consume ')'
                node = ("pow", node, "omega-1")
            elif tokens[i] == 'w':
                node = ("pow", node, "omega")
                i += 1
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


def omega(x):
    """
    Compute x^w (the idempotent) for a semigroup element x
    """
    cycle = compute_cycle(x)

    # find idempotent in the cycle
    for e in cycle:
        if mul(e, e) == e:
            return e
    raise RuntimeError("No idempotent found in cycle")


def omega_minus_1(x):
    """
    Compute x^(w-1) for a semigroup element x.
    """
    cycle = compute_cycle(x)

    # find the idempotent e = x^w in the cycle
    e = None
    for elem in cycle:
        if mul(elem, elem) == elem:
            e = elem
            break
    if e is None:
        raise RuntimeError("No idempotent found in cycle")

    # x^(w-1) is the element y in the cycle such that y * x = e
    for y in cycle:
        if mul(y, x) == e:
            return y

    raise RuntimeError("x^(w-1) not found in cycle")


def repeat(x, n):
    r = x
    for _ in range(n - 1):
        r = mul(r, x)
    return r


def eval_expr(expr, assignment, omega_f):
    # variable
    if isinstance(expr, str):
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
        elif exp == "omega-1":
            return omega_minus_1(base)
        else:
            return repeat(base, exp)
    return None


def compute_cycle(x):
    seen = {}
    seq = []

    cur = x
    while tuple(cur) not in seen:
        seen[tuple(cur)] = len(seq)
        seq.append(cur)
        cur = mul(cur, x)

    cycle_start = seen[tuple(cur)]
    return seq[cycle_start:]


def vars_in(expr):
    if isinstance(expr, str):
        return {expr}

    kind = expr[0]

    if kind == "concat":
        return vars_in(expr[1]) | vars_in(expr[2])

    if kind == "pow":
        return vars_in(expr[1])

    raise ValueError(f"Unknown expression node: {expr}")


# =============================================================================
# ADD SPACING BETWEEN SYMBOLS IN EQUATION
# =============================================================================
def add_spacing_to_equation(s):
    out = []
    i = 0
    n = len(s)

    def emit_and_maybe_space(tmp, next_char):
        out.append(tmp)
        # space if next char starts a new symbol (letter or '(')
        if next_char and (next_char.isalpha() or next_char == '('):
            out.append(' ')

    while i < n:
        c = s[i]

        # '^(w-1)' — copy verbatim
        if c == '^' and i + 1 < n and s[i + 1] == '(':
            j = s.find(')', i)
            if j == -1:
                out.append(s[i:])
                break
            chunk = s[i:j + 1]
            i = j + 1
            emit_and_maybe_space(chunk, s[i] if i < n else '')
            continue

        # '^w' or '^<digits>'
        if c == '^':
            i += 1
            exp_start = i
            if i < n and s[i] == 'w':
                i += 1
            else:
                while i < n and s[i].isdigit():
                    i += 1
            chunk = '^' + s[exp_start:i]
            emit_and_maybe_space(chunk, s[i] if i < n else '')
            continue

        # plain char
        out.append(c)
        i += 1
        if i < n:
            nxt = s[i]
            if c.isalpha() and (nxt.isalpha() or nxt == '('):
                out.append(' ')

    result = ''.join(out)
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s*=\s*', ' = ', result)
    return result.strip()


# ==============================================================================
# EQUATION CHECKING LOGIC
# ==============================================================================
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


# ==============================================================================
# CHECK A GIVEN BATCH OF EQUATIONS FOR A SYNTACTIC SEMIGROUP
# ==============================================================================
def check_equations_batch_semigroup(min_dfa, equations_text):
    reps, _alphabet = compute_syntactic_semigroup(min_dfa)
    return check_equations_with_reps(reps, equations_text)


# ==============================================================================
# CHECK A GIVEN BATCH OF EQUATIONS FOR A SYNTACTIC MONOID
# ==============================================================================
def check_equations_batch_monoid(min_dfa, equations_text):
    reps = compute_syntactic_monoid(min_dfa)
    return check_equations_with_reps(reps, equations_text)


# ==============================================================================
# CHECK A GIVEN BATCH OF EQUATIONS FOR A STABLE SEMIGROUP
# ==============================================================================
def check_equations_batch_stable_semigroup(min_dfa, equations_text):
    reps, alphabet = compute_syntactic_semigroup(min_dfa)
    stable = compute_stable_semigroup(reps)
    reps_stable = {e: w for e, w in reps.items() if e in stable}
    return check_equations_with_reps(reps_stable, equations_text)
