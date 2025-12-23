"""
the method gets a String that is a list of equations of the form R = AR + B that are separated by semicolons
and returns a regex that is equivalent to the equations

Example input:
A = aA + bB + epsilon;
B = aA + aB;
"""
import sympy as sp
import re


def arden_to_regex(equations: dict[str, list[str]]):
    # Copy so original is not modified
    eq = {v: terms[:] for v, terms in equations.items()}
    solved = {}  # variable → regex solution
    eq_dup = eq.copy()

    while len(solved) < len(eq):
        progress = False

        # Try to find a variable in Arden form
        for var, terms in list(eq.items()):
            if var in solved:
                continue

            in_form, r, s = is_arden_form(var, terms)
            if not in_form:
                continue

            # Solve using Arden
            solution = arden_solution(r, s)
            solved[var] = solution
            print(f"Solved {var} = {solution}")

            # Substitute solution into all other equations
            substitute(eq, solution, var)

            # Mark this equation solved
            del eq[var]
            progress = True
            break

        if not progress:
            #debug:
            print("Current equations:")
            for v, t in eq.items():
                print(f"  {v} = {' + '.join(t)}")
            print("No Arden form — performing substitution to create one.")

            # Perform substitution to create Arden form
            # pick variable to eliminate
            if len(eq_dup) == 0:
                eq_dup = eq.copy()
            elim = next(iter(eq_dup))
            elim_terms = eq_dup[elim]

            # substitute elim everywhere else
            substitute(eq, f"({'+'.join(elim_terms)})", elim)

            # remove elim from the system
            del eq_dup[elim]

            continue
    # Return regex for the FIRST variable in original equations (A by default)
    start_var = list(equations.keys())[0]
    print(f"Final solution for {start_var}: {solved[start_var]}")

    return solved[start_var]


def arden_solution(r, s):
    if r == "":
        r = "ε"  # meaning empty string concatenation
    return f"({r})*({s})"


def is_arden_form(variable, terms):
    # separate self-loop terms and others
    self_terms = [t for t in terms if variable in t]
    other_terms = [t for t in terms if variable not in t]

    # must have at least one self-loop
    if len(self_terms) == 0:
        return False, None, None

    # R is union of all prefixes before variable
    r_parts = []
    for t in self_terms:
        prefix = t.replace(variable, "")
        r_parts.append(prefix)

    r = "|".join(r_parts)

    # S is union of all other terms
    s = "|".join(other_terms) if other_terms else "ε"

    return True, r, s


def substitute(eq, solution, var):
    for other in eq:
        # if the variable == other, skip
        if other == var:
            continue
        new_terms = []
        for t in eq[other]:
            new_terms.append(t.replace(var, f"{solution}"))
        # expand the new terms
        expanded_terms = []
        for t in new_terms:
            expanded = expand_regex(t)
            ##spltit by '+' to get individual terms
            expanded_terms.extend(expanded.split("+"))
        eq[other] = expanded_terms


def to_sympy(expr: str):
    """Convert automata regex to sympy algebra."""
    expr = expr.replace(" ", "")  # remove spaces
    expr = expr.replace("ε", "1")  # epsilon → identity

    # repeatedly insert '*' between adjacent symbols until stable
    prev = None
    while prev != expr:
        prev = expr

        # 1. letter followed by letter
        expr = re.sub(r"([A-Za-z0-9])([A-Za-z0-9])", r"\1*\2", expr)

        # 2. letter followed by '('
        expr = re.sub(r"([A-Za-z0-9])(\()", r"\1*\2", expr)

        # 3. ')' followed by letter
        expr = re.sub(r"(\))([A-Za-z0-9])", r"\1*\2", expr)

        # 4. ')' followed by '('
        expr = re.sub(r"(\))(\()", r"\1*\2", expr)
    return expr


def from_sympy(expr):
    """Convert sympy-expanded algebra back to automata-style regex."""
    # convert sympy expression to string
    s = str(expr)

    # remove '*', restore concatenation convention
    s = s.replace("*", "")

    # restore epsilon
    s = s.replace("1", "ε")

    return s


def expand_regex(expr: str):
    """Expand an automata-style regex using SymPy algebra."""
    sym_expr = to_sympy(expr)

    # find all symbols used
    letters = sorted(set(re.findall(r"[A-Za-z]", sym_expr)))

    # create noncommutative symbols
    local_dict = {c: sp.Symbol(c, commutative=False) for c in letters}

    algebra = sp.sympify(sym_expr, locals=local_dict)
    expanded = sp.expand(algebra)
    return from_sympy(expanded)

"""
## make this a main method for testing
if __name__ == "__main__":
    ##dictionary for the entry equations
    equations_test = {
        "A": ["bB", "ε"],
        "B": ["aA", "aB"],
        "C": ["bC", "cD"],
        "D": ["dE", "eD", "fD"],
        "E": ["gE", "hE", "iE"],
    }
    for test_var, test_terms in equations_test.items():
        test_in_form, test_R, test_S = is_arden_form(test_var, test_terms)
        print(f"Variable: {test_var}, In Arden Form: {test_in_form}, R: {test_R}, S: {test_S}")

    regex = arden_to_regex(equations_test)
    print(f"Final Regex: {regex}")
    
      exp = "a(b + c)d"

    print(expand_regex(exp))

    equations = {
        "A": ["aB"],
        "B": ["bC", "b"],
        "C": ["cA"],
    }
    regex = arden_to_regex(equations)
    print(f"Final Regex: {regex}")
    """

"""
def extract_syms(terms_str):
    terms_dup = terms_str
    symbols = ""
    for c in terms_dup:
        if c.isalpha() and c not in symbols:
            symbols += c + " "

    return symbols

def expand(expr):
    # 1. extract symbol names
    names = extract_syms(expr)

    # 2. build symbol dictionary
    symbol_dict = {name: sp.Symbol(name, commutative=False) for name in names}

    # 3. parse expression with these symbols
    expr = sp.sympify(expr, locals=symbol_dict)

    # 4. expand (order preserved)
    expanded = sp.expand(expr)
    print(expanded)
    
    
    
    
def expand_again(exprr):
    expr = to_sympy(exprr)
    sym_expr = sp.sympify(exprr)
    expanded = sp.expand(sym_expr)
    return expanded
"""
