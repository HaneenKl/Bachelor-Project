import sympy as sp

a, b, c, C = sp.symbols("a b c C", commutative=False)

expr = c * a * (b*C + b)
expanded = sp.expand(c * a * (b*C + b))

print(expanded)



exp = "a*(b + c)*d"
sym_expr = sp.sympify(exp)
expanded = sp.expand(sym_expr)
print(expanded)
