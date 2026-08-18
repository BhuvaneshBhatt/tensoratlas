from __future__ import annotations

import sympy as sp

from tensoratlas import (
    exterior_form_nf,
    hodge_star_nf,
    codifferential_nf,
    clifford_algebra,
    gamma_string_simplify,
)


def bench_hodge_and_codifferential(iterations: int = 50):
    x, y, z = sp.symbols('x y z')
    form = exterior_form_nf({(0, 1): x*y + z, (1, 2): x + y*z}, dimension=3)
    for _ in range(iterations):
        hodge_star_nf(form)
        codifferential_nf(form, (x, y, z))


def bench_gamma_strings(iterations: int = 200):
    cl = clifford_algebra(4, (4, 0, 0), basis_labels=('0', '1', '2', '3'))
    g0, g1, g2, g3 = [sp.Symbol(f'gamma{i}', commutative=False) for i in ('0', '1', '2', '3')]
    expr = (g3*g2*g1*g0 + g0*g1*g2*g3 + g1*g1 + g2*g2) * sp.Symbol('a')
    for _ in range(iterations):
        gamma_string_simplify(expr, cl)
