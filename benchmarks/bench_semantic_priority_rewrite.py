from __future__ import annotations

import sympy as sp

from ._common import BenchmarkCase
from tensoratlas import (
    clifford_algebra,
    exterior_form_nf,
    hodge_expr,
    gamma_string,
    semantic_operator_rules,
    semantic_rewrite,
)


def build_cases():
    x, y = sp.symbols('x y')
    form = exterior_form_nf({(0,): x, (1,): y}, dimension=2, basis_labels=('dx', 'dy'))
    cl = clifford_algebra(2, (2, 0, 0), basis_labels=('0', '1'))
    return [
        BenchmarkCase('semantic_hodge_eval', lambda: semantic_rewrite(hodge_expr(form, metric_signature=(1,1)), semantic_operator_rules())),
        BenchmarkCase('semantic_gamma_eval', lambda: semantic_rewrite(gamma_string(cl, [0,0,1,1]), semantic_operator_rules())),
    ]
