from __future__ import annotations

from typing import Any

import sympy as sp


def sympify_expr(expr: Any) -> sp.Expr:
    return sp.sympify(expr)


def light_simplify(expr: Any) -> sp.Expr:
    expr = sympify_expr(expr)
    if expr.is_Atom:
        return expr
    out = sp.cancel(expr)
    if out == expr:
        out = sp.together(expr)
    out2 = sp.factor_terms(out)
    if out2 != out:
        out = out2
    return out


def canonical_simplify(expr: Any, *, final: bool = False) -> sp.Expr:
    expr = light_simplify(expr)
    if final:
        return sp.simplify(expr)
    return expr
