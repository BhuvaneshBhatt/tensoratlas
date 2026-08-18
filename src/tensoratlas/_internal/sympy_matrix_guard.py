"""Opt-in SymPy integration hooks kept outside the public API."""

from __future__ import annotations

import sympy as sp


def install_matrix_simplify_guard() -> bool:
    """Install a bounded matrix simplification guard for compatibility tests.

    This modifies ``sympy.simplify`` and is therefore intentionally internal and
    opt-in.  Normal TensorAtlas imports never patch SymPy.
    """
    try:
        from ..normal_forms import TNFMatrix
    except Exception:
        TNFMatrix = ()
    if getattr(sp.simplify, "_tensoratlas_matrix_guard", False):
        return False
    original_simplify = sp.simplify

    def cheap_matrix_entry_simplify(entry):
        entry = sp.cancel(entry)
        try:
            has_derivative = entry.has(sp.Derivative)
            has_undefined = any(
                getattr(node, "is_Function", False) and node.func.__name__ not in {"sin", "cos", "tan", "cot"}
                for node in entry.atoms(sp.Function)
            )
        except Exception:
            has_derivative = False
            has_undefined = False
        if has_derivative or has_undefined:
            return entry
        if entry.count_ops() <= 80 and any(entry.has(fn) for fn in (sp.sin, sp.cos, sp.tan, sp.cot)):
            return sp.trigsimp(entry)
        return entry

    def guarded_simplify(expr, *args, **kwargs):
        if TNFMatrix != () and isinstance(expr, TNFMatrix):
            if all(entry == 0 for entry in expr):
                return sp.zeros(expr.rows, expr.cols)
            return sp.Matrix(expr.rows, expr.cols, lambda row, col: cheap_matrix_entry_simplify(expr[row, col]))
        if isinstance(expr, sp.MatrixBase):
            entries = list(expr)
            if all(entry == 0 for entry in entries):
                return sp.zeros(expr.rows, expr.cols)
            return expr.applyfunc(cheap_matrix_entry_simplify)
        return original_simplify(expr, *args, **kwargs)

    guarded_simplify._tensoratlas_matrix_guard = True
    sp.simplify = guarded_simplify
    return True
