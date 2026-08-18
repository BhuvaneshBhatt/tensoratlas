from __future__ import annotations

from typing import Any

import sympy as sp

from .simplification_core import canonical_simplify
from .symbolic_simplification_policy import bounded_algebraic_simplify_expr
from .normal_forms import TNFMatrix, TNFTensorArray, as_tnf_array, as_tnf_matrix, tnf_build_array, tnf_build_matrix, tnf_iter_indices


SIMPLIFICATION_LEVELS = ("cheap", "normal", "strong")


def cheap_simplify(expr: Any):
    return bounded_algebraic_simplify_expr(expr)


def normal_simplify(expr: Any):
    return canonical_simplify(expr, final=False)


def strong_simplify(expr: Any):
    return canonical_simplify(expr, final=True)


def simplify_expr(expr: Any, *, level: str = "normal"):
    if level == "cheap":
        return cheap_simplify(expr)
    if level == "strong":
        return strong_simplify(expr)
    return normal_simplify(expr)



def simplify_object(obj: Any, *, level: str = "normal"):
    if isinstance(obj, TNFMatrix):
        return tnf_build_matrix(obj.rows, obj.cols, lambda i, j: simplify_expr(obj[i, j], level=level))
    if isinstance(obj, TNFTensorArray):
        return tnf_build_array(obj.shape, lambda idx: simplify_expr(obj[idx], level=level))
    if isinstance(obj, sp.MatrixBase):
        return sp.Matrix(obj.rows, obj.cols, lambda i, j: simplify_expr(obj[i, j], level=level))
    if isinstance(obj, sp.NDimArray):
        arr = as_tnf_array(obj)
        cleaned = tnf_build_array(arr.shape, lambda idx: simplify_expr(arr[idx], level=level))
        return cleaned.to_sympy()
    return simplify_expr(obj, level=level)


def normalize_structural(obj: Any):
    return simplify_object(obj, level="cheap")


def normalize_refined(obj: Any):
    return simplify_object(obj, level="normal")


def normalize_presentation(obj: Any):
    return simplify_object(obj, level="strong")


def normalize_simplifier(simplify, *, true_strategy=None):
    """Return a callable simplifier from a bool/callable policy.

    Parameters
    ----------
    simplify:
        ``False`` returns the identity function.  A callable is returned
        unchanged.  ``True`` returns ``true_strategy`` when supplied, otherwise
        ``sympy.simplify``.
    true_strategy:
        Optional callable used for the ``True`` case.  Domain packages use this
        to share policy normalization while keeping mathematically appropriate
        default simplification strategies.
    """
    if callable(simplify):
        return simplify
    if not simplify:
        return lambda expr: expr
    return true_strategy or sp.simplify


def trigonometric_rational_simplifier(simplify):
    """Normalize a policy for common metric/curvature expressions.

    The ``True`` strategy uses ``factor(cancel(expr))``.  It avoids broad
    trigonometric rewrites so familiar textbook factors such as
    ``sin(theta)*cos(theta)`` remain readable unless callers request a custom
    simplifier.
    """

    def clean(expr: Any):
        return sp.factor(sp.cancel(expr))

    return normalize_simplifier(simplify, true_strategy=clean)
