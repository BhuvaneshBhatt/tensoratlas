"""Small display and validation helpers for notebooks and examples."""

from __future__ import annotations

from typing import Any, Callable

import sympy as sp


def to_latex(obj: Any) -> str:
    """Return a LaTeX representation using SymPy when possible."""
    try:
        if hasattr(obj, "components") and not isinstance(obj, sp.Basic):
            return sp.latex(getattr(obj, "components"))
        return sp.latex(obj)
    except Exception:
        return str(obj)


def display_components(obj: Any) -> dict[str, Any]:
    """Return a compact dictionary suitable for notebook display."""
    if hasattr(obj, "summary"):
        data = obj.summary()
        if isinstance(data, dict):
            return data
    if hasattr(obj, "components"):
        return {"components": getattr(obj, "components")}
    if isinstance(obj, sp.MatrixBase):
        return {"shape": obj.shape, "components": obj}
    return {"value": obj}


def _is_seq(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def nonzero_components(array: Any, *, simplify: bool | Callable[[Any], Any] = False) -> dict[tuple[int, ...], Any]:
    """Return nonzero entries of a matrix, nested list, or TensorArray-like object.

    Simplification is disabled by default because display helpers are often
    called on large symbolic arrays in notebooks. Pass ``simplify=True`` or a
    callable simplifier when zero detection needs algebraic cleanup.
    """
    clean = simplify if callable(simplify) else (sp.simplify if simplify else (lambda x: x))
    if hasattr(array, "components") and not isinstance(array, sp.Basic):
        array = getattr(array, "components")
    out: dict[tuple[int, ...], Any] = {}
    if isinstance(array, sp.MatrixBase):
        for i in range(array.rows):
            for j in range(array.cols):
                value = clean(array[i, j])
                if value != 0:
                    out[(i, j)] = value
        return out
    def visit(obj: Any, prefix: tuple[int, ...]) -> None:
        if _is_seq(obj):
            for i, item in enumerate(obj):
                visit(item, prefix + (i,))
        else:
            value = clean(obj)
            if value != 0:
                out[prefix] = value
    visit(array, ())
    return out


def display_nonzero_components(array: Any, *, name: str = "T", simplify: bool | Callable[[Any], Any] = False) -> dict[str, Any]:
    """Return labeled nonzero components for notebook display.

    Simplification is disabled by default for predictable notebook latency.
    Pass ``simplify=True`` when components require simplification before zero
    filtering.
    """
    comps = nonzero_components(array, simplify=simplify)
    return {f"{name}{key}": value for key, value in comps.items()}
