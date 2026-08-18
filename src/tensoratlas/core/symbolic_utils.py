"""Small shared helpers for optional SymPy-backed component code."""
from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable, Sequence

from .manifolds import TensorKernelError

Scalar = Any
Index = tuple[int, ...]


def require_sympy(context: str = "symbolic operation"):
    """Import SymPy with a consistent TensorAtlas error message."""
    try:
        import sympy as sp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise TensorKernelError(f"{context} requires SymPy.") from exc
    return sp


def as_matrix(value: Any, *, context: str = "matrix operation"):
    """Return ``value`` as a SymPy matrix."""
    return require_sympy(context).Matrix(value)


def cancel_safely(value: Scalar) -> Scalar:
    """Apply a cheap rational simplification, falling back to the input."""
    try:
        return require_sympy("scalar simplification").cancel(value)
    except Exception:
        return value


def trig_cancel_safely(value: Scalar) -> Scalar:
    """Apply bounded trig simplification followed by cancellation."""
    sp = require_sympy("scalar simplification")
    try:
        return sp.cancel(sp.trigsimp(value))
    except Exception:
        return value


def build_nested(shape: Sequence[int], getter: Callable[[Index], Scalar], prefix: Index = ()) -> Any:
    """Build a rectangular nested tuple from a component getter."""
    if not shape:
        return getter(prefix)
    return tuple(build_nested(shape[1:], getter, prefix + (i,)) for i in range(shape[0]))


def nested_shape(values: Any) -> tuple[int, ...]:
    """Return the rectangular shape of nested tuple/list values."""
    if not isinstance(values, (tuple, list)):
        return ()
    length = len(values)
    if length == 0:
        return (0,)
    child = nested_shape(values[0])
    for item in values[1:]:
        if nested_shape(item) != child:
            raise TensorKernelError("Tensor components must be rectangular.")
    return (length,) + child


def get_nested(values: Any, key: Index) -> Scalar:
    """Return one component from nested tuple/list values."""
    current = values
    for item in key:
        current = current[item]
    return current


def iter_indices(shape: Sequence[int]) -> Iterable[Index]:
    """Iterate component keys for a rectangular shape."""
    return product(*(range(dim) for dim in shape)) if shape else ((),)


def is_zero(value: Scalar) -> bool:
    """Conservatively test whether a symbolic component is zero."""
    if value == 0:
        return True
    flag = getattr(value, "is_zero", None)
    return bool(flag is True)


def normalize_variance(value: str | None) -> str | None:
    """Normalize variance spellings to ``up`` / ``down`` / ``None``."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"up", "upper", "contravariant", "+"}:
        return "up"
    if normalized in {"down", "lower", "covariant", "-"}:
        return "down"
    raise TensorKernelError("Variance entries must be up/down or contravariant/covariant.")
