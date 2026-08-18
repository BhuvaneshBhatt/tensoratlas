"""Inspection helpers for tensor component arrays and curvature objects."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import sympy as sp

from .curvature import (
    CurvatureComputer,
    christoffel_symbols,
    einstein_tensor,
    ricci_tensor,
    riemann_tensor,
)
from .metrics import MetricModel, simplifier


def nonzero_components(array: Any, *, simplify: bool | Callable[[Any], Any] = True) -> dict[tuple[int, ...], Any]:
    """Return nonzero entries from matrices or nested component arrays."""
    clean = simplifier(simplify)
    out: dict[tuple[int, ...], Any] = {}
    if hasattr(array, "components") and not isinstance(array, sp.Basic):
        array = getattr(array, "components")
    if isinstance(array, sp.MatrixBase):
        for row in range(array.rows):
            for col in range(array.cols):
                value = clean(array[row, col])
                if value != 0:
                    out[(row, col)] = value
        return out

    def visit(obj: Any, prefix: tuple[int, ...]) -> None:
        if isinstance(obj, (list, tuple)):
            for index, item in enumerate(obj):
                visit(item, prefix + (index,))
        else:
            value = clean(obj)
            if value != 0:
                out[prefix] = value

    visit(array, ())
    return out


def nonzero_christoffel(metric: MetricModel | sp.Matrix, coordinates: Iterable[Any] | None = None, *, simplify: bool | Callable[[Any], Any] = True) -> dict[tuple[int, ...], Any]:
    """Return nonzero Christoffel symbols keyed by ``(upper, first, second)``.

    This helper computes the dense Christoffel table and then filters nonzero
    entries. For large symbolic metrics, prefer ``christoffel_component``.
    """
    return nonzero_components(christoffel_symbols(metric, coordinates, simplify=simplify), simplify=simplify)


def nonzero_riemann(metric: MetricModel | sp.Matrix, coordinates: Iterable[Any] | None = None, *, simplify: bool | Callable[[Any], Any] = True) -> dict[tuple[int, ...], Any]:
    """Return nonzero Riemann components keyed by ``(upper, lower, first, second)``.

    This helper computes the dense Riemann tensor and then filters nonzero
    entries. For large symbolic metrics, prefer selected component calls or
    ``sparse_nonzero_riemann``.
    """
    return nonzero_components(riemann_tensor(metric, coordinates, simplify=simplify), simplify=simplify)


def nonzero_ricci(metric: MetricModel | sp.Matrix, coordinates: Iterable[Any] | None = None, *, simplify: bool | Callable[[Any], Any] = True) -> dict[tuple[int, ...], Any]:
    """Return nonzero Ricci components keyed by ``(first, second)``.

    This helper computes the dense Ricci tensor and then filters nonzero
    entries. For large symbolic metrics, prefer ``ricci_component`` or
    ``sparse_nonzero_ricci``.
    """
    return nonzero_components(ricci_tensor(metric, coordinates, simplify=simplify), simplify=simplify)


def nonzero_einstein(metric: MetricModel | sp.Matrix, coordinates: Iterable[Any] | None = None, *, simplify: bool | Callable[[Any], Any] = True) -> dict[tuple[int, ...], Any]:
    """Return nonzero Einstein-tensor components keyed by ``(first, second)``.

    This helper computes the dense Einstein tensor and then filters nonzero
    entries. For large symbolic metrics, prefer ``einstein_component`` or
    ``sparse_nonzero_einstein``.
    """
    return nonzero_components(einstein_tensor(metric, coordinates, simplify=simplify), simplify=simplify)


def _is_nonzero(value: Any, clean: Callable[[Any], Any]) -> tuple[bool, Any]:
    cleaned = clean(value)
    return cleaned != 0, cleaned


def sparse_nonzero_riemann(
    metric: MetricModel | sp.Matrix,
    coordinates: Iterable[Any] | None = None,
    *,
    simplify: bool | Callable[[Any], Any] = True,
) -> dict[tuple[int, int, int, int], Any]:
    """Return nonzero Riemann components without constructing a dense Riemann tensor.

    The helper still computes and caches the Christoffel table, but it evaluates
    Riemann components one at a time and skips the antisymmetric ``c == d``
    components. This is preferable to ``nonzero_riemann`` for exploratory
    four-dimensional metrics.
    """
    computer = CurvatureComputer(metric, coordinates, simplify=simplify)
    clean = simplifier(simplify)
    out: dict[tuple[int, int, int, int], Any] = {}
    dim = computer.dimension
    for upper in range(dim):
        for lower in range(dim):
            for first in range(dim):
                for second in range(dim):
                    if first == second:
                        continue
                    ok, value = _is_nonzero(computer.riemann(upper, lower, first, second), clean)
                    if ok:
                        out[(upper, lower, first, second)] = value
    return out


def sparse_nonzero_ricci(
    metric: MetricModel | sp.Matrix,
    coordinates: Iterable[Any] | None = None,
    *,
    simplify: bool | Callable[[Any], Any] = True,
) -> dict[tuple[int, int], Any]:
    """Return nonzero Ricci components without constructing a dense Riemann tensor."""
    computer = CurvatureComputer(metric, coordinates, simplify=simplify)
    clean = simplifier(simplify)
    out: dict[tuple[int, int], Any] = {}
    for first in range(computer.dimension):
        for second in range(computer.dimension):
            ok, value = _is_nonzero(computer.ricci(first, second), clean)
            if ok:
                out[(first, second)] = value
    return out


def sparse_nonzero_einstein(
    metric: MetricModel | sp.Matrix,
    coordinates: Iterable[Any] | None = None,
    *,
    simplify: bool | Callable[[Any], Any] = True,
) -> dict[tuple[int, int], Any]:
    """Return nonzero Einstein components using cached selected components."""
    computer = CurvatureComputer(metric, coordinates, simplify=simplify)
    clean = simplifier(simplify)
    out: dict[tuple[int, int], Any] = {}
    for first in range(computer.dimension):
        for second in range(computer.dimension):
            ok, value = _is_nonzero(computer.einstein(first, second), clean)
            if ok:
                out[(first, second)] = value
    return out
