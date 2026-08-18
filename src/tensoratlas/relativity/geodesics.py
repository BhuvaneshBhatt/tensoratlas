"""Symbolic and numerical geodesic helpers."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import sympy as sp

from .curvature import christoffel_symbols
from .metrics import MetricModel, metric_data, simplifier
from tensoratlas.errors import TensorShapeError
from tensoratlas.validation import check_indices


def geodesic_equations(
    metric: MetricModel | sp.Matrix,
    parameter: Any | None = None,
    coordinates: Iterable[Any] | None = None,
    *,
    simplify: bool | Callable[[Any], Any] = True,
) -> tuple[Any, ...]:
    _g, coords, _cached = metric_data(metric, coordinates)
    lam = sp.Symbol("lambda") if parameter is None else parameter
    clean = simplifier(simplify)
    dim = len(coords)
    gamma = christoffel_symbols(metric, coords, simplify=simplify)
    funcs = tuple(sp.Function(str(coord))(lam) for coord in coords)
    subs = {coords[index]: funcs[index] for index in range(dim)}
    equations = []
    for upper in range(dim):
        expr = sp.diff(funcs[upper], lam, 2)
        expr += sum(
            gamma[upper][first][second].subs(subs)
            * sp.diff(funcs[first], lam)
            * sp.diff(funcs[second], lam)
            for first in range(dim)
            for second in range(dim)
        )
        equations.append(clean(expr))
    return tuple(equations)


def geodesic_rhs(
    metric: MetricModel | sp.Matrix,
    state: Iterable[Any],
    coordinates: Iterable[Any] | None = None,
    *,
    modules: str | list[Any] = "math",
    simplify: bool | Callable[[Any], Any] = False,
):
    _g, coords, _cached = metric_data(metric, coordinates)
    dim = len(coords)
    state = tuple(state)
    if len(state) != 2 * dim:
        raise TensorShapeError(f"geodesic state length must be {2 * dim}: {dim} positions followed by {dim} velocities")
    gamma = christoffel_symbols(metric, coords, simplify=simplify)
    position_syms = state[:dim]
    velocity_syms = state[dim:]
    substitutions = {coords[index]: position_syms[index] for index in range(dim)}
    accelerations = []
    for upper in range(dim):
        expr = -sum(
            gamma[upper][first][second].subs(substitutions) * velocity_syms[first] * velocity_syms[second]
            for first in range(dim)
            for second in range(dim)
        )
        accelerations.append(expr)
    rhs = tuple(velocity_syms) + tuple(accelerations)
    return sp.lambdify(state, rhs, modules=modules)


def geodesic_equation(
    metric: MetricModel | sp.Matrix,
    index: int,
    parameter: Any | None = None,
    coordinates: Iterable[Any] | None = None,
    *,
    simplify: bool | Callable[[Any], Any] = True,
) -> Any:
    """Return a single geodesic equation component."""
    _g, coords, _cached = metric_data(metric, coordinates)
    dim = len(coords)
    check_indices("geodesic equation", dim, index)
    return geodesic_equations(metric, parameter, coords, simplify=simplify)[index]
