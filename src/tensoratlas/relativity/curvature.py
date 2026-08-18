"""Curvature tensors for symbolic metrics.

The Riemann tensor convention is
R^a{}_{bcd} = d_c Gamma^a{}_{bd} - d_d Gamma^a{}_{bc}
              + Gamma^a{}_{ce} Gamma^e{}_{bd}
              - Gamma^a{}_{de} Gamma^e{}_{bc}.
Ricci is R_{bd} = R^a{}_{bad}; scalar curvature is g^{ab} R_{ab}.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import sympy as sp

from .metrics import MetricModel, inverse_metric, metric_data, simplifier
from tensoratlas.validation import check_indices


def christoffel_symbols(
    metric: MetricModel | sp.Matrix,
    coordinates: Iterable[Any] | None = None,
    *,
    simplify: bool | Callable[[Any], Any] = True,
) -> list[list[list[Any]]]:
    g, coords, cached = metric_data(metric, coordinates)
    ginv = inverse_metric(metric, g, cached, simplify=simplify)
    clean = simplifier(simplify)
    dim = len(coords)
    gamma = [[[sp.Integer(0) for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    for upper in range(dim):
        for first in range(dim):
            for second in range(first, dim):
                expr = sp.Rational(1, 2) * sum(
                    ginv[upper, deriv]
                    * (
                        sp.diff(g[deriv, second], coords[first])
                        + sp.diff(g[deriv, first], coords[second])
                        - sp.diff(g[first, second], coords[deriv])
                    )
                    for deriv in range(dim)
                )
                value = clean(expr)
                gamma[upper][first][second] = value
                gamma[upper][second][first] = value
    return gamma


def christoffel_component(
    metric: MetricModel | sp.Matrix,
    upper: int,
    first: int,
    second: int,
    coordinates: Iterable[Any] | None = None,
    *,
    simplify: bool | Callable[[Any], Any] = True,
) -> Any:
    g, coords, cached = metric_data(metric, coordinates)
    dim = len(coords)
    check_indices("Christoffel component", dim, upper, first, second)
    ginv = inverse_metric(metric, g, cached, simplify=simplify)
    clean = simplifier(simplify)
    return clean(
        sp.Rational(1, 2)
        * sum(
            ginv[upper, deriv]
            * (
                sp.diff(g[deriv, second], coords[first])
                + sp.diff(g[deriv, first], coords[second])
                - sp.diff(g[first, second], coords[deriv])
            )
            for deriv in range(dim)
        )
    )


def riemann_tensor(
    metric: MetricModel | sp.Matrix,
    coordinates: Iterable[Any] | None = None,
    *,
    gamma: list[list[list[Any]]] | None = None,
    simplify: bool | Callable[[Any], Any] = True,
) -> list[list[list[list[Any]]]]:
    _g, coords, _cached = metric_data(metric, coordinates)
    clean = simplifier(simplify)
    dim = len(coords)
    if gamma is None:
        gamma = christoffel_symbols(metric, coords, simplify=simplify)
    riemann = [[[[sp.Integer(0) for _ in range(dim)] for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    for upper in range(dim):
        for lower in range(dim):
            for first in range(dim):
                for second in range(first + 1, dim):
                    expr = sp.diff(gamma[upper][lower][second], coords[first])
                    expr -= sp.diff(gamma[upper][lower][first], coords[second])
                    expr += sum(
                        gamma[upper][first][contract] * gamma[contract][lower][second]
                        - gamma[upper][second][contract] * gamma[contract][lower][first]
                        for contract in range(dim)
                    )
                    value = clean(expr)
                    riemann[upper][lower][first][second] = value
                    riemann[upper][lower][second][first] = -value
    return riemann


def riemann_component(
    metric: MetricModel | sp.Matrix,
    upper: int,
    lower: int,
    first: int,
    second: int,
    coordinates: Iterable[Any] | None = None,
    *,
    gamma: list[list[list[Any]]] | None = None,
    simplify: bool | Callable[[Any], Any] = True,
) -> Any:
    """Return one Riemann tensor component ``R^upper{}_{lower first second}``.

    The convention is ``R^a{}_{bcd} = partial_c Gamma^a{}_{bd} -
    partial_d Gamma^a{}_{bc} + Gamma^a{}_{ce} Gamma^e{}_{bd} -
    Gamma^a{}_{de} Gamma^e{}_{bc}``.  The last two slots are
    antisymmetric, so equal derivative slots return zero immediately.
    """
    _g, coords, _cached = metric_data(metric, coordinates)
    dim = len(coords)
    check_indices("Riemann component", dim, upper, lower, first, second)
    if first == second:
        return sp.Integer(0)
    clean = simplifier(simplify)
    if gamma is None:
        gamma = christoffel_symbols(metric, coords, simplify=simplify)
    expr = sp.diff(gamma[upper][lower][second], coords[first])
    expr -= sp.diff(gamma[upper][lower][first], coords[second])
    expr += sum(
        gamma[upper][first][contract] * gamma[contract][lower][second]
        - gamma[upper][second][contract] * gamma[contract][lower][first]
        for contract in range(dim)
    )
    return clean(expr)


def ricci_component(
    metric: MetricModel | sp.Matrix,
    first: int,
    second: int,
    coordinates: Iterable[Any] | None = None,
    *,
    riemann: list[list[list[list[Any]]]] | None = None,
    simplify: bool | Callable[[Any], Any] = True,
) -> Any:
    """Return one Ricci tensor component ``R_{first second}``.

    Ricci contraction uses ``R_{bd} = R^a{}_{bad}`` with the same Riemann
    convention used by :func:`riemann_component`.
    """
    _g, coords, _cached = metric_data(metric, coordinates)
    dim = len(coords)
    check_indices("Ricci component", dim, first, second)
    clean = simplifier(simplify)
    if riemann is not None:
        return clean(sum(riemann[upper][first][upper][second] for upper in range(dim)))
    gamma = christoffel_symbols(metric, coords, simplify=simplify)
    return clean(
        sum(
            riemann_component(metric, upper, first, upper, second, coords, gamma=gamma, simplify=simplify)
            for upper in range(dim)
        )
    )


def einstein_component(
    metric: MetricModel | sp.Matrix,
    first: int,
    second: int,
    coordinates: Iterable[Any] | None = None,
    *,
    ricci: sp.Matrix | None = None,
    scalar: Any | None = None,
    simplify: bool | Callable[[Any], Any] = True,
) -> Any:
    """Return one Einstein tensor component ``G_{first second}``.

    ``G_ab = R_ab - (1/2) g_ab R`` using the covariant Ricci tensor and
    scalar curvature convention documented in this module.
    """
    g, coords, _cached = metric_data(metric, coordinates)
    dim = len(coords)
    check_indices("Einstein component", dim, first, second)
    clean = simplifier(simplify)
    if ricci is None:
        ricci_value = ricci_component(metric, first, second, coords, simplify=simplify)
    else:
        ricci_value = ricci[first, second]
    if scalar is None:
        scalar = scalar_curvature(metric, coords, ricci=ricci, simplify=simplify)
    return clean(ricci_value - sp.Rational(1, 2) * g[first, second] * scalar)


def ricci_tensor(
    metric: MetricModel | sp.Matrix,
    coordinates: Iterable[Any] | None = None,
    *,
    riemann: list[list[list[list[Any]]]] | None = None,
    simplify: bool | Callable[[Any], Any] = True,
) -> sp.Matrix:
    _g, coords, _cached = metric_data(metric, coordinates)
    clean = simplifier(simplify)
    dim = len(coords)
    if riemann is None:
        riemann = riemann_tensor(metric, coords, simplify=simplify)
    return sp.Matrix(dim, dim, lambda lower, second: clean(sum(riemann[upper][lower][upper][second] for upper in range(dim))))


def scalar_curvature(
    metric: MetricModel | sp.Matrix,
    coordinates: Iterable[Any] | None = None,
    *,
    ricci: sp.Matrix | None = None,
    simplify: bool | Callable[[Any], Any] = True,
) -> Any:
    g, coords, cached = metric_data(metric, coordinates)
    ginv = inverse_metric(metric, g, cached, simplify=simplify)
    clean = simplifier(simplify)
    if ricci is None:
        ricci = ricci_tensor(metric, coords, simplify=simplify)
    dim = len(coords)
    return clean(sum(ginv[first, second] * ricci[first, second] for first in range(dim) for second in range(dim)))


def einstein_tensor(
    metric: MetricModel | sp.Matrix,
    coordinates: Iterable[Any] | None = None,
    *,
    ricci: sp.Matrix | None = None,
    scalar: Any | None = None,
    simplify: bool | Callable[[Any], Any] = True,
) -> sp.Matrix:
    g, coords, _cached = metric_data(metric, coordinates)
    clean = simplifier(simplify)
    if ricci is None:
        ricci = ricci_tensor(metric, coords, simplify=simplify)
    if scalar is None:
        scalar = scalar_curvature(metric, coords, ricci=ricci, simplify=simplify)
    dim = len(coords)
    return sp.Matrix(dim, dim, lambda first, second: clean(ricci[first, second] - sp.Rational(1, 2) * g[first, second] * scalar))


def metric_component(
    metric: MetricModel | sp.Matrix,
    first: int,
    second: int,
    coordinates: Iterable[Any] | None = None,
) -> Any:
    """Return one covariant metric component ``g[first, second]``."""
    g, coords, _cached = metric_data(metric, coordinates)
    dim = len(coords)
    check_indices("metric component", dim, first, second)
    return g[first, second]


def inverse_metric_component(
    metric: MetricModel | sp.Matrix,
    first: int,
    second: int,
    coordinates: Iterable[Any] | None = None,
    *,
    simplify: bool | Callable[[Any], Any] = True,
) -> Any:
    """Return one inverse-metric component ``g^{first, second}``."""
    g, coords, cached = metric_data(metric, coordinates)
    dim = len(coords)
    check_indices("inverse metric component", dim, first, second)
    return inverse_metric(metric, g, cached, simplify=simplify)[first, second]

class CurvatureComputer:
    """Cached selected-component curvature workflow.

    The object caches the inverse metric, Christoffel table, selected Riemann
    components, Ricci components, scalar curvature, and Einstein components for
    one metric/simplification policy.  It is useful when a notebook or script
    needs several related components without constructing every dense tensor.
    """

    def __init__(self, metric: MetricModel | sp.Matrix, coordinates: Iterable[Any] | None = None, *, simplify: bool | Callable[[Any], Any] = True):
        self.metric = metric
        self.coordinates_arg = None if coordinates is None else tuple(coordinates)
        self.simplify = simplify
        self.g, self.coordinates, self.cached_metric = metric_data(metric, coordinates)
        self.dimension = len(self.coordinates)
        self._inverse_metric: sp.Matrix | None = None
        self._gamma: list[list[list[Any]]] | None = None
        self._riemann_cache: dict[tuple[int, int, int, int], Any] = {}
        self._ricci_cache: dict[tuple[int, int], Any] = {}
        self._scalar_cache: Any | None = None
        self._einstein_cache: dict[tuple[int, int], Any] = {}

    @property
    def inverse_metric(self) -> sp.Matrix:
        if self._inverse_metric is None:
            self._inverse_metric = inverse_metric(self.metric, self.g, self.cached_metric, simplify=self.simplify)
        return self._inverse_metric

    @property
    def gamma(self) -> list[list[list[Any]]]:
        if self._gamma is None:
            self._gamma = christoffel_symbols(self.metric, self.coordinates, simplify=self.simplify)
        return self._gamma

    def christoffel(self, upper: int, first: int, second: int) -> Any:
        check_indices("Christoffel component", self.dimension, upper, first, second)
        return self.gamma[upper][first][second]

    def riemann(self, upper: int, lower: int, first: int, second: int) -> Any:
        check_indices("Riemann component", self.dimension, upper, lower, first, second)
        if first == second:
            return sp.Integer(0)
        key = (upper, lower, first, second)
        if key not in self._riemann_cache:
            self._riemann_cache[key] = riemann_component(
                self.metric,
                upper,
                lower,
                first,
                second,
                self.coordinates,
                gamma=self.gamma,
                simplify=self.simplify,
            )
        return self._riemann_cache[key]

    def ricci(self, first: int, second: int) -> Any:
        check_indices("Ricci component", self.dimension, first, second)
        key = (first, second)
        if key not in self._ricci_cache:
            clean = simplifier(self.simplify)
            self._ricci_cache[key] = clean(sum(self.riemann(upper, first, upper, second) for upper in range(self.dimension)))
        return self._ricci_cache[key]

    def scalar(self) -> Any:
        if self._scalar_cache is None:
            clean = simplifier(self.simplify)
            self._scalar_cache = clean(
                sum(
                    self.inverse_metric[first, second] * self.ricci(first, second)
                    for first in range(self.dimension)
                    for second in range(self.dimension)
                )
            )
        return self._scalar_cache

    def einstein(self, first: int, second: int) -> Any:
        check_indices("Einstein component", self.dimension, first, second)
        key = (first, second)
        if key not in self._einstein_cache:
            clean = simplifier(self.simplify)
            self._einstein_cache[key] = clean(
                self.ricci(first, second) - sp.Rational(1, 2) * self.g[first, second] * self.scalar()
            )
        return self._einstein_cache[key]

    def ricci_matrix(self) -> sp.Matrix:
        return sp.Matrix(self.dimension, self.dimension, lambda i, j: self.ricci(i, j))

    def einstein_matrix(self) -> sp.Matrix:
        return sp.Matrix(self.dimension, self.dimension, lambda i, j: self.einstein(i, j))
