"""Metric models and standard relativity metric catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Iterable

import sympy as sp


from tensoratlas.errors import MetricError
from tensoratlas.simplification_policy import trigonometric_rational_simplifier
from tensoratlas.validation import ValidationReport, invalid_report, valid_report


def relativity_simplifier(simplify: bool | Callable[[Any], Any]):
    """Return the standard simplifier for metric and curvature expressions."""
    return trigonometric_rational_simplifier(simplify)


# Backwards-compatible internal alias used by curvature/geodesic modules.
simplifier = relativity_simplifier


@dataclass(frozen=True)
class MetricModel:
    """Symbolic pseudo-Riemannian metric with cached inverse metric.

    Coordinates are ordered exactly as supplied, and all curvature routines use
    this order for tensor components.  The optional signature tuple records the
    diagonal sign convention when it is known.
    """

    name: str
    coordinates: tuple[Any, ...]
    metric: sp.Matrix
    parameters: tuple[Any, ...] = ()
    signature: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metric = sp.Matrix(self.metric)
        if metric.shape != (len(self.coordinates), len(self.coordinates)):
            raise MetricError("metric shape must match number of coordinates")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        object.__setattr__(self, "parameters", tuple(self.parameters))

    @cached_property
    def inverse_metric(self) -> sp.Matrix:
        """Unsimplified inverse metric cached for low-level computations."""
        return self.metric.inv()

    @cached_property
    def simplified_inverse_metric(self) -> sp.Matrix:
        """Simplified inverse metric for presentation-oriented workflows."""
        return self.metric.inv().applyfunc(relativity_simplifier(True))

    @property
    def dimension(self) -> int:
        return len(self.coordinates)

    def summary(self) -> dict[str, Any]:
        """Return a compact metric summary for notebooks and debugging."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "coordinates": self.coordinates,
            "signature": self.signature,
            "parameters": self.parameters,
            "metadata": dict(self.metadata),
        }

    def validation_report(self) -> ValidationReport:
        """Return structured validation diagnostics without raising."""
        errors = []
        if self.metric.shape != (self.dimension, self.dimension):
            errors.append("metric shape must match coordinate dimension")
        if self.metric != self.metric.T:
            errors.append("metric matrix must be symmetric")
        return invalid_report(*errors) if errors else valid_report()

    def validate(self) -> bool:
        """Validate metric dimensions and symmetry."""
        report = self.validation_report()
        if not report.ok:
            report.raise_as(MetricError)
        return True

    def to_matrix(self) -> sp.Matrix:
        """Return the SymPy metric matrix."""
        return sp.Matrix(self.metric)

    def coordinate_index(self, coordinate: Any) -> int:
        """Return the integer position of a coordinate symbol or name."""
        if isinstance(coordinate, int):
            index = int(coordinate)
        elif isinstance(coordinate, str):
            names = [str(item) for item in self.coordinates]
            if coordinate not in names:
                raise MetricError(f"unknown coordinate name {coordinate!r}")
            index = names.index(coordinate)
        else:
            if coordinate not in self.coordinates:
                raise MetricError(f"unknown coordinate {coordinate!r}")
            index = self.coordinates.index(coordinate)
        if index < 0 or index >= self.dimension:
            raise IndexError(f"coordinate index out of range for dimension {self.dimension}: {index}")
        return index

    @classmethod
    def from_matrix(cls, name: str, coordinates: Iterable[Any], metric: Iterable[Iterable[Any]] | sp.Matrix, **kwargs: Any) -> "MetricModel":
        """Build a metric model from a SymPy matrix or nested sequence."""
        return cls(name, tuple(coordinates), sp.Matrix(metric), **kwargs)

    def christoffel(self, *, simplify: bool | Callable[[Any], Any] = True):
        from .curvature import christoffel_symbols
        return christoffel_symbols(self, simplify=simplify)

    def christoffel_component(self, upper: Any, first: Any, second: Any, *, simplify: bool | Callable[[Any], Any] = True) -> Any:
        from .curvature import christoffel_component
        return christoffel_component(self, self.coordinate_index(upper), self.coordinate_index(first), self.coordinate_index(second), simplify=simplify)

    def riemann(self, *, simplify: bool | Callable[[Any], Any] = True):
        from .curvature import riemann_tensor
        return riemann_tensor(self, simplify=simplify)

    def ricci(self, *, simplify: bool | Callable[[Any], Any] = True) -> sp.Matrix:
        from .curvature import ricci_tensor
        return ricci_tensor(self, simplify=simplify)

    def scalar_curvature(self, *, simplify: bool | Callable[[Any], Any] = True) -> Any:
        from .curvature import scalar_curvature
        return scalar_curvature(self, simplify=simplify)

    def einstein(self, *, simplify: bool | Callable[[Any], Any] = True) -> sp.Matrix:
        from .curvature import einstein_tensor
        return einstein_tensor(self, simplify=simplify)

    def geodesic_equation(self, coordinate: Any, **kwargs: Any):
        from .geodesics import geodesic_equation
        return geodesic_equation(self, self.coordinate_index(coordinate), **kwargs)

    def geodesic_equations(self, **kwargs: Any):
        from .geodesics import geodesic_equations
        return geodesic_equations(self, **kwargs)

    def curvature(self, *, simplify: bool | Callable[[Any], Any] = True):
        """Return a cached selected-component curvature computer."""
        from .curvature import CurvatureComputer
        return CurvatureComputer(self, simplify=simplify)


def metric_data(
    metric: MetricModel | sp.Matrix | Iterable[Iterable[Any]],
    coordinates: Iterable[Any] | None = None,
) -> tuple[sp.Matrix, tuple[Any, ...], MetricModel | None]:
    if isinstance(metric, MetricModel):
        return metric.metric, metric.coordinates, metric
    if coordinates is None:
        raise MetricError("coordinates are required when metric is not a MetricModel")
    coords = tuple(coordinates)
    g = sp.Matrix(metric)
    if g.shape != (len(coords), len(coords)):
        raise MetricError("metric shape must match number of coordinates")
    return g, coords, None


def inverse_metric(
    metric: MetricModel | sp.Matrix | Iterable[Iterable[Any]],
    g: sp.Matrix,
    cached: MetricModel | None,
    *,
    simplify: bool | Callable[[Any], Any] = True,
) -> sp.Matrix:
    if cached is not None:
        if simplify is True:
            return cached.simplified_inverse_metric
        if simplify is False:
            return cached.inverse_metric
        clean = simplifier(simplify)
        return cached.inverse_metric.applyfunc(clean)
    if simplify is False:
        return g.inv()
    clean = simplifier(simplify)
    return g.inv().applyfunc(clean)


def minkowski_metric(dimension: int = 4, *, names: tuple[str, ...] | None = None) -> MetricModel:
    """Return flat Minkowski spacetime with mostly-plus signature."""
    if names is None:
        names = tuple(["t"] + [f"x{i}" for i in range(1, dimension)])
    coords = sp.symbols(" ".join(names))
    return MetricModel(
        "Minkowski",
        tuple(coords),
        sp.diag(-1, *([1] * (dimension - 1))),
        signature=(-1,) + (1,) * (dimension - 1),
    )


def two_sphere_metric(radius: Any = None) -> MetricModel:
    theta, phi = sp.symbols("theta phi")
    radius = sp.Symbol("R", positive=True) if radius is None else sp.sympify(radius)
    metric = sp.diag(radius**2, radius**2 * sp.sin(theta) ** 2)
    return MetricModel("two_sphere", (theta, phi), metric, parameters=(radius,), signature=(1, 1))


def schwarzschild_metric(mass: Any = None) -> MetricModel:
    """Return Schwarzschild coordinates with signature (-,+,+,+)."""
    t, r, theta, phi = sp.symbols("t r theta phi")
    mass = sp.Symbol("M", positive=True) if mass is None else sp.sympify(mass)
    factor = 1 - 2 * mass / r
    metric = sp.diag(-factor, 1 / factor, r**2, r**2 * sp.sin(theta) ** 2)
    return MetricModel("Schwarzschild", (t, r, theta, phi), metric, parameters=(mass,), signature=(-1, 1, 1, 1))


def flrw_metric(scale_factor: Any = None, curvature: Any = None) -> MetricModel:
    """Return the FLRW metric in spherical comoving coordinates."""
    t, r, theta, phi = sp.symbols("t r theta phi")
    scale = sp.Function("a")(t) if scale_factor is None else scale_factor
    k = sp.Symbol("k") if curvature is None else sp.sympify(curvature)
    metric = sp.diag(-1, scale**2 / (1 - k * r**2), scale**2 * r**2, scale**2 * r**2 * sp.sin(theta) ** 2)
    return MetricModel("FLRW", (t, r, theta, phi), metric, parameters=(k,), signature=(-1, 1, 1, 1))
