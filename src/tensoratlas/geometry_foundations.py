from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Any

import sympy as sp

from .basis import TensorBasis, tangent_basis
from .charts import CoordinateChart
from .fields import ScalarField, TensorField, VectorField
from .calculus import covariant_derivative as _covariant_derivative, lie_derivative as _lie_derivative
from .exterior import exterior_derivative as _exterior_derivative
from .normal_forms import TNFTensorArray, as_tnf_array, tnf_build_array


@dataclass(frozen=True)
class ManifoldDef:
    """Typed manifold descriptor for the geometry layer."""

    name: str
    dimension: int
    charts: tuple[CoordinateChart, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_chart(self, chart: CoordinateChart) -> "ManifoldDef":
        if chart.dimension != self.dimension:
            raise ValueError("Chart dimension must match manifold dimension.")
        if chart in self.charts:
            return self
        return ManifoldDef(self.name, self.dimension, self.charts + (chart,), dict(self.metadata))


@dataclass(frozen=True)
class BundleDef:
    """Typed bundle descriptor for tensor and form slots."""

    name: str
    manifold: ManifoldDef
    rank: int
    kind: str = "vector"
    dual_of: Optional[str] = None
    metric_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def compatible_with(self, other: "BundleDef") -> bool:
        return (
            isinstance(other, BundleDef)
            and self.manifold.dimension == other.manifold.dimension
            and self.kind == other.kind
            and self.rank == other.rank
            and self.manifold.name == other.manifold.name
        )


@dataclass(frozen=True)
class ChartDef:
    name: str
    manifold: ManifoldDef
    chart: CoordinateChart
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chart.dimension != self.manifold.dimension:
            raise ValueError("Chart dimension must match manifold dimension.")

    @property
    def coordinates(self) -> tuple[sp.Symbol, ...]:
        return self.chart.symbols()


@dataclass(frozen=True)
class MetricDef:
    name: str
    bundle: BundleDef
    chart: Optional[CoordinateChart] = None
    signature: Optional[tuple[int, int, int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def coefficients(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if self.chart is None:
            return None
        return self.chart.metric_tnf(coords)

    def inverse_coefficients(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if self.chart is None:
            return None
        return self.chart.inverse_metric_tnf(coords)

    def determinant_density(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if self.chart is None:
            return None
        return self.chart.sqrt_metric_det(coords)

    def scalar_curvature(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if self.chart is None:
            return None
        return self.chart.scalar_curvature(coords)

    def geodesic_equations(self, functions=None, parameter: Optional[sp.Symbol] = None):
        if self.chart is None:
            raise ValueError("A chart-backed metric is required for geodesic equations.")
        return self.chart.geodesic_equations(functions=functions, parameter=parameter)


@dataclass(frozen=True)
class ConnectionDef:
    name: str
    bundle: BundleDef
    chart: Optional[CoordinateChart] = None
    coefficients_func: Optional[Callable[[Tuple[sp.Symbol, ...]], TNFTensorArray | Any]] = None
    torsion_free: bool = False
    metric_compatible: bool = False
    metric: Optional[MetricDef] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def coefficients(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if self.coefficients_func is not None:
            actual = coords or (self.chart.symbols() if self.chart is not None else None)
            if actual is None:
                raise ValueError("coords are required for coefficient evaluation.")
            return as_tnf_array(self.coefficients_func(actual))
        if self.chart is None:
            return None
        return self.chart.christoffel_symbols(coords)

    def torsion_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if self.chart is None:
            return None
        coeffs = self.coefficients(coords)
        if coeffs is None:
            return None
        return self.chart.torsion_tensor(coords, coeffs)

    def riemann_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if self.chart is None:
            return None
        if coords is None:
            coords = self.chart.symbols()
        coeffs = self.coefficients(coords)
        if coeffs is None:
            return None
        dim = self.chart.dimension
        return tnf_build_array(
            (dim, dim, dim, dim),
            lambda idx: (
                sp.diff(coeffs[idx[0], idx[1], idx[3]], coords[idx[2]])
                - sp.diff(coeffs[idx[0], idx[1], idx[2]], coords[idx[3]])
                + sum(
                    coeffs[idx[0], m, idx[2]] * coeffs[m, idx[1], idx[3]]
                    - coeffs[idx[0], m, idx[3]] * coeffs[m, idx[1], idx[2]]
                    for m in range(dim)
                )
            ),
        )

    def ricci_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if self.chart is None:
            return None
        riem = self.riemann_tensor(coords)
        if riem is None:
            return None
        dim = self.chart.dimension
        return tnf_build_array((dim, dim), lambda idx: sum(riem[i, idx[0], i, idx[1]] for i in range(dim)))

    def scalar_curvature(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if self.chart is None:
            return None
        ric = self.ricci_tensor(coords)
        inverse_metric = self.chart.inverse_metric(coords)
        if ric is None or inverse_metric is None:
            return None
        dim = self.chart.dimension
        return sp.simplify(sum(inverse_metric[i, j] * ric[i, j] for i in range(dim) for j in range(dim)))

    def nonmetricity_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if self.chart is None:
            return None
        coeffs = self.coefficients(coords)
        if coeffs is None:
            return None
        return self.chart.nonmetricity_tensor(coords, coeffs)

    def metric_compatibility_residual(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        return self.nonmetricity_tensor(coords)

    def bianchi_residual(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if self.chart is None:
            return None
        riem = self.riemann_tensor(coords)
        if riem is None:
            return None
        dim = self.chart.dimension
        return tnf_build_array(
            (dim, dim, dim, dim),
            lambda idx: riem[idx[0], idx[1], idx[2], idx[3]]
            + riem[idx[0], idx[2], idx[3], idx[1]]
            + riem[idx[0], idx[3], idx[1], idx[2]],
        )

    def validate(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> dict[str, bool | None]:
        coeffs = self.coefficients(coords)
        torsion = self.torsion_tensor(coords)
        nonmetricity = self.nonmetricity_tensor(coords) if self.metric is not None else None

        def all_zero(tensor):
            if tensor is None:
                return None
            return all(sp.simplify(entry) == 0 for entry in tensor.entries)

        return {
            "has_coefficients": coeffs is not None,
            "torsion_free_declared": self.torsion_free,
            "torsion_free_verified": all_zero(torsion),
            "metric_compatible_declared": self.metric_compatible,
            "metric_compatible_verified": all_zero(nonmetricity),
        }


@dataclass(frozen=True)
class FrameDef:
    name: str
    chart: ChartDef
    basis: TensorBasis
    orthonormal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.basis.chart is not None and self.basis.chart != self.chart.chart:
            raise ValueError("Frame basis chart must match frame chart.")


@dataclass(frozen=True)
class DifferentialOperatorDef:
    name: str
    kind: str
    connection: Optional[ConnectionDef] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply(self, field: ScalarField | VectorField | TensorField, /, *, vector: Optional[VectorField] = None):
        if self.kind == "covariant":
            return _covariant_derivative(field)
        if self.kind == "exterior":
            return _exterior_derivative(field)
        if self.kind == "lie":
            if vector is None:
                raise ValueError("Lie derivative requires a vector= argument.")
            return _lie_derivative(field, vector)
        raise ValueError(f"Unsupported operator kind {self.kind!r}.")


# Constructors / convenience layer

def manifold(name: str, dimension: int, *, metadata: Optional[dict[str, Any]] = None) -> ManifoldDef:
    return ManifoldDef(name=name, dimension=dimension, metadata=dict(metadata or {}))


def chart_definition(manifold: ManifoldDef, chart: CoordinateChart, *, name: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> ChartDef:
    chart_name = name or f"{manifold.name}:{chart.chart_name}"
    return ChartDef(chart_name, manifold.with_chart(chart), chart, dict(metadata or {}))


def tangent_bundle(manifold: ManifoldDef, *, name: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> BundleDef:
    return BundleDef(name or f"T({manifold.name})", manifold, manifold.dimension, kind="vector", metadata=dict(metadata or {}))


def cotangent_bundle(manifold: ManifoldDef, *, name: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> BundleDef:
    tangent = name or f"T*({manifold.name})"
    return BundleDef(tangent, manifold, manifold.dimension, kind="covector", dual_of=f"T({manifold.name})", metadata=dict(metadata or {}))


def riemannian_metric_from_chart(chart: CoordinateChart, *, manifold_name: Optional[str] = None, name: str = "g", signature: Optional[tuple[int, int, int]] = None, metadata: Optional[dict[str, Any]] = None) -> MetricDef:
    base = manifold(manifold_name or chart.metric_name or "M", chart.dimension)
    bundle = tangent_bundle(base, metadata={"chart_name": chart.chart_name})
    sig = signature or (chart.dimension, 0, 0)
    return MetricDef(name=name, bundle=bundle, chart=chart, signature=sig, metadata=dict(metadata or {}))


def levi_civita_connection(metric: MetricDef, *, name: str = "∇", metadata: Optional[dict[str, Any]] = None) -> ConnectionDef:
    if metric.chart is None:
        raise ValueError("Levi-Civita construction currently requires a chart-backed metric.")
    return ConnectionDef(
        name=name,
        bundle=metric.bundle,
        chart=metric.chart,
        torsion_free=True,
        metric_compatible=True,
        metric=metric,
        metadata=dict(metadata or {}),
    )



def affine_connection_from_coefficients(
    bundle: BundleDef,
    chart: CoordinateChart,
    coefficients_func: Callable[[Tuple[sp.Symbol, ...]], TNFTensorArray | Any],
    *,
    name: str = "D",
    torsion_free: bool = False,
    metric_compatible: bool = False,
    metric: Optional[MetricDef] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ConnectionDef:
    """Build a chart-backed affine connection from explicit Christoffel-like coefficients.

    Coefficients use the same convention as ``CoordinateChart.christoffel_symbols``:
    ``Gamma[upper, lower, derivative_direction]``.  This makes the object usable
    by torsion, curvature, Ricci, scalar-curvature and validation routines.
    """
    if chart.dimension != bundle.manifold.dimension:
        raise ValueError("Connection chart dimension must match the bundle manifold dimension.")
    return ConnectionDef(
        name=name,
        bundle=bundle,
        chart=chart,
        coefficients_func=coefficients_func,
        torsion_free=torsion_free,
        metric_compatible=metric_compatible,
        metric=metric,
        metadata=dict(metadata or {}),
    )


def connection_geometry_report(connection: ConnectionDef, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> dict[str, Any]:
    """Return executable diagnostic facts for an affine connection."""
    coeffs = connection.coefficients(coords)
    torsion = connection.torsion_tensor(coords)
    nonmetricity = connection.nonmetricity_tensor(coords)
    riem = connection.riemann_tensor(coords)
    ric = connection.ricci_tensor(coords)
    return {
        "name": connection.name,
        "dimension": None if connection.chart is None else connection.chart.dimension,
        "has_coefficients": coeffs is not None,
        "coefficient_shape": None if coeffs is None else coeffs.shape,
        "torsion_shape": None if torsion is None else torsion.shape,
        "nonmetricity_shape": None if nonmetricity is None else nonmetricity.shape,
        "riemann_shape": None if riem is None else riem.shape,
        "ricci_shape": None if ric is None else ric.shape,
        "scalar_curvature": connection.scalar_curvature(coords),
        "validation": connection.validate(coords),
    }

def frame_definition(chart: ChartDef, basis: Optional[TensorBasis] = None, *, name: Optional[str] = None, orthonormal: bool = False, metadata: Optional[dict[str, Any]] = None) -> FrameDef:
    active_basis = basis or tangent_basis(chart.chart)
    return FrameDef(name or active_basis.name, chart, active_basis, orthonormal=orthonormal, metadata=dict(metadata or {}))


def covariant_derivative_operator(connection: Optional[ConnectionDef] = None, *, name: str = "∇", metadata: Optional[dict[str, Any]] = None) -> DifferentialOperatorDef:
    return DifferentialOperatorDef(name=name, kind="covariant", connection=connection, metadata=dict(metadata or {}))


def exterior_derivative_operator(*, name: str = "d", metadata: Optional[dict[str, Any]] = None) -> DifferentialOperatorDef:
    return DifferentialOperatorDef(name=name, kind="exterior", metadata=dict(metadata or {}))


def lie_derivative_operator(*, name: str = "L", metadata: Optional[dict[str, Any]] = None) -> DifferentialOperatorDef:
    return DifferentialOperatorDef(name=name, kind="lie", metadata=dict(metadata or {}))


def geometry_summary(chart: CoordinateChart) -> dict[str, Any]:
    metric = riemannian_metric_from_chart(chart)
    connection = levi_civita_connection(metric)
    return {
        "chart": chart.chart_name,
        "metric_name": chart.metric_name,
        "dimension": chart.dimension,
        "coordinate_names": chart.coordinate_names,
        "is_orthogonal": chart.is_orthogonal_metric(),
        "cyclic_coordinates": chart.cyclic_coordinates(),
        "signature": metric.signature,
        "connection": connection.name,
        "torsion_free": connection.torsion_free,
        "metric_compatible": connection.metric_compatible,
    }


__all__ = [
    "ManifoldDef",
    "BundleDef",
    "ChartDef",
    "MetricDef",
    "ConnectionDef",
    "FrameDef",
    "DifferentialOperatorDef",
    "manifold",
    "chart_definition",
    "tangent_bundle",
    "cotangent_bundle",
    "riemannian_metric_from_chart",
    "levi_civita_connection",
    "frame_definition",
    "covariant_derivative_operator",
    "exterior_derivative_operator",
    "lie_derivative_operator",
    "geometry_summary",
    "geometry_covariant_derivative_operator",
    "geometry_exterior_derivative_operator",
    "geometry_lie_derivative_operator",
]


# Stable aliases that avoid collisions with the abstract-tensor operator layer
geometry_covariant_derivative_operator = covariant_derivative_operator
geometry_exterior_derivative_operator = exterior_derivative_operator
geometry_lie_derivative_operator = lie_derivative_operator
