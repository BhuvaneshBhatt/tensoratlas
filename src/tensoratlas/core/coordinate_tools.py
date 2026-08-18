"""High-level coordinate systems, maps, and standard chart catalog.

This module complements the lower-level component layer.  It stores coordinate
maps as symbolic data, records chart-domain and singularity metadata, and can
produce Jacobian-based component transforms for the existing component tensor
machinery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from .components import CoordinateSystem, CoordinateTransform
from tensoratlas.validation import ValidationReport, invalid_report, valid_report
from .manifolds import Manifold, TensorKernelError

Scalar = Any
MetricBuilder = Callable[[tuple[Scalar, ...], Mapping[str, Scalar]], Any]
MapBuilder = Callable[[tuple[Scalar, ...], Mapping[str, Scalar]], tuple[Scalar, ...]]


def _require_sympy():
    try:
        import sympy as sp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise TensorKernelError("Coordinate catalog operations require SymPy.") from exc
    return sp


@dataclass(frozen=True, slots=True)
class CoordinateDomain:
    """Domain metadata for one coordinate symbol."""

    kind: str
    lower: Scalar | None = None
    upper: Scalar | None = None
    lower_inclusive: bool = False
    upper_inclusive: bool = False
    periodic: bool = False

    def assumptions(self, symbol: Scalar) -> tuple[Scalar, ...]:
        sp = _require_sympy()
        clauses: list[Scalar] = []
        if self.kind == "real_line":
            return tuple()
        if self.lower is not None:
            clauses.append(sp.Ge(symbol, self.lower) if self.lower_inclusive else sp.Gt(symbol, self.lower))
        if self.upper is not None:
            clauses.append(sp.Le(symbol, self.upper) if self.upper_inclusive else sp.Lt(symbol, self.upper))
        return tuple(clauses)

    def as_dict(self) -> dict[str, Scalar]:
        return {
            "kind": self.kind,
            "lower": self.lower,
            "upper": self.upper,
            "lower_inclusive": self.lower_inclusive,
            "upper_inclusive": self.upper_inclusive,
            "periodic": self.periodic,
        }


@dataclass(frozen=True, slots=True)
class CoordinateSingularity:
    """A symbolic description of a coordinate singularity or excluded locus."""

    expression: Scalar
    reason: str = "coordinate singularity"


@dataclass(frozen=True, slots=True)
class CoordinateMap:
    """High-level symbolic coordinate transformation.

    ``forward`` stores target coordinates as functions of source coordinates.
    ``inverse`` stores source coordinates as functions of target coordinates
    when a branch choice is available.
    """

    source: CoordinateSystem
    target: CoordinateSystem
    forward: tuple[Scalar, ...]
    inverse: tuple[Scalar, ...] | None = None
    singularities: tuple[CoordinateSingularity, ...] = ()
    domain_notes: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)
    name: str | None = None

    def __post_init__(self) -> None:
        if self.source.manifold != self.target.manifold:
            raise TensorKernelError("Coordinate maps require source and target on the same manifold.")
        if len(self.forward) != self.target.dimension:
            raise TensorKernelError("Forward map length must match target dimension.")
        if self.inverse is not None and len(self.inverse) != self.source.dimension:
            raise TensorKernelError("Inverse map length must match source dimension.")
        object.__setattr__(self, "forward", tuple(self.forward))
        if self.inverse is not None:
            object.__setattr__(self, "inverse", tuple(self.inverse))
        object.__setattr__(self, "singularities", tuple(self.singularities))
        object.__setattr__(self, "domain_notes", dict(self.domain_notes))

    @property
    def source_symbols(self) -> tuple[Scalar, ...]:
        if self.source.coordinate_symbols is not None:
            return self.source.coordinate_symbols
        sp = _require_sympy()
        return tuple(sp.Symbol(name, real=True) for name in self.source.coordinate_names)

    @property
    def target_symbols(self) -> tuple[Scalar, ...]:
        if self.target.coordinate_symbols is not None:
            return self.target.coordinate_symbols
        sp = _require_sympy()
        return tuple(sp.Symbol(name, real=True) for name in self.target.coordinate_names)


    @property
    def inverse_map_expressions(self) -> tuple[Scalar, ...] | None:
        """Alias for the explicit inverse coordinate expressions, when present."""
        return self.inverse

    def domain_conditions(self) -> dict[str, Any]:
        """Return source and target coordinate-domain metadata and map notes."""
        return {
            "source_domain": dict(getattr(self.source, "domain", {}) or {}),
            "target_domain": dict(getattr(self.target, "domain", {}) or {}),
            "map_notes": dict(self.domain_notes),
            "singularities": tuple((s.expression, s.reason) for s in self.singularities),
        }

    def summary(self) -> dict[str, Any]:
        """Return a compact description of the coordinate map."""
        return {
            "name": self.name,
            "source": self.source.name,
            "target": self.target.name,
            "source_coordinates": self.source.coordinate_names,
            "target_coordinates": self.target.coordinate_names,
            "mapping": self.forward,
            "inverse_available": self.inverse is not None,
            "singularities": tuple((s.expression, s.reason) for s in self.singularities),
        }

    def validation_report(self) -> ValidationReport:
        """Return structured validation diagnostics without raising."""
        errors = []
        if self.source.manifold != self.target.manifold:
            errors.append("source and target must lie on the same manifold")
        if len(self.forward) != self.target.dimension:
            errors.append("forward map length must match target dimension")
        if self.inverse is not None and len(self.inverse) != self.source.dimension:
            errors.append("inverse map length must match source dimension")
        return invalid_report(*errors) if errors else valid_report()

    def validate(self) -> bool:
        """Validate map dimensions and manifold compatibility."""
        report = self.validation_report()
        if not report.ok:
            raise TensorKernelError("; ".join(report.errors))
        return True

    @property
    def mapping(self) -> tuple[Scalar, ...]:
        """Target-coordinate expressions as functions of source coordinates.

        This read-only alias mirrors the mathematical term "mapping" while
        keeping ``forward`` as the canonical storage field used internally.
        """
        return self.forward

    def jacobian(self) -> tuple[tuple[Scalar, ...], ...]:
        sp = _require_sympy()
        src = self.source_symbols
        matrix = sp.Matrix([[sp.diff(expr, coord) for coord in src] for expr in self.forward])
        return tuple(tuple(sp.cancel(matrix[i, j]) for j in range(matrix.cols)) for i in range(matrix.rows))

    def inverse_jacobian(self) -> tuple[tuple[Scalar, ...], ...] | None:
        sp = _require_sympy()
        if self.inverse is not None:
            tgt = self.target_symbols
            matrix = sp.Matrix([[sp.diff(expr, coord) for coord in tgt] for expr in self.inverse])
            return tuple(tuple(sp.cancel(matrix[i, j]) for j in range(matrix.cols)) for i in range(matrix.rows))
        try:
            inv = sp.Matrix(self.jacobian()).inv()
        except Exception:
            return None
        return tuple(tuple(sp.cancel(inv[i, j]) for j in range(inv.cols)) for i in range(inv.rows))

    def jacobian_determinant(self) -> Scalar:
        sp = _require_sympy()
        det = sp.Matrix(self.jacobian()).det()
        if det.has(sp.atan2) or det.has(sp.Abs):
            try:
                return sp.factor(det)
            except Exception:
                return det
        try:
            factored = sp.factor(det)
            if hasattr(factored, "count_ops") and factored.count_ops() <= 64:
                return sp.factor(sp.trigsimp(factored))
            return factored
        except Exception:
            return det

    def is_locally_invertible_condition(self) -> Scalar:
        sp = _require_sympy()
        return sp.Ne(self.jacobian_determinant(), 0)

    def as_component_transform(self) -> CoordinateTransform:
        return CoordinateTransform(self.source, self.target, self.jacobian(), self.inverse_jacobian())

    def inverse_map(self, *, name: str | None = None) -> "CoordinateMap":
        if self.inverse is None:
            raise TensorKernelError("This coordinate map does not have an explicit inverse branch.")
        return CoordinateMap(
            self.target,
            self.source,
            self.inverse,
            inverse=self.forward,
            singularities=self.singularities,
            domain_notes=self.domain_notes,
            name=name or (f"{self.target.name}_to_{self.source.name}"),
        )

    def transform_scalar_to_target(self, expression: Scalar) -> Scalar:
        if self.inverse is None:
            raise TensorKernelError("Scalar transport to target coordinates requires an inverse branch.")
        src = self.source_symbols
        return expression.subs(dict(zip(src, self.inverse)))

    def transform_scalar_to_source(self, expression: Scalar) -> Scalar:
        tgt = self.target_symbols
        return expression.subs(dict(zip(tgt, self.forward)))


@dataclass(frozen=True, slots=True)
class StandardCoordinateEntry:
    """Catalog metadata for a standard coordinate system."""

    name: str
    dimension: int
    coordinate_names: tuple[str, ...]
    domains: Mapping[str, CoordinateDomain]
    metric_builder: MetricBuilder
    to_cartesian_builder: MapBuilder | None = None
    from_cartesian_builder: MapBuilder | None = None
    singularity_builder: Callable[[tuple[Scalar, ...], Mapping[str, Scalar]], tuple[CoordinateSingularity, ...]] | None = None
    parameters: Mapping[str, Scalar] = field(default_factory=dict, compare=False, hash=False)
    description: str = ""

    def coordinate_system(self, manifold: Manifold | None = None, *, index_name: str | None = None) -> CoordinateSystem:
        manifold = manifold or Manifold(f"M{self.dimension}", self.dimension)
        index_type = manifold.index_type(index_name or f"{self.name}_coord")
        return CoordinateSystem(
            self.name,
            manifold,
            self.coordinate_names,
            index_type=index_type,
            domain={name: domain.as_dict() for name, domain in self.domains.items()},
            coordinate_symbols=self.symbols(),
        )

    def symbols(self) -> tuple[Scalar, ...]:
        sp = _require_sympy()
        return tuple(sp.Symbol(name, real=True) for name in self.coordinate_names)

    def metric(self, coordinates: Sequence[Scalar] | None = None, parameters: Mapping[str, Scalar] | None = None) -> Any:
        coords = tuple(coordinates) if coordinates is not None else self.symbols()
        params = dict(self.parameters)
        if parameters:
            params.update(parameters)
        return self.metric_builder(coords, params)

    def singularities(self, coordinates: Sequence[Scalar] | None = None, parameters: Mapping[str, Scalar] | None = None) -> tuple[CoordinateSingularity, ...]:
        coords = tuple(coordinates) if coordinates is not None else self.symbols()
        params = dict(self.parameters)
        if parameters:
            params.update(parameters)
        if self.singularity_builder is None:
            return tuple()
        return self.singularity_builder(coords, params)

    def map_to_cartesian(self, cartesian: CoordinateSystem | None = None, *, manifold: Manifold | None = None) -> CoordinateMap:
        if self.to_cartesian_builder is None:
            raise TensorKernelError(f"{self.name} has no catalogued Cartesian map.")
        source = self.coordinate_system(manifold)
        target = cartesian or standard_coordinate_system(f"cartesian{self.dimension}", manifold=source.manifold)
        forward = self.to_cartesian_builder(self.symbols(), self.parameters)
        inverse = None
        if self.from_cartesian_builder is not None:
            target_symbols = target.coordinate_symbols or tuple(_require_sympy().Symbol(name, real=True) for name in target.coordinate_names)
            inverse = self.from_cartesian_builder(target_symbols, self.parameters)
        return CoordinateMap(source, target, forward, inverse=inverse, singularities=self.singularities(), name=f"{self.name}_to_cartesian")


def _diag(*entries: Scalar):
    sp = _require_sympy()
    return sp.diag(*entries)


def _cartesian_metric(coords: tuple[Scalar, ...], params: Mapping[str, Scalar]):
    return _diag(*([1] * len(coords)))


def _minkowski_metric(coords: tuple[Scalar, ...], params: Mapping[str, Scalar]):
    return _diag(-1, *([1] * (len(coords) - 1)))


def _domain_real(names: Sequence[str]) -> dict[str, CoordinateDomain]:
    return {name: CoordinateDomain("real_line") for name in names}


def _angle_domain(*, lower=None, upper=None) -> CoordinateDomain:
    sp = _require_sympy()
    return CoordinateDomain("open_interval", sp.Integer(0) if lower is None else lower, 2 * sp.pi if upper is None else upper, periodic=True)


def _positive_domain() -> CoordinateDomain:
    sp = _require_sympy()
    return CoordinateDomain("open_interval", sp.Integer(0), None)


@lru_cache(maxsize=1)
def standard_coordinate_catalog() -> dict[str, StandardCoordinateEntry]:
    """Return a catalog of common coordinate systems with metrics and maps."""
    sp = _require_sympy()
    a = sp.Symbol("a", positive=True, real=True)
    R = sp.Symbol("R", positive=True, real=True)
    t = sp.Symbol("t", real=True)

    catalog: dict[str, StandardCoordinateEntry] = {}
    for dim in (2, 3, 4):
        names = tuple("xyzw"[:dim]) if dim <= 4 else tuple(f"x{i}" for i in range(dim))
        catalog[f"cartesian{dim}"] = StandardCoordinateEntry(
            f"cartesian{dim}", dim, names, _domain_real(names), _cartesian_metric,
            to_cartesian_builder=lambda coords, params: tuple(coords),
            from_cartesian_builder=lambda coords, params: tuple(coords),
            description=f"{dim}-dimensional Cartesian coordinates",
        )

    catalog["polar"] = StandardCoordinateEntry(
        "polar", 2, ("r", "theta"),
        {"r": _positive_domain(), "theta": _angle_domain(lower=-sp.pi, upper=sp.pi)},
        lambda c, p: _diag(1, c[0] ** 2),
        to_cartesian_builder=lambda c, p: (c[0] * sp.cos(c[1]), c[0] * sp.sin(c[1])),
        from_cartesian_builder=lambda c, p: (sp.sqrt(c[0] ** 2 + c[1] ** 2), sp.atan2(c[1], c[0])),
        singularity_builder=lambda c, p: (CoordinateSingularity(c[0], "origin r=0"),),
        description="Plane polar coordinates",
    )
    catalog["cylindrical"] = StandardCoordinateEntry(
        "cylindrical", 3, ("rho", "phi", "z"),
        {"rho": _positive_domain(), "phi": _angle_domain(lower=-sp.pi, upper=sp.pi), "z": CoordinateDomain("real_line")},
        lambda c, p: _diag(1, c[0] ** 2, 1),
        to_cartesian_builder=lambda c, p: (c[0] * sp.cos(c[1]), c[0] * sp.sin(c[1]), c[2]),
        from_cartesian_builder=lambda c, p: (sp.sqrt(c[0] ** 2 + c[1] ** 2), sp.atan2(c[1], c[0]), c[2]),
        singularity_builder=lambda c, p: (CoordinateSingularity(c[0], "axis rho=0"),),
        description="Cylindrical coordinates in Euclidean 3-space",
    )
    catalog["spherical"] = StandardCoordinateEntry(
        "spherical", 3, ("r", "theta", "phi"),
        {"r": _positive_domain(), "theta": CoordinateDomain("open_interval", 0, sp.pi), "phi": _angle_domain(lower=-sp.pi, upper=sp.pi)},
        lambda c, p: _diag(1, c[0] ** 2, c[0] ** 2 * sp.sin(c[1]) ** 2),
        to_cartesian_builder=lambda c, p: (c[0] * sp.sin(c[1]) * sp.cos(c[2]), c[0] * sp.sin(c[1]) * sp.sin(c[2]), c[0] * sp.cos(c[1])),
        from_cartesian_builder=lambda c, p: (
            sp.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2),
            sp.acos(c[2] / sp.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)),
            sp.atan2(c[1], c[0]),
        ),
        singularity_builder=lambda c, p: (CoordinateSingularity(c[0], "origin r=0"), CoordinateSingularity(sp.sin(c[1]), "polar axis sin(theta)=0")),
        description="Spherical coordinates in Euclidean 3-space",
    )
    catalog["parabolic_cylindrical"] = StandardCoordinateEntry(
        "parabolic_cylindrical", 3, ("u", "v", "z"),
        {"u": CoordinateDomain("real_line"), "v": CoordinateDomain("real_line"), "z": CoordinateDomain("real_line")},
        lambda c, p: _diag(c[0] ** 2 + c[1] ** 2, c[0] ** 2 + c[1] ** 2, 1),
        to_cartesian_builder=lambda c, p: ((c[0] ** 2 - c[1] ** 2) / 2, c[0] * c[1], c[2]),
        from_cartesian_builder=lambda c, p: (
            sp.sqrt(sp.sqrt(c[0] ** 2 + c[1] ** 2) + c[0]),
            c[1] / sp.sqrt(sp.sqrt(c[0] ** 2 + c[1] ** 2) + c[0]),
            c[2],
        ),
        singularity_builder=lambda c, p: (CoordinateSingularity(c[0] ** 2 + c[1] ** 2, "parabolic focal line"),),
        description="Parabolic cylindrical coordinates",
    )
    catalog["parabolic"] = StandardCoordinateEntry(
        "parabolic", 2, ("sigma", "tau"),
        {"sigma": CoordinateDomain("real_line"), "tau": CoordinateDomain("real_line")},
        lambda c, p: _diag(c[0] ** 2 + c[1] ** 2, c[0] ** 2 + c[1] ** 2),
        to_cartesian_builder=lambda c, p: ((c[0] ** 2 - c[1] ** 2) / 2, c[0] * c[1]),
        from_cartesian_builder=lambda c, p: (
            sp.sqrt(sp.sqrt(c[0] ** 2 + c[1] ** 2) + c[0]),
            c[1] / sp.sqrt(sp.sqrt(c[0] ** 2 + c[1] ** 2) + c[0]),
        ),
        singularity_builder=lambda c, p: (CoordinateSingularity(c[0] ** 2 + c[1] ** 2, "parabolic origin"),),
        description="Plane parabolic coordinates",
    )
    catalog["elliptic_cylindrical"] = StandardCoordinateEntry(
        "elliptic_cylindrical", 3, ("mu", "nu", "z"),
        {"mu": CoordinateDomain("open_interval", 0, None), "nu": _angle_domain(lower=-sp.pi, upper=sp.pi), "z": CoordinateDomain("real_line")},
        lambda c, p: _diag(p.get("a", a) ** 2 * (sp.sinh(c[0]) ** 2 + sp.sin(c[1]) ** 2), p.get("a", a) ** 2 * (sp.sinh(c[0]) ** 2 + sp.sin(c[1]) ** 2), 1),
        to_cartesian_builder=lambda c, p: (p.get("a", a) * sp.cosh(c[0]) * sp.cos(c[1]), p.get("a", a) * sp.sinh(c[0]) * sp.sin(c[1]), c[2]),
        parameters={"a": a},
        singularity_builder=lambda c, p: (CoordinateSingularity(sp.sinh(c[0]) ** 2 + sp.sin(c[1]) ** 2, "elliptic coordinate focal set"),),
        description="Elliptic cylindrical coordinates",
    )
    catalog["prolate_spheroidal"] = StandardCoordinateEntry(
        "prolate_spheroidal", 3, ("mu", "nu", "phi"),
        {"mu": CoordinateDomain("open_interval", 0, None), "nu": CoordinateDomain("open_interval", 0, sp.pi), "phi": _angle_domain(lower=-sp.pi, upper=sp.pi)},
        lambda c, p: _diag(
            p.get("a", a) ** 2 * (sp.sinh(c[0]) ** 2 + sp.sin(c[1]) ** 2),
            p.get("a", a) ** 2 * (sp.sinh(c[0]) ** 2 + sp.sin(c[1]) ** 2),
            p.get("a", a) ** 2 * sp.sinh(c[0]) ** 2 * sp.sin(c[1]) ** 2,
        ),
        to_cartesian_builder=lambda c, p: (
            p.get("a", a) * sp.sinh(c[0]) * sp.sin(c[1]) * sp.cos(c[2]),
            p.get("a", a) * sp.sinh(c[0]) * sp.sin(c[1]) * sp.sin(c[2]),
            p.get("a", a) * sp.cosh(c[0]) * sp.cos(c[1]),
        ),
        parameters={"a": a},
        singularity_builder=lambda c, p: (CoordinateSingularity(sp.sinh(c[0]) * sp.sin(c[1]), "symmetry axis or focal segment"),),
        description="Prolate spheroidal coordinates",
    )
    catalog["oblate_spheroidal"] = StandardCoordinateEntry(
        "oblate_spheroidal", 3, ("mu", "nu", "phi"),
        {"mu": CoordinateDomain("open_interval", 0, None), "nu": CoordinateDomain("open_interval", -sp.pi / 2, sp.pi / 2), "phi": _angle_domain(lower=-sp.pi, upper=sp.pi)},
        lambda c, p: _diag(
            p.get("a", a) ** 2 * (sp.sinh(c[0]) ** 2 + sp.sin(c[1]) ** 2),
            p.get("a", a) ** 2 * (sp.sinh(c[0]) ** 2 + sp.sin(c[1]) ** 2),
            p.get("a", a) ** 2 * sp.cosh(c[0]) ** 2 * sp.cos(c[1]) ** 2,
        ),
        to_cartesian_builder=lambda c, p: (
            p.get("a", a) * sp.cosh(c[0]) * sp.cos(c[1]) * sp.cos(c[2]),
            p.get("a", a) * sp.cosh(c[0]) * sp.cos(c[1]) * sp.sin(c[2]),
            p.get("a", a) * sp.sinh(c[0]) * sp.sin(c[1]),
        ),
        parameters={"a": a},
        singularity_builder=lambda c, p: (CoordinateSingularity(sp.cosh(c[0]) * sp.cos(c[1]), "symmetry axis or disk branch set"),),
        description="Oblate spheroidal coordinates",
    )
    catalog["bipolar"] = StandardCoordinateEntry(
        "bipolar", 2, ("sigma", "tau"),
        {"sigma": _angle_domain(lower=-sp.pi, upper=sp.pi), "tau": CoordinateDomain("real_line")},
        lambda c, p: _diag((p.get("a", a) / (sp.cosh(c[1]) - sp.cos(c[0]))) ** 2, (p.get("a", a) / (sp.cosh(c[1]) - sp.cos(c[0]))) ** 2),
        to_cartesian_builder=lambda c, p: (
            p.get("a", a) * sp.sinh(c[1]) / (sp.cosh(c[1]) - sp.cos(c[0])),
            p.get("a", a) * sp.sin(c[0]) / (sp.cosh(c[1]) - sp.cos(c[0])),
        ),
        parameters={"a": a},
        singularity_builder=lambda c, p: (CoordinateSingularity(sp.cosh(c[1]) - sp.cos(c[0]), "bipolar denominator vanishes"),),
        description="Bipolar coordinates",
    )
    catalog["toroidal"] = StandardCoordinateEntry(
        "toroidal", 3, ("tau", "sigma", "phi"),
        {"tau": CoordinateDomain("open_interval", 0, None), "sigma": _angle_domain(lower=-sp.pi, upper=sp.pi), "phi": _angle_domain(lower=-sp.pi, upper=sp.pi)},
        lambda c, p: _diag(
            (p.get("a", a) / (sp.cosh(c[0]) - sp.cos(c[1]))) ** 2,
            (p.get("a", a) / (sp.cosh(c[0]) - sp.cos(c[1]))) ** 2,
            (p.get("a", a) * sp.sinh(c[0]) / (sp.cosh(c[0]) - sp.cos(c[1]))) ** 2,
        ),
        to_cartesian_builder=lambda c, p: (
            p.get("a", a) * sp.sinh(c[0]) * sp.cos(c[2]) / (sp.cosh(c[0]) - sp.cos(c[1])),
            p.get("a", a) * sp.sinh(c[0]) * sp.sin(c[2]) / (sp.cosh(c[0]) - sp.cos(c[1])),
            p.get("a", a) * sp.sin(c[1]) / (sp.cosh(c[0]) - sp.cos(c[1])),
        ),
        parameters={"a": a},
        singularity_builder=lambda c, p: (CoordinateSingularity(sp.cosh(c[0]) - sp.cos(c[1]), "toroidal denominator vanishes"), CoordinateSingularity(sp.sinh(c[0]), "symmetry axis")),
        description="Toroidal coordinates",
    )
    catalog["bispherical"] = StandardCoordinateEntry(
        "bispherical", 3, ("sigma", "tau", "phi"),
        {"sigma": _angle_domain(lower=-sp.pi, upper=sp.pi), "tau": CoordinateDomain("real_line"), "phi": _angle_domain(lower=-sp.pi, upper=sp.pi)},
        lambda c, p: _diag(
            (p.get("a", a) / (sp.cosh(c[1]) - sp.cos(c[0]))) ** 2,
            (p.get("a", a) / (sp.cosh(c[1]) - sp.cos(c[0]))) ** 2,
            (p.get("a", a) * sp.sin(c[0]) / (sp.cosh(c[1]) - sp.cos(c[0]))) ** 2,
        ),
        to_cartesian_builder=lambda c, p: (
            p.get("a", a) * sp.sin(c[0]) * sp.cos(c[2]) / (sp.cosh(c[1]) - sp.cos(c[0])),
            p.get("a", a) * sp.sin(c[0]) * sp.sin(c[2]) / (sp.cosh(c[1]) - sp.cos(c[0])),
            p.get("a", a) * sp.sinh(c[1]) / (sp.cosh(c[1]) - sp.cos(c[0])),
        ),
        parameters={"a": a},
        singularity_builder=lambda c, p: (CoordinateSingularity(sp.cosh(c[1]) - sp.cos(c[0]), "bispherical denominator vanishes"), CoordinateSingularity(sp.sin(c[0]), "symmetry axis")),
        description="Bispherical coordinates",
    )
    catalog["minkowski_cartesian4"] = StandardCoordinateEntry(
        "minkowski_cartesian4", 4, ("t", "x", "y", "z"), _domain_real(("t", "x", "y", "z")), _minkowski_metric,
        description="Minkowski spacetime in Cartesian inertial coordinates",
    )
    catalog["schwarzschild"] = StandardCoordinateEntry(
        "schwarzschild", 4, ("t", "r", "theta", "phi"),
        {"t": CoordinateDomain("real_line"), "r": CoordinateDomain("open_interval", 2 * sp.Symbol("M", positive=True, real=True), None), "theta": CoordinateDomain("open_interval", 0, sp.pi), "phi": _angle_domain(lower=-sp.pi, upper=sp.pi)},
        lambda c, p: _diag(-(1 - 2 * p.get("M", sp.Symbol("M", positive=True, real=True)) / c[1]), 1 / (1 - 2 * p.get("M", sp.Symbol("M", positive=True, real=True)) / c[1]), c[1] ** 2, c[1] ** 2 * sp.sin(c[2]) ** 2),
        parameters={"M": sp.Symbol("M", positive=True, real=True)},
        singularity_builder=lambda c, p: (CoordinateSingularity(c[1] - 2 * p.get("M", sp.Symbol("M", positive=True, real=True)), "Schwarzschild horizon in this chart"), CoordinateSingularity(c[1], "curvature singularity"), CoordinateSingularity(sp.sin(c[2]), "polar axis")),
        description="Schwarzschild coordinates with signature (-,+,+,+)",
    )
    catalog["flrw_flat"] = StandardCoordinateEntry(
        "flrw_flat", 4, ("t", "r", "theta", "phi"),
        {"t": CoordinateDomain("real_line"), "r": CoordinateDomain("open_interval", 0, None), "theta": CoordinateDomain("open_interval", 0, sp.pi), "phi": _angle_domain(lower=-sp.pi, upper=sp.pi)},
        lambda c, p: _diag(-1, sp.Function("a")(c[0]) ** 2, sp.Function("a")(c[0]) ** 2 * c[1] ** 2, sp.Function("a")(c[0]) ** 2 * c[1] ** 2 * sp.sin(c[2]) ** 2),
        singularity_builder=lambda c, p: (CoordinateSingularity(c[1], "spherical origin"), CoordinateSingularity(sp.sin(c[2]), "polar axis")),
        description="Spatially flat FLRW coordinates with signature (-,+,+,+)",
    )
    catalog["rindler2"] = StandardCoordinateEntry(
        "rindler2", 2, ("eta", "rho"),
        {"eta": CoordinateDomain("real_line"), "rho": _positive_domain()},
        lambda c, p: _diag(-c[1] ** 2, 1),
        to_cartesian_builder=lambda c, p: (c[1] * sp.sinh(c[0]), c[1] * sp.cosh(c[0])),
        from_cartesian_builder=lambda c, p: (sp.atanh(c[0] / c[1]), sp.sqrt(c[1] ** 2 - c[0] ** 2)),
        singularity_builder=lambda c, p: (CoordinateSingularity(c[1], "Rindler horizon rho=0"),),
        description="Two-dimensional Rindler coordinates on a wedge of Minkowski space",
    )
    catalog["hyperbolic2_upper_half_plane"] = StandardCoordinateEntry(
        "hyperbolic2_upper_half_plane", 2, ("x", "y"),
        {"x": CoordinateDomain("real_line"), "y": _positive_domain()},
        lambda c, p: _diag(1 / c[1] ** 2, 1 / c[1] ** 2),
        singularity_builder=lambda c, p: (CoordinateSingularity(c[1], "conformal boundary y=0"),),
        description="Hyperbolic plane in upper-half-plane coordinates",
    )
    catalog["hyperbolic3_upper_half_space"] = StandardCoordinateEntry(
        "hyperbolic3_upper_half_space", 3, ("x", "y", "z"),
        {"x": CoordinateDomain("real_line"), "y": CoordinateDomain("real_line"), "z": _positive_domain()},
        lambda c, p: _diag(1 / c[2] ** 2, 1 / c[2] ** 2, 1 / c[2] ** 2),
        singularity_builder=lambda c, p: (CoordinateSingularity(c[2], "conformal boundary z=0"),),
        description="Hyperbolic 3-space in upper-half-space coordinates",
    )
    catalog["isotropic_schwarzschild"] = StandardCoordinateEntry(
        "isotropic_schwarzschild", 4, ("t", "rho", "theta", "phi"),
        {"t": CoordinateDomain("real_line"), "rho": CoordinateDomain("open_interval", sp.Symbol("M", positive=True, real=True) / 2, None), "theta": CoordinateDomain("open_interval", 0, sp.pi), "phi": _angle_domain(lower=-sp.pi, upper=sp.pi)},
        lambda c, p: _diag(
            -((1 - p.get("M", sp.Symbol("M", positive=True, real=True)) / (2 * c[1])) / (1 + p.get("M", sp.Symbol("M", positive=True, real=True)) / (2 * c[1]))) ** 2,
            (1 + p.get("M", sp.Symbol("M", positive=True, real=True)) / (2 * c[1])) ** 4,
            (1 + p.get("M", sp.Symbol("M", positive=True, real=True)) / (2 * c[1])) ** 4 * c[1] ** 2,
            (1 + p.get("M", sp.Symbol("M", positive=True, real=True)) / (2 * c[1])) ** 4 * c[1] ** 2 * sp.sin(c[2]) ** 2,
        ),
        parameters={"M": sp.Symbol("M", positive=True, real=True)},
        singularity_builder=lambda c, p: (CoordinateSingularity(c[1] - p.get("M", sp.Symbol("M", positive=True, real=True)) / 2, "Schwarzschild horizon in isotropic radius"), CoordinateSingularity(sp.sin(c[2]), "polar axis")),
        description="Schwarzschild metric in isotropic spatial coordinates",
    )
    catalog["eddington_finkelstein_ingoing"] = StandardCoordinateEntry(
        "eddington_finkelstein_ingoing", 4, ("v", "r", "theta", "phi"),
        {"v": CoordinateDomain("real_line"), "r": _positive_domain(), "theta": CoordinateDomain("open_interval", 0, sp.pi), "phi": _angle_domain(lower=-sp.pi, upper=sp.pi)},
        lambda c, p: sp.Matrix([
            [-(1 - 2 * p.get("M", sp.Symbol("M", positive=True, real=True)) / c[1]), 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, c[1] ** 2, 0],
            [0, 0, 0, c[1] ** 2 * sp.sin(c[2]) ** 2],
        ]),
        parameters={"M": sp.Symbol("M", positive=True, real=True)},
        singularity_builder=lambda c, p: (CoordinateSingularity(c[1], "curvature singularity"), CoordinateSingularity(sp.sin(c[2]), "polar axis")),
        description="Ingoing Eddington-Finkelstein coordinates",
    )
    catalog["de_sitter_flat"] = StandardCoordinateEntry(
        "de_sitter_flat", 4, ("t", "x", "y", "z"),
        _domain_real(("t", "x", "y", "z")),
        lambda c, p: _diag(-1, sp.exp(2 * p.get("H", sp.Symbol("H", positive=True, real=True)) * c[0]), sp.exp(2 * p.get("H", sp.Symbol("H", positive=True, real=True)) * c[0]), sp.exp(2 * p.get("H", sp.Symbol("H", positive=True, real=True)) * c[0])),
        parameters={"H": sp.Symbol("H", positive=True, real=True)},
        description="Flat slicing of de Sitter spacetime",
    )
    return catalog


def standard_coordinate_entry(name: str) -> StandardCoordinateEntry:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cartesian": "cartesian3",
        "cartesian_2d": "cartesian2",
        "cartesian_3d": "cartesian3",
        "cartesian_4d": "cartesian4",
        "spherical3": "spherical",
        "polar2": "polar",
        "rindler": "rindler2",
        "hyperbolic2": "hyperbolic2_upper_half_plane",
        "hyperbolic3": "hyperbolic3_upper_half_space",
    }
    key = aliases.get(key, key)
    catalog = standard_coordinate_catalog()
    if key not in catalog:
        raise TensorKernelError(f"Unknown standard coordinate system {name!r}.")


    return catalog[key]


def list_standard_coordinates() -> tuple[str, ...]:
    return tuple(sorted(standard_coordinate_catalog()))


def standard_coordinate_system(name: str, *, manifold: Manifold | None = None, index_name: str | None = None) -> CoordinateSystem:
    return standard_coordinate_entry(name).coordinate_system(manifold, index_name=index_name)


def standard_metric(name: str, coordinates: Sequence[Scalar] | None = None, parameters: Mapping[str, Scalar] | None = None) -> Any:
    return standard_coordinate_entry(name).metric(coordinates, parameters)


def standard_coordinate_map_to_cartesian(name: str, *, manifold: Manifold | None = None) -> CoordinateMap:
    return standard_coordinate_entry(name).map_to_cartesian(manifold=manifold)


def _coordinate_domain_boundary_expressions(domain: CoordinateDomain, symbol: Scalar) -> tuple[Scalar, ...]:
    bounds: list[Scalar] = []
    if domain.lower is not None and not domain.lower_inclusive:
        bounds.append(symbol - domain.lower)
    if domain.upper is not None and not domain.upper_inclusive:
        bounds.append(symbol - domain.upper)
    return tuple(bounds)


def coordinate_domain_assumptions(entry_or_name: StandardCoordinateEntry | str, coordinates: Sequence[Scalar] | None = None) -> tuple[Scalar, ...]:
    """Return symbolic coordinate-domain assumptions for a catalog entry."""
    entry = standard_coordinate_entry(entry_or_name) if isinstance(entry_or_name, str) else entry_or_name
    coords = tuple(coordinates) if coordinates is not None else entry.symbols()
    clauses: list[Scalar] = []
    for name, coord in zip(entry.coordinate_names, coords):
        domain = entry.domains.get(name)
        if domain is not None:
            clauses.extend(domain.assumptions(coord))
    return tuple(clauses)


def infer_coordinate_singularities(metric=None, coordinates: Sequence[Scalar] | None = None, jacobian_det=None, known: Sequence[CoordinateSingularity] = ()) -> tuple[CoordinateSingularity, ...]:
    """Infer obvious singular loci from metric determinant and/or Jacobian determinant.

    This is intentionally conservative: it records determinant expressions rather
    than attempting algebraic variety decomposition.
    """
    sp = _require_sympy()
    items = list(known)
    if jacobian_det is not None:
        items.append(CoordinateSingularity(sp.factor(jacobian_det), "Jacobian determinant vanishes"))
    if metric is not None:
        det = sp.factor(sp.Matrix(metric).det())
        items.append(CoordinateSingularity(det, "metric determinant vanishes or changes signature"))
    if coordinates is not None:
        for coord in coordinates:
            if getattr(coord, "is_positive", None) is False:
                continue
    # Deduplicate by symbolic string and reason, preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[CoordinateSingularity] = []
    for item in items:
        key = (str(item.expression), item.reason)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return tuple(unique)


def coordinate_map_between(source_name: str, target_name: str, *, manifold: Manifold | None = None) -> CoordinateMap:
    """Return a catalogued transition map between two standard coordinate systems.

    The implementation composes source -> Cartesian with the inverse branch of
    target -> Cartesian.  It therefore returns only one branch where the catalog
    has one.
    """
    source_entry = standard_coordinate_entry(source_name)
    target_entry = standard_coordinate_entry(target_name)
    sp = _require_sympy()
    if source_entry.name == "spherical" and target_entry.name == "cylindrical":
        source = source_entry.coordinate_system(manifold)
        target = target_entry.coordinate_system(source.manifold)
        r, theta, phi = source_entry.symbols()
        rho, phic, z = target_entry.symbols()
        return CoordinateMap(
            source,
            target,
            (r * sp.sin(theta), phi, r * sp.cos(theta)),
            inverse=(sp.sqrt(rho ** 2 + z ** 2), sp.atan2(rho, z), phic),
            singularities=infer_coordinate_singularities(jacobian_det=-r, known=source_entry.singularities((r, theta, phi)) + target_entry.singularities((rho, phic, z))),
            domain_notes={"constructed_from": "catalogued spherical-cylindrical transition branch"},
            name="spherical_to_cylindrical",
        )
    if source_entry.name == "cylindrical" and target_entry.name == "spherical":
        source = source_entry.coordinate_system(manifold)
        target = target_entry.coordinate_system(source.manifold)
        rho, phi, z = source_entry.symbols()
        r, theta, phis = target_entry.symbols()
        return CoordinateMap(
            source,
            target,
            (sp.sqrt(rho ** 2 + z ** 2), sp.atan2(rho, z), phi),
            inverse=(r * sp.sin(theta), phis, r * sp.cos(theta)),
            singularities=infer_coordinate_singularities(jacobian_det=-rho / sp.sqrt(rho ** 2 + z ** 2), known=source_entry.singularities((rho, phi, z)) + target_entry.singularities((r, theta, phis))),
            domain_notes={"constructed_from": "catalogued cylindrical-spherical transition branch"},
            name="cylindrical_to_spherical",
        )
    if source_entry.dimension != target_entry.dimension:
        raise TensorKernelError("Coordinate transition maps require equal dimensions.")
    if source_entry.name == target_entry.name:
        system = source_entry.coordinate_system(manifold)
        return CoordinateMap(system, system, system.coordinate_symbols if hasattr(system, "coordinate_symbols") else source_entry.symbols(), inverse=source_entry.symbols(), name=f"{source_entry.name}_identity")
    source_to_cart = source_entry.map_to_cartesian(manifold=manifold)
    target_to_cart = target_entry.map_to_cartesian(manifold=source_to_cart.source.manifold)
    if target_to_cart.inverse is None:
        raise TensorKernelError(f"No inverse Cartesian branch is catalogued for {target_entry.name!r}.")
    cart_symbols = target_to_cart.target_symbols
    forward = tuple(sp.cancel(expr.subs(dict(zip(cart_symbols, source_to_cart.forward)))) for expr in target_to_cart.inverse)
    inverse = None
    if source_to_cart.inverse is not None:
        target_symbols = source_to_cart.target_symbols
        inverse = tuple(sp.cancel(expr.subs(dict(zip(target_symbols, target_to_cart.forward)))) for expr in source_to_cart.inverse)
    singularities = infer_coordinate_singularities(jacobian_det=sp.Matrix([[sp.diff(expr, coord) for coord in source_to_cart.source_symbols] for expr in forward]).det(), known=source_to_cart.singularities + target_to_cart.singularities)
    return CoordinateMap(source_to_cart.source, target_to_cart.source, forward, inverse=inverse, singularities=singularities, domain_notes={"constructed_from": "cartesian transition branch"}, name=f"{source_entry.name}_to_{target_entry.name}")


def complete_coordinate_metadata(name: str, coordinates: Sequence[Scalar] | None = None, parameters: Mapping[str, Scalar] | None = None) -> dict[str, Any]:
    """Return metric, domain, and inferred singularity metadata for a catalog entry."""
    entry = standard_coordinate_entry(name)
    coords = tuple(coordinates) if coordinates is not None else entry.symbols()
    metric = entry.metric(coords, parameters)
    known = entry.singularities(coords, parameters)
    return {
        "name": entry.name,
        "dimension": entry.dimension,
        "coordinates": coords,
        "domains": dict(entry.domains),
        "assumptions": coordinate_domain_assumptions(entry, coords),
        "metric": metric,
        "singularities": infer_coordinate_singularities(metric=metric, coordinates=coords, known=known),
        "description": entry.description,
    }
