from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Iterable, Mapping, Sequence

import sympy as sp

from .charts import CoordinateChart
from .geometry_foundations import (
    ManifoldDef,
    BundleDef,
    ChartDef,
    MetricDef,
    ConnectionDef,
    FrameDef,
    DifferentialOperatorDef,
    manifold,
    chart_definition,
    tangent_bundle,
    cotangent_bundle,
)
from .tensor_indices import DifferentialForm
from .tensor_core import TensorObject
from .fields import TensorField
from .basis import TensorBasis, TensorFrame


@dataclass(frozen=True)
class SpinStructureDef:
    name: str
    manifold: ManifoldDef
    signature: tuple[int, int, int]
    oriented: bool = True
    time_oriented: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return self.manifold.dimension


@dataclass(frozen=True)
class SpinorBundleDef:
    name: str
    manifold: ManifoldDef
    spin_structure: SpinStructureDef
    complex_dimension: int
    chirality: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CliffordAlgebraDef:
    name: str
    dimension: int
    signature: tuple[int, int, int]
    generator_prefix: str = "gamma"
    basis_labels: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if sum(self.signature) != self.dimension:
            raise ValueError("Clifford signature must sum to the algebra dimension.")
        if self.basis_labels and len(self.basis_labels) != self.dimension:
            raise ValueError("basis_labels length must equal the dimension.")

    @property
    def diagonal_metric(self) -> tuple[int, ...]:
        p, q, r = self.signature
        return (1,) * p + (-1,) * q + (0,) * r

    def eta(self, i: int, j: int) -> sp.Expr:
        if i < 0 or j < 0 or i >= self.dimension or j >= self.dimension:
            raise IndexError("generator index out of range")
        return sp.Integer(self.diagonal_metric[i]) if i == j else sp.Integer(0)


@dataclass(frozen=True)
class GeometryArchive:
    version: str
    objects: tuple[Any, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def spin_structure(manifold_obj: ManifoldDef, signature: tuple[int, int, int], *, name: str | None = None, oriented: bool = True, time_oriented: bool = False, metadata: Mapping[str, Any] | None = None) -> SpinStructureDef:
    return SpinStructureDef(
        name=name or f"Spin({manifold_obj.name})",
        manifold=manifold_obj,
        signature=signature,
        oriented=oriented,
        time_oriented=time_oriented,
        metadata=dict(metadata or {}),
    )


def spinor_bundle(manifold_obj: ManifoldDef, signature: tuple[int, int, int], *, name: str = "S", chirality: str | None = None, metadata: Mapping[str, Any] | None = None) -> SpinorBundleDef:
    spin = spin_structure(manifold_obj, signature, metadata=metadata)
    complex_dimension = 2 ** (manifold_obj.dimension // 2)
    return SpinorBundleDef(
        name=name,
        manifold=manifold_obj,
        spin_structure=spin,
        complex_dimension=complex_dimension,
        chirality=chirality,
        metadata=dict(metadata or {}),
    )


def clifford_algebra(dimension: int, signature: tuple[int, int, int], *, name: str = "Cl", generator_prefix: str = "gamma", basis_labels: Sequence[str] | None = None, metadata: Mapping[str, Any] | None = None) -> CliffordAlgebraDef:
    return CliffordAlgebraDef(
        name=name,
        dimension=dimension,
        signature=signature,
        generator_prefix=generator_prefix,
        basis_labels=tuple(basis_labels or tuple(str(i) for i in range(dimension))),
        metadata=dict(metadata or {}),
    )


def gamma_generators(clifford: CliffordAlgebraDef) -> tuple[sp.Symbol, ...]:
    labels = clifford.basis_labels or tuple(str(i) for i in range(clifford.dimension))
    return tuple(sp.Symbol(f"{clifford.generator_prefix}{label}", commutative=False) for label in labels)


def gamma_anticommutator(left: sp.Symbol, right: sp.Symbol, clifford: CliffordAlgebraDef) -> sp.Expr:
    gens = gamma_generators(clifford)
    index = {g: i for i, g in enumerate(gens)}
    if left not in index or right not in index:
        raise ValueError("Both inputs must be generators from the supplied Clifford algebra.")
    return sp.Integer(2) * clifford.eta(index[left], index[right])


def _reduce_generator_product(factors: Sequence[sp.Expr], gens: tuple[sp.Symbol, ...], clifford: CliffordAlgebraDef) -> sp.Expr:
    index = {g: i for i, g in enumerate(gens)}
    sign = sp.Integer(1)
    scalar = sp.Integer(1)
    current = list(factors)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(current) - 1:
            a = current[i]
            b = current[i + 1]
            if a in index and b in index:
                ia = index[a]
                ib = index[b]
                if ia == ib:
                    scalar *= clifford.eta(ia, ib)
                    del current[i:i + 2]
                    changed = True
                    continue
                if ia > ib:
                    current[i], current[i + 1] = current[i + 1], current[i]
                    sign *= -1
                    changed = True
            i += 1
    if scalar == 0:
        return sp.Integer(0)
    return sp.expand(sign * scalar * sp.Mul(*current))


def clifford_reduce(expr: Any, clifford: CliffordAlgebraDef) -> sp.Expr:
    expr = sp.expand(sp.sympify(expr))
    gens = gamma_generators(clifford)

    def reduce_mul(term: sp.Expr) -> sp.Expr:
        coeff, tail = term.as_coeff_Mul()
        raw_factors = list(sp.Mul.make_args(tail))
        factors: list[sp.Expr] = []
        for fac in raw_factors:
            if isinstance(fac, sp.Pow) and fac.base in gens and fac.exp.is_Integer and fac.exp >= 0:
                factors.extend([fac.base] * int(fac.exp))
            else:
                factors.append(fac)
        non_gamma = [f for f in factors if f not in gens]
        gamma = [f for f in factors if f in gens]
        ordered = _reduce_generator_product(gamma, gens, clifford)
        return sp.expand(coeff * sp.Mul(*non_gamma) * ordered)

    pieces = [reduce_mul(term) for term in sp.Add.make_args(expr)]
    return sp.expand(sum(pieces, sp.Integer(0)))


def as_differential_form(obj: TensorObject | TensorField | DifferentialForm) -> DifferentialForm:
    if isinstance(obj, DifferentialForm):
        return obj
    if isinstance(obj, TensorObject):
        return DifferentialForm(obj)
    if isinstance(obj, TensorField):
        return DifferentialForm.from_tensor_field(obj)
    raise TypeError("Unsupported input for as_differential_form")


def de_rham_complex_report(obj: TensorObject | TensorField | DifferentialForm) -> dict[str, Any]:
    form = as_differential_form(obj)
    first = form.d()
    second = first.d()
    return {
        "degree": form.degree,
        "d_degree": first.degree,
        "d_squared_degree": second.degree,
        "d_squared_tensor": second.tensor,
        "provenance": {
            "input": str(form),
            "check": "constructed d(d(omega)) using existing DifferentialForm API",
        },
    }


# JSON import/export helpers

def _jsonify_expr(expr: Any) -> Any:
    if expr is None:
        return None
    if isinstance(expr, (int, float, str, bool)):
        return expr
    return sp.srepr(sp.sympify(expr))


def _dejsonify_expr(expr: Any) -> Any:
    if expr is None or isinstance(expr, (int, float, bool)):
        return expr
    if isinstance(expr, str):
        try:
            return sp.sympify(expr)
        except Exception:
            return expr
    return expr


def _serialize_chart(chart: CoordinateChart) -> dict[str, Any]:
    coords = chart.symbols()
    metric = chart.metric(coords)
    return {
        "type": "CoordinateChart",
        "metric_name": chart.metric_name,
        "chart_name": chart.chart_name,
        "dimension": chart.dimension,
        "coordinate_names": list(chart.coordinate_names),
        "metric_entries": None if metric is None else [[_jsonify_expr(metric[i, j]) for j in range(metric.cols)] for i in range(metric.rows)],
        "metadata": dict(chart.metadata),
    }


def _deserialize_chart(payload: Mapping[str, Any]) -> CoordinateChart:
    metric_entries = payload.get("metric_entries")
    metric_func = None
    if metric_entries is not None:
        matrix_entries = [[_dejsonify_expr(entry) for entry in row] for row in metric_entries]
        frozen_matrix = sp.Matrix(matrix_entries)
        metric_func = lambda coords, frozen_matrix=frozen_matrix: frozen_matrix
    return CoordinateChart(
        metric_name=payload["metric_name"],
        chart_name=payload["chart_name"],
        dimension=int(payload["dimension"]),
        coordinate_names=tuple(payload["coordinate_names"]),
        metric_func=metric_func,
        metadata=dict(payload.get("metadata", {})),
        parameter_func=None,
    )


def geometry_to_data(obj: Any) -> dict[str, Any]:
    if isinstance(obj, CoordinateChart):
        return _serialize_chart(obj)
    if isinstance(obj, ManifoldDef):
        return {"type": "ManifoldDef", "name": obj.name, "dimension": obj.dimension, "charts": [geometry_to_data(chart) for chart in obj.charts], "metadata": dict(obj.metadata)}
    if isinstance(obj, BundleDef):
        return {
            "type": "BundleDef",
            "name": obj.name,
            "manifold": geometry_to_data(obj.manifold),
            "rank": obj.rank,
            "kind": obj.kind,
            "dual_of": obj.dual_of,
            "metric_name": obj.metric_name,
            "metadata": dict(obj.metadata),
        }
    if isinstance(obj, ChartDef):
        return {"type": "ChartDef", "name": obj.name, "manifold": geometry_to_data(obj.manifold), "chart": geometry_to_data(obj.chart), "metadata": dict(obj.metadata)}
    if isinstance(obj, MetricDef):
        return {
            "type": "MetricDef",
            "name": obj.name,
            "bundle": geometry_to_data(obj.bundle),
            "chart": None if obj.chart is None else geometry_to_data(obj.chart),
            "signature": obj.signature,
            "metadata": dict(obj.metadata),
        }
    if isinstance(obj, ConnectionDef):
        return {
            "type": "ConnectionDef",
            "name": obj.name,
            "bundle": geometry_to_data(obj.bundle),
            "chart": None if obj.chart is None else geometry_to_data(obj.chart),
            "torsion_free": obj.torsion_free,
            "metric_compatible": obj.metric_compatible,
            "metric": None if obj.metric is None else geometry_to_data(obj.metric),
            "metadata": dict(obj.metadata),
        }
    if isinstance(obj, FrameDef):
        return {
            "type": "FrameDef",
            "name": obj.name,
            "chart": geometry_to_data(obj.chart),
            "basis": geometry_to_data(obj.basis),
            "orthonormal": obj.orthonormal,
            "metadata": dict(obj.metadata),
        }
    if isinstance(obj, TensorBasis):
        from .exterior_geometry import serialize_basis
        payload = serialize_basis(obj)
        payload["chart"] = None if obj.chart is None else geometry_to_data(obj.chart)
        return payload
    if isinstance(obj, TensorFrame):
        from .exterior_geometry import serialize_frame
        payload = serialize_frame(obj)
        payload["chart"] = geometry_to_data(obj.chart)
        return payload
    if isinstance(obj, DifferentialOperatorDef):
        return {
            "type": "DifferentialOperatorDef",
            "name": obj.name,
            "kind": obj.kind,
            "connection": None if obj.connection is None else geometry_to_data(obj.connection),
            "metadata": dict(obj.metadata),
        }
    if isinstance(obj, SpinStructureDef):
        return {
            "type": "SpinStructureDef",
            "name": obj.name,
            "manifold": geometry_to_data(obj.manifold),
            "signature": obj.signature,
            "oriented": obj.oriented,
            "time_oriented": obj.time_oriented,
            "metadata": dict(obj.metadata),
        }
    if isinstance(obj, SpinorBundleDef):
        return {
            "type": "SpinorBundleDef",
            "name": obj.name,
            "manifold": geometry_to_data(obj.manifold),
            "spin_structure": geometry_to_data(obj.spin_structure),
            "complex_dimension": obj.complex_dimension,
            "chirality": obj.chirality,
            "metadata": dict(obj.metadata),
        }
    if isinstance(obj, CliffordAlgebraDef):
        return {
            "type": "CliffordAlgebraDef",
            "name": obj.name,
            "dimension": obj.dimension,
            "signature": obj.signature,
            "generator_prefix": obj.generator_prefix,
            "basis_labels": list(obj.basis_labels),
            "metadata": dict(obj.metadata),
        }
    raise TypeError(f"Unsupported geometry object type: {type(obj)!r}")


def geometry_from_data(payload: Mapping[str, Any]) -> Any:
    typ = payload["type"]
    if typ == "CoordinateChart":
        return _deserialize_chart(payload)
    if typ == "ManifoldDef":
        base = manifold(payload["name"], int(payload["dimension"]), metadata=payload.get("metadata", {}))
        out = base
        for chart_payload in payload.get("charts", []):
            out = out.with_chart(geometry_from_data(chart_payload))
        return out
    if typ == "BundleDef":
        return BundleDef(
            name=payload["name"],
            manifold=geometry_from_data(payload["manifold"]),
            rank=int(payload["rank"]),
            kind=payload.get("kind", "vector"),
            dual_of=payload.get("dual_of"),
            metric_name=payload.get("metric_name"),
            metadata=dict(payload.get("metadata", {})),
        )
    if typ == "ChartDef":
        return ChartDef(
            name=payload["name"],
            manifold=geometry_from_data(payload["manifold"]),
            chart=geometry_from_data(payload["chart"]),
            metadata=dict(payload.get("metadata", {})),
        )
    if typ == "MetricDef":
        return MetricDef(
            name=payload["name"],
            bundle=geometry_from_data(payload["bundle"]),
            chart=None if payload.get("chart") is None else geometry_from_data(payload["chart"]),
            signature=None if payload.get("signature") is None else tuple(int(x) for x in payload["signature"]),
            metadata=dict(payload.get("metadata", {})),
        )
    if typ == "ConnectionDef":
        return ConnectionDef(
            name=payload["name"],
            bundle=geometry_from_data(payload["bundle"]),
            chart=None if payload.get("chart") is None else geometry_from_data(payload["chart"]),
            torsion_free=bool(payload.get("torsion_free", False)),
            metric_compatible=bool(payload.get("metric_compatible", False)),
            metric=None if payload.get("metric") is None else geometry_from_data(payload["metric"]),
            metadata=dict(payload.get("metadata", {})),
        )
    if typ == "FrameDef":
        chart_obj = geometry_from_data(payload["chart"])
        basis_payload = payload.get("basis")
        if basis_payload is None:
            raise ValueError("FrameDef payload is missing serialized basis data.")
        basis_obj = geometry_from_data(basis_payload)
        return FrameDef(
            name=payload["name"],
            chart=chart_obj,
            basis=basis_obj,
            orthonormal=bool(payload.get("orthonormal", False)),
            metadata=dict(payload.get("metadata", {})),
        )
    if typ == "TensorBasis":
        from .exterior_geometry import deserialize_basis
        chart_obj = None if payload.get("chart") is None else geometry_from_data(payload["chart"])
        return deserialize_basis(payload, chart_obj)
    if typ == "TensorFrame":
        from .exterior_geometry import deserialize_frame
        chart_obj = geometry_from_data(payload["chart"])
        return deserialize_frame(payload, chart_obj)
    if typ == "DifferentialOperatorDef":
        return DifferentialOperatorDef(
            name=payload["name"],
            kind=payload["kind"],
            connection=None if payload.get("connection") is None else geometry_from_data(payload["connection"]),
            metadata=dict(payload.get("metadata", {})),
        )
    if typ == "SpinStructureDef":
        return SpinStructureDef(
            name=payload["name"],
            manifold=geometry_from_data(payload["manifold"]),
            signature=tuple(int(x) for x in payload["signature"]),
            oriented=bool(payload.get("oriented", True)),
            time_oriented=bool(payload.get("time_oriented", False)),
            metadata=dict(payload.get("metadata", {})),
        )
    if typ == "SpinorBundleDef":
        return SpinorBundleDef(
            name=payload["name"],
            manifold=geometry_from_data(payload["manifold"]),
            spin_structure=geometry_from_data(payload["spin_structure"]),
            complex_dimension=int(payload["complex_dimension"]),
            chirality=payload.get("chirality"),
            metadata=dict(payload.get("metadata", {})),
        )
    if typ == "CliffordAlgebraDef":
        return CliffordAlgebraDef(
            name=payload["name"],
            dimension=int(payload["dimension"]),
            signature=tuple(int(x) for x in payload["signature"]),
            generator_prefix=payload.get("generator_prefix", "gamma"),
            basis_labels=tuple(payload.get("basis_labels", [])),
            metadata=dict(payload.get("metadata", {})),
        )
    raise ValueError(f"Unsupported geometry data type {typ!r}")


def export_geometry_archive(objects: Iterable[Any], destination: str | Path, *, metadata: Mapping[str, Any] | None = None) -> Path:
    archive = {
        "version": "geometry-archive-v1",
        "metadata": dict(metadata or {}),
        "objects": [geometry_to_data(obj) for obj in objects],
    }
    destination = Path(destination)
    destination.write_text(json.dumps(archive, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def import_geometry_archive(source: str | Path) -> GeometryArchive:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    objects = tuple(geometry_from_data(item) for item in payload.get("objects", []))
    return GeometryArchive(version=payload.get("version", "unknown"), objects=objects, metadata=dict(payload.get("metadata", {})))


__all__ = [
    "SpinStructureDef",
    "SpinorBundleDef",
    "CliffordAlgebraDef",
    "GeometryArchive",
    "spin_structure",
    "spinor_bundle",
    "clifford_algebra",
    "gamma_generators",
    "gamma_anticommutator",
    "clifford_reduce",
    "as_differential_form",
    "de_rham_complex_report",
    "geometry_to_data",
    "geometry_from_data",
    "export_geometry_archive",
    "import_geometry_archive",
]
