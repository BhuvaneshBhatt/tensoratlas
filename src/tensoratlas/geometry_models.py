
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .geometry_components import component_geometry_report
from .semantic_ir import (
    TensorExpr,
    compile_tensor_expr,
    normalize_tensor_expr,
    materialize_tensor_expr,
)


@dataclass(frozen=True)
class SymbolicConnection:
    chart_name: str
    dimension: int
    kind: str = "levi_civita"
    torsion_free: bool = True
    metric_compatible: bool = True
    coefficients: Mapping[tuple[int, int, int], Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolicCurvatureHierarchy:
    riemann: Mapping[tuple[int, int, int, int], Any] = field(default_factory=dict)
    ricci: Mapping[tuple[int, int], Any] = field(default_factory=dict)
    scalar_curvature: Any = None
    einstein: Mapping[tuple[int, int], Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryModel:
    connection: SymbolicConnection
    curvature: SymbolicCurvatureHierarchy
    torsion: Mapping[tuple[int, int, int], Any] = field(default_factory=dict)
    nonmetricity: Mapping[tuple[int, int, int], Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryMaturationReport:
    original: Any
    geometry_model: GeometryModel
    ir: TensorExpr
    normalized_ir: TensorExpr
    materialized: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _symbolic_zero_tensor(rank: int, dim: int):
    if rank == 0:
        return sp.Integer(0)
    out = {}
    if rank == 1:
        for i in range(dim):
            out[(i,)] = sp.Integer(0)
    elif rank == 2:
        for i in range(dim):
            for j in range(dim):
                out[(i, j)] = sp.Integer(0)
    elif rank == 3:
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    out[(i, j, k)] = sp.Integer(0)
    elif rank == 4:
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        out[(i, j, k, l)] = sp.Integer(0)
    return out


def _build_symbolic_connection(chart, *, kind: str = "levi_civita", torsion_free: bool = True, metric_compatible: bool = True) -> SymbolicConnection:
    dim = chart.dimension
    coeffs = _symbolic_zero_tensor(3, dim)
    return SymbolicConnection(
        chart_name=getattr(chart, "chart_name", ""),
        dimension=dim,
        kind=kind,
        torsion_free=torsion_free,
        metric_compatible=metric_compatible,
        coefficients=coeffs,
        metadata={},
    )


def _build_symbolic_curvature(chart) -> SymbolicCurvatureHierarchy:
    dim = chart.dimension
    rep = component_geometry_report(chart, include_curvature=True)
    scalar = getattr(rep, "scalar_curvature", None)
    riemann = _symbolic_zero_tensor(4, dim)
    ricci = _symbolic_zero_tensor(2, dim)
    einstein = _symbolic_zero_tensor(2, dim)
    return SymbolicCurvatureHierarchy(
        riemann=riemann,
        ricci=ricci,
        scalar_curvature=scalar,
        einstein=einstein,
        metadata={},
    )


def build_geometry_model(
    chart,
    *,
    kind: str = "levi_civita",
    torsion_free: bool = True,
    metric_compatible: bool = True,
    include_torsion: bool = False,
    include_nonmetricity: bool = False,
) -> GeometryModel:
    conn = _build_symbolic_connection(chart, kind=kind, torsion_free=torsion_free, metric_compatible=metric_compatible)
    curv = _build_symbolic_curvature(chart)
    dim = chart.dimension
    torsion = _symbolic_zero_tensor(3, dim) if include_torsion else {}
    nonmetricity = _symbolic_zero_tensor(3, dim) if include_nonmetricity else {}
    return GeometryModel(
        connection=conn,
        curvature=curv,
        torsion=torsion,
        nonmetricity=nonmetricity,
        metadata={
            "torsion_enabled": include_torsion,
            "nonmetricity_enabled": include_nonmetricity,
        },
    )


def transport_geometry_model(geometry: GeometryModel, *, target_chart_name: str) -> GeometryModel:
    conn = SymbolicConnection(
        chart_name=target_chart_name,
        dimension=geometry.connection.dimension,
        kind=geometry.connection.kind,
        torsion_free=geometry.connection.torsion_free,
        metric_compatible=geometry.connection.metric_compatible,
        coefficients=dict(geometry.connection.coefficients),
        metadata=dict(geometry.connection.metadata),
    )
    curv = SymbolicCurvatureHierarchy(
        riemann=dict(geometry.curvature.riemann),
        ricci=dict(geometry.curvature.ricci),
        scalar_curvature=geometry.curvature.scalar_curvature,
        einstein=dict(geometry.curvature.einstein),
        metadata=dict(geometry.curvature.metadata),
    )
    return GeometryModel(
        connection=conn,
        curvature=curv,
        torsion=dict(geometry.torsion),
        nonmetricity=dict(geometry.nonmetricity),
        metadata=dict(geometry.metadata),
    )


def compile_geometry_ir(geometry: GeometryModel) -> TensorExpr:
    conn_payload = tuple(sorted(((k, sp.sympify(v)) for k, v in geometry.connection.coefficients.items()), key=repr))
    curv_payload = {
        "riemann": tuple(sorted(((k, sp.sympify(v)) for k, v in geometry.curvature.riemann.items()), key=repr)),
        "ricci": tuple(sorted(((k, sp.sympify(v)) for k, v in geometry.curvature.ricci.items()), key=repr)),
        "scalar_curvature": geometry.curvature.scalar_curvature,
        "einstein": tuple(sorted(((k, sp.sympify(v)) for k, v in geometry.curvature.einstein.items()), key=repr)),
    }
    return TensorExpr(
        kind="geometry_model",
        children=(
            TensorExpr(
                kind="symbolic_connection",
                payload=conn_payload,
                metadata={
                    "chart_name": geometry.connection.chart_name,
                    "dimension": geometry.connection.dimension,
                    "connection_kind": geometry.connection.kind,
                    "torsion_free": geometry.connection.torsion_free,
                    "metric_compatible": geometry.connection.metric_compatible,
                },
            ),
            TensorExpr(
                kind="symbolic_curvature",
                payload=curv_payload,
                metadata={},
            ),
        ),
        metadata=dict(geometry.metadata),
    )


def execute_geometry_maturation(
    obj: Any,
    *,
    kind: str = "levi_civita",
    torsion_free: bool = True,
    metric_compatible: bool = True,
    include_torsion: bool = False,
    include_nonmetricity: bool = False,
) -> GeometryMaturationReport:
    chart = getattr(obj, "chart", None)
    if chart is None:
        raise ValueError("Phase E currently expects an object with a chart attribute, such as a component tensor field.")
    geometry = build_geometry_model(
        chart,
        kind=kind,
        torsion_free=torsion_free,
        metric_compatible=metric_compatible,
        include_torsion=include_torsion,
        include_nonmetricity=include_nonmetricity,
    )
    obj_ir = compile_tensor_expr(obj)
    geom_ir = compile_geometry_ir(geometry)
    combined = TensorExpr(
        kind="geometry_maturation_bundle",
        children=(obj_ir, geom_ir),
        metadata={},
    )
    normalized = normalize_tensor_expr(combined)
    materialized = {
        "original": materialize_tensor_expr(obj_ir),
        "geometry_ir_kind": geom_ir.kind,
        "geometry_connection_kind": geometry.connection.kind,
    }
    return GeometryMaturationReport(
        original=obj,
        geometry_model=geometry,
        ir=combined,
        normalized_ir=normalized,
        materialized=materialized,
        metadata={
            "torsion_enabled": include_torsion,
            "nonmetricity_enabled": include_nonmetricity,
            "scalar_curvature": geometry.curvature.scalar_curvature,
        },
    )
