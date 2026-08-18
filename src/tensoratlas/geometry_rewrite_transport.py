
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import sympy as sp

from .conflict_priority_geometry_engine import PriorityRewriteRule, conflict_aware_priority_reduce
from .geometry_models import (
    SymbolicConnection,
    SymbolicCurvatureHierarchy,
    GeometryModel,
    build_geometry_model,
    transport_geometry_model,
)


@dataclass(frozen=True)
class GeometryRewriteReport:
    original: GeometryModel
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurvatureRelationReport:
    riemann_name: str
    ricci_name: str
    scalar_name: str
    weyl_name: str
    einstein_name: str
    relations: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DerivedGeometryTransportReport:
    source_chart: str
    target_chart: str
    transported_geometry: GeometryModel
    transported_relations: CurvatureRelationReport
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _normalize_terms(weighted_terms):
    groups = {}
    for c, t in weighted_terms:
        key = repr(t)
        groups.setdefault(key, []).append((sp.sympify(c), t))
    out = []
    for items in groups.values():
        coeff = sp.simplify(sum(c for c, _ in items))
        if coeff != 0:
            out.append((coeff, items[0][1]))
    out.sort(key=lambda x: (repr(x[1]), sp.srepr(sp.sympify(x[0]))))
    return out


def _connection_key(obj: Any):
    if not isinstance(obj, GeometryModel):
        return None
    conn = obj.connection
    return (
        conn.kind,
        conn.chart_name,
        conn.dimension,
        bool(conn.torsion_free),
        bool(conn.metric_compatible),
    )


def _curvature_key(curv: SymbolicCurvatureHierarchy):
    return (
        "curvature",
        curv.scalar_curvature,
        tuple(sorted(curv.riemann.keys())),
        tuple(sorted(curv.ricci.keys())),
        tuple(sorted(curv.einstein.keys())),
    )


def _geometry_model_key(obj: Any):
    if not isinstance(obj, GeometryModel):
        return None
    geom = obj
    return (_connection_key(geom), _curvature_key(geom.curvature), bool(geom.torsion), bool(geom.nonmetricity))


def _geometry_orbit_reduce(weighted_terms, key_fn, threshold):
    buckets = {}
    rest = []
    for c, t in weighted_terms:
        key = key_fn(t)
        if key is None:
            rest.append((c, t))
        else:
            buckets.setdefault(key, []).append((c, t))
    reduced = list(rest)
    changed = False
    for _, items in buckets.items():
        coeff = sp.simplify(sum(c for c, _ in items))
        if len(items) >= threshold:
            changed = True
            if coeff != 0:
                reduced.append((coeff, items[0][1]))
        else:
            reduced.extend(items)
    return _normalize_terms(reduced), changed


def _apply_connection_family(terms):
    return _geometry_orbit_reduce(terms, _connection_key, 2)


def _apply_curvature_family(terms):
    def key(obj):
        if not isinstance(obj, GeometryModel):
            return None
        return _curvature_key(obj.curvature)
    return _geometry_orbit_reduce(terms, key, 2)


def _apply_geometry_model_family(terms):
    return _geometry_orbit_reduce(terms, _geometry_model_key, 2)


GEOMETRY_REWRITE_RULES: tuple[PriorityRewriteRule, ...] = (
    PriorityRewriteRule("rewrite_symbolic_connection_family", "geometry_connection", 100, ("geometry", "connection", 1), _apply_connection_family, {"family_set": "geometry"}),
    PriorityRewriteRule("rewrite_symbolic_curvature_family", "geometry_curvature", 90, ("geometry", "curvature", 2), _apply_curvature_family, {"family_set": "geometry"}),
    PriorityRewriteRule("rewrite_geometry_model_family", "geometry_model", 80, ("geometry", "model", 3), _apply_geometry_model_family, {"family_set": "geometry"}),
)


def rewrite_geometry_model(expr_or_terms: Any) -> GeometryRewriteReport:
    report = conflict_aware_priority_reduce(expr_or_terms, rules=GEOMETRY_REWRITE_RULES)
    if isinstance(expr_or_terms, (list, tuple)) and expr_or_terms and isinstance(expr_or_terms[0], tuple):
        original = expr_or_terms[0][1]
    else:
        original = expr_or_terms
    return GeometryRewriteReport(
        original=original,
        reduced_terms=report.reduced_terms,
        applied_rules=report.applied_rules,
        metadata={"blocked_rules": report.blocked_rules, "iterations": report.iterations},
    )


def build_active_curvature_relations(chart, *, riemann_name: str = "Riemann", ricci_name: str = "Ricci", scalar_name: str = "ScalarCurvature", weyl_name: str = "Weyl", einstein_name: str = "Einstein") -> CurvatureRelationReport:
    dim = chart.dimension
    relations = {
        "riemann_to_ricci": f"{ricci_name}[ab] := contraction of {riemann_name}[acbd]",
        "ricci_to_scalar": f"{scalar_name} := contraction of {ricci_name}[ab] with metric^ab",
        "riemann_decomposition": f"{riemann_name} = {weyl_name} + metric/Ricci/scalar pieces in dimension {dim}",
        "einstein_from_ricci_scalar": f"{einstein_name}[ab] := {ricci_name}[ab] - 1/2 g[ab] {scalar_name}",
    }
    return CurvatureRelationReport(
        riemann_name=riemann_name,
        ricci_name=ricci_name,
        scalar_name=scalar_name,
        weyl_name=weyl_name,
        einstein_name=einstein_name,
        relations=relations,
    )


def transport_derived_geometry_explicit(geometry: GeometryModel, *, target_chart_name: str) -> DerivedGeometryTransportReport:
    moved = transport_geometry_model(geometry, target_chart_name=target_chart_name)
    rel = build_active_curvature_relations(
        type("ChartStub", (), {"dimension": moved.connection.dimension})(),
        riemann_name="Riemann",
        ricci_name="Ricci",
        scalar_name="ScalarCurvature",
        weyl_name="Weyl",
        einstein_name="Einstein",
    )
    return DerivedGeometryTransportReport(
        source_chart=geometry.connection.chart_name,
        target_chart=target_chart_name,
        transported_geometry=moved,
        transported_relations=rel,
        metadata={
            "torsion_enabled": moved.metadata.get("torsion_enabled", False),
            "nonmetricity_enabled": moved.metadata.get("nonmetricity_enabled", False),
            "scalar_curvature": moved.curvature.scalar_curvature,
        },
    )


def build_and_rewrite_geometry_model(
    chart,
    *,
    kind: str = "levi_civita",
    torsion_free: bool = True,
    metric_compatible: bool = True,
    include_torsion: bool = False,
    include_nonmetricity: bool = False,
) -> GeometryRewriteReport:
    geom = build_geometry_model(
        chart,
        kind=kind,
        torsion_free=torsion_free,
        metric_compatible=metric_compatible,
        include_torsion=include_torsion,
        include_nonmetricity=include_nonmetricity,
    )
    return rewrite_geometry_model([(sp.Integer(1), geom), (sp.Integer(0), geom)])
