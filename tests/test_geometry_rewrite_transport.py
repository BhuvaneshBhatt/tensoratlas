
from __future__ import annotations

from tensoratlas.charts import get_chart
from tensoratlas.geometry_models import build_geometry_model
from tensoratlas.geometry_rewrite_transport import (
    GEOMETRY_REWRITE_RULES,
    rewrite_geometry_model,
    build_active_curvature_relations,
    transport_derived_geometry_explicit,
    build_and_rewrite_geometry_model,
)

def test_geometry_rewrite_rules_exist():
    assert len(GEOMETRY_REWRITE_RULES) >= 3
    assert GEOMETRY_REWRITE_RULES[0].family == "geometry_connection"

def test_build_active_curvature_relations():
    chart = get_chart("Euclidean", "Cartesian", 3)
    rel = build_active_curvature_relations(chart)
    assert "riemann_to_ricci" in rel.relations
    assert "einstein_from_ricci_scalar" in rel.relations

def test_rewrite_geometry_model_runs():
    chart = get_chart("Euclidean", "Cartesian", 2)
    geom = build_geometry_model(chart)
    rep = rewrite_geometry_model([(1, geom), (1, geom)])
    assert "rewrite_symbolic_connection_family" in rep.applied_rules or len(rep.reduced_terms) >= 1

def test_transport_derived_geometry_explicit():
    chart = get_chart("Euclidean", "Cartesian", 2)
    geom = build_geometry_model(chart, include_torsion=True)
    rep = transport_derived_geometry_explicit(geom, target_chart_name="Polar")
    assert rep.target_chart == "Polar"
    assert rep.transported_relations.einstein_name == "Einstein"

def test_build_and_rewrite_geometry_model():
    chart = get_chart("Euclidean", "Cartesian", 2)
    rep = build_and_rewrite_geometry_model(chart, include_nonmetricity=True)
    assert rep.original.connection.chart_name == "Cartesian"
