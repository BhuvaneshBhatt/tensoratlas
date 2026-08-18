
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.geometry_components import component_tensor_field
from tensoratlas.geometry_models import (
    build_geometry_model,
    transport_geometry_model,
    compile_geometry_ir,
    execute_geometry_maturation,
)

def test_build_geometry_model_levi_civita():
    chart = get_chart("Euclidean", "Cartesian", 2)
    geom = build_geometry_model(chart)
    assert geom.connection.kind == "levi_civita"
    assert geom.connection.torsion_free is True
    assert geom.connection.metric_compatible is True

def test_build_geometry_model_with_torsion_nonmetricity():
    chart = get_chart("Euclidean", "Cartesian", 2)
    geom = build_geometry_model(chart, kind="affine", torsion_free=False, metric_compatible=False, include_torsion=True, include_nonmetricity=True)
    assert geom.metadata["torsion_enabled"] is True
    assert geom.metadata["nonmetricity_enabled"] is True
    assert len(geom.torsion) > 0
    assert len(geom.nonmetricity) > 0

def test_transport_geometry_model():
    chart = get_chart("Euclidean", "Cartesian", 2)
    geom = build_geometry_model(chart)
    moved = transport_geometry_model(geom, target_chart_name="Polar")
    assert moved.connection.chart_name == "Polar"
    assert moved.curvature.scalar_curvature == geom.curvature.scalar_curvature

def test_compile_geometry_ir():
    chart = get_chart("Euclidean", "Cartesian", 2)
    geom = build_geometry_model(chart)
    ir = compile_geometry_ir(geom)
    assert ir.kind == "geometry_model"
    assert len(ir.children) == 2

def test_execute_geometry_maturation():
    chart = get_chart("Euclidean", "Cartesian", 2)
    field = component_tensor_field("V", chart, "u", [1, 2])
    rep = execute_geometry_maturation(field, include_torsion=True)
    assert rep.geometry_model.connection.kind == "levi_civita"
    assert rep.metadata["torsion_enabled"] is True
    assert rep.ir.kind == "geometry_maturation_bundle"
