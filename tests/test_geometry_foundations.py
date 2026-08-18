from __future__ import annotations

import sympy as sp

from tensoratlas import (
    coordinate_chart,
    ScalarField,
    VectorField,
    manifold,
    chart_definition,
    tangent_bundle,
    cotangent_bundle,
    riemannian_metric_from_chart,
    levi_civita_connection,
    frame_definition,
    geometry_covariant_derivative_operator,
    geometry_exterior_derivative_operator,
    geometry_lie_derivative_operator,
    geometry_summary,
)
from tensoratlas.basis import tangent_basis, orthonormal_tangent_basis
from tensoratlas.normal_forms import tnf_column_from_entries


def test_geometry_typed_geometry_descriptors_basic():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    M = manifold("R2", 2)
    cdef = chart_definition(M, polar)
    tb = tangent_bundle(cdef.manifold)
    cb = cotangent_bundle(cdef.manifold)
    metric = riemannian_metric_from_chart(polar, manifold_name=M.name)
    conn = levi_civita_connection(metric)
    frame = frame_definition(cdef, tangent_basis(polar))

    assert cdef.manifold.dimension == 2
    assert tb.rank == 2
    assert cb.dual_of == "T(R2)"
    assert metric.signature == (2, 0, 0)
    assert conn.torsion_free is True
    assert conn.metric_compatible is True
    assert frame.basis.chart == polar


def test_geometry_geometry_summary_and_metric_curvature():
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    summary = geometry_summary(spherical)
    metric = riemannian_metric_from_chart(spherical, manifold_name="R3")

    assert summary["dimension"] == 3
    assert summary["is_orthogonal"] is True
    assert summary["torsion_free"] is True
    assert sp.simplify(metric.scalar_curvature()) == 0


def test_geometry_differential_operators_dispatch():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    scalar = ScalarField(polar, r**2 * sp.sin(theta))
    vector = VectorField(polar, tnf_column_from_entries([r, 0]), "contravariant")

    nabla = geometry_covariant_derivative_operator()
    ext = geometry_exterior_derivative_operator()
    lie = geometry_lie_derivative_operator()

    grad_like = nabla.apply(scalar)
    two_form_seed = ext.apply(scalar)
    dragged = lie.apply(vector, vector=vector)

    assert grad_like.chart == polar
    assert two_form_seed.chart == polar
    assert dragged.chart == polar


def test_geometry_frame_definition_with_orthonormal_basis():
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    M = manifold("R3", 3)
    cdef = chart_definition(M, spherical)
    frame = frame_definition(cdef, orthonormal_tangent_basis(spherical), orthonormal=True)
    assert frame.orthonormal is True
    assert frame.basis.kind == "orthonormal_tangent"


def test_geometry_connection_validation():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    metric = riemannian_metric_from_chart(polar, manifold_name="R2")
    conn = levi_civita_connection(metric)
    report = conn.validate()
    assert report["has_coefficients"] is True
    assert report["torsion_free_declared"] is True
    assert report["metric_compatible_declared"] is True
