from __future__ import annotations

import pytest
import sympy as sp

import tensoratlas
from tensoratlas.display import display_nonzero_components, nonzero_components
from tensoratlas.geometric_algebra import GeometricAlgebra, project_vector_onto_vector, rotate
from tensoratlas.relativity import (
    CurvatureComputer,
    sparse_nonzero_einstein,
    sparse_nonzero_ricci,
    sparse_nonzero_riemann,
    two_sphere_metric,
)


def test_root_all_is_curated_and_hides_internal_priority_names():
    assert len(tensoratlas.__all__) < 150
    assert not any("priority" in name.lower() for name in tensoratlas.__all__)
    assert "GeometricAlgebra" in tensoratlas.__all__
    assert "MetricModel" in tensoratlas.__all__


def test_geometric_algebra_basis_names_norm_inverse_rotation_projection():
    ga = GeometricAlgebra.euclidean(2)
    e1, e2 = ga.basis_vectors()
    assert ga.blade("e1", "e2") == ga.blade(0, 1)
    assert ga.basis_product("e1", "e1").scalar_part() == 1
    assert e1.norm_squared(require_scalar=True) == 1
    assert rotate(e1, ga.scalar(2)) == e1
    with pytest.raises(ZeroDivisionError):
        project_vector_onto_vector(e1, ga.zero())


def test_display_nonzero_helpers_do_not_simplify_by_default():
    x = sp.symbols("x")
    expr = sp.Add(x, -x, evaluate=False)
    assert nonzero_components([[expr]]) == {(0, 0): expr}
    assert nonzero_components([[expr]], simplify=True) == {}
    assert display_nonzero_components([[expr]], name="A") == {"A(0, 0)": expr}


def test_sparse_nonzero_curvature_helpers_match_dense_for_two_sphere():
    model = two_sphere_metric()
    curv = CurvatureComputer(model, simplify=True)
    sparse_ricci = sparse_nonzero_ricci(model, simplify=True)
    sparse_einstein = sparse_nonzero_einstein(model, simplify=True)
    sparse_riemann = sparse_nonzero_riemann(model, simplify=True)
    assert sparse_ricci[(0, 0)] == curv.ricci(0, 0)
    assert sparse_ricci[(1, 1)] == curv.ricci(1, 1)
    assert sparse_einstein == {}
    assert sparse_riemann
