from __future__ import annotations

import sympy as sp

from tensoratlas import (
    coordinate_chart,
    ScalarField,
    TensorField,
    TensorObject,
    tangent_basis,
    cotangent_basis,
    sparse_tensor,
    tensor_to_structured,
    tensor_from_structured,
    tensor_from_components,
    tensor_reduce,
    tensor_symmetry,
    array_transform,
    basis_pair_contraction_matrix,
    tensor_bases,
)
from tensoratlas.normal_forms import tnf_build_array


def test_structured_roundtrip_and_domain_metadata():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    arr = tnf_build_array((2, 2), lambda idx: sp.Integer(1) if idx == (0, 0) else sp.Integer(0))
    t = tensor_from_components(chart, arr, "ul", (tangent_basis(chart), cotangent_basis(chart)), domain_metadata={"manifold": "R2"})
    st = tensor_to_structured(t)
    assert st.domain_metadata["manifold"] == "R2"
    back = tensor_from_structured(chart, st, "ul", (tangent_basis(chart), cotangent_basis(chart)))
    assert back.domain_metadata["manifold"] == "R2"
    assert back.components[(0,0)] == 1


def test_tolerance_aware_symmetry_check():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    eps = sp.Float('1e-10')
    arr = tnf_build_array((2,2), lambda idx: { (0,0):1, (0,1):2+eps, (1,0):2, (1,1):3 }[idx])
    tf = TensorField(chart, arr, 'll')
    assert tensor_symmetry(tf, tolerance=1e-8)["symmetric"] == ((0,1),)


def test_array_transform_identity_on_vector():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    arr = tnf_build_array((2,), lambda idx: sp.Integer(idx[0]+1))
    tf = TensorField(chart, arr, 'u')
    ident = sp.eye(2)
    out = array_transform(tf, [ident], slots=[0])
    assert out.components[(0,)] == 1 and out.components[(1,)] == 2


def test_basis_pair_and_tensor_reduce():
    chart = coordinate_chart("Euclidean", "Polar", 2)
    mat = basis_pair_contraction_matrix(tangent_basis(chart), cotangent_basis(chart))
    assert mat.shape == (2,2)
    s = ScalarField(chart, chart.symbols()[0]**2)
    out = tensor_reduce(s.exterior_derivative())
    assert out.variance_spec == 'l'
