
import pytest
import sympy as sp

from tensoratlas import (
    coordinate_chart,
    TensorObject,
    TensorIndex,
    indexed,
    indices,
    indexed_equal,
    stronger_indexed_equal,
    normalize_indexed_expression,
    indexed_canonical_report,
    frame_basis,
    frame_commutator_coefficients,
    coframe_connection_one_forms,
    orthonormal_tangent_basis,
    orthonormal_cotangent_basis,
)
from tensoratlas.tensor_algebra import metric_tensor, permutation_tensor, kronecker_delta_tensor
from tensoratlas.tensor_indices import BundleCompatibilityError


def test_more_complete_abstract_index_canonicalization():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i,j = indices("i^ j_")
    a,b = indices("a^ b_")
    assert indexed_equal(indexed(d,i,j), indexed(d,a,b))
    rep = indexed_canonical_report(indexed(d,i,j))
    assert rep.tensor_kinds


def test_stronger_symmetry_and_irreducible_canonicalization():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0,1),)})
    i,j = indices("i_ j_")
    assert stronger_indexed_equal(indexed(g,i,j), indexed(g,j,i))


def test_deeper_bundle_multi_space_reasoning():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    d_std = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    d_orth = d_std.change_basis((orthonormal_tangent_basis(chart), orthonormal_cotangent_basis(chart)))
    iT = TensorIndex("i","u","T(Cartesian)")
    jT = TensorIndex("j","l","T(Cartesian)")
    iE = TensorIndex("i","u","e(Cartesian)")
    jE = TensorIndex("j","l","e(Cartesian)")
    good = indexed(d_std, iT, jT)
    bad = good + indexed(d_orth, iE, jE)
    with pytest.raises(BundleCompatibilityError):
        normalize_indexed_expression(bad)


def test_more_complete_metric_delta_levicivita_simplification():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    gU = TensorObject.from_tensor_field(metric_tensor(chart, "uu"), name="gU")
    gL = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="gL")
    epsL = TensorObject.from_tensor_field(permutation_tensor(chart, "lll"), name="eps")
    i,a,b,c = indices("i^ a^ b_ c_")
    expr = indexed(gU, i, a) * indexed(epsL, a.dual(), b, c)
    norm = normalize_indexed_expression(expr)
    assert norm is not None
    expr2 = indexed(gU, i, a) * indexed(gL, a.dual(), b)
    norm2 = normalize_indexed_expression(expr2)
    assert norm2 is not None


def test_frame_coframe_connection_formalism_helpers():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    fr = frame_basis("e", chart, lambda coords: sp.eye(chart.dimension))
    coeffs = frame_commutator_coefficients(chart, fr)
    oneforms = coframe_connection_one_forms(chart, fr)
    assert coeffs is not None
    assert oneforms is not None
