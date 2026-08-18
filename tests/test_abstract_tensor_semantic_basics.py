import sympy as sp

from sympy.tensor.tensor import tensor_indices

from tensoratlas import (
    TensorObject,
    TensorAtlasAbstractExpr,
    abstract_index_type,
    abstract_to_indexed,
    canonicalize_abstract_tensor_expr_with_report,
    coordinate_chart,
    cotangent_basis,
    indexed,
    indices,
    indexed_to_abstract,
    multi_term_tensor_reduce,
    riemann_tensor_head,
    weyl_tensor_head,
)


def test_tensoratlas_abstract_wrapper_and_report_capture_contractions():
    lor = abstract_index_type('Lorentz', dummy_name='L')
    a, b = tensor_indices('a,b', lor)
    R = riemann_tensor_head('R', lor)
    wrapped = canonicalize_abstract_tensor_expr_with_report(R(-a, b, a, -b))
    assert isinstance(wrapped, TensorAtlasAbstractExpr)
    assert wrapped.report is not None
    assert wrapped.report.tensor_heads == ('R',)
    assert wrapped.report.contraction_pairs_before
    assert 'R' in wrapped.report.slot_symmetries


def test_indexed_to_abstract_and_back_for_simple_leaf():
    cart = coordinate_chart('Euclidean', 'Cartesian', 2)
    arr = sp.MutableDenseNDimArray.zeros(2, 2)
    arr[0, 0] = 1
    arr[1, 1] = 2
    T = TensorObject(cart, arr, 'll', (cotangent_basis(cart), cotangent_basis(cart)), name='T')
    i, j = indices('i_ j_')
    leaf = indexed(T, i, j)
    abstract = indexed_to_abstract(leaf)
    roundtrip = abstract_to_indexed(abstract, tensor_registry={'T': T})
    assert str(roundtrip) == str(leaf)


def test_multi_term_bianchi_reduction_for_riemann_tensor():
    lor = abstract_index_type('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a,b,c,d', lor)
    R = riemann_tensor_head('R', lor)
    expr = R(a, b, c, d) + R(a, c, d, b) + R(a, d, b, c)
    reduced, used = multi_term_tensor_reduce(expr)
    assert reduced == 0
    assert 'first Bianchi cyclic reduction' in used


def test_dimension_dependent_weyl_vanishes_in_three_dimensions():
    lor3 = abstract_index_type('Lorentz3', dummy_name='L', dim=3)
    a, b, c, d = tensor_indices('a,b,c,d', lor3)
    C = weyl_tensor_head('C', lor3)
    expr = C(a, b, c, d)
    reduced, used = multi_term_tensor_reduce(expr, dimension=3)
    assert reduced == 0
    assert 'dimension-dependent Weyl vanishing' in used
