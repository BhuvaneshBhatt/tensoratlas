import sympy as sp

from tensoratlas import (
    ScalarField,
    TensorField,
    TensorObject,
    coordinate_chart,
    diagonal_tensor,
    tensor_product,
    tensor_reduce,
    tensor_sort,
    tensor_transpose,
    tensor_conjugate_transpose,
    tensor_symmetry_report,
    symmetric_tensor_q,
    antisymmetric_tensor_q,
    hermitian_tensor_q,
    antihermitian_tensor_q,
    last_tensor_reduction_report,
    tensor_from_components,
    indices,
)
from tensoratlas.tensor_core import TensorExpr
from tensoratlas.normal_forms import tnf_build_array


def test_symmetry_predicates_and_report_for_basic_rank2_tensors():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    sym = diagonal_tensor(chart, [1, 2], variance_spec='ll')
    asym_arr = tnf_build_array((2, 2), lambda idx: {(0, 1): 1, (1, 0): -1}.get(idx, 0))
    asym = tensor_from_components(chart, asym_arr, 'll')
    h_arr = tnf_build_array((2, 2), lambda idx: {(0, 0): 1, (1, 1): 2, (0, 1): sp.I, (1, 0): -sp.I}[idx])
    h = tensor_from_components(chart, h_arr, 'll')
    ah_arr = tnf_build_array((2, 2), lambda idx: {(0, 0): sp.I, (1, 1): 2 * sp.I, (0, 1): 1, (1, 0): -1}[idx])
    ah = tensor_from_components(chart, ah_arr, 'll')

    assert symmetric_tensor_q(sym) is True
    assert antisymmetric_tensor_q(sym) is False
    assert antisymmetric_tensor_q(asym) is True
    assert hermitian_tensor_q(h) is True
    assert antihermitian_tensor_q(ah) is True
    rep = tensor_symmetry_report(sym)
    assert rep['is_symmetric'] is True
    assert rep['rank'] == 2
    assert rep['dimensions'] == (2, 2)


def test_tensor_sort_canonicalizes_tensor_expr_product_order():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    a = TensorObject.from_tensor_field(diagonal_tensor(chart, [1, 2], variance_spec='ll'), name='A')
    b = TensorObject.from_tensor_field(diagonal_tensor(chart, [3, 4], variance_spec='ll'), name='B')
    expr = TensorExpr('tensor_product', (b, a))
    out = tensor_sort(expr)
    assert isinstance(out, TensorExpr)
    assert out.args[0].name == 'A'
    assert out.args[1].name == 'B'


def test_tensor_reduce_separates_scalar_factors_in_tensor_expr_products():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    a = TensorObject.from_tensor_field(diagonal_tensor(chart, [1, 2], variance_spec='ll'), name='A')
    scalar_obj = TensorObject.from_scalar_field(ScalarField(chart, 3), name='s')
    expr = TensorExpr('tensor_product', (scalar_obj, a))
    out = tensor_reduce(expr)
    report = last_tensor_reduction_report()
    assert isinstance(out, TensorField)
    assert out.components[(0, 0)] == 3
    assert out.components[(1, 1)] == 6
    assert report.scalar_factor == 3
    assert 'separated scalar tensor-product factors' in report.notes


def test_transpose_and_conjugate_transpose_use_basic_symmetry_shortcuts():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    sym = diagonal_tensor(chart, [1, 2], variance_spec='ll')
    asym_arr = tnf_build_array((2, 2), lambda idx: {(0, 1): 1, (1, 0): -1}.get(idx, 0))
    asym = TensorField(chart, asym_arr, 'll')
    h_arr = tnf_build_array((2, 2), lambda idx: {(0, 0): 1, (1, 1): 2, (0, 1): sp.I, (1, 0): -sp.I}[idx])
    h = tensor_from_components(chart, h_arr, 'll')

    assert tensor_transpose(sym, (1, 0)).components[(1, 0)] == sym.components[(1, 0)]
    t_asym = tensor_transpose(asym, (1, 0))
    assert t_asym.components[(0, 1)] == -asym.components[(0, 1)]
    h_ct = tensor_conjugate_transpose(h)
    assert h_ct.components[(0, 1)] == h.components[(0, 1)]


def test_tensor_reduce_splits_disconnected_indexed_components_and_records_report():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    delta = TensorObject.from_tensor_field(diagonal_tensor(chart, [1, 1], variance_spec='ul'), name='D')
    i_u, i_l, j_u, j_l = indices('i^ i_ j^ j_')
    expr = delta.with_indices(i_u, i_l) * delta.with_indices(j_u, j_l)
    out = tensor_reduce(expr)
    report = last_tensor_reduction_report()
    assert report.disconnected_components == ((0,), (1,))
    assert 'separated disconnected contraction components' in report.notes
    assert out is not None
