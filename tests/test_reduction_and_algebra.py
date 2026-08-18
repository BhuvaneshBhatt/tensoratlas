import sympy as sp

from tensoratlas import (
    TensorObject,
    coordinate_chart,
    diagonal_tensor,
    indices,
    indexed,
    tensor_antisymmetrize,
    tensor_graph,
    tensor_has_symmetry,
    tensor_permute,
    tensor_project_symmetry,
    tensor_reduce,
    tensor_sort,
    tensor_symmetrize,
    tensor_symmetry_class,
    tensor_transpose,
    last_tensor_reduction_report,
)
from tensoratlas.tensor_core import TensorExpr
from tensoratlas.normal_forms import tnf_build_array
from tensoratlas import tensor_from_components


def test_tensor_permute_matches_transpose_for_rank2_tensorfield():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    arr = tnf_build_array((2, 2), lambda idx: sp.Integer(10 * idx[0] + idx[1]))
    tensor = tensor_from_components(chart, arr, 'll').to_tensor_field()
    a = tensor_transpose(tensor, (1, 0))
    b = tensor_permute(tensor, (1, 0))
    assert a.components[(0, 1)] == b.components[(0, 1)]
    assert a.components[(1, 0)] == b.components[(1, 0)]


def test_tensor_symmetrize_and_antisymmetrize_public_wrappers_work():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    arr = tnf_build_array((2, 2), lambda idx: { (0, 1): 2, (1, 0): 0 }.get(idx, 0))
    obj = TensorObject.from_tensor_field(tensor_from_components(chart, arr, 'll').to_tensor_field())
    sym = tensor_symmetrize(obj, (0, 1))
    asym = tensor_antisymmetrize(obj, (0, 1))
    assert sp.simplify(sym.components[(0, 1)] - sym.components[(1, 0)]) == 0
    assert sp.simplify(asym.components[(0, 1)] + asym.components[(1, 0)]) == 0


def test_tensor_symmetry_subsystem_helpers_report_and_project():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    sym = TensorObject.from_tensor_field(diagonal_tensor(chart, [1, 2], variance_spec='ll'))
    assert tensor_has_symmetry(sym, 'symmetric') is True
    assert tensor_symmetry_class(sym) == 'symmetric'
    arr = tnf_build_array((2, 2), lambda idx: { (0, 1): 1, (1, 0): -1 }.get(idx, 0))
    generic = TensorObject.from_tensor_field(tensor_from_components(chart, arr, 'll').to_tensor_field())
    projected = tensor_project_symmetry(generic, {'antisymmetric': ((0, 1),)})
    assert tensor_has_symmetry(projected, {'antisymmetric': ((0, 1),)}) is True


def test_tensor_graph_includes_plan_and_summary_for_indexed_products():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    A = diagonal_tensor(chart, [1, 2, 3], variance_spec='ul')
    B = diagonal_tensor(chart, [4, 5, 6], variance_spec='ul')
    i, j_down, j_up, k = indices('i^ j_ j^ k_')
    expr = TensorExpr('tensor_product', (indexed(A, i, j_down), indexed(B, j_up, k)))
    graph = tensor_graph(expr)
    assert 'summary' in graph
    assert graph['summary']['contraction_edges'] >= 1
    assert 'plan' in graph


def test_tensor_reduce_reports_stages_and_costs_on_indexed_input():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    delta = TensorObject.from_tensor_field(diagonal_tensor(chart, [1, 1], variance_spec='ul'), name='D')
    i_u, i_l, j_u, j_l = indices('i^ i_ j^ j_')
    expr = delta.with_indices(i_u, i_l) * delta.with_indices(j_u, j_l)
    out = tensor_reduce(expr)
    rep = last_tensor_reduction_report()
    assert out is not None
    assert 'split_components' in rep.stages
    assert 'tensorform' in rep.stages
    assert 'sort_products' in rep.stage_counts
    assert rep.stage_durations_ms['sort_products'] >= 0.0


def test_tensor_reduce_sorts_tensor_products_before_evaluation():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    a = TensorObject.from_tensor_field(diagonal_tensor(chart, [1, 2], variance_spec='ll'), name='A')
    b = TensorObject.from_tensor_field(diagonal_tensor(chart, [3, 4], variance_spec='ll'), name='B')
    expr = TensorExpr('tensor_product', (b, a))
    sorted_expr = tensor_sort(expr)
    reduced = tensor_reduce(sorted_expr, stages=('sort_products', 'simplify'))
    assert reduced.components.shape == (2, 2, 2, 2)
