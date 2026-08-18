from __future__ import annotations

import sympy as sp

from tensoratlas import (
    TensorObject,
    coordinate_chart,
    diagonal_tensor,
    indices,
    indexed,
    tensor_array,
    tensor_diagonal,
    tensor_element,
    tensor_flatten,
    tensor_graph,
    tensor_interop_report,
    tensor_map,
    tensor_reduce,
    tensor_reshape,
    tensor_roundtrip_structured,
    zero_tensor_like,
)
from tensoratlas.tensor_core import TensorExpr, tensor_from_components
from tensoratlas.tensor_indices import canonical_indexed_form, normalize_indexed_expression
from tensoratlas.normal_forms import tnf_build_array


def test_public_convenience_tensor_diagonal_flatten_reshape_and_map():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    arr = tnf_build_array((2, 2), lambda idx: sp.Integer(10 * idx[0] + idx[1]))
    obj = tensor_from_components(chart, arr, 'ul', name='T')

    diag = tensor_diagonal(obj, (0, 1))
    assert tensor_array(diag).shape == (2,)
    assert tensor_element(diag, (1,)) == 11

    flat = tensor_flatten(obj)
    assert flat.shape == (4, 1)

    reshaped = tensor_reshape(obj, (4,), variance_spec='u', slot_bases=obj.slot_bases[:1])
    assert reshaped.shape == (4,)

    mapped = tensor_map(obj, lambda e: e + 1)
    assert tensor_element(mapped, (1, 1)) == 12


def test_zero_tensor_like_and_strict_roundtrip_interop():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    obj = tensor_from_components(
        chart,
        [[1, 2], [2, 3]],
        'uu',
        symmetry_metadata={'symmetric': ((0, 1),)},
        domain_metadata={'domain': 'demo'},
    )
    zero = zero_tensor_like(obj)
    assert tensor_array(zero).shape == (2, 2)
    assert all(tensor_element(zero, idx) == 0 for idx in [(0, 0), (0, 1), (1, 0), (1, 1)])
    rebuilt = tensor_roundtrip_structured(obj, strict=True)
    rep = tensor_interop_report(rebuilt)
    assert rep['lossless_roundtrip'] is True
    assert rep['metadata_view_equal'] is True


def test_authoritative_indexed_public_reducers_agree():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    A = TensorObject.from_tensor_field(diagonal_tensor(chart, [1, 2], variance_spec='ul'), name='A')
    B = TensorObject.from_tensor_field(diagonal_tensor(chart, [3, 4], variance_spec='ul'), name='B')
    i, j, k, l = indices('i^ j_ k^ l_')
    expr = TensorExpr('add', (indexed(B, k, l), indexed(A, i, j)))
    left = canonical_indexed_form(expr)
    right = normalize_indexed_expression(expr)
    assert str(left) == str(right)


def test_tensor_reduce_detects_conflicting_symmetry_zero_case():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    base = diagonal_tensor(chart, [1, 2], variance_spec='ll')
    obj = TensorObject.from_tensor_field(base, symmetry_metadata={'symmetric': ((0, 1),), 'antisymmetric': ((0, 1),)})
    reduced = tensor_reduce(obj)
    assert all(reduced.components[idx] == 0 for idx in [(0, 0), (0, 1), (1, 0), (1, 1)])


def test_tensor_graph_exposes_contraction_planner_steps():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    A = TensorObject.from_tensor_field(diagonal_tensor(chart, [1, 2, 3], variance_spec='ul'), name='A')
    B = TensorObject.from_tensor_field(diagonal_tensor(chart, [4, 5, 6], variance_spec='ul'), name='B')
    C = TensorObject.from_tensor_field(diagonal_tensor(chart, [7, 8, 9], variance_spec='ul'), name='C')
    i, j_down, j_up, k_down, k_up, l = indices('i^ j_ j^ k_ k^ l_')
    expr = TensorExpr('tensor_product', (
        indexed(A, i, j_down),
        TensorExpr('tensor_product', (indexed(B, j_up, k_down), indexed(C, k_up, l))),
    ))
    graph = tensor_graph(expr)
    assert 'plan' in graph
    assert 'estimated_cost' in graph['plan']
    assert 'steps' in graph['plan']
