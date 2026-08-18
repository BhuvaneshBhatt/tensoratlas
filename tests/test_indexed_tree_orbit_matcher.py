
from __future__ import annotations

import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.semantic_core import compile_semantic_node
from tensoratlas.semantic_rewrite import semantic_match, spat


def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart)
    cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    shape = tuple([2] * len(variance))
    arr = sp.MutableDenseNDimArray.zeros(*shape) if shape else sp.MutableDenseNDimArray([0])
    md = {} if symmetry is None else symmetry
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=md)


def test_indexed_tensor_product_matches_modulo_commutativity_and_slot_symmetry():
    S = _tensor("S", "ll", symmetry={"symmetric": ((0, 1),)})
    T = _tensor("T", "ll")
    a, b, c, d = [TensorIndex(x, "l") for x in "abcd"]
    expr = IndexedTensorExpr("tensor_product", (
        IndexedTensor(S, (a, b)),
        IndexedTensor(T, (c, d)),
    ))
    pattern_expr = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (c, d)),
        IndexedTensor(S, (b, a)),
    ))
    pattern_node = compile_semantic_node(pattern_expr)
    env = semantic_match(expr, pattern_node)
    assert env is not None


def test_indexed_add_matches_modulo_commutativity_and_leaf_slot_symmetry():
    A = _tensor("A", "ll", symmetry={"antisymmetric": ((0, 1),)})
    B = _tensor("B", "ll")
    i, j, k, l = [TensorIndex(x, "l") for x in "ijkl"]
    expr = IndexedTensorExpr("add", (
        IndexedTensor(A, (i, j)),
        IndexedTensor(B, (k, l)),
    ))
    pattern_expr = IndexedTensorExpr("add", (
        IndexedTensor(B, (k, l)),
        IndexedTensor(A, (j, i)),
    ))
    pattern_node = compile_semantic_node(pattern_expr)
    env = semantic_match(expr, pattern_node)
    assert env is not None
    assert env.get("__orbit_sign__", 1) in (-1, 1)


def test_bound_tree_pattern_records_orbit_sign():
    A = _tensor("A", "ll", symmetry={"antisymmetric": ((0, 1),)})
    i, j = [TensorIndex(x, "l") for x in "ij"]
    B = _tensor("B", "ll")
    k, l = [TensorIndex(x, "l") for x in "kl"]
    expr = IndexedTensorExpr("tensor_product", (IndexedTensor(A, (i, j)), IndexedTensor(B, (k, l))))
    pattern = spat("indexed_tensor_product", spat("indexed_tensor", bind="leaf"), spat("indexed_tensor"), bind="tree")
    env = semantic_match(expr, pattern)
    assert env is not None
    assert "tree" in env
