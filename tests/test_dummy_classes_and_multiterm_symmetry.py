
from __future__ import annotations

import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.semantic_core import compile_semantic_node
from tensoratlas.semantic_rewrite import semantic_match
from tensoratlas.semantic_matching import indexed_tree_dummy_normalize


def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart)
    cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    shape = tuple([2] * len(variance))
    arr = sp.MutableDenseNDimArray.zeros(*shape) if shape else sp.MutableDenseNDimArray([0])
    md = {} if symmetry is None else symmetry
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=md)


def test_dummy_classes_normalize_across_multiple_leaves():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    i, j, a, b = [TensorIndex(x, "u" if k % 2 == 0 else "l") for k, x in enumerate(["i", "i", "a", "a"])]
    # Build T^i_i U^a_a and T^j_j U^b_b using distinct dummy names across leaves
    expr1 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l"))),
        IndexedTensor(U, (TensorIndex("a","u"), TensorIndex("a","l"))),
    ))
    expr2 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("j","u"), TensorIndex("j","l"))),
        IndexedTensor(U, (TensorIndex("b","u"), TensorIndex("b","l"))),
    ))
    node1 = indexed_tree_dummy_normalize(compile_semantic_node(expr1))
    node2 = indexed_tree_dummy_normalize(compile_semantic_node(expr2))
    assert semantic_match(expr1, compile_semantic_node(expr2)) is not None
    assert node1.children[0].children[0].children[0].value == node2.children[0].children[0].children[0].value


def test_cyclic_multiterm_symmetry_matches_rotated_slots():
    C = _tensor("C", "lll", symmetry={"cyclic": ((0, 1, 2),)})
    a, b, c = [TensorIndex(x, "l") for x in "abc"]
    expr = IndexedTensor(C, (a, b, c))
    pattern = compile_semantic_node(IndexedTensor(C, (b, c, a)))
    assert semantic_match(expr, pattern) is not None


def test_pair_antisymmetric_symmetry_matches_pair_swap_with_sign():
    R = _tensor("R", "llll", symmetry={"pair_antisymmetric": (((0, 1), (2, 3)),)})
    a, b, c, d = [TensorIndex(x, "l") for x in "abcd"]
    expr = IndexedTensor(R, (a, b, c, d))
    env = semantic_match(expr, compile_semantic_node(IndexedTensor(R, (c, d, a, b))))
    assert env is not None
    assert env.get("__orbit_sign__", 1) in (-1, 1)
