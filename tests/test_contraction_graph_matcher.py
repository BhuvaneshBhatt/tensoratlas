
from __future__ import annotations

import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.semantic_core import compile_semantic_node
from tensoratlas.semantic_matching import indexed_contraction_graph_signature, indexed_graph_equivalent
from tensoratlas.semantic_rewrite import semantic_match


def _tensor(name: str, variance: str):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart)
    cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    shape = tuple([2] * len(variance))
    arr = sp.MutableDenseNDimArray.zeros(*shape) if shape else sp.MutableDenseNDimArray([0])
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name)


def test_contraction_graph_equivalence_ignores_dummy_names_across_tree():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    expr1 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l"))),
        IndexedTensor(U, (TensorIndex("a","u"), TensorIndex("a","l"))),
    ))
    expr2 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("x","u"), TensorIndex("x","l"))),
        IndexedTensor(U, (TensorIndex("y","u"), TensorIndex("y","l"))),
    ))
    assert indexed_graph_equivalent(expr1, expr2)


def test_contraction_graph_distinguishes_different_contraction_structure():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    expr1 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("j","l"))),
        IndexedTensor(U, (TensorIndex("j","u"), TensorIndex("i","l"))),
    ))
    expr2 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l"))),
        IndexedTensor(U, (TensorIndex("j","u"), TensorIndex("j","l"))),
    ))
    assert not indexed_graph_equivalent(expr1, expr2)


def test_semantic_match_uses_contraction_graph_for_indexed_nodes():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    expr = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("j","l"))),
        IndexedTensor(U, (TensorIndex("j","u"), TensorIndex("i","l"))),
    ))
    pat = compile_semantic_node(IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("a","u"), TensorIndex("b","l"))),
        IndexedTensor(U, (TensorIndex("b","u"), TensorIndex("a","l"))),
    )))
    assert semantic_match(expr, pat) is not None


def test_contraction_graph_signature_stable_under_dummy_renaming():
    T = _tensor("T", "ul")
    expr1 = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    expr2 = IndexedTensor(T, (TensorIndex("z","u"), TensorIndex("z","l")))
    s1 = indexed_contraction_graph_signature(compile_semantic_node(expr1))
    s2 = indexed_contraction_graph_signature(compile_semantic_node(expr2))
    assert s1 == s2
