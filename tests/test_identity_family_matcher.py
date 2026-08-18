
from __future__ import annotations

import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.semantic_core import compile_semantic_node
from tensoratlas.semantic_matching import (
    indexed_identity_family_signature,
    indexed_graph_family_signature,
    indexed_graph_equivalent,
)
from tensoratlas.semantic_rewrite import semantic_match


def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart)
    cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    md = {} if symmetry is None else symmetry
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=md)


def test_identity_family_signature_detects_riemann_metadata():
    R = _tensor("R", "llll", symmetry={"riemann": True, "antisymmetric": ((0, 1), (2, 3)), "pair_symmetric": (((0, 1), (2, 3)),)})
    expr = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    sig = indexed_identity_family_signature(compile_semantic_node(expr))
    assert sig
    text = repr(sig).lower()
    assert "riemann" in text or "riemann_like" in text


def test_graph_family_signature_changes_when_identity_family_changes():
    R = _tensor("R", "llll", symmetry={"riemann": True, "antisymmetric": ((0, 1), (2, 3)), "pair_symmetric": (((0, 1), (2, 3)),)})
    T = _tensor("T", "llll", symmetry={"antisymmetric": ((0, 1), (2, 3))})
    expr_r = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    expr_t = IndexedTensor(T, tuple(TensorIndex(x, "l") for x in "abcd"))
    s_r = indexed_graph_family_signature(compile_semantic_node(expr_r))
    s_t = indexed_graph_family_signature(compile_semantic_node(expr_t))
    assert s_r != s_t


def test_indexed_graph_equivalence_requires_matching_identity_family():
    R = _tensor("R", "llll", symmetry={"riemann": True, "antisymmetric": ((0, 1), (2, 3)), "pair_symmetric": (((0, 1), (2, 3)),)})
    T = _tensor("T", "llll", symmetry={"antisymmetric": ((0, 1), (2, 3)), "pair_symmetric": (((0, 1), (2, 3)),)})
    left = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    right = IndexedTensor(T, tuple(TensorIndex(x, "l") for x in "abcd"))
    assert not indexed_graph_equivalent(left, right)


def test_semantic_match_preserves_identity_family_requirement():
    R = _tensor("R", "llll", symmetry={"riemann": True, "antisymmetric": ((0, 1), (2, 3)), "pair_symmetric": (((0, 1), (2, 3)),)})
    T = _tensor("T", "llll", symmetry={"antisymmetric": ((0, 1), (2, 3)), "pair_symmetric": (((0, 1), (2, 3)),)})
    expr = IndexedTensorExpr("tensor_product", (
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd")),
    ))
    pat = compile_semantic_node(IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, tuple(TensorIndex(x, "l") for x in "wxyz")),
    )))
    assert semantic_match(expr, pat) is None


def test_semantic_match_succeeds_with_same_family_and_dummy_renaming():
    R = _tensor("R", "llll", symmetry={"riemann": True, "antisymmetric": ((0, 1), (2, 3)), "pair_symmetric": (((0, 1), (2, 3)),)})
    expr = IndexedTensorExpr("tensor_product", (
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd")),
    ))
    pat = compile_semantic_node(IndexedTensorExpr("tensor_product", (
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "wxyz")),
    )))
    assert semantic_match(expr, pat) is not None
