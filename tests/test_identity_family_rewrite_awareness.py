
from __future__ import annotations

import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.semantic_core import compile_semantic_node
from tensoratlas.semantic_matching import indexed_identity_rewrite_signatures, indexed_graph_equivalent
from tensoratlas.semantic_rewrite import semantic_match


def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart)
    cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))


def test_riemann_bianchi_sum_gets_identity_rewrite_signature():
    R = _tensor("R", "llll", symmetry={"riemann": True, "antisymmetric": ((0,1),(2,3)), "pair_symmetric": (((0,1),(2,3)),), "bianchi": True})
    expr = IndexedTensorExpr("add", (
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd")),
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","c","d","b"))),
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","d","b","c"))),
    ))
    sigs = indexed_identity_rewrite_signatures(compile_semantic_node(expr))
    assert any("riemann_bianchi_sum" in repr(s) for s in sigs)


def test_bianchi_sums_match_under_term_reordering_and_dummy_renaming():
    R = _tensor("R", "llll", symmetry={"riemann": True, "antisymmetric": ((0,1),(2,3)), "pair_symmetric": (((0,1),(2,3)),), "bianchi": True})
    expr1 = IndexedTensorExpr("add", (
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd")),
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","c","d","b"))),
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","d","b","c"))),
    ))
    expr2 = IndexedTensorExpr("add", (
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("w","y","z","x"))),
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("w","x","y","z"))),
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("w","z","x","y"))),
    ))
    assert indexed_graph_equivalent(expr1, expr2)
    assert semantic_match(expr1, compile_semantic_node(expr2)) is not None


def test_non_bianchi_sum_does_not_match_bianchi_family_rewrite():
    R = _tensor("R", "llll", symmetry={"riemann": True, "antisymmetric": ((0,1),(2,3)), "pair_symmetric": (((0,1),(2,3)),), "bianchi": True})
    expr1 = IndexedTensorExpr("add", (
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd")),
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","c","d","b"))),
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","d","b","c"))),
    ))
    expr_bad = IndexedTensorExpr("add", (
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd")),
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","b","d","c"))),
        IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","d","b","c"))),
    ))
    assert not indexed_graph_equivalent(expr1, expr_bad)
