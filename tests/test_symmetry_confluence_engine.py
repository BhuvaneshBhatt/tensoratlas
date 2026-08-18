
from __future__ import annotations
import sympy as sp
from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.symmetry_confluence_engine import (
    full_tree_canonical_key,
    semantic_native_conflict_reduce,
    confluence_completed_reduce,
    BROAD_IDENTITY_RULES,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_more_complete_symmetry_family():
    eps = _tensor("eps", "lll", symmetry={"epsilon": True, "antisymmetric": True})
    t1 = IndexedTensor(eps, tuple(TensorIndex(x, "l") for x in "abc"))
    t2 = IndexedTensor(eps, tuple(TensorIndex(x, "l") for x in ("b","c","a")))
    assert full_tree_canonical_key(t1) == full_tree_canonical_key(t2)

def test_stronger_contraction_graph_tree_dummy_handling():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    V = _tensor("V", "ul")
    e1 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("j","l"))),
        IndexedTensor(U, (TensorIndex("j","u"), TensorIndex("k","l"))),
        IndexedTensor(V, (TensorIndex("k","u"), TensorIndex("i","l"))),
    ))
    e2 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("a","u"), TensorIndex("b","l"))),
        IndexedTensor(U, (TensorIndex("b","u"), TensorIndex("c","l"))),
        IndexedTensor(V, (TensorIndex("c","u"), TensorIndex("a","l"))),
    ))
    assert full_tree_canonical_key(e1) == full_tree_canonical_key(e2)

def test_semantic_native_execution_path_runs():
    T = _tensor("T", "ul")
    term = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    expr = IndexedTensorExpr("add", (term, term))
    rep = semantic_native_conflict_reduce(expr)
    assert isinstance(rep.semantic_fingerprint, tuple)

def test_confluence_completion_runs():
    Ric = _tensor("Ricci", "ll", symmetry={"ricci_symmetric": True, "symmetric": True})
    t1 = IndexedTensor(Ric, (TensorIndex("a","l"), TensorIndex("b","l")))
    t2 = IndexedTensor(Ric, (TensorIndex("b","l"), TensorIndex("a","l")))
    rep = confluence_completed_reduce([(2, t1), (3, t2)])
    assert len(rep.canonicalized_terms) == 1
    assert "priority_conflict_rewrite_integration" in rep.applied_rules
    assert len(BROAD_IDENTITY_RULES) >= 8
