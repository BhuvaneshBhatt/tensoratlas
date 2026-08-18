
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.priority_reduction import (
    broader_tree_canonical_key,
    contraction_graph_canonicalize_report,
    priority_canonicalization_integrated_reduce,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_broader_slot_symmetry_across_larger_family():
    Ric = _tensor("Ricci", "ll", symmetry={"ricci_symmetric": True, "symmetric": True})
    t1 = IndexedTensor(Ric, (TensorIndex("a","l"), TensorIndex("b","l")))
    t2 = IndexedTensor(Ric, (TensorIndex("b","l"), TensorIndex("a","l")))
    assert broader_tree_canonical_key(t1) == broader_tree_canonical_key(t2)

def test_stronger_dummy_handling_contraction_graph():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    e1 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("j","l"))),
        IndexedTensor(U, (TensorIndex("j","u"), TensorIndex("i","l"))),
    ))
    e2 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(T, (TensorIndex("a","u"), TensorIndex("b","l"))),
        IndexedTensor(U, (TensorIndex("b","u"), TensorIndex("a","l"))),
    ))
    assert broader_tree_canonical_key(e1) == broader_tree_canonical_key(e2)

def test_contraction_graph_report_has_fingerprint():
    T = _tensor("T", "ul")
    t = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    rep = contraction_graph_canonicalize_report(t)
    assert isinstance(rep.semantic_fingerprint, tuple)

def test_tighter_priority_rewrite_integration():
    Ric = _tensor("Ricci", "ll", symmetry={"ricci_symmetric": True, "symmetric": True})
    t1 = IndexedTensor(Ric, (TensorIndex("a","l"), TensorIndex("b","l")))
    t2 = IndexedTensor(Ric, (TensorIndex("b","l"), TensorIndex("a","l")))
    rep = priority_canonicalization_integrated_reduce([(2, t1), (3, t2)])
    assert len(rep.canonicalized_terms) == 1
    assert "broader_slot_symmetry" in rep.applied_rules
