
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.dummy_multiterm_reduction import (
    tree_canonical_key,
    multiterm_canonicalize,
    canonicalize_expression_tree_report,
    canonicalization_integrated_rewrite,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_deeper_dummy_relabeling_across_tree():
    T = _tensor("T", "ul")
    a = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    b = IndexedTensor(T, (TensorIndex("x","u"), TensorIndex("x","l")))
    e1 = IndexedTensorExpr("tensor_product", (a, b))
    e2 = IndexedTensorExpr("tensor_product", (b, a))
    assert tree_canonical_key(e1) == tree_canonical_key(e2)

def test_multiterm_canonicalization_modulo_symmetry():
    Ric = _tensor("Ricci", "ll", symmetry={"ricci_symmetric": True, "symmetric": True})
    t1 = IndexedTensor(Ric, (TensorIndex("a","l"), TensorIndex("b","l")))
    t2 = IndexedTensor(Ric, (TensorIndex("b","l"), TensorIndex("a","l")))
    red = multiterm_canonicalize([(2, t1), (3, t2)])
    assert len(red) == 1
    assert red[0][0] == 5

def test_tree_report_has_fingerprint():
    T = _tensor("T", "ul")
    t = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    rep = canonicalize_expression_tree_report(t)
    assert isinstance(rep.semantic_fingerprint, tuple)

def test_rewrite_integration_uses_canonicalization():
    Ric = _tensor("Ricci", "ll", symmetry={"ricci_symmetric": True, "symmetric": True})
    t1 = IndexedTensor(Ric, (TensorIndex("a","l"), TensorIndex("b","l")))
    t2 = IndexedTensor(Ric, (TensorIndex("b","l"), TensorIndex("a","l")))
    rep = canonicalization_integrated_rewrite([(2, t1), (3, t2)])
    assert len(rep.reduced_terms) == 1
    assert "rewrite_engine_integration" in rep.applied_rules
