
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.canonicalization_core import (
    abstract_index_canonicalize,
    abstract_index_equivalent,
    expr_canonical_key,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_ricci_slot_symmetry_equivalent():
    Ric = _tensor("Ricci", "ll", symmetry={"ricci_symmetric": True, "symmetric": True})
    t1 = IndexedTensor(Ric, (TensorIndex("a","l"), TensorIndex("b","l")))
    t2 = IndexedTensor(Ric, (TensorIndex("b","l"), TensorIndex("a","l")))
    assert abstract_index_equivalent(t1, t2)

def test_riemann_pair_exchange_equivalent():
    R = _tensor("R", "llll", symmetry={"riemann": True, "pair_symmetric": True, "bianchi": True})
    t1 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    t2 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("c","d","a","b")))
    assert abstract_index_equivalent(t1, t2)

def test_add_expression_canonical_order():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    t1 = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    t2 = IndexedTensor(U, (TensorIndex("a","u"), TensorIndex("a","l")))
    e1 = IndexedTensorExpr("add", (t1, t2))
    e2 = IndexedTensorExpr("add", (t2, t1))
    assert expr_canonical_key(e1) == expr_canonical_key(e2)

def test_canonicalization_report_has_fingerprint():
    T = _tensor("T", "ul")
    t = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    rep = abstract_index_canonicalize(t)
    assert isinstance(rep.semantic_fingerprint, tuple)
