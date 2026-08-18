
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.signed_contraction_canonicalization import (
    signed_contraction_canonical_key,
    signed_contraction_canonicalize_report,
    signed_contraction_equivalent,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 3)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([3] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_signed_permutation_antisymmetric_family():
    eps = _tensor("eps", "lll", symmetry={"epsilon": True, "antisymmetric": True})
    t1 = IndexedTensor(eps, tuple(TensorIndex(x, "l") for x in ("a","b","c")))
    t2 = IndexedTensor(eps, tuple(TensorIndex(x, "l") for x in ("b","a","c")))
    k1 = signed_contraction_canonical_key(t1)
    k2 = signed_contraction_canonical_key(t2)
    assert k1[0] == k2[0] == "IndexedTensor"
    assert k1[1] == k2[1] == "eps"
    assert k1[3] in (-1, 1)
    assert k2[3] in (-1, 1)

def test_global_dummy_relabeling_across_whole_tree():
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
    assert signed_contraction_equivalent(e1, e2)

def test_multi_factor_contraction_graph_large_tree():
    A = _tensor("A", "ul")
    B = _tensor("B", "ul")
    C = _tensor("C", "ul")
    D = _tensor("D", "ul")
    e1 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(A, (TensorIndex("i","u"), TensorIndex("j","l"))),
        IndexedTensor(B, (TensorIndex("j","u"), TensorIndex("k","l"))),
        IndexedTensor(C, (TensorIndex("k","u"), TensorIndex("m","l"))),
        IndexedTensor(D, (TensorIndex("m","u"), TensorIndex("i","l"))),
    ))
    e2 = IndexedTensorExpr("tensor_product", (
        IndexedTensor(A, (TensorIndex("a","u"), TensorIndex("b","l"))),
        IndexedTensor(B, (TensorIndex("b","u"), TensorIndex("c","l"))),
        IndexedTensor(C, (TensorIndex("c","u"), TensorIndex("d","l"))),
        IndexedTensor(D, (TensorIndex("d","u"), TensorIndex("a","l"))),
    ))
    assert signed_contraction_equivalent(e1, e2)

def test_signed_contraction_report_has_semantic_fingerprint():
    T = _tensor("T", "ul")
    t = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    rep = signed_contraction_canonicalize_report(t)
    assert isinstance(rep.semantic_fingerprint, tuple)
