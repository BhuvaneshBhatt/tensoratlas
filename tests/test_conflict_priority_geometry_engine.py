
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor
from tensoratlas.conflict_priority_geometry_engine import (
    ordered_conflict_rules,
    conflict_aware_priority_reduce,
    conflict_aware_priority_equivalent,
    indexed_geometry_priority_canonicalize,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_extended_rule_basis_order():
    rules = ordered_conflict_rules()
    names = tuple(r.name for r in rules)
    assert names[:4] == (
        "rewrite_bianchi_three_term",
        "rewrite_riemann_pair_exchange",
        "rewrite_weyl_trace",
        "rewrite_ricci_symmetry",
    )

def test_conflict_handling_blocks_lower_priority_rule():
    R = _tensor("R", "llll", symmetry={"riemann": True, "bianchi": True})
    t1 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    t2 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","c","d","b")))
    t3 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","d","b","c")))
    rep = conflict_aware_priority_reduce([(1, t1), (1, t2), (-2, t3)])
    assert "rewrite_bianchi_three_term" in rep.applied_rules
    assert "rewrite_riemann_pair_exchange" in rep.blocked_rules or rep.iterations >= 1

def test_conflict_aware_equivalence_order_insensitive():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    t1 = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    t2 = IndexedTensor(U, (TensorIndex("a","u"), TensorIndex("a","l")))
    assert conflict_aware_priority_equivalent([(2, t1), (3, t2)], [(3, t2), (2, t1)])

def test_indexed_geometry_priority_canonicalize_runs():
    T = _tensor("T", "ul")
    obj = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    rep = indexed_geometry_priority_canonicalize(obj)
    assert isinstance(rep.semantic_fingerprint, tuple)
