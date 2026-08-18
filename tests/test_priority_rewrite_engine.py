
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor
from tensoratlas.priority_rewrite_engine import (
    DEFAULT_PRIORITY_RULES,
    ordered_rules,
    priority_rewrite_reduce,
    priority_rewrite_equivalent,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_rule_order_is_priority_sorted():
    rules = ordered_rules()
    assert tuple(r.name for r in rules) == (
        "rewrite_bianchi_three_term",
        "rewrite_riemann_pair_exchange",
        "rewrite_ricci_symmetry",
        "rewrite_metric_family",
    )

def test_priority_rewrite_reduce_bianchi():
    R = _tensor("R", "llll", symmetry={"riemann": True, "bianchi": True})
    t1 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    t2 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","c","d","b")))
    t3 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","d","b","c")))
    rep = priority_rewrite_reduce([(1, t1), (1, t2), (-2, t3)])
    assert "rewrite_bianchi_three_term" in rep.applied_rules or rep.iterations >= 1

def test_priority_rewrite_reduce_ricci_symmetry():
    Ric = _tensor("Ricci", "ll", symmetry={"ricci_symmetric": True})
    t1 = IndexedTensor(Ric, (TensorIndex("a","l"), TensorIndex("b","l")))
    t2 = IndexedTensor(Ric, (TensorIndex("b","l"), TensorIndex("a","l")))
    rep = priority_rewrite_reduce([(2, t1), (3, t2)])
    assert any(name == "rewrite_ricci_symmetry" for name in rep.applied_rules) or len(rep.reduced_terms) == 1

def test_priority_rewrite_equivalent_order_insensitive():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    t1 = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    t2 = IndexedTensor(U, (TensorIndex("a","u"), TensorIndex("a","l")))
    assert priority_rewrite_equivalent([(2, t1), (3, t2)], [(3, t2), (2, t1)])
