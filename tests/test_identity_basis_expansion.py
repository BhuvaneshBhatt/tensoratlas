
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor
from tensoratlas.identity_basis_expansion import (
    get_expanded_identity_basis,
    get_expanded_identity_rule_sets,
    expand_identity_basis_reduce,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_expanded_identity_basis_is_broader():
    basis = get_expanded_identity_basis()
    assert len(basis) >= 8
    families = {r.family for r in basis}
    assert {"curvature", "metric", "epsilon_delta", "derivative", "exterior"} <= families

def test_expanded_identity_rule_sets_exist():
    names = tuple(rs.name for rs in get_expanded_identity_rule_sets())
    assert names == ("curvature", "metric", "epsilon_delta", "derivative", "exterior")

def test_expand_identity_basis_reduce_runs_on_curvature():
    R = _tensor("R", "llll", symmetry={"riemann": True, "bianchi": True})
    t1 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    t2 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","c","d","b")))
    t3 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","d","b","c")))
    rep = expand_identity_basis_reduce([(1, t1), (1, t2), (-2, t3)])
    assert "curvature" in rep.family_sets
    assert isinstance(rep.completion_summary["confluence_agrees"], bool)

def test_expand_identity_basis_reduce_runs_on_metric():
    g = _tensor("g", "ll", symmetry={"metric": True, "symmetric": True})
    t1 = IndexedTensor(g, (TensorIndex("a","l"), TensorIndex("b","l")))
    t2 = IndexedTensor(g, (TensorIndex("b","l"), TensorIndex("a","l")))
    rep = expand_identity_basis_reduce([(2, t1), (3, t2)])
    assert len(rep.reduced_terms) >= 1
