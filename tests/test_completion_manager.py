
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor
from tensoratlas.completion_manager import (
    DEFAULT_REWRITE_STRATEGIES,
    family_clustered_rules,
    compute_normal_form,
    generate_completion_issues,
    completion_manager,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_family_clustered_rules_runs():
    rules = family_clustered_rules()
    assert len(rules) >= 4
    assert rules[0].family in {"riemann", "weyl", "ricci", "metric", "epsilon", "delta"}

def test_compute_normal_form_runs():
    Ric = _tensor("Ricci", "ll", symmetry={"ricci_symmetric": True, "symmetric": True})
    t1 = IndexedTensor(Ric, (TensorIndex("a","l"), TensorIndex("b","l")))
    t2 = IndexedTensor(Ric, (TensorIndex("b","l"), TensorIndex("a","l")))
    rep = compute_normal_form([(2, t1), (3, t2)], strategy="priority_forward")
    assert rep.strategy == "priority_forward"

def test_generate_completion_issues_runs():
    issues = generate_completion_issues()
    assert len(issues) > 0
    assert any(i.status in {"resolved-by-priority", "needs-completion"} for i in issues)

def test_completion_manager_runs():
    R = _tensor("R", "llll", symmetry={"riemann": True, "bianchi": True})
    t1 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    t2 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","c","d","b")))
    t3 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","d","b","c")))
    rep = completion_manager([(1, t1), (1, t2), (-2, t3)])
    assert len(rep.alternate_normal_forms) >= 1
    assert isinstance(rep.confluence_agrees, bool)
