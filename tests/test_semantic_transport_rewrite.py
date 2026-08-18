
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor
from tensoratlas.exterior_geometry import ExteriorFormNF
from tensoratlas.geometry_components import component_tensor_field
from tensoratlas.semantic_transport_rewrite import (
    rewrite_system_identity_reduce,
    rewrite_system_identity_equivalent,
    semantic_core_native_canonicalize,
    deeper_cross_chart_transport_to_derived,
    mixed_basis_hodge,
    mixed_basis_codifferential,
    mixed_basis_interior_product,
    mixed_basis_lie_derivative,
    MixedBasisMetricContext,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_rewrite_system_rule_basis():
    R = _tensor("R", "llll", symmetry={"riemann": True, "bianchi": True})
    t1 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    t2 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","c","d","b")))
    t3 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","d","b","c")))
    rep = rewrite_system_identity_reduce([(1, t1), (1, t2), (-2, t3)])
    assert "rewrite_bianchi_three_term" in rep.applied_rules or rep.iterations >= 1

def test_rewrite_system_equivalence():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    t1 = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    t2 = IndexedTensor(U, (TensorIndex("a","u"), TensorIndex("a","l")))
    assert rewrite_system_identity_equivalent([(2, t1), (3, t2)], [(3, t2), (2, t1)])

def test_semantic_core_native_canonicalize():
    rep = semantic_core_native_canonicalize(sp.Add(sp.Symbol("x"), 0, evaluate=False), subsystem="scalar")
    assert rep.subsystem == "scalar"

def test_transport_to_derived_identity():
    chart = get_chart("Euclidean", "Cartesian", 2)
    field = component_tensor_field("V", chart, "u", [1, 2])
    class DummyMap:
        target = chart
        def jacobian(self, coords):
            return sp.eye(2)
    rep = deeper_cross_chart_transport_to_derived(field, DummyMap())
    assert list(rep.transported_tensor.components) == [1, 2]
    assert rep.curvature_report is not None

def test_mixed_basis_ops():
    ctx = MixedBasisMetricContext(2, sp.Matrix([[2,1],[1,3]]), ("b0","b1"), ("e0","e1"))
    form = ExteriorFormNF(2, {(0,): sp.Integer(1)}, basis_labels=("b0","b1"), metadata={})
    assert isinstance(mixed_basis_hodge(form, ctx).result.terms, dict)
    x0, x1 = sp.symbols("x0 x1")
    form2 = ExteriorFormNF(2, {(0,): x0, (1,): x1}, basis_labels=("b0","b1"), metadata={})
    assert isinstance(mixed_basis_codifferential(form2, ctx, coordinates=(x0, x1)).result.terms, dict)
    assert isinstance(mixed_basis_interior_product([1, 0], form2, ctx).result.terms, dict)
    assert isinstance(mixed_basis_lie_derivative([1, 0], form2, ctx, coordinates=(x0, x1)).result.terms, dict)
