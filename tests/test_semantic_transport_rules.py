
from __future__ import annotations
import sympy as sp
from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor
from tensoratlas.exterior_geometry import ExteriorFormNF
from tensoratlas.geometry_components import component_tensor_field
from tensoratlas.semantic_transport_rules import (
    iterative_algebraic_identity_reduce,
    iterative_algebraic_identity_equivalent,
    semantic_native_internal_execute,
    deeper_cross_chart_transport,
    deep_frame_hodge,
    deep_frame_codifferential,
    deep_frame_interior_product,
    deep_frame_lie_derivative,
    DeepFrameExteriorContext,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_iterative_rule_driven_reduction():
    R = _tensor("R", "llll", symmetry={"riemann": True, "bianchi": True})
    t1 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    t2 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","c","d","b")))
    t3 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","d","b","c")))
    rep = iterative_algebraic_identity_reduce([(1, t1), (1, t2), (-2, t3)])
    assert "linear_bianchi_reduction" in rep.applied_rules

def test_iterative_equivalence_order_insensitive():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    t1 = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    t2 = IndexedTensor(U, (TensorIndex("a","u"), TensorIndex("a","l")))
    assert iterative_algebraic_identity_equivalent([(2, t1), (3, t2)], [(3, t2), (2, t1)])

def test_semantic_native_internal_execute():
    rep = semantic_native_internal_execute(sp.Add(sp.Symbol("x"), 0, evaluate=False), subsystem="scalar")
    assert rep.subsystem == "scalar"

def test_deeper_cross_chart_transport_identity():
    chart = get_chart("Euclidean", "Cartesian", 2)
    field = component_tensor_field("V", chart, "u", [1, 2])
    class DummyMap:
        target = chart
        def jacobian(self, coords):
            return sp.eye(2)
    rep = deeper_cross_chart_transport(field, DummyMap())
    assert list(rep.transported.components) == [1, 2]

def test_deep_frame_ops():
    ctx = DeepFrameExteriorContext(2, sp.Matrix([[2,1],[1,3]]), ("e0","e1"), ("e0","e1"))
    form = ExteriorFormNF(2, {(0,): sp.Integer(1)}, basis_labels=("e0","e1"), metadata={})
    assert isinstance(deep_frame_hodge(form, ctx).result.terms, dict)
    x0, x1 = sp.symbols("x0 x1")
    form2 = ExteriorFormNF(2, {(0,): x0, (1,): x1}, basis_labels=("e0","e1"), metadata={})
    assert isinstance(deep_frame_codifferential(form2, ctx, coordinates=(x0, x1)).result.terms, dict)
    assert isinstance(deep_frame_interior_product([1, 0], form2, ctx).result.terms, dict)
    assert isinstance(deep_frame_lie_derivative([1, 0], form2, ctx, coordinates=(x0, x1)).result.terms, dict)
