
from __future__ import annotations
import sympy as sp

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor
from tensoratlas.exterior_geometry import ExteriorFormNF
from tensoratlas.geometry_components import component_tensor_field
from tensoratlas.semantic_identity_components import (
    general_multi_term_tensor_identity_engine,
    general_multi_term_identity_equivalent,
    semantic_core_native_execute,
    component_tensor_change_basis,
    EnhancedFrameGeometryContext,
    advanced_frame_metric_hodge,
    advanced_frame_metric_codifferential,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_coefficient_aware_cancellation():
    T = _tensor("T", "ul")
    term = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    red = general_multi_term_tensor_identity_engine([(1, term), (-1, term)])
    assert red.terms == tuple()

def test_bianchi_family_reduction():
    R = _tensor("R", "llll", symmetry={"riemann": True, "bianchi": True, "antisymmetric": ((0,1),(2,3)), "pair_symmetric": (((0,1),(2,3)),)})
    t1 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    t2 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","c","d","b")))
    t3 = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in ("a","d","b","c")))
    red = general_multi_term_tensor_identity_engine([(1, t1), (1, t2), (-2, t3)])
    assert isinstance(red.applied_rules, tuple)

def test_multi_term_equivalence_under_order():
    T = _tensor("T", "ul")
    U = _tensor("U", "ul")
    t1 = IndexedTensor(T, (TensorIndex("i","u"), TensorIndex("i","l")))
    t2 = IndexedTensor(U, (TensorIndex("a","u"), TensorIndex("a","l")))
    assert general_multi_term_identity_equivalent([(2, t1), (3, t2)], [(3, t2), (2, t1)])

def test_semantic_core_native_execute_runs():
    rep = semantic_core_native_execute(sp.Add(sp.Symbol("x"), 0, evaluate=False), subsystem="scalar")
    assert rep.subsystem == "scalar"

def test_component_tensor_change_basis_identity():
    chart = get_chart("Euclidean", "Cartesian", 2)
    field = component_tensor_field("V", chart, "u", [1, 2])
    out = component_tensor_change_basis(field, sp.eye(2))
    assert list(out.components) == [1, 2]

def test_advanced_frame_metric_hodge_and_codifferential():
    ctx = EnhancedFrameGeometryContext(2, sp.Matrix([[2,1],[1,3]]), ("e0","e1"), ("e0","e1"))
    form = ExteriorFormNF(2, {(0,): sp.Integer(1)}, basis_labels=("e0","e1"), metadata={})
    star = advanced_frame_metric_hodge(form, ctx)
    assert isinstance(star.terms, dict)
    x0, x1 = sp.symbols("x0 x1")
    form2 = ExteriorFormNF(2, {(0,): x0, (1,): x1}, basis_labels=("e0","e1"), metadata={})
    delta = advanced_frame_metric_codifferential(form2, ctx, coordinates=(x0, x1))
    assert isinstance(delta.terms, dict)
