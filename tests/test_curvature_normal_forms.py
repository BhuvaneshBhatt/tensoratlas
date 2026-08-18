
from __future__ import annotations

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor
from tensoratlas.semantic_ir import TensorExpr
from tensoratlas.ir_curvature_rewriting import ir_curvature_symbol
from tensoratlas.ir_curvature_completion import ir_hodge, ir_lie, ir_interior, ir_wedge
from tensoratlas.curvature_normal_forms import (
    canonicalize_indexed_tensor_expr,
    curvature_normal_form,
    execute_advanced_completion_workflow,
    execute_semantic_operator_interactions,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    import sympy as sp
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_indexed_tensor_expr_canonicalization():
    R = _tensor("Riemann", "llll", symmetry={"riemann": True, "bianchi": True})
    t = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    rep = canonicalize_indexed_tensor_expr(t)
    assert rep.canonical_ir.kind == "curvature_symbol"
    assert rep.metadata["ir_kind"] == "curvature_symbol"

def test_curvature_normal_form_has_richer_metadata():
    node = ir_curvature_symbol("Riemann", 4, name="R")
    rep = curvature_normal_form(node)
    assert rep.normal_form_metadata.preferred_basis in {"contracted_curvature", "decomposed_curvature", "linear_combo", "curvature_family", "unknown"}
    assert isinstance(rep.normal_form_key, tuple)

def test_advanced_completion_workflow_has_witnesses_and_diagnostics():
    node = ir_curvature_symbol("Riemann", 4, name="R")
    rep = execute_advanced_completion_workflow(node)
    assert isinstance(rep.confluence_agrees, bool)
    assert isinstance(rep.overlap_witnesses, tuple)
    assert isinstance(rep.family_diagnostics, tuple)

def test_semantic_operator_interactions_metric_curvature():
    metric = TensorExpr(kind="curvature_symbol", metadata={"family": "Metric", "dimension": 2, "name": "g"})
    deriv = TensorExpr(kind="derivative", children=(metric,), metadata={"operator": "covariant"})
    rep = execute_semantic_operator_interactions(deriv)
    assert "metric_derivative_to_connection_semantics" in rep.applied_rules

    curv = ir_curvature_symbol("Ricci", 2, name="Ric")
    lie = ir_lie(curv, vector_name="X")
    rep2 = execute_semantic_operator_interactions(lie)
    assert "curvature_lie_semantic_lift" in rep2.applied_rules

def test_semantic_operator_interactions_hodge_interior():
    a = TensorExpr(kind="scalar:symbol", payload="a")
    b = TensorExpr(kind="scalar:symbol", payload="b")
    rep = execute_semantic_operator_interactions(ir_hodge(ir_wedge(a, b)))
    assert "hodge_wedge_semantic_lift" in rep.applied_rules

    rep2 = execute_semantic_operator_interactions(ir_interior(ir_wedge(a, b), vector_name="X"))
    assert "interior_wedge_semantic_lift" in rep2.applied_rules
