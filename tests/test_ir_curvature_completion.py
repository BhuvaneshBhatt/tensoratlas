
from __future__ import annotations

from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis, cotangent_basis
from tensoratlas.tensor_core import TensorObject
from tensoratlas.tensor_indices import TensorIndex, IndexedTensor, IndexedTensorExpr
from tensoratlas.semantic_ir import TensorExpr
from tensoratlas.ir_curvature_rewriting import ir_curvature_symbol
from tensoratlas.ir_curvature_completion import (
    ir_wedge,
    ir_hodge,
    ir_lie,
    ir_interior,
    compile_indexed_tensor_to_curvature_ir,
    curvature_ir_canonicalization_report,
    rewrite_operator_ir_node,
    execute_ir_completion_workflow,
    execute_ir_operator_rewriting,
)

def _tensor(name: str, variance: str, *, symmetry=None):
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart); cb = cotangent_basis(chart)
    slot_bases = tuple(tb if v == "u" else cb for v in variance)
    import sympy as sp
    arr = sp.MutableDenseNDimArray.zeros(*([2] * len(variance)))
    return TensorObject(chart=chart, components=arr, variance_spec=variance, slot_bases=slot_bases, name=name, symmetry_metadata=dict(symmetry or {}))

def test_compile_indexed_tensor_to_curvature_ir():
    R = _tensor("Riemann", "llll", symmetry={"riemann": True, "bianchi": True})
    t = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    ir = compile_indexed_tensor_to_curvature_ir(t)
    assert ir.kind == "curvature_symbol"
    assert ir.metadata["family"] == "Riemann"

def test_curvature_ir_canonicalization_report():
    Ric = _tensor("Ricci", "ll", symmetry={"ricci_symmetric": True, "symmetric": True})
    t = IndexedTensor(Ric, (TensorIndex("a", "l"), TensorIndex("b", "l")))
    rep = curvature_ir_canonicalization_report(t)
    assert rep.metadata["ir_kind"] == "curvature_symbol"

def test_operator_rewriting_wedge_lie_interior_hodge():
    a = TensorExpr(kind="scalar:symbol", payload="a")
    b = TensorExpr(kind="scalar:symbol", payload="b")
    node = ir_lie(ir_wedge(a, b), vector_name="X")
    rewritten, applied = rewrite_operator_ir_node(node)
    assert rewritten.kind == "wedge"
    assert "distribute_lie_over_wedge" in applied

    node2 = ir_interior(ir_wedge(a, b), vector_name="X")
    rewritten2, applied2 = rewrite_operator_ir_node(node2)
    assert rewritten2.kind == "wedge"
    assert "distribute_interior_over_wedge" in applied2

    node3 = ir_hodge(ir_hodge(a))
    rewritten3, applied3 = rewrite_operator_ir_node(node3)
    assert "cancel_double_hodge" in applied3

def test_completion_workflow_curvature_ir():
    node = ir_curvature_symbol("Riemann", 4, name="R")
    rep = execute_ir_completion_workflow(node)
    assert isinstance(rep.confluence_agrees, bool)
    assert rep.primary_rewritten_ir.kind in {"curvature_contraction", "curvature_linear_combo"}

def test_execute_ir_operator_rewriting_runs():
    x = TensorExpr(kind="scalar:symbol", payload="x")
    node = ir_hodge(ir_hodge(x))
    rep = execute_ir_operator_rewriting(node)
    assert "cancel_double_hodge" in rep.applied_rules

def test_completion_workflow_on_indexed_expr():
    R = _tensor("Riemann", "llll", symmetry={"riemann": True, "bianchi": True})
    t = IndexedTensor(R, tuple(TensorIndex(x, "l") for x in "abcd"))
    expr = IndexedTensorExpr("add", (t,))
    rep = execute_ir_completion_workflow(expr)
    assert isinstance(rep.confluence_agrees, bool)
