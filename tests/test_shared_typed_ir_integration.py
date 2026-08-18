from __future__ import annotations

from tensoratlas.semantic_ir import TensorExpr, ir_to_dict, normalize_tensor_expr
from tensoratlas.variational_gr import add, symbol, variation, covariant_derivative
from tensoratlas.curvature_relations import Riemann, curvature_object_to_ir, curvature_reduce_to_ir
from tensoratlas.curvature_normal_forms import curvature_normal_form


def test_variational_constructors_use_shared_tensor_expr():
    expr = covariant_derivative(variation(symbol("g")), index="a")
    assert isinstance(expr, TensorExpr)
    assert ir_to_dict(expr)["kind"] == "covariant_derivative"


def test_curvature_objects_compile_to_shared_tensor_expr():
    ir = curvature_object_to_ir(Riemann(4))
    assert isinstance(ir, TensorExpr)
    assert ir.kind == "curvature_symbol"
    assert ir.metadata["family"] == "Riemann"


def test_curvature_reduction_returns_normalizable_shared_ir():
    ir = curvature_reduce_to_ir(Riemann(4))
    normalized = normalize_tensor_expr(ir)
    assert isinstance(normalized, TensorExpr)
    assert normalized.kind == "curvature_linear_combo"


def test_curvature_normal_form_accepts_shared_ir_directly():
    ir = curvature_object_to_ir(Riemann(4))
    report = curvature_normal_form(ir)
    assert isinstance(report.normalized_ir, TensorExpr)
    assert report.normal_form_metadata.target_families


def test_generic_add_normalization_is_shared_across_clients():
    expr = add(symbol("b"), add(symbol("a"), symbol("c")))
    normalized = normalize_tensor_expr(expr)
    assert normalized.kind == "add"
    assert len(normalized.children) == 3
