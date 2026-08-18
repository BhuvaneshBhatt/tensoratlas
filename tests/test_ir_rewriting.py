
from __future__ import annotations
import sympy as sp

from tensoratlas.semantic_ir import TensorExpr
from tensoratlas.ir_rewriting import (
    IR_NATIVE_RULES,
    ir_contraction,
    ir_raise_lower,
    ir_derivative,
    canonicalize_tensor_expr,
    rewrite_tensor_expr,
    execute_ir_native_rewriting,
)

def test_ir_native_rules_exist():
    assert len(IR_NATIVE_RULES) >= 5
    assert IR_NATIVE_RULES[0].name == "flatten_nested_contraction"

def test_contraction_flattening():
    a = TensorExpr(kind="indexed_tensor", metadata={"tensor_name": "A"})
    b = TensorExpr(kind="indexed_tensor", metadata={"tensor_name": "B"})
    c = TensorExpr(kind="indexed_tensor", metadata={"tensor_name": "C"})
    node = ir_contraction(a, ir_contraction(b, c))
    rewritten, applied = rewrite_tensor_expr(node)
    assert rewritten.kind == "contraction"
    assert len(rewritten.children) == 3
    assert "flatten_nested_contraction" in applied

def test_raise_lower_cancellation():
    a = TensorExpr(kind="indexed_tensor", metadata={"tensor_name": "A"})
    node = ir_raise_lower(ir_raise_lower(a, mode="raise"), mode="lower")
    rewritten, applied = rewrite_tensor_expr(node)
    assert rewritten.kind == "indexed_tensor"
    assert "cancel_raise_lower_pair" in applied

def test_derivative_distribution():
    a = TensorExpr(kind="scalar:symbol", payload="x")
    b = TensorExpr(kind="scalar:symbol", payload="y")
    add = TensorExpr(kind="scalar:add", children=(b, a))
    node = ir_derivative(add, operator="covariant")
    rewritten, applied = rewrite_tensor_expr(node)
    assert rewritten.kind == "scalar:add"
    assert all(ch.kind == "derivative" for ch in rewritten.children)
    assert "distribute_derivative_over_add" in applied

def test_ir_canonicalization_sorts_children():
    a = TensorExpr(kind="scalar:symbol", payload="a")
    b = TensorExpr(kind="scalar:symbol", payload="b")
    node = TensorExpr(kind="scalar:add", children=(b, a))
    out = canonicalize_tensor_expr(node)
    assert len(out.children) == 2

def test_execute_ir_native_rewriting_runs():
    a = TensorExpr(kind="scalar:symbol", payload="x")
    b = TensorExpr(kind="scalar:symbol", payload="y")
    add = TensorExpr(kind="scalar:add", children=(b, a))
    node = ir_derivative(add, operator="covariant")
    rep = execute_ir_native_rewriting(node)
    assert rep.rewritten_ir.kind == "scalar:add"
    assert isinstance(rep.applied_rules, tuple)
    assert isinstance(rep.materialized, (dict, sp.Basic))
