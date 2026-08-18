
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import sympy as sp

from .semantic_ir import TensorExpr, compile_tensor_expr, normalize_tensor_expr, materialize_tensor_expr


@dataclass(frozen=True)
class IRRewriteRule:
    name: str
    priority: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IRNativeRewriteReport:
    original: Any
    input_ir: TensorExpr
    normalized_ir: TensorExpr
    rewritten_ir: TensorExpr
    materialized: Any
    applied_rules: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def ir_contraction(*children: TensorExpr, metadata: Mapping[str, Any] | None = None) -> TensorExpr:
    return TensorExpr(kind="contraction", children=tuple(children), metadata=dict(metadata or {}))


def ir_raise_lower(child: TensorExpr, *, mode: str, metadata: Mapping[str, Any] | None = None) -> TensorExpr:
    md = dict(metadata or {})
    md["mode"] = mode
    return TensorExpr(kind="raise_lower", children=(child,), metadata=md)


def ir_derivative(child: TensorExpr, *, operator: str = "covariant", metadata: Mapping[str, Any] | None = None) -> TensorExpr:
    md = dict(metadata or {})
    md["operator"] = operator
    return TensorExpr(kind="derivative", children=(child,), metadata=md)


def _ir_key(node: TensorExpr) -> tuple[Any, ...]:
    return (
        node.kind,
        repr(node.payload),
        tuple(sorted(node.metadata.items(), key=repr)),
        tuple(_ir_key(ch) for ch in node.children),
    )


def _normalize_ir_deep(node: TensorExpr) -> TensorExpr:
    children = tuple(_normalize_ir_deep(ch) for ch in node.children)
    if node.kind in {"indexed_expr:add", "indexed_expr:tensor_product", "indexed_expr:mul", "scalar:add", "scalar:mul", "contraction"}:
        children = tuple(sorted(children, key=_ir_key))
    return TensorExpr(kind=node.kind, payload=node.payload, children=children, metadata=dict(node.metadata))


def _rewrite_add_identities(node: TensorExpr) -> tuple[TensorExpr, bool]:
    if node.kind in {"scalar:add", "indexed_expr:add"} and len(node.children) == 1:
        return node.children[0], True
    return node, False


def _rewrite_mul_identities(node: TensorExpr) -> tuple[TensorExpr, bool]:
    if node.kind in {"scalar:mul", "indexed_expr:mul", "indexed_expr:tensor_product"} and len(node.children) == 1:
        return node.children[0], True
    return node, False


def _rewrite_nested_contraction(node: TensorExpr) -> tuple[TensorExpr, bool]:
    if node.kind != "contraction":
        return node, False
    new_children = []
    changed = False
    for ch in node.children:
        if ch.kind == "contraction":
            new_children.extend(ch.children)
            changed = True
        else:
            new_children.append(ch)
    if changed:
        return TensorExpr(kind="contraction", children=tuple(new_children), metadata=dict(node.metadata)), True
    return node, False


def _rewrite_raise_lower_cancel(node: TensorExpr) -> tuple[TensorExpr, bool]:
    if node.kind != "raise_lower" or len(node.children) != 1:
        return node, False
    child = node.children[0]
    if child.kind != "raise_lower":
        return node, False
    outer = node.metadata.get("mode")
    inner = child.metadata.get("mode")
    if {outer, inner} == {"raise", "lower"}:
        return child.children[0], True
    return node, False


def _rewrite_derivative_distribution(node: TensorExpr) -> tuple[TensorExpr, bool]:
    if node.kind != "derivative" or len(node.children) != 1:
        return node, False
    child = node.children[0]
    if child.kind not in {"scalar:add", "indexed_expr:add"}:
        return node, False
    new_children = tuple(
        TensorExpr(kind="derivative", children=(c,), metadata=dict(node.metadata))
        for c in child.children
    )
    return TensorExpr(kind=child.kind, children=new_children, metadata=dict(child.metadata)), True


IR_NATIVE_RULES: tuple[IRRewriteRule, ...] = (
    IRRewriteRule("flatten_nested_contraction", 100),
    IRRewriteRule("cancel_raise_lower_pair", 90),
    IRRewriteRule("distribute_derivative_over_add", 80),
    IRRewriteRule("collapse_singleton_add", 70),
    IRRewriteRule("collapse_singleton_mul", 60),
)


def _apply_rules_once(node: TensorExpr) -> tuple[TensorExpr, list[str]]:
    applied: list[str] = []

    node2, changed = _rewrite_nested_contraction(node)
    if changed:
        applied.append("flatten_nested_contraction")
        node = node2

    node2, changed = _rewrite_raise_lower_cancel(node)
    if changed:
        applied.append("cancel_raise_lower_pair")
        node = node2

    node2, changed = _rewrite_derivative_distribution(node)
    if changed:
        applied.append("distribute_derivative_over_add")
        node = node2

    node2, changed = _rewrite_add_identities(node)
    if changed:
        applied.append("collapse_singleton_add")
        node = node2

    node2, changed = _rewrite_mul_identities(node)
    if changed:
        applied.append("collapse_singleton_mul")
        node = node2

    return node, applied


def _rewrite_ir_recursive(node: TensorExpr) -> tuple[TensorExpr, tuple[str, ...]]:
    rewritten_children = []
    applied_all: list[str] = []
    for ch in node.children:
        new_ch, applied = _rewrite_ir_recursive(ch)
        rewritten_children.append(new_ch)
        applied_all.extend(applied)
    node = TensorExpr(kind=node.kind, payload=node.payload, children=tuple(rewritten_children), metadata=dict(node.metadata))
    node = _normalize_ir_deep(node)

    changed = True
    while changed:
        new_node, applied = _apply_rules_once(node)
        if applied:
            applied_all.extend(applied)
            node = _normalize_ir_deep(new_node)
        else:
            changed = False
    return node, tuple(applied_all)


def compile_tensor_expr_extended(obj: Any) -> TensorExpr:
    if isinstance(obj, TensorExpr):
        return obj
    return compile_tensor_expr(obj)


def canonicalize_tensor_expr(node: TensorExpr) -> TensorExpr:
    return _normalize_ir_deep(node)


def rewrite_tensor_expr(node: TensorExpr) -> tuple[TensorExpr, tuple[str, ...]]:
    node = canonicalize_tensor_expr(node)
    return _rewrite_ir_recursive(node)


def materialize_ir_native(node: TensorExpr) -> Any:
    if node.kind == "contraction":
        return {"op": "contraction", "children": tuple(materialize_ir_native(ch) for ch in node.children), "metadata": dict(node.metadata)}
    if node.kind == "raise_lower":
        return {"op": "raise_lower", "mode": node.metadata.get("mode"), "child": materialize_ir_native(node.children[0])}
    if node.kind == "derivative":
        return {"op": "derivative", "operator": node.metadata.get("operator"), "child": materialize_ir_native(node.children[0])}
    if node.kind == "scalar:symbol":
        return sp.Symbol(str(node.payload))
    if node.kind.startswith("scalar:") and node.kind not in {"scalar:add", "scalar:mul", "scalar:symbol"}:
        if node.payload is not None:
            try:
                return sp.sympify(node.payload)
            except Exception:
                return {"kind": node.kind, "payload": node.payload}
    if node.kind == "scalar:add":
        mats = [materialize_ir_native(ch) for ch in node.children]
        if all(isinstance(m, sp.Basic) for m in mats):
            return sp.Add(*mats)
        return {"op": "add", "children": tuple(mats)}
    if node.kind == "scalar:mul":
        mats = [materialize_ir_native(ch) for ch in node.children]
        if all(isinstance(m, sp.Basic) for m in mats):
            return sp.Mul(*mats)
        return {"op": "mul", "children": tuple(mats)}
    try:
        return materialize_tensor_expr(node)
    except Exception:
        return {"kind": node.kind, "payload": node.payload, "children": tuple(materialize_ir_native(ch) for ch in node.children), "metadata": dict(node.metadata)}


def execute_ir_native_rewriting(obj: Any) -> IRNativeRewriteReport:
    ir = compile_tensor_expr_extended(obj)
    normalized = canonicalize_tensor_expr(ir)
    rewritten, applied = rewrite_tensor_expr(normalized)
    materialized = materialize_ir_native(rewritten)
    return IRNativeRewriteReport(
        original=obj,
        input_ir=ir,
        normalized_ir=normalized,
        rewritten_ir=rewritten,
        materialized=materialized,
        applied_rules=applied,
        metadata={"rule_count": len(applied)},
    )
