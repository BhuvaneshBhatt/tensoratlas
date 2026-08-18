
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .semantic_ir import TensorExpr
from .ir_rewriting import (
    IRRewriteRule,
    rewrite_tensor_expr,
    materialize_ir_native,
)


@dataclass(frozen=True)
class IRCurvatureRewriteReport:
    original: Any
    input_ir: TensorExpr
    normalized_ir: TensorExpr
    rewritten_ir: TensorExpr
    materialized: Any
    applied_rules: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def ir_curvature_symbol(family: str, dimension: int, *, name: str | None = None, metadata: Mapping[str, Any] | None = None) -> TensorExpr:
    md = dict(metadata or {})
    md.update({"family": family, "dimension": int(dimension), "name": name or family})
    return TensorExpr(kind="curvature_symbol", metadata=md)


def ir_curvature_linear_combo(*children: TensorExpr, metadata: Mapping[str, Any] | None = None) -> TensorExpr:
    return TensorExpr(kind="curvature_linear_combo", children=tuple(children), metadata=dict(metadata or {}))


def ir_curvature_contraction(child: TensorExpr, *, target_family: str, metadata: Mapping[str, Any] | None = None) -> TensorExpr:
    md = dict(metadata or {})
    md["target_family"] = target_family
    return TensorExpr(kind="curvature_contraction", children=(child,), metadata=md)


def ir_curvature_decomposition(child: TensorExpr, *, target_family: str, metadata: Mapping[str, Any] | None = None) -> TensorExpr:
    md = dict(metadata or {})
    md["target_family"] = target_family
    return TensorExpr(kind="curvature_decomposition", children=(child,), metadata=md)


def _ir_curvature_key(node: TensorExpr):
    return (
        node.kind,
        tuple(sorted(node.metadata.items(), key=repr)),
        tuple(_ir_curvature_key(ch) for ch in node.children),
    )


def _normalize_curvature_ir_deep(node: TensorExpr) -> TensorExpr:
    children = tuple(_normalize_curvature_ir_deep(ch) for ch in node.children)
    if node.kind in {"curvature_linear_combo", "contraction", "indexed_expr:add", "indexed_expr:tensor_product", "indexed_expr:mul", "scalar:add", "scalar:mul"}:
        children = tuple(sorted(children, key=_ir_curvature_key))
    return TensorExpr(kind=node.kind, payload=node.payload, children=children, metadata=dict(node.metadata))


def _is_curvature_family(node: TensorExpr, family: str) -> bool:
    return node.kind == "curvature_symbol" and node.metadata.get("family") == family


def _rewrite_riemann_to_ricci_contraction(node: TensorExpr):
    if not _is_curvature_family(node, "Riemann"):
        return node, False
    return ir_curvature_contraction(node, target_family="Ricci"), True


def _rewrite_ricci_to_scalar_contraction(node: TensorExpr):
    if not _is_curvature_family(node, "Ricci"):
        return node, False
    return ir_curvature_contraction(node, target_family="ScalarCurvature"), True


def _rewrite_ricci_to_einstein_decomposition(node: TensorExpr):
    if not _is_curvature_family(node, "Ricci"):
        return node, False
    dim = int(node.metadata.get("dimension", 0))
    return ir_curvature_linear_combo(
        ir_curvature_symbol("Einstein", dim, name="Einstein"),
        ir_curvature_decomposition(node, target_family="MetricScalarPart"),
    ), True


def _rewrite_riemann_to_weyl_ricci_scalar(node: TensorExpr):
    if not _is_curvature_family(node, "Riemann"):
        return node, False
    dim = int(node.metadata.get("dimension", 0))
    return ir_curvature_linear_combo(
        ir_curvature_symbol("Weyl", dim, name="Weyl"),
        ir_curvature_decomposition(node, target_family="RicciPart"),
        ir_curvature_decomposition(node, target_family="ScalarPart"),
    ), True


def _rewrite_curvature_singleton_combo(node: TensorExpr):
    if node.kind == "curvature_linear_combo" and len(node.children) == 1:
        return node.children[0], True
    return node, False


IR_CURVATURE_RULES: tuple[IRRewriteRule, ...] = (
    IRRewriteRule("ir_rewrite_riemann_to_ricci_contraction", 140, {"orientation": "lower_rank", "terminating": True}),
    IRRewriteRule("ir_rewrite_ricci_to_scalar_contraction", 130, {"orientation": "lower_rank", "terminating": True}),
    IRRewriteRule("ir_rewrite_ricci_to_einstein_decomposition", 120, {"orientation": "decompose", "terminating": True}),
    IRRewriteRule("ir_rewrite_riemann_to_weyl_ricci_scalar_decomposition", 110, {"orientation": "decompose", "terminating": True}),
    IRRewriteRule("ir_rewrite_curvature_singleton_combo", 100, {"orientation": "simplify", "terminating": True}),
)


def _apply_curvature_rules_once(node: TensorExpr):
    applied: list[str] = []
    for name, fn in (
        ("ir_rewrite_riemann_to_ricci_contraction", _rewrite_riemann_to_ricci_contraction),
        ("ir_rewrite_ricci_to_scalar_contraction", _rewrite_ricci_to_scalar_contraction),
        ("ir_rewrite_ricci_to_einstein_decomposition", _rewrite_ricci_to_einstein_decomposition),
        ("ir_rewrite_riemann_to_weyl_ricci_scalar_decomposition", _rewrite_riemann_to_weyl_ricci_scalar),
        ("ir_rewrite_curvature_singleton_combo", _rewrite_curvature_singleton_combo),
    ):
        node2, changed = fn(node)
        if changed:
            applied.append(name)
            node = node2
    return node, applied


def _rewrite_ir_curvature_recursive(node: TensorExpr):
    rewritten_children = []
    applied_all: list[str] = []
    for ch in node.children:
        new_ch, applied = _rewrite_ir_curvature_recursive(ch)
        rewritten_children.append(new_ch)
        applied_all.extend(applied)
    node = TensorExpr(kind=node.kind, payload=node.payload, children=tuple(rewritten_children), metadata=dict(node.metadata))
    node = _normalize_curvature_ir_deep(node)
    changed = True
    while changed:
        new_node, applied = _apply_curvature_rules_once(node)
        if applied:
            applied_all.extend(applied)
            node = _normalize_curvature_ir_deep(new_node)
        else:
            changed = False
    return node, tuple(applied_all)


def compile_curvature_symbol_to_ir(obj: Any) -> TensorExpr:
    if isinstance(obj, TensorExpr):
        return obj
    fam = getattr(obj, "family", None)
    dim = getattr(obj, "dimension", None)
    name = getattr(obj, "name", None)
    if fam is not None and dim is not None:
        return ir_curvature_symbol(str(fam), int(dim), name=name)
    raise TypeError("Expected a TensorExpr curvature object or a curvature-like object with family/dimension attributes.")


def execute_ir_curvature_rewriting(obj: Any) -> IRCurvatureRewriteReport:
    ir = compile_curvature_symbol_to_ir(obj)
    normalized = _normalize_curvature_ir_deep(ir)
    generic_rewritten, generic_applied = rewrite_tensor_expr(normalized)
    rewritten, curvature_applied = _rewrite_ir_curvature_recursive(generic_rewritten)
    materialized = materialize_ir_native(rewritten)
    return IRCurvatureRewriteReport(
        original=obj,
        input_ir=ir,
        normalized_ir=normalized,
        rewritten_ir=rewritten,
        materialized=materialized,
        applied_rules=tuple(generic_applied) + tuple(curvature_applied),
        metadata={"generic_rule_count": len(generic_applied), "curvature_rule_count": len(curvature_applied)},
    )
