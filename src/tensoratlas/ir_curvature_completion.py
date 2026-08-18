
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .semantic_ir import TensorExpr
from .ir_rewriting import rewrite_tensor_expr, materialize_ir_native
from .ir_curvature_rewriting import execute_ir_curvature_rewriting


@dataclass(frozen=True)
class IRCurvatureCanonicalizationReport:
    original: Any
    canonical_key: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IRCompletionWorkflowReport:
    original: Any
    primary_rewritten_ir: TensorExpr
    alternate_rewritten_ir: TensorExpr
    confluence_agrees: bool
    applied_rules_primary: tuple[str, ...]
    applied_rules_alternate: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IROperatorRewriteReport:
    original: Any
    rewritten_ir: TensorExpr
    applied_rules: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def ir_wedge(*children: TensorExpr, metadata: Mapping[str, Any] | None = None) -> TensorExpr:
    return TensorExpr(kind="wedge", children=tuple(children), metadata=dict(metadata or {}))


def ir_hodge(child: TensorExpr, metadata: Mapping[str, Any] | None = None) -> TensorExpr:
    return TensorExpr(kind="hodge", children=(child,), metadata=dict(metadata or {}))


def ir_lie(child: TensorExpr, *, vector_name: str = "X", metadata: Mapping[str, Any] | None = None) -> TensorExpr:
    md = dict(metadata or {})
    md["vector_name"] = vector_name
    return TensorExpr(kind="lie", children=(child,), metadata=md)


def ir_interior(child: TensorExpr, *, vector_name: str = "X", metadata: Mapping[str, Any] | None = None) -> TensorExpr:
    md = dict(metadata or {})
    md["vector_name"] = vector_name
    return TensorExpr(kind="interior", children=(child,), metadata=md)


def _curvature_meta_signature(node: TensorExpr):
    md = dict(node.metadata)
    family = md.get("family")
    dim = md.get("dimension")
    name = md.get("name")
    target = md.get("target_family")
    tensor_name = md.get("tensor_name")
    variance = md.get("variance_spec")
    indices = tuple(md.get("indices", ()))
    sym = tuple(sorted((md.get("symmetry_metadata", {}) or {}).items()))
    return (family, dim, name, target, tensor_name, variance, indices, sym)


def _ir_deep_key(node: TensorExpr):
    special = _curvature_meta_signature(node)
    if node.kind in {
        "curvature_linear_combo", "indexed_expr:add", "indexed_expr:tensor_product", "indexed_expr:mul",
        "scalar:add", "scalar:mul", "contraction", "wedge"
    }:
        child_keys = tuple(sorted((_ir_deep_key(ch) for ch in node.children), key=repr))
    else:
        child_keys = tuple(_ir_deep_key(ch) for ch in node.children)
    return (node.kind, special, child_keys, tuple(sorted(node.metadata.items(), key=repr)))


def canonicalize_curvature_ir_node(node: TensorExpr) -> TensorExpr:
    children = tuple(canonicalize_curvature_ir_node(ch) for ch in node.children)
    if node.kind in {
        "curvature_linear_combo", "indexed_expr:add", "indexed_expr:tensor_product", "indexed_expr:mul",
        "scalar:add", "scalar:mul", "contraction", "wedge"
    }:
        children = tuple(sorted(children, key=_ir_deep_key))
    return TensorExpr(kind=node.kind, payload=node.payload, children=children, metadata=dict(node.metadata))


def compile_indexed_tensor_to_curvature_ir(obj: Any) -> TensorExpr:
    if isinstance(obj, TensorExpr):
        return obj
    cls = type(obj).__name__
    if cls == "IndexedTensor":
        t = getattr(obj, "tensor", None)
        md = dict(getattr(t, "symmetry_metadata", {}) or {})
        name = getattr(t, "name", "")
        variance = getattr(t, "variance_spec", "")
        indices = tuple((getattr(i, "name", str(i)), getattr(i, "variance", "")) for i in getattr(obj, "indices", ()))
        family = None
        if md.get("riemann") or name.lower() in {"r", "riemann"}:
            family = "Riemann"
        elif md.get("ricci_symmetric") or "ricci" in name.lower():
            family = "Ricci"
        elif md.get("weyl") or name.lower() in {"c", "weyl"}:
            family = "Weyl"
        elif md.get("metric") or name.lower() in {"g", "metric"}:
            family = "Metric"
        if family is not None:
            return TensorExpr(
                kind="curvature_symbol",
                metadata={
                    "family": family,
                    "dimension": max(len(indices), 1),
                    "name": name or family,
                    "tensor_name": name,
                    "variance_spec": variance,
                    "indices": indices,
                    "symmetry_metadata": md,
                },
            )
        return TensorExpr(
            kind="indexed_tensor",
            metadata={
                "tensor_name": name,
                "variance_spec": variance,
                "indices": indices,
                "symmetry_metadata": md,
            },
        )
    if cls == "IndexedTensorExpr":
        op = getattr(obj, "op", None)
        return TensorExpr(
            kind=f"indexed_expr:{op}",
            children=tuple(compile_indexed_tensor_to_curvature_ir(a) for a in getattr(obj, "args", ())),
            metadata={"op": op},
        )
    fam = getattr(obj, "family", None)
    dim = getattr(obj, "dimension", None)
    name = getattr(obj, "name", None)
    if fam is not None and dim is not None:
        return TensorExpr(kind="curvature_symbol", metadata={"family": str(fam), "dimension": int(dim), "name": name or fam})
    raise TypeError("Expected indexed tensor/expression, curvature-like object, or TensorExpr.")


def curvature_ir_canonicalization_report(obj: Any) -> IRCurvatureCanonicalizationReport:
    ir = compile_indexed_tensor_to_curvature_ir(obj)
    canon = canonicalize_curvature_ir_node(ir)
    return IRCurvatureCanonicalizationReport(
        original=obj,
        canonical_key=_ir_deep_key(canon),
        metadata={"ir_kind": canon.kind},
    )


def _rewrite_operator_once(node: TensorExpr):
    if node.kind == "hodge" and len(node.children) == 1 and node.children[0].kind == "hodge":
        return node.children[0].children[0], ("cancel_double_hodge",)

    if node.kind == "wedge" and len(node.children) == 1:
        return node.children[0], ("collapse_singleton_wedge",)

    if node.kind == "derivative" and len(node.children) == 1 and node.children[0].kind in {"wedge", "scalar:add", "indexed_expr:add", "curvature_linear_combo"}:
        child = node.children[0]
        new_children = tuple(TensorExpr(kind="derivative", children=(c,), metadata=dict(node.metadata)) for c in child.children)
        return TensorExpr(kind=child.kind, children=new_children, metadata=dict(child.metadata)), ("distribute_derivative_over_ir_operator",)

    if node.kind == "lie" and len(node.children) == 1 and node.children[0].kind == "wedge":
        child = node.children[0]
        new_children = tuple(TensorExpr(kind="lie", children=(c,), metadata=dict(node.metadata)) for c in child.children)
        return TensorExpr(kind="wedge", children=new_children, metadata=dict(child.metadata)), ("distribute_lie_over_wedge",)

    if node.kind == "interior" and len(node.children) == 1 and node.children[0].kind == "wedge":
        child = node.children[0]
        new_children = tuple(TensorExpr(kind="interior", children=(c,), metadata=dict(node.metadata)) for c in child.children)
        return TensorExpr(kind="wedge", children=new_children, metadata=dict(child.metadata)), ("distribute_interior_over_wedge",)

    return node, tuple()


def rewrite_operator_ir_node(node: TensorExpr):
    children = []
    applied_all = []
    for ch in node.children:
        nch, appl = rewrite_operator_ir_node(ch)
        children.append(nch)
        applied_all.extend(appl)
    node = canonicalize_curvature_ir_node(TensorExpr(kind=node.kind, payload=node.payload, children=tuple(children), metadata=dict(node.metadata)))
    changed = True
    while changed:
        node2, appl = _rewrite_operator_once(node)
        if appl:
            applied_all.extend(appl)
            node = canonicalize_curvature_ir_node(node2)
        else:
            changed = False
    return node, tuple(applied_all)


def execute_ir_completion_workflow(obj: Any) -> IRCompletionWorkflowReport:
    ir = canonicalize_curvature_ir_node(compile_indexed_tensor_to_curvature_ir(obj))

    generic1, applied_g1 = rewrite_tensor_expr(ir)
    curv_rep1 = execute_ir_curvature_rewriting(generic1)
    op1, applied_o1 = rewrite_operator_ir_node(curv_rep1.rewritten_ir)
    primary = canonicalize_curvature_ir_node(op1)

    op2, applied_o2 = rewrite_operator_ir_node(ir)
    generic2, applied_g2 = rewrite_tensor_expr(op2)
    curv_rep2 = execute_ir_curvature_rewriting(generic2)
    alternate = canonicalize_curvature_ir_node(curv_rep2.rewritten_ir)

    agree = _ir_deep_key(primary) == _ir_deep_key(alternate)

    return IRCompletionWorkflowReport(
        original=obj,
        primary_rewritten_ir=primary,
        alternate_rewritten_ir=alternate,
        confluence_agrees=agree,
        applied_rules_primary=tuple(applied_g1) + tuple(curv_rep1.applied_rules) + tuple(applied_o1),
        applied_rules_alternate=tuple(applied_o2) + tuple(applied_g2) + tuple(curv_rep2.applied_rules),
        metadata={
            "primary_materialized": materialize_ir_native(primary),
            "alternate_materialized": materialize_ir_native(alternate),
        },
    )


def execute_ir_operator_rewriting(obj: Any) -> IROperatorRewriteReport:
    ir = canonicalize_curvature_ir_node(compile_indexed_tensor_to_curvature_ir(obj))
    rewritten, applied = rewrite_operator_ir_node(ir)
    return IROperatorRewriteReport(
        original=obj,
        rewritten_ir=rewritten,
        applied_rules=applied,
        metadata={"materialized": materialize_ir_native(rewritten)},
    )
