
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .semantic_ir import TensorExpr, TensorExpr, canonical_ir_key, rewrite_with_provenance
from .ir_rewriting import rewrite_tensor_expr, materialize_ir_native
from .ir_curvature_rewriting import execute_ir_curvature_rewriting
from .ir_curvature_completion import compile_indexed_tensor_to_curvature_ir, rewrite_operator_ir_node


@dataclass(frozen=True)
class CurvatureNormalFormMetadata:
    preferred_basis: str
    decomposition_rank: int
    contraction_depth: int
    symmetry_signature: tuple[Any, ...]
    target_families: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurvatureNormalFormReport:
    original: Any
    normalized_ir: TensorExpr
    normal_form_key: tuple[Any, ...]
    normal_form_metadata: CurvatureNormalFormMetadata
    materialized: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexedTensorExprCanonicalizationReport:
    original: Any
    ir: TensorExpr
    canonical_ir: TensorExpr
    canonical_key: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OverlapWitness:
    family_a: str
    family_b: str
    witness_ir: TensorExpr
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleFamilyDiagnostic:
    family: str
    rule_names: tuple[str, ...]
    overlap_count: int
    unresolved_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdvancedCompletionWorkflowReport:
    original: Any
    primary_rewritten_ir: TensorExpr
    alternate_rewritten_ir: TensorExpr
    overlap_witnesses: tuple[OverlapWitness, ...]
    family_diagnostics: tuple[RuleFamilyDiagnostic, ...]
    confluence_agrees: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticOperatorInteractionReport:
    original: Any
    rewritten_ir: TensorExpr
    applied_rules: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _symmetry_signature_from_metadata(md: Mapping[str, Any]) -> tuple[Any, ...]:
    sym = md.get("symmetry_metadata", {}) or {}
    return tuple(sorted(sym.items(), key=repr))


def _curvature_nf_key(node: TensorExpr) -> tuple[Any, ...]:
    """Curvature normal-form key derived from the canonical TensorExpr key."""
    return canonical_ir_key(node)


def _annotate_curvature_metadata(node: TensorExpr) -> TensorExpr:
    children = tuple(_annotate_curvature_metadata(ch) for ch in node.children)
    md = dict(node.metadata)
    if node.kind in {"curvature_symbol", "curvature"}:
        fam = md.get("family", "")
        md.setdefault("preferred_basis", "curvature_family")
        md.setdefault("target_family", fam)
        md.setdefault("decomposition_rank", 0 if fam in {"ScalarCurvature"} else (1 if fam in {"Ricci", "Einstein", "Metric"} else 2))
        md.setdefault("contraction_depth", 0)
    elif node.kind == "curvature_contraction":
        md.setdefault("preferred_basis", "contracted_curvature")
        md["decomposition_rank"] = 1
        md["contraction_depth"] = 1 + sum(ch.metadata.get("contraction_depth", 0) for ch in children)
    elif node.kind == "curvature_decomposition":
        md.setdefault("preferred_basis", "decomposed_curvature")
        md["decomposition_rank"] = 2
        md["contraction_depth"] = sum(ch.metadata.get("contraction_depth", 0) for ch in children)
    elif node.kind == "curvature_linear_combo":
        md.setdefault("preferred_basis", "linear_combo")
        md["decomposition_rank"] = max([ch.metadata.get("decomposition_rank", 0) for ch in children] or [0])
        md["contraction_depth"] = max([ch.metadata.get("contraction_depth", 0) for ch in children] or [0])
        targets = []
        for ch in children:
            tf = ch.metadata.get("target_family") or ch.metadata.get("family")
            if tf is not None:
                targets.append(tf)
        if targets:
            md["target_families"] = tuple(sorted(set(targets)))
    elif node.kind == "indexed_tensor":
        sym = md.get("symmetry_metadata", {}) or {}
        fam = None
        tname = str(md.get("tensor_name", ""))
        if sym.get("riemann") or tname.lower() in {"r", "riemann"}:
            fam = "Riemann"
        elif sym.get("ricci_symmetric") or "ricci" in tname.lower():
            fam = "Ricci"
        elif sym.get("weyl") or tname.lower() in {"c", "weyl"}:
            fam = "Weyl"
        elif sym.get("metric") or tname.lower() in {"g", "metric"}:
            fam = "Metric"
        if fam is not None:
            md["family"] = fam
            md.setdefault("preferred_basis", "indexed_curvature")
            md.setdefault("target_family", fam)
            md.setdefault("decomposition_rank", 0)
            md.setdefault("contraction_depth", 0)
    return TensorExpr(kind=node.kind, payload=node.payload, children=children, metadata=md, provenance=node.provenance)


def canonicalize_curvature_ir_node(node: TensorExpr) -> TensorExpr:
    children = tuple(canonicalize_curvature_ir_node(ch) for ch in node.children)
    if node.kind in {
        "curvature_linear_combo", "indexed_expr:add", "indexed_expr:tensor_product", "indexed_expr:mul",
        "scalar:add", "scalar:mul", "contraction", "wedge"
    }:
        children = tuple(sorted(children, key=_curvature_nf_key))
    rebuilt = TensorExpr(kind=node.kind, payload=node.payload, children=children, metadata=dict(node.metadata), provenance=node.provenance)
    return rewrite_with_provenance(node, rebuilt, rule="canonicalize_curvature_ir_node", source="curvature_normal_forms")


def canonicalize_indexed_tensor_expr(obj: Any) -> IndexedTensorExprCanonicalizationReport:
    ir = compile_indexed_tensor_to_curvature_ir(obj)
    annotated = _annotate_curvature_metadata(ir)
    canon = canonicalize_curvature_ir_node(annotated)
    return IndexedTensorExprCanonicalizationReport(
        original=obj,
        ir=annotated,
        canonical_ir=canon,
        canonical_key=_curvature_nf_key(canon),
        metadata={"ir_kind": canon.kind},
    )


def curvature_normal_form(obj: Any) -> CurvatureNormalFormReport:
    ir = compile_indexed_tensor_to_curvature_ir(obj)
    annotated = _annotate_curvature_metadata(ir)
    generic, _ = rewrite_tensor_expr(annotated)
    curv = execute_ir_curvature_rewriting(generic)
    canon = canonicalize_curvature_ir_node(_annotate_curvature_metadata(curv.rewritten_ir))
    md = canon.metadata
    target_families = tuple(md.get("target_families", ()))
    if not target_families and md.get("target_family") is not None:
        target_families = (md.get("target_family"),)
    nfmd = CurvatureNormalFormMetadata(
        preferred_basis=md.get("preferred_basis", "unknown"),
        decomposition_rank=int(md.get("decomposition_rank", 0)),
        contraction_depth=int(md.get("contraction_depth", 0)),
        symmetry_signature=_symmetry_signature_from_metadata(md),
        target_families=target_families,
        metadata={"ir_kind": canon.kind},
    )
    return CurvatureNormalFormReport(
        original=obj,
        normalized_ir=canon,
        normal_form_key=_curvature_nf_key(canon),
        normal_form_metadata=nfmd,
        materialized=materialize_ir_native(canon),
        metadata={},
    )


def _build_overlap_witnesses(primary: TensorExpr, alternate: TensorExpr) -> tuple[OverlapWitness, ...]:
    witnesses = []
    if _curvature_nf_key(primary) != _curvature_nf_key(alternate):
        witnesses.append(
            OverlapWitness(
                family_a=str(primary.metadata.get("family") or primary.kind),
                family_b=str(alternate.metadata.get("family") or alternate.kind),
                witness_ir=primary,
                reason="normal forms diverged under alternate rewrite order",
                metadata={"alternate_kind": alternate.kind},
            )
        )
    fams = []
    for ch in primary.children:
        fam = str(ch.metadata.get("family") or ch.metadata.get("target_family") or ch.kind)
        if fam not in fams:
            fams.append(fam)
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            witnesses.append(
                OverlapWitness(
                    family_a=fams[i],
                    family_b=fams[j],
                    witness_ir=primary,
                    reason="coexisting families in canonicalized curvature expression",
                    metadata={},
                )
            )
    return tuple(witnesses)


def _family_diagnostics(primary: TensorExpr, witnesses: tuple[OverlapWitness, ...]) -> tuple[RuleFamilyDiagnostic, ...]:
    families = {}
    seed_fams = [str(primary.metadata.get("family") or primary.kind)] + [str(ch.metadata.get("family") or ch.metadata.get("target_family") or ch.kind) for ch in primary.children]
    for fam in seed_fams:
        if fam not in families:
            families[fam] = {"rules": [], "overlap": 0, "unresolved": 0}
    for w in witnesses:
        for fam in (w.family_a, w.family_b):
            families.setdefault(fam, {"rules": [], "overlap": 0, "unresolved": 0})
            families[fam]["overlap"] += 1
            if "diverged" in w.reason:
                families[fam]["unresolved"] += 1
    out = []
    for fam, stats in sorted(families.items()):
        out.append(RuleFamilyDiagnostic(
            family=fam,
            rule_names=tuple(sorted(set(stats["rules"]))),
            overlap_count=int(stats["overlap"]),
            unresolved_count=int(stats["unresolved"]),
            metadata={},
        ))
    return tuple(out)


def execute_advanced_completion_workflow(obj: Any) -> AdvancedCompletionWorkflowReport:
    ir = canonicalize_curvature_ir_node(_annotate_curvature_metadata(compile_indexed_tensor_to_curvature_ir(obj)))
    generic1, _ = rewrite_tensor_expr(ir)
    curv1 = execute_ir_curvature_rewriting(generic1)
    op1, _ = rewrite_operator_ir_node(curv1.rewritten_ir)
    primary = canonicalize_curvature_ir_node(_annotate_curvature_metadata(op1))

    op2, _ = rewrite_operator_ir_node(ir)
    generic2, _ = rewrite_tensor_expr(op2)
    curv2 = execute_ir_curvature_rewriting(generic2)
    alternate = canonicalize_curvature_ir_node(_annotate_curvature_metadata(curv2.rewritten_ir))

    witnesses = _build_overlap_witnesses(primary, alternate)
    diags = _family_diagnostics(primary, witnesses)
    agree = _curvature_nf_key(primary) == _curvature_nf_key(alternate)

    return AdvancedCompletionWorkflowReport(
        original=obj,
        primary_rewritten_ir=primary,
        alternate_rewritten_ir=alternate,
        overlap_witnesses=witnesses,
        family_diagnostics=diags,
        confluence_agrees=agree,
        metadata={
            "primary_materialized": materialize_ir_native(primary),
            "alternate_materialized": materialize_ir_native(alternate),
        },
    )


def _rewrite_metric_curvature_operator_once(node: TensorExpr):
    if node.kind == "derivative" and len(node.children) == 1:
        child = node.children[0]
        fam = child.metadata.get("family")
        if fam == "Metric":
            return TensorExpr(kind="metric_connection_derivative", children=(child,), metadata=dict(node.metadata)), ("metric_derivative_to_connection_semantics",)
        if fam in {"Riemann", "Ricci", "Weyl", "Einstein", "ScalarCurvature"}:
            return TensorExpr(kind="curvature_connection_derivative", children=(child,), metadata=dict(node.metadata)), ("curvature_derivative_to_connection_semantics",)

    if node.kind == "hodge" and len(node.children) == 1 and node.children[0].kind == "wedge":
        return TensorExpr(kind="hodge_wedge_semantic", children=node.children[0].children, metadata=dict(node.metadata)), ("hodge_wedge_semantic_lift",)

    if node.kind == "lie" and len(node.children) == 1 and node.children[0].metadata.get("family") in {"Riemann", "Ricci", "Weyl", "Einstein", "ScalarCurvature"}:
        return TensorExpr(kind="curvature_lie_semantic", children=node.children, metadata=dict(node.metadata)), ("curvature_lie_semantic_lift",)

    if node.kind == "interior" and len(node.children) == 1 and node.children[0].kind == "wedge":
        return TensorExpr(kind="interior_wedge_semantic", children=node.children[0].children, metadata=dict(node.metadata)), ("interior_wedge_semantic_lift",)

    if node.kind == "wedge" and node.children and all(ch.kind == "interior" for ch in node.children):
        return TensorExpr(kind="interior_wedge_semantic", children=tuple(ch.children[0] for ch in node.children), metadata=dict(node.metadata)), ("interior_wedge_semantic_lift",)

    return node, tuple()


def _rewrite_semantic_ops_recursive(node: TensorExpr):
    children = []
    applied_all = []
    for ch in node.children:
        nch, appl = _rewrite_semantic_ops_recursive(ch)
        children.append(nch)
        applied_all.extend(appl)
    node = canonicalize_curvature_ir_node(TensorExpr(kind=node.kind, payload=node.payload, children=tuple(children), metadata=dict(node.metadata), provenance=node.provenance))
    changed = True
    while changed:
        node2, appl = _rewrite_metric_curvature_operator_once(node)
        if appl:
            applied_all.extend(appl)
            node = canonicalize_curvature_ir_node(_annotate_curvature_metadata(node2))
        else:
            changed = False
    return node, tuple(applied_all)


def execute_semantic_operator_interactions(obj: Any) -> SemanticOperatorInteractionReport:
    ir = canonicalize_curvature_ir_node(_annotate_curvature_metadata(compile_indexed_tensor_to_curvature_ir(obj)))
    basic, applied_basic = rewrite_operator_ir_node(ir)
    semantic, applied_sem = _rewrite_semantic_ops_recursive(basic)
    return SemanticOperatorInteractionReport(
        original=obj,
        rewritten_ir=semantic,
        applied_rules=tuple(applied_basic) + tuple(applied_sem),
        metadata={"materialized": materialize_ir_native(semantic)},
    )
