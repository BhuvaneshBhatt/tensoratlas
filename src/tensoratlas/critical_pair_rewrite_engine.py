
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .semantic_core import (
    SemanticNode,
    compile_semantic_node,
    normalize_semantic_node,
    semantic_node_fingerprint,
)
from .conflict_priority_geometry_engine import (
    PriorityRewriteRule,
    DEFAULT_EXTENDED_PRIORITY_RULES,
    conflict_aware_priority_reduce,
    ordered_conflict_rules,
)


@dataclass(frozen=True)
class CriticalPairReport:
    rule_a: str
    rule_b: str
    overlap_kind: str
    shared_family: str
    is_conflicting: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CriticalPairAnalysisReport:
    rules_considered: tuple[str, ...]
    pairs: tuple[CriticalPairReport, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticNativeIndexedNormalizationReport:
    original: Any
    semantic_node_kind: str
    semantic_fingerprint: tuple[Any, ...]
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _family_overlap(rule_a: PriorityRewriteRule, rule_b: PriorityRewriteRule) -> tuple[str, str, bool]:
    fa = rule_a.family
    fb = rule_b.family
    conflicts_a = tuple(rule_a.metadata.get("conflicts_with", ()))
    conflicts_b = tuple(rule_b.metadata.get("conflicts_with", ()))
    if fa == fb:
        return fa, "same_family", True
    if rule_b.name in conflicts_a or rule_a.name in conflicts_b:
        return fa if fa == fb else f"{fa}/{fb}", "declared_conflict", True
    if fa in {"riemann", "ricci", "metric", "weyl", "delta"} and fb in {"riemann", "ricci", "metric", "weyl", "delta"}:
        return f"{fa}/{fb}", "tensor_identity_family", False
    return f"{fa}/{fb}", "disjoint", False


def analyze_critical_pairs(rules: Sequence[PriorityRewriteRule] | None = None) -> CriticalPairAnalysisReport:
    rules_ord = ordered_conflict_rules(rules or DEFAULT_EXTENDED_PRIORITY_RULES)
    pairs: list[CriticalPairReport] = []
    for i, ra in enumerate(rules_ord):
        for rb in rules_ord[i + 1:]:
            fam, kind, conflict = _family_overlap(ra, rb)
            pairs.append(CriticalPairReport(
                rule_a=ra.name,
                rule_b=rb.name,
                overlap_kind=kind,
                shared_family=fam,
                is_conflicting=conflict,
                metadata={
                    "priority_gap": abs(ra.priority - rb.priority),
                    "order_a": ra.normal_order_key,
                    "order_b": rb.normal_order_key,
                },
            ))
    return CriticalPairAnalysisReport(
        rules_considered=tuple(r.name for r in rules_ord),
        pairs=tuple(pairs),
        metadata={"pair_count": len(pairs)},
    )


def _semantic_native_linear_terms(obj: Any) -> list[tuple[sp.Expr, Any]]:
    node = normalize_semantic_node(compile_semantic_node(obj))
    # stay as semantic-core-native as long as possible, only materialize leaf-level children
    if node.kind == "add":
        out = []
        for ch in node.children:
            out.append((sp.Integer(1), ch))
        return out
    return [(sp.Integer(1), node)]


def semantic_native_indexed_geometry_reduce(
    obj: Any,
    *,
    rules: Sequence[PriorityRewriteRule] | None = None,
    subsystem: str = "indexed_geometry_semantic_native",
):
    node = normalize_semantic_node(compile_semantic_node(obj))
    weighted = _semantic_native_linear_terms(obj)
    # reuse conflict-aware reducer, but feed normalized semantic children when possible
    report = conflict_aware_priority_reduce(weighted, rules=rules)
    return SemanticNativeIndexedNormalizationReport(
        original=obj,
        semantic_node_kind=node.kind,
        semantic_fingerprint=semantic_node_fingerprint(node),
        reduced_terms=report.reduced_terms,
        applied_rules=report.applied_rules,
        metadata={
            "subsystem": subsystem,
            "blocked_rules": report.blocked_rules,
            "iterations": report.iterations,
        },
    )
