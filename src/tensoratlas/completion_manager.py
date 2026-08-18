
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .conflict_priority_geometry_engine import (
    DEFAULT_EXTENDED_PRIORITY_RULES,
    PriorityRewriteRule,
    ordered_conflict_rules,
    conflict_aware_priority_reduce,
)
from .critical_pair_rewrite_engine import analyze_critical_pairs


@dataclass(frozen=True)
class RewriteStrategy:
    name: str
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionIssue:
    pair: tuple[str, str]
    kind: str
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalFormResult:
    strategy: str
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    blocked_rules: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionManagerReport:
    original: Any
    strategy: str
    default_normal_form: NormalFormResult
    alternate_normal_forms: tuple[NormalFormResult, ...]
    completion_issues: tuple[CompletionIssue, ...]
    confluence_agrees: bool
    provenance: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


DEFAULT_REWRITE_STRATEGIES: tuple[RewriteStrategy, ...] = (
    RewriteStrategy(
        name="priority_forward",
        description="Apply the ordered rule basis from highest priority to lowest priority.",
    ),
    RewriteStrategy(
        name="priority_reverse",
        description="Apply the ordered rule basis from lowest priority to highest priority.",
    ),
    RewriteStrategy(
        name="family_clustered",
        description="Apply rules grouped by family, with geometry-heavy families first.",
    ),
)


def family_clustered_rules(rules: Sequence[PriorityRewriteRule] | None = None) -> tuple[PriorityRewriteRule, ...]:
    rules = tuple(rules or DEFAULT_EXTENDED_PRIORITY_RULES)
    family_order = {"riemann": 0, "weyl": 1, "ricci": 2, "metric": 3, "epsilon": 4, "delta": 5}
    return tuple(sorted(rules, key=lambda r: (family_order.get(r.family, 99), -r.priority, r.normal_order_key, r.name)))


def strategy_rule_order(strategy: str, rules: Sequence[PriorityRewriteRule] | None = None) -> tuple[PriorityRewriteRule, ...]:
    if strategy == "priority_forward":
        return ordered_conflict_rules(rules or DEFAULT_EXTENDED_PRIORITY_RULES)
    if strategy == "priority_reverse":
        return tuple(reversed(ordered_conflict_rules(rules or DEFAULT_EXTENDED_PRIORITY_RULES)))
    if strategy == "family_clustered":
        return family_clustered_rules(rules or DEFAULT_EXTENDED_PRIORITY_RULES)
    raise ValueError(f"Unknown strategy: {strategy}")


def _normalize_terms_for_compare(terms: Sequence[tuple[sp.Expr, Any]]) -> tuple[tuple[str, str], ...]:
    out = []
    for coeff, term in terms:
        out.append((sp.srepr(sp.sympify(coeff)), repr(term)))
    return tuple(sorted(out))


def compute_normal_form(
    expr_or_terms: Any,
    *,
    strategy: str = "priority_forward",
    rules: Sequence[PriorityRewriteRule] | None = None,
) -> NormalFormResult:
    ordered = strategy_rule_order(strategy, rules)
    report = conflict_aware_priority_reduce(expr_or_terms, rules=ordered)
    return NormalFormResult(
        strategy=strategy,
        reduced_terms=report.reduced_terms,
        applied_rules=report.applied_rules,
        blocked_rules=report.blocked_rules,
        metadata={
            "iterations": report.iterations,
            "rule_order": report.rule_order,
        },
    )


def generate_completion_issues(rules: Sequence[PriorityRewriteRule] | None = None) -> tuple[CompletionIssue, ...]:
    analysis = analyze_critical_pairs(rules or DEFAULT_EXTENDED_PRIORITY_RULES)
    issues: list[CompletionIssue] = []
    for pair in analysis.pairs:
        status = "resolved-by-priority" if pair.is_conflicting else "needs-completion"
        issues.append(
            CompletionIssue(
                pair=(pair.rule_a, pair.rule_b),
                kind=pair.overlap_kind,
                status=status,
                metadata={
                    "shared_family": pair.shared_family,
                    **dict(pair.metadata),
                },
            )
        )
    return tuple(issues)


def completion_manager(
    expr_or_terms: Any,
    *,
    rules: Sequence[PriorityRewriteRule] | None = None,
    primary_strategy: str = "priority_forward",
) -> CompletionManagerReport:
    rules = tuple(rules or DEFAULT_EXTENDED_PRIORITY_RULES)
    issues = generate_completion_issues(rules)
    primary = compute_normal_form(expr_or_terms, strategy=primary_strategy, rules=rules)

    alternates: list[NormalFormResult] = []
    for strategy in ("priority_reverse", "family_clustered"):
        if strategy == primary_strategy:
            continue
        alternates.append(compute_normal_form(expr_or_terms, strategy=strategy, rules=rules))

    primary_nf = _normalize_terms_for_compare(primary.reduced_terms)
    confluence = all(_normalize_terms_for_compare(a.reduced_terms) == primary_nf for a in alternates)

    provenance = ["critical-pair-analysis", f"normal-form:{primary_strategy}"]
    provenance.extend(f"alternate-normal-form:{a.strategy}" for a in alternates)
    if confluence:
        provenance.append("all-normal-forms-agree")
    else:
        provenance.append("normal-form-divergence-detected")

    return CompletionManagerReport(
        original=expr_or_terms,
        strategy=primary_strategy,
        default_normal_form=primary,
        alternate_normal_forms=tuple(alternates),
        completion_issues=issues,
        confluence_agrees=confluence,
        provenance=tuple(provenance),
        metadata={
            "rule_count": len(rules),
            "issue_count": len(issues),
        },
    )
