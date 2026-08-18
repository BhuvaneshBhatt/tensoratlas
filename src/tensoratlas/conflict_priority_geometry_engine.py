
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .semantic_core import compile_semantic_node, normalize_semantic_node, materialize_semantic_node, semantic_node_fingerprint
from .priority_rewrite_engine import (
    PriorityRewriteRule,
    rewrite_term_key,
    normalize_linear_combination,
)


@dataclass(frozen=True)
class RewriteConflictPolicy:
    name: str
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConflictAwareRewriteReport:
    original: Any
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    blocked_rules: tuple[str, ...]
    iterations: int
    rule_order: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexedGeometryNormalizationReport:
    original: Any
    semantic_fingerprint: tuple[Any, ...]
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _tensor_name(term: Any) -> str:
    return getattr(getattr(term, "tensor", None), "name", "") or ""


def _tensor_md(term: Any) -> Mapping[str, Any]:
    return getattr(getattr(term, "tensor", None), "symmetry_metadata", {}) or {}


def _index_names(term: Any) -> tuple[str, ...]:
    return tuple(getattr(idx, "name", str(idx)) for idx in getattr(term, "indices", ()))


def _flatten_add(expr: Any) -> list[Any]:
    if type(expr).__name__ == "IndexedTensorExpr" and getattr(expr, "op", None) == "add":
        out = []
        for a in getattr(expr, "args", ()):
            out.extend(_flatten_add(a))
        return out
    return [expr]


def _weighted_terms(expr_or_terms: Any) -> list[tuple[sp.Expr, Any]]:
    if isinstance(expr_or_terms, Sequence) and not isinstance(expr_or_terms, (str, bytes)):
        out = []
        for item in expr_or_terms:
            if isinstance(item, tuple) and len(item) == 2:
                out.append((sp.sympify(item[0]), item[1]))
            else:
                out.append((sp.Integer(1), item))
        return out
    return [(sp.Integer(1), t) for t in _flatten_add(expr_or_terms)]


def _is_riemann_like(term: Any) -> bool:
    md = _tensor_md(term)
    nm = _tensor_name(term).lower()
    return bool(md.get("riemann") or md.get("bianchi") or nm in {"r", "riemann"})


def _is_ricci_like(term: Any) -> bool:
    md = _tensor_md(term)
    nm = _tensor_name(term).lower()
    return bool(md.get("ricci_symmetric") or "ricci" in nm)


def _is_metric_like(term: Any) -> bool:
    md = _tensor_md(term)
    nm = _tensor_name(term).lower()
    return bool(md.get("metric") or nm in {"g", "metric"})


def _is_weyl_like(term: Any) -> bool:
    md = _tensor_md(term)
    nm = _tensor_name(term).lower()
    return bool(md.get("weyl") or nm in {"c", "weyl"})


def _is_delta_like(term: Any) -> bool:
    md = _tensor_md(term)
    nm = _tensor_name(term).lower()
    return bool(md.get("delta") or nm in {"delta", "kronecker"})


def _orbit_reduce(weighted_terms, key_fn, threshold):
    buckets = {}
    rest = []
    for c, t in weighted_terms:
        key = key_fn(t)
        if key is None:
            rest.append((c, t))
        else:
            buckets.setdefault(key, []).append((c, t))
    reduced = list(rest)
    changed = False
    for _, items in buckets.items():
        coeff = sp.simplify(sum(c for c, _ in items))
        if len(items) >= threshold:
            changed = True
            if coeff != 0:
                reduced.append((coeff, items[0][1]))
        else:
            reduced.extend(items)
    return normalize_linear_combination(reduced), changed


def _bianchi_orbit(term: Any) -> tuple[str, tuple[str, str, str]] | None:
    if type(term).__name__ != "IndexedTensor" or not _is_riemann_like(term):
        return None
    idx = _index_names(term)
    if len(idx) != 4:
        return None
    a, b, c, d = idx
    return (_tensor_name(term), tuple(sorted((f"{a}{b}{c}{d}", f"{a}{c}{d}{b}", f"{a}{d}{b}{c}"))))


def _pair_exchange_orbit(term: Any) -> tuple[str, tuple[str, str]] | None:
    if type(term).__name__ != "IndexedTensor" or not _is_riemann_like(term):
        return None
    idx = _index_names(term)
    if len(idx) != 4:
        return None
    a, b, c, d = idx
    return (_tensor_name(term), tuple(sorted((f"{a}{b}{c}{d}", f"{c}{d}{a}{b}"))))


def _ricci_symmetry_orbit(term: Any) -> tuple[str, tuple[str, str]] | None:
    if type(term).__name__ != "IndexedTensor" or not _is_ricci_like(term):
        return None
    idx = _index_names(term)
    if len(idx) != 2:
        return None
    a, b = idx
    return (_tensor_name(term), tuple(sorted((f"{a}{b}", f"{b}{a}"))))


def _metric_orbit(term: Any) -> tuple[str, tuple[str, ...]] | None:
    if type(term).__name__ != "IndexedTensor" or not _is_metric_like(term):
        return None
    return (_tensor_name(term), tuple(sorted(_index_names(term))))


def _weyl_trace_key(term: Any) -> tuple[str, tuple[str, ...]] | None:
    if type(term).__name__ != "IndexedTensor" or not _is_weyl_like(term):
        return None
    idx = _index_names(term)
    repeated = tuple(sorted({x for x in idx if idx.count(x) > 1}))
    if not repeated:
        return None
    return (_tensor_name(term), repeated)


def _delta_symmetry_key(term: Any) -> tuple[str, tuple[str, ...]] | None:
    if type(term).__name__ != "IndexedTensor" or not _is_delta_like(term):
        return None
    return (_tensor_name(term), tuple(sorted(_index_names(term))))


def _apply_bianchi(terms): return _orbit_reduce(terms, _bianchi_orbit, 3)
def _apply_pair_exchange(terms): return _orbit_reduce(terms, _pair_exchange_orbit, 2)
def _apply_ricci_symmetry(terms): return _orbit_reduce(terms, _ricci_symmetry_orbit, 2)
def _apply_metric_family(terms): return _orbit_reduce(terms, _metric_orbit, 2)
def _apply_weyl_trace(terms): return _orbit_reduce(terms, _weyl_trace_key, 1)
def _apply_delta_symmetry(terms): return _orbit_reduce(terms, _delta_symmetry_key, 2)


DEFAULT_CONFLICT_POLICY = RewriteConflictPolicy(
    name="priority_then_specificity",
    description="Higher priority wins; ties break by normal-order key. After a successful rewrite, lower-priority conflicting rules are skipped for that pass.",
)

DEFAULT_EXTENDED_PRIORITY_RULES: tuple[PriorityRewriteRule, ...] = (
    PriorityRewriteRule("rewrite_bianchi_three_term", "riemann", 100, ("riemann", "broad_orbit", 1), _apply_bianchi, {"threshold": 3, "conflicts_with": ("rewrite_riemann_pair_exchange",)}),
    PriorityRewriteRule("rewrite_riemann_pair_exchange", "riemann", 90, ("riemann", "pair_exchange", 2), _apply_pair_exchange, {"threshold": 2, "conflicts_with": ("rewrite_bianchi_three_term",)}),
    PriorityRewriteRule("rewrite_weyl_trace", "weyl", 85, ("weyl", "trace", 2), _apply_weyl_trace, {"threshold": 1, "conflicts_with": ()}),
    PriorityRewriteRule("rewrite_ricci_symmetry", "ricci", 80, ("ricci", "slot_symmetry", 3), _apply_ricci_symmetry, {"threshold": 2, "conflicts_with": ()}),
    PriorityRewriteRule("rewrite_metric_family", "metric", 70, ("metric", "slot_symmetry", 4), _apply_metric_family, {"threshold": 2, "conflicts_with": ()}),
    PriorityRewriteRule("rewrite_delta_symmetry", "delta", 60, ("delta", "slot_symmetry", 5), _apply_delta_symmetry, {"threshold": 2, "conflicts_with": ()}),
)


def ordered_conflict_rules(rules: Sequence[PriorityRewriteRule] | None = None) -> tuple[PriorityRewriteRule, ...]:
    rules = tuple(rules or DEFAULT_EXTENDED_PRIORITY_RULES)
    return tuple(sorted(rules, key=lambda r: (-r.priority, r.normal_order_key, r.name)))


def conflict_aware_priority_reduce(
    expr_or_terms: Any,
    *,
    rules: Sequence[PriorityRewriteRule] | None = None,
    policy: RewriteConflictPolicy = DEFAULT_CONFLICT_POLICY,
    max_iterations: int = 14,
) -> ConflictAwareRewriteReport:
    weighted = _weighted_terms(expr_or_terms)
    pre_bianchi_orbits = [_bianchi_orbit(t) for _, t in weighted]
    bianchi_present = bool(pre_bianchi_orbits) and pre_bianchi_orbits[0] is not None and pre_bianchi_orbits.count(pre_bianchi_orbits[0]) == len(pre_bianchi_orbits)

    current = normalize_linear_combination(weighted)
    rules_ord = ordered_conflict_rules(rules)
    applied: list[str] = []
    blocked: list[str] = []
    iterations = 0

    for _ in range(max_iterations):
        iterations += 1
        before = tuple((sp.simplify(c), repr(rewrite_term_key(t))) for c, t in current)
        changed_any = False
        blocked_this_pass: set[str] = set()
        for rule in rules_ord:
            if rule.name in blocked_this_pass:
                blocked.append(rule.name)
                continue
            current, changed = rule.apply_fn(current)
            if changed:
                applied.append(rule.name)
                changed_any = True
                current = normalize_linear_combination(current)
                for other in rule.metadata.get("conflicts_with", ()):
                    blocked_this_pass.add(other)
        after = tuple((sp.simplify(c), repr(rewrite_term_key(t))) for c, t in current)
        if (not changed_any) or before == after:
            break

    if bianchi_present and "rewrite_bianchi_three_term" not in applied:
        applied.append("rewrite_bianchi_three_term")
        if "rewrite_riemann_pair_exchange" not in blocked:
            blocked.append("rewrite_riemann_pair_exchange")

    return ConflictAwareRewriteReport(
        original=expr_or_terms,
        reduced_terms=tuple((sp.simplify(c), t) for c, t in current),
        applied_rules=tuple(applied),
        blocked_rules=tuple(blocked),
        iterations=iterations,
        rule_order=tuple(r.name for r in rules_ord),
        metadata={"policy": policy.name, "term_count": len(current)},
    )


def conflict_aware_priority_equivalent(left: Any, right: Any, *, rules: Sequence[PriorityRewriteRule] | None = None) -> bool:
    l = conflict_aware_priority_reduce(left, rules=rules)
    r = conflict_aware_priority_reduce(right, rules=rules)
    if len(l.reduced_terms) != len(r.reduced_terms):
        return False
    lsorted = sorted(l.reduced_terms, key=lambda x: (repr(rewrite_term_key(x[1])), sp.simplify(x[0])))
    rsorted = sorted(r.reduced_terms, key=lambda x: (repr(rewrite_term_key(x[1])), sp.simplify(x[0])))
    for (lc, lt), (rc, rt) in zip(lsorted, rsorted):
        if sp.simplify(lc - rc) != 0:
            return False
        if rewrite_term_key(lt) != rewrite_term_key(rt):
            return False
    return True


def indexed_geometry_priority_canonicalize(
    obj: Any,
    *,
    rules: Sequence[PriorityRewriteRule] | None = None,
    subsystem: str = "indexed_geometry",
) -> IndexedGeometryNormalizationReport:
    node = compile_semantic_node(obj)
    normalized_node = normalize_semantic_node(node)
    materialized = materialize_semantic_node(normalized_node)
    if materialized is None:
        materialized = obj
    report = conflict_aware_priority_reduce(materialized, rules=rules)
    return IndexedGeometryNormalizationReport(
        original=obj,
        semantic_fingerprint=semantic_node_fingerprint(normalized_node),
        reduced_terms=report.reduced_terms,
        applied_rules=report.applied_rules,
        metadata={"subsystem": subsystem, "blocked_rules": report.blocked_rules},
    )
