
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .conflict_priority_geometry_engine import (
    PriorityRewriteRule,
    conflict_aware_priority_reduce,
)
from .canonicalization_core import _tensor_name, _tensor_md, _index_objects, _index_name
from .completion_manager import completion_manager


@dataclass(frozen=True)
class IdentityFamilyRuleSet:
    name: str
    rules: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpandedIdentityBasisReport:
    original: Any
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    family_sets: tuple[str, ...]
    completion_summary: Mapping[str, Any] = field(default_factory=dict)


def _normalize_terms(weighted_terms):
    groups = {}
    for c, t in weighted_terms:
        key = repr(t)
        groups.setdefault(key, []).append((sp.sympify(c), t))
    out = []
    for items in groups.values():
        coeff = sp.simplify(sum(c for c, _ in items))
        if coeff != 0:
            out.append((coeff, items[0][1]))
    out.sort(key=lambda x: (repr(x[1]), sp.srepr(sp.sympify(x[0]))))
    return out


def _index_names(term: Any) -> tuple[str, ...]:
    return tuple(_index_name(i) for i in _index_objects(term))


def _is_riemann_like(term: Any) -> bool:
    md = _tensor_md(term); nm = _tensor_name(term).lower()
    return bool(md.get("riemann") or md.get("bianchi") or nm in {"r", "riemann"})


def _is_ricci_like(term: Any) -> bool:
    md = _tensor_md(term); nm = _tensor_name(term).lower()
    return bool(md.get("ricci_symmetric") or "ricci" in nm)


def _is_metric_like(term: Any) -> bool:
    md = _tensor_md(term); nm = _tensor_name(term).lower()
    return bool(md.get("metric") or nm in {"g", "metric"})


def _is_epsilon_like(term: Any) -> bool:
    md = _tensor_md(term); nm = _tensor_name(term).lower()
    return bool(md.get("epsilon") or md.get("levi_civita") or nm in {"eps", "epsilon"})


def _is_delta_like(term: Any) -> bool:
    md = _tensor_md(term); nm = _tensor_name(term).lower()
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
    return _normalize_terms(reduced), changed


# Curvature family
def _bianchi_orbit(term):
    if type(term).__name__ != "IndexedTensor" or not _is_riemann_like(term):
        return None
    idx = _index_names(term)
    if len(idx) != 4:
        return None
    a,b,c,d = idx
    return (_tensor_name(term), tuple(sorted((f"{a}{b}{c}{d}", f"{a}{c}{d}{b}", f"{a}{d}{b}{c}"))))

def _pair_exchange_orbit(term):
    if type(term).__name__ != "IndexedTensor" or not _is_riemann_like(term):
        return None
    idx = _index_names(term)
    if len(idx) != 4:
        return None
    a,b,c,d = idx
    return (_tensor_name(term), tuple(sorted((f"{a}{b}{c}{d}", f"{c}{d}{a}{b}"))))

def _apply_bianchi(terms): return _orbit_reduce(terms, _bianchi_orbit, 3)
def _apply_pair_exchange(terms): return _orbit_reduce(terms, _pair_exchange_orbit, 2)

# Metric family
def _metric_orbit(term):
    if type(term).__name__ != "IndexedTensor" or not _is_metric_like(term):
        return None
    return (_tensor_name(term), tuple(sorted(_index_names(term))))

def _metric_trace_key(term):
    if type(term).__name__ != "IndexedTensor" or not _is_metric_like(term):
        return None
    idx = _index_names(term)
    repeated = tuple(sorted({x for x in idx if idx.count(x) > 1}))
    return (_tensor_name(term), repeated) if repeated else None

def _apply_metric_family(terms): return _orbit_reduce(terms, _metric_orbit, 2)
def _apply_metric_trace(terms): return _orbit_reduce(terms, _metric_trace_key, 1)

# Ricci / Einstein-like family
def _ricci_symmetry_orbit(term):
    if type(term).__name__ != "IndexedTensor" or not _is_ricci_like(term):
        return None
    idx = _index_names(term)
    if len(idx) != 2:
        return None
    a,b = idx
    return (_tensor_name(term), tuple(sorted((f"{a}{b}", f"{b}{a}"))))

def _apply_ricci_symmetry(terms): return _orbit_reduce(terms, _ricci_symmetry_orbit, 2)

# Epsilon / delta family
def _epsilon_antisymmetry_key(term):
    if type(term).__name__ != "IndexedTensor" or not _is_epsilon_like(term):
        return None
    idx = _index_names(term)
    return (_tensor_name(term), tuple(sorted(idx))) if len(idx) >= 2 else None

def _delta_symmetry_key(term):
    if type(term).__name__ != "IndexedTensor" or not _is_delta_like(term):
        return None
    return (_tensor_name(term), tuple(sorted(_index_names(term))))

def _apply_epsilon_antisymmetry(terms): return _orbit_reduce(terms, _epsilon_antisymmetry_key, 2)
def _apply_delta_symmetry(terms): return _orbit_reduce(terms, _delta_symmetry_key, 2)

# Differential/derivative family scaffold
def _derivative_commutator_key(term):
    if type(term).__name__ != "IndexedTensorExpr":
        return None
    if getattr(term, "op", None) not in {"covariant_derivative", "derivative"}:
        return None
    return ("derivative", repr(term))

def _apply_derivative_commutator(terms): return _orbit_reduce(terms, _derivative_commutator_key, 2)

# Exterior family scaffold
def _hodge_square_key(term):
    if type(term).__name__ != "InteroperableExteriorReport":
        return None
    return ("hodge_square", repr(term))

def _apply_hodge_square(terms): return _orbit_reduce(terms, _hodge_square_key, 2)


def _build_identity_rule_families() -> Mapping[str, tuple[PriorityRewriteRule, ...]]:
    return {
        "curvature": (
            PriorityRewriteRule("rewrite_bianchi_three_term", "curvature", 120, ("curvature", "bianchi", 1), _apply_bianchi, {"family_set": "curvature"}),
            PriorityRewriteRule("rewrite_riemann_pair_exchange", "curvature", 110, ("curvature", "pair_exchange", 2), _apply_pair_exchange, {"family_set": "curvature"}),
            PriorityRewriteRule("rewrite_ricci_symmetry", "curvature", 100, ("curvature", "ricci", 3), _apply_ricci_symmetry, {"family_set": "curvature"}),
        ),
        "metric": (
            PriorityRewriteRule("rewrite_metric_trace", "metric", 95, ("metric", "trace", 1), _apply_metric_trace, {"family_set": "metric"}),
            PriorityRewriteRule("rewrite_metric_family", "metric", 90, ("metric", "family", 2), _apply_metric_family, {"family_set": "metric"}),
        ),
        "epsilon_delta": (
            PriorityRewriteRule("rewrite_epsilon_antisymmetry", "epsilon_delta", 85, ("epsilon_delta", "epsilon", 1), _apply_epsilon_antisymmetry, {"family_set": "epsilon_delta"}),
            PriorityRewriteRule("rewrite_delta_symmetry", "epsilon_delta", 80, ("epsilon_delta", "delta", 2), _apply_delta_symmetry, {"family_set": "epsilon_delta"}),
        ),
        "derivative": (
            PriorityRewriteRule("rewrite_derivative_commutator_scaffold", "derivative", 70, ("derivative", "commutator", 1), _apply_derivative_commutator, {"family_set": "derivative"}),
        ),
        "exterior": (
            PriorityRewriteRule("rewrite_hodge_square_scaffold", "exterior", 60, ("exterior", "hodge", 1), _apply_hodge_square, {"family_set": "exterior"}),
        ),
    }


def get_identity_rule_families() -> Mapping[str, tuple[PriorityRewriteRule, ...]]:
    cached = globals().get("_IDENTITY_RULE_FAMILIES_CACHE")
    if cached is None:
        cached = _build_identity_rule_families()
        globals()["_IDENTITY_RULE_FAMILIES_CACHE"] = cached
    return cached


def get_expanded_identity_basis() -> tuple[PriorityRewriteRule, ...]:
    families = get_identity_rule_families()
    return families["curvature"] + families["metric"] + families["epsilon_delta"] + families["derivative"] + families["exterior"]


def get_expanded_identity_rule_sets() -> tuple[IdentityFamilyRuleSet, ...]:
    families = get_identity_rule_families()
    return tuple(IdentityFamilyRuleSet(name, tuple(rule.name for rule in rules)) for name, rules in families.items())



def expand_identity_basis_reduce(expr_or_terms: Any) -> ExpandedIdentityBasisReport:
    basis = get_expanded_identity_basis()
    rewrite_report = conflict_aware_priority_reduce(expr_or_terms, rules=basis)
    manager_report = completion_manager(expr_or_terms, rules=basis)
    return ExpandedIdentityBasisReport(
        original=expr_or_terms,
        reduced_terms=rewrite_report.reduced_terms,
        applied_rules=rewrite_report.applied_rules,
        family_sets=tuple(rs.name for rs in get_expanded_identity_rule_sets()),
        completion_summary={
            "confluence_agrees": manager_report.confluence_agrees,
            "issue_count": len(manager_report.completion_issues),
            "primary_strategy": manager_report.strategy,
        },
    )
