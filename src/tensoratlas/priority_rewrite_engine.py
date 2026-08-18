
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import sympy as sp


@dataclass(frozen=True)
class PriorityRewriteRule:
    name: str
    family: str
    priority: int
    normal_order_key: tuple[Any, ...]
    apply_fn: Callable[[list[tuple[sp.Expr, Any]]], tuple[list[tuple[sp.Expr, Any]], bool]]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PriorityRewriteEngineReport:
    original: Any
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    iterations: int
    rule_order: tuple[str, ...]
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


def _dummy_pattern(term: Any) -> tuple[tuple[str, int], ...]:
    remap = {}
    nxt = 0
    out = []
    for idx in getattr(term, "indices", ()):
        nm = getattr(idx, "name", str(idx))
        if nm not in remap:
            remap[nm] = nxt
            nxt += 1
        out.append((getattr(idx, "variance", ""), remap[nm]))
    return tuple(out)


def rewrite_term_key(term: Any) -> tuple[Any, ...]:
    cls = type(term).__name__
    if cls == "IndexedTensor":
        return (
            "IndexedTensor",
            _tensor_name(term),
            getattr(getattr(term, "tensor", None), "variance_spec", ""),
            _dummy_pattern(term),
            tuple(sorted(k for k, v in _tensor_md(term).items() if v)),
        )
    if cls == "IndexedTensorExpr":
        return ("IndexedTensorExpr", getattr(term, "op", None), tuple(rewrite_term_key(a) for a in getattr(term, "args", ())))
    try:
        return ("sympy", sp.srepr(sp.sympify(term)))
    except Exception:
        return ("repr", repr(term))


def normalize_linear_combination(weighted_terms: list[tuple[sp.Expr, Any]]) -> list[tuple[sp.Expr, Any]]:
    groups: dict[tuple[Any, ...], list[tuple[sp.Expr, Any]]] = {}
    for c, t in weighted_terms:
        groups.setdefault(rewrite_term_key(t), []).append((sp.sympify(c), t))
    out = []
    for items in groups.values():
        coeff = sp.simplify(sum(c for c, _ in items))
        if coeff != 0:
            out.append((coeff, items[0][1]))
    out.sort(key=lambda x: (repr(rewrite_term_key(x[1])), sp.srepr(sp.sympify(x[0]))))
    return out


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


def _orbit_reduce(weighted_terms: list[tuple[sp.Expr, Any]], key_fn, threshold: int) -> tuple[list[tuple[sp.Expr, Any]], bool]:
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


def _apply_bianchi_three_term(terms):
    return _orbit_reduce(terms, _bianchi_orbit, 3)


def _apply_pair_exchange(terms):
    return _orbit_reduce(terms, _pair_exchange_orbit, 2)


def _apply_ricci_symmetry(terms):
    return _orbit_reduce(terms, _ricci_symmetry_orbit, 2)


def _apply_metric_family(terms):
    return _orbit_reduce(terms, _metric_orbit, 2)


DEFAULT_PRIORITY_RULES: tuple[PriorityRewriteRule, ...] = (
    PriorityRewriteRule(
        name="rewrite_bianchi_three_term",
        family="riemann",
        priority=100,
        normal_order_key=("riemann", "broad_orbit", 1),
        apply_fn=_apply_bianchi_three_term,
        metadata={"threshold": 3},
    ),
    PriorityRewriteRule(
        name="rewrite_riemann_pair_exchange",
        family="riemann",
        priority=90,
        normal_order_key=("riemann", "pair_exchange", 2),
        apply_fn=_apply_pair_exchange,
        metadata={"threshold": 2},
    ),
    PriorityRewriteRule(
        name="rewrite_ricci_symmetry",
        family="ricci",
        priority=80,
        normal_order_key=("ricci", "slot_symmetry", 3),
        apply_fn=_apply_ricci_symmetry,
        metadata={"threshold": 2},
    ),
    PriorityRewriteRule(
        name="rewrite_metric_family",
        family="metric",
        priority=70,
        normal_order_key=("metric", "slot_symmetry", 4),
        apply_fn=_apply_metric_family,
        metadata={"threshold": 2},
    ),
)


def ordered_rules(rules: Sequence[PriorityRewriteRule] | None = None) -> tuple[PriorityRewriteRule, ...]:
    rules = tuple(rules or DEFAULT_PRIORITY_RULES)
    return tuple(sorted(rules, key=lambda r: (-r.priority, r.normal_order_key, r.name)))


def priority_rewrite_reduce(expr_or_terms: Any, *, rules: Sequence[PriorityRewriteRule] | None = None, max_iterations: int = 12) -> PriorityRewriteEngineReport:
    current = normalize_linear_combination(_weighted_terms(expr_or_terms))
    applied: list[str] = []
    rules_ord = ordered_rules(rules)
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        before = tuple((sp.simplify(c), repr(rewrite_term_key(t))) for c, t in current)
        changed_any = False
        for rule in rules_ord:
            current, changed = rule.apply_fn(current)
            if changed:
                applied.append(rule.name)
                changed_any = True
                current = normalize_linear_combination(current)
        after = tuple((sp.simplify(c), repr(rewrite_term_key(t))) for c, t in current)
        if (not changed_any) or before == after:
            break
    return PriorityRewriteEngineReport(
        original=expr_or_terms,
        reduced_terms=tuple((sp.simplify(c), t) for c, t in current),
        applied_rules=tuple(applied),
        iterations=iterations,
        rule_order=tuple(r.name for r in rules_ord),
        metadata={"term_count": len(current)},
    )


def priority_rewrite_equivalent(left: Any, right: Any, *, rules: Sequence[PriorityRewriteRule] | None = None) -> bool:
    l = priority_rewrite_reduce(left, rules=rules)
    r = priority_rewrite_reduce(right, rules=rules)
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
