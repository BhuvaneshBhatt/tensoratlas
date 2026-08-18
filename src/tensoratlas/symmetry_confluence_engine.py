
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Mapping, Sequence
import sympy as sp
from .semantic_core import compile_semantic_node, normalize_semantic_node, semantic_node_fingerprint, SemanticNode
from .canonicalization_core import _tensor_name, _tensor_variance, _tensor_md, _index_objects, _index_name, _index_var
from .conflict_priority_geometry_engine import DEFAULT_CONFLICT_POLICY, conflict_aware_priority_reduce, PriorityRewriteRule

@dataclass(frozen=True)
class SymmetryFamilyCanonicalizationReport:
    original: Any
    canonical_key: tuple[Any, ...]
    semantic_fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SemanticNativeRewriteReport:
    original: Any
    semantic_kind: str
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    semantic_fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ConfluenceCompletionReport:
    original: Any
    canonicalized_terms: tuple[tuple[sp.Expr, Any], ...]
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    alternate_paths_agree: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

def _symmetry_flags(term: Any) -> tuple[str, ...]:
    return tuple(sorted(k for k, v in _tensor_md(term).items() if v))

def _renumber_indices(indices) -> tuple[tuple[str, int], ...]:
    remap = {}
    nxt = 0
    out = []
    for idx in indices:
        nm = _index_name(idx)
        if nm not in remap:
            remap[nm] = nxt
            nxt += 1
        out.append((_index_var(idx), remap[nm]))
    return tuple(out)

def _whole_tree_index_histogram(expr: Any) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    def walk(obj: Any):
        cls = type(obj).__name__
        if cls == "IndexedTensor":
            for idx in _index_objects(obj):
                nm = _index_name(idx)
                counts[nm] = counts.get(nm, 0) + 1
        elif cls == "IndexedTensorExpr":
            for a in getattr(obj, "args", ()):
                walk(a)
        elif isinstance(obj, SemanticNode):
            for a in getattr(obj, "children", ()):
                walk(a)
    walk(expr)
    return counts

def _complete_slot_orbits(term: Any):
    idx = list(_index_objects(term))
    if not idx:
        return (tuple(),)
    md = _tensor_md(term)
    cands = {tuple(idx)}
    n = len(idx)
    if n == 2:
        if md.get("symmetric") or md.get("ricci_symmetric") or md.get("metric"):
            cands.add((idx[1], idx[0]))
        if md.get("antisymmetric") or md.get("delta"):
            cands.add((idx[1], idx[0]))
    if n == 3 and (md.get("antisymmetric") or md.get("epsilon")):
        cands.update(tuple(p) for p in permutations(idx, 3))
    if n == 4 and (md.get("riemann") or md.get("bianchi") or md.get("pair_symmetric") or md.get("antisymmetric") or md.get("weyl")):
        a, b, c, d = idx
        cands.update({
            (a,b,c,d), (b,a,c,d), (a,b,d,c), (b,a,d,c),
            (c,d,a,b), (d,c,a,b), (c,d,b,a), (d,c,b,a),
            (a,c,d,b), (a,d,b,c), (b,c,d,a), (b,d,a,c),
        })
    return tuple(cands)

def _difficult_tensor_key(term: Any, global_counts: Mapping[str, int]) -> tuple[Any, ...]:
    candidates = []
    for orbit in _complete_slot_orbits(term):
        base = _renumber_indices(orbit)
        graph_marks = tuple((i, global_counts.get(_index_name(ix), 0)) for i, ix in enumerate(orbit))
        candidates.append(("IndexedTensor", _tensor_name(term), _tensor_variance(term), base, graph_marks, tuple(sorted(k for k, v in _tensor_md(term).items() if v))))
    return min(candidates, key=repr)

def full_tree_canonical_key(expr: Any, *, global_counts: Mapping[str, int] | None = None) -> tuple[Any, ...]:
    if global_counts is None:
        global_counts = _whole_tree_index_histogram(expr)
    cls = type(expr).__name__
    if cls == "IndexedTensor":
        return _difficult_tensor_key(expr, global_counts)
    if cls == "IndexedTensorExpr":
        op = getattr(expr, "op", None)
        args = list(getattr(expr, "args", ()))
        child_keys = [full_tree_canonical_key(a, global_counts=global_counts) for a in args]
        if op in {"add", "tensor_product", "mul"}:
            child_keys = sorted(child_keys, key=repr)
        return ("IndexedTensorExpr", op, tuple(child_keys))
    if isinstance(expr, SemanticNode):
        child_keys = [full_tree_canonical_key(a, global_counts=global_counts) for a in getattr(expr, "children", ())]
        return ("SemanticNode", expr.kind, tuple(sorted(child_keys, key=repr)), expr.value)
    try:
        return ("sympy", sp.srepr(sp.sympify(expr)))
    except Exception:
        return ("repr", repr(expr))

def full_tree_canonicalize(expr: Any, *, global_counts: Mapping[str, int] | None = None):
    if global_counts is None:
        global_counts = _whole_tree_index_histogram(expr)
    cls = type(expr).__name__
    if cls == "IndexedTensor":
        return expr
    if cls == "IndexedTensorExpr":
        op = getattr(expr, "op", None)
        args = [full_tree_canonicalize(a, global_counts=global_counts) for a in getattr(expr, "args", ())]
        if op in {"add", "tensor_product", "mul"}:
            args = sorted(args, key=lambda x: repr(full_tree_canonical_key(x, global_counts=global_counts)))
        return type(expr)(op, tuple(args))
    return expr

def symmetry_family_canonicalize_report(expr: Any) -> SymmetryFamilyCanonicalizationReport:
    semantic = normalize_semantic_node(compile_semantic_node(expr))
    counts = _whole_tree_index_histogram(expr)
    return SymmetryFamilyCanonicalizationReport(expr, full_tree_canonical_key(expr, global_counts=counts), semantic_node_fingerprint(semantic), {"semantic_kind": semantic.kind, "index_histogram": dict(counts)})

def _semantic_native_linear_terms(obj: Any):
    node = normalize_semantic_node(compile_semantic_node(obj))
    if node.kind == "add":
        return [(sp.Integer(1), ch) for ch in node.children], node
    return [(sp.Integer(1), node)], node

def _weighted_seq(expr_or_terms: Any):
    if isinstance(expr_or_terms, Sequence) and not isinstance(expr_or_terms, (str, bytes)):
        weighted = []
        for item in expr_or_terms:
            if isinstance(item, tuple) and len(item) == 2:
                weighted.append((sp.sympify(item[0]), item[1]))
            else:
                weighted.append((sp.Integer(1), item))
    else:
        weighted = [(sp.Integer(1), expr_or_terms)]
    return weighted

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
def _is_weyl_like(term: Any) -> bool:
    md = _tensor_md(term); nm = _tensor_name(term).lower()
    return bool(md.get("weyl") or nm in {"c", "weyl"})
def _is_delta_like(term: Any) -> bool:
    md = _tensor_md(term); nm = _tensor_name(term).lower()
    return bool(md.get("delta") or nm in {"delta", "kronecker"})
def _is_epsilon_like(term: Any) -> bool:
    md = _tensor_md(term); nm = _tensor_name(term).lower()
    return bool(md.get("epsilon") or md.get("levi_civita") or nm in {"eps", "epsilon"})

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
    # normalize by difficult tree key
    groups = {}
    for c, t in reduced:
        groups.setdefault(full_tree_canonical_key(t), []).append((sp.sympify(c), t))
    out = []
    for items in groups.values():
        coeff = sp.simplify(sum(c for c, _ in items))
        if coeff != 0:
            out.append((coeff, items[0][1]))
    out.sort(key=lambda x: (repr(full_tree_canonical_key(x[1])), sp.srepr(sp.sympify(x[0]))))
    return out, changed

def _bianchi_orbit(term):
    if type(term).__name__ != "IndexedTensor" or not _is_riemann_like(term): return None
    idx = _index_names(term)
    if len(idx) != 4: return None
    a,b,c,d = idx
    return (_tensor_name(term), tuple(sorted((f"{a}{b}{c}{d}", f"{a}{c}{d}{b}", f"{a}{d}{b}{c}"))))
def _pair_exchange_orbit(term):
    if type(term).__name__ != "IndexedTensor" or not _is_riemann_like(term): return None
    idx = _index_names(term)
    if len(idx) != 4: return None
    a,b,c,d = idx
    return (_tensor_name(term), tuple(sorted((f"{a}{b}{c}{d}", f"{c}{d}{a}{b}"))))
def _ricci_symmetry_orbit(term):
    if type(term).__name__ != "IndexedTensor" or not _is_ricci_like(term): return None
    idx = _index_names(term)
    if len(idx) != 2: return None
    a,b = idx
    return (_tensor_name(term), tuple(sorted((f"{a}{b}", f"{b}{a}"))))
def _metric_orbit(term):
    if type(term).__name__ != "IndexedTensor" or not _is_metric_like(term): return None
    return (_tensor_name(term), tuple(sorted(_index_names(term))))
def _weyl_trace_key(term):
    if type(term).__name__ != "IndexedTensor" or not _is_weyl_like(term): return None
    idx = _index_names(term)
    repeated = tuple(sorted({x for x in idx if idx.count(x) > 1}))
    return (_tensor_name(term), repeated) if repeated else None
def _delta_symmetry_key(term):
    if type(term).__name__ != "IndexedTensor" or not _is_delta_like(term): return None
    return (_tensor_name(term), tuple(sorted(_index_names(term))))
def _epsilon_antisymmetry_key(term):
    if type(term).__name__ != "IndexedTensor" or not _is_epsilon_like(term): return None
    idx = _index_names(term)
    return (_tensor_name(term), tuple(sorted(idx))) if len(idx) >= 2 else None
def _metric_trace_key(term):
    if type(term).__name__ != "IndexedTensor" or not _is_metric_like(term): return None
    idx = _index_names(term)
    repeated = tuple(sorted({x for x in idx if idx.count(x) > 1}))
    return (_tensor_name(term), repeated) if repeated else None

def _apply_bianchi(terms): return _orbit_reduce(terms, _bianchi_orbit, 3)
def _apply_pair_exchange(terms): return _orbit_reduce(terms, _pair_exchange_orbit, 2)
def _apply_ricci_symmetry(terms): return _orbit_reduce(terms, _ricci_symmetry_orbit, 2)
def _apply_metric_family(terms): return _orbit_reduce(terms, _metric_orbit, 2)
def _apply_weyl_trace(terms): return _orbit_reduce(terms, _weyl_trace_key, 1)
def _apply_delta_symmetry(terms): return _orbit_reduce(terms, _delta_symmetry_key, 2)
def _apply_epsilon_antisymmetry(terms): return _orbit_reduce(terms, _epsilon_antisymmetry_key, 2)
def _apply_metric_trace(terms): return _orbit_reduce(terms, _metric_trace_key, 1)

BROAD_IDENTITY_RULES = (
    PriorityRewriteRule("rewrite_bianchi_three_term", "riemann", 120, ("riemann", "broad_orbit", 1), _apply_bianchi, {"threshold": 3, "conflicts_with": ("rewrite_riemann_pair_exchange",)}),
    PriorityRewriteRule("rewrite_riemann_pair_exchange", "riemann", 110, ("riemann", "pair_exchange", 2), _apply_pair_exchange, {"threshold": 2, "conflicts_with": ("rewrite_bianchi_three_term",)}),
    PriorityRewriteRule("rewrite_weyl_trace", "weyl", 105, ("weyl", "trace", 2), _apply_weyl_trace, {"threshold": 1, "conflicts_with": ()}),
    PriorityRewriteRule("rewrite_ricci_symmetry", "ricci", 100, ("ricci", "slot_symmetry", 3), _apply_ricci_symmetry, {"threshold": 2, "conflicts_with": ()}),
    PriorityRewriteRule("rewrite_metric_trace", "metric", 95, ("metric", "trace", 3), _apply_metric_trace, {"threshold": 1, "conflicts_with": ()}),
    PriorityRewriteRule("rewrite_metric_family", "metric", 90, ("metric", "slot_symmetry", 4), _apply_metric_family, {"threshold": 2, "conflicts_with": ()}),
    PriorityRewriteRule("rewrite_epsilon_antisymmetry", "epsilon", 85, ("epsilon", "antisymmetry", 4), _apply_epsilon_antisymmetry, {"threshold": 2, "conflicts_with": ()}),
    PriorityRewriteRule("rewrite_delta_symmetry", "delta", 80, ("delta", "slot_symmetry", 5), _apply_delta_symmetry, {"threshold": 2, "conflicts_with": ()}),
)

def semantic_native_conflict_reduce(obj: Any, *, rules=None, subsystem: str = "semantic_native_indexed_geometry") -> SemanticNativeRewriteReport:
    weighted, node = _semantic_native_linear_terms(obj)
    report = conflict_aware_priority_reduce(weighted, rules=rules or BROAD_IDENTITY_RULES, policy=DEFAULT_CONFLICT_POLICY)
    return SemanticNativeRewriteReport(obj, node.kind, report.reduced_terms, report.applied_rules, semantic_node_fingerprint(node), {"subsystem": subsystem, "blocked_rules": report.blocked_rules, "iterations": report.iterations})

def _canonicalize_weighted_terms(expr_or_terms: Any):
    weighted = _weighted_seq(expr_or_terms)
    all_expr = [t for _, t in weighted]
    counts = {}
    for expr in all_expr:
        h = _whole_tree_index_histogram(expr)
        for k, v in h.items():
            counts[k] = counts.get(k, 0) + v
    canon_terms = [(c, full_tree_canonicalize(t, global_counts=counts)) for c, t in weighted]
    groups = {}
    for c, t in canon_terms:
        key = full_tree_canonical_key(t, global_counts=counts)
        groups.setdefault(key, []).append((sp.sympify(c), t))
    out = []
    for items in groups.values():
        coeff = sp.simplify(sum(c for c, _ in items))
        if coeff != 0:
            out.append((coeff, items[0][1]))
    out.sort(key=lambda x: (repr(full_tree_canonical_key(x[1], global_counts=counts)), sp.srepr(sp.sympify(x[0]))))
    return tuple(out), counts

def confluence_completed_reduce(expr_or_terms: Any, *, rules=None) -> ConfluenceCompletionReport:
    rules = tuple(rules or BROAD_IDENTITY_RULES)
    canon_terms, counts = _canonicalize_weighted_terms(expr_or_terms)
    report_default = conflict_aware_priority_reduce(canon_terms, rules=rules)
    report_reverse = conflict_aware_priority_reduce(canon_terms, rules=tuple(reversed(rules)))
    def normalize_report_terms(terms):
        norm = []
        for c, t in terms:
            norm.append((sp.simplify(c), full_tree_canonical_key(t, global_counts=counts)))
        return tuple(sorted(norm, key=repr))
    agree = normalize_report_terms(report_default.reduced_terms) == normalize_report_terms(report_reverse.reduced_terms)
    applied = ("expanded_symmetry_families", "contraction_graph_tree_dummy_handling", "priority_conflict_rewrite_integration") + report_default.applied_rules
    applied = applied + (("confluence_check_agree",) if agree else ("confluence_check_diverge",))
    return ConfluenceCompletionReport(expr_or_terms, canon_terms, report_default.reduced_terms, applied, agree, {"blocked_rules": report_default.blocked_rules, "iterations": report_default.iterations, "index_histogram": dict(counts)})
