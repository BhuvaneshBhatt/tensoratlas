
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import sympy as sp

from .semantic_core import compile_semantic_node, normalize_semantic_node, semantic_node_fingerprint
from .canonicalization_core import _tensor_name, _tensor_variance, _tensor_md, _index_objects, _index_name, _index_var
from .conflict_priority_geometry_engine import conflict_aware_priority_reduce


@dataclass(frozen=True)
class ContractionGraphCanonicalizationReport:
    original: Any
    canonical_key: tuple[Any, ...]
    semantic_fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PriorityCanonicalRewriteReport:
    original: Any
    canonicalized_terms: tuple[tuple[sp.Expr, Any], ...]
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    semantic_fingerprint: tuple[Any, ...]
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
    walk(expr)
    return counts


def _contraction_graph_key_for_tensor(term: Any, global_counts: Mapping[str, int]) -> tuple[Any, ...]:
    indices = _index_objects(term)
    base = _renumber_indices(indices)
    graph_marks = tuple((i, global_counts.get(_index_name(idx), 0)) for i, idx in enumerate(indices))
    return (
        "IndexedTensor",
        _tensor_name(term),
        _tensor_variance(term),
        base,
        graph_marks,
        _symmetry_flags(term),
    )


def _orbit_candidates(term: Any, global_counts: Mapping[str, int]) -> tuple[tuple[Any, ...], ...]:
    idx = list(_index_objects(term))
    if not idx:
        return (("IndexedTensor", _tensor_name(term), _tensor_variance(term), tuple(), tuple(), _symmetry_flags(term)),)
    cands = {tuple(idx)}
    md = _tensor_md(term)

    if len(idx) == 2 and (md.get("ricci_symmetric") or md.get("symmetric")):
        cands.add((idx[1], idx[0]))
    if len(idx) == 2 and md.get("antisymmetric"):
        cands.add((idx[1], idx[0]))

    if len(idx) == 4 and (md.get("riemann") or md.get("bianchi") or md.get("pair_symmetric") or md.get("antisymmetric")):
        a, b, c, d = idx
        cands.update({
            (a, b, c, d),
            (c, d, a, b),
            (b, a, c, d),
            (a, b, d, c),
            (a, c, d, b),
            (a, d, b, c),
        })

    if len(idx) >= 2 and md.get("delta"):
        cands.add(tuple(reversed(idx)))

    out = []
    for cand in cands:
        base = _renumber_indices(cand)
        graph_marks = tuple((i, global_counts.get(_index_name(ix), 0)) for i, ix in enumerate(cand))
        out.append(("IndexedTensor", _tensor_name(term), _tensor_variance(term), base, graph_marks, _symmetry_flags(term)))
    return tuple(out)


def broader_tree_canonical_key(expr: Any, *, global_counts: Mapping[str, int] | None = None) -> tuple[Any, ...]:
    if global_counts is None:
        global_counts = _whole_tree_index_histogram(expr)
    cls = type(expr).__name__
    if cls == "IndexedTensor":
        cands = _orbit_candidates(expr, global_counts)
        return min(cands, key=repr)
    if cls == "IndexedTensorExpr":
        op = getattr(expr, "op", None)
        args = list(getattr(expr, "args", ()))
        child_keys = [broader_tree_canonical_key(a, global_counts=global_counts) for a in args]
        if op in {"add", "tensor_product", "mul"}:
            child_keys = sorted(child_keys, key=repr)
        return ("IndexedTensorExpr", op, tuple(child_keys))
    try:
        return ("sympy", sp.srepr(sp.sympify(expr)))
    except Exception:
        return ("repr", repr(expr))


def broader_tree_canonicalize(expr: Any, *, global_counts: Mapping[str, int] | None = None):
    if global_counts is None:
        global_counts = _whole_tree_index_histogram(expr)
    cls = type(expr).__name__
    if cls == "IndexedTensor":
        return expr
    if cls == "IndexedTensorExpr":
        op = getattr(expr, "op", None)
        args = [broader_tree_canonicalize(a, global_counts=global_counts) for a in getattr(expr, "args", ())]
        if op in {"add", "tensor_product", "mul"}:
            args = sorted(args, key=lambda x: repr(broader_tree_canonical_key(x, global_counts=global_counts)))
        return type(expr)(op, tuple(args))
    return expr


def contraction_graph_canonicalize_report(expr: Any) -> ContractionGraphCanonicalizationReport:
    semantic = normalize_semantic_node(compile_semantic_node(expr))
    counts = _whole_tree_index_histogram(expr)
    return ContractionGraphCanonicalizationReport(
        original=expr,
        canonical_key=broader_tree_canonical_key(expr, global_counts=counts),
        semantic_fingerprint=semantic_node_fingerprint(semantic),
        metadata={"semantic_kind": semantic.kind, "index_histogram": dict(counts)},
    )


def _canonicalize_weighted_terms(expr_or_terms: Any):
    if isinstance(expr_or_terms, Sequence) and not isinstance(expr_or_terms, (str, bytes)):
        weighted = []
        for item in expr_or_terms:
            if isinstance(item, tuple) and len(item) == 2:
                weighted.append((sp.sympify(item[0]), item[1]))
            else:
                weighted.append((sp.Integer(1), item))
    else:
        weighted = [(sp.Integer(1), expr_or_terms)]
    all_expr = [t for _, t in weighted]
    counts = {}
    for expr in all_expr:
        h = _whole_tree_index_histogram(expr)
        for k, v in h.items():
            counts[k] = counts.get(k, 0) + v
    canon_terms = [(c, broader_tree_canonicalize(t, global_counts=counts)) for c, t in weighted]
    groups: dict[tuple[Any, ...], list[tuple[sp.Expr, Any]]] = {}
    for c, t in canon_terms:
        key = broader_tree_canonical_key(t, global_counts=counts)
        groups.setdefault(key, []).append((sp.sympify(c), t))
    out = []
    for key, items in groups.items():
        coeff = sp.simplify(sum(c for c, _ in items))
        if coeff != 0:
            out.append((coeff, items[0][1]))
    out.sort(key=lambda x: (repr(broader_tree_canonical_key(x[1], global_counts=counts)), sp.srepr(sp.sympify(x[0]))))
    return tuple(out), counts


def priority_canonicalization_integrated_reduce(expr_or_terms: Any, *, rules=None) -> PriorityCanonicalRewriteReport:
    semantic = normalize_semantic_node(compile_semantic_node(expr_or_terms))
    canon_terms, counts = _canonicalize_weighted_terms(expr_or_terms)
    conflict_report = conflict_aware_priority_reduce(canon_terms, rules=rules)
    return PriorityCanonicalRewriteReport(
        original=expr_or_terms,
        canonicalized_terms=canon_terms,
        reduced_terms=conflict_report.reduced_terms,
        applied_rules=("broader_slot_symmetry", "whole_tree_dummy_contraction_graph") + conflict_report.applied_rules,
        semantic_fingerprint=semantic_node_fingerprint(semantic),
        metadata={
            "semantic_kind": semantic.kind,
            "blocked_rules": conflict_report.blocked_rules,
            "iterations": conflict_report.iterations,
            "index_histogram": dict(counts),
        },
    )
