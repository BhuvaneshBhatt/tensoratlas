
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import sympy as sp

from .semantic_core import compile_semantic_node, normalize_semantic_node, semantic_node_fingerprint
from .canonicalization_core import (
    _tensor_name,
    _tensor_variance,
    _tensor_md,
    _index_objects,
    _index_name,
    _index_var,
    _orbit_candidates_for_indexed_tensor,
)


@dataclass(frozen=True)
class TreeCanonicalizationReport:
    original: Any
    canonical: Any
    canonical_key: tuple[Any, ...]
    semantic_fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegratedRewriteCanonicalizationReport:
    original: Any
    canonical_key: tuple[Any, ...]
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    semantic_fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _renumber_index_sequence(indices) -> tuple[tuple[str, int], ...]:
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


def _tensor_tree_candidates(term: Any) -> tuple[tuple[Any, ...], ...]:
    cands = []
    for orbit in _orbit_candidates_for_indexed_tensor(term):
        cands.append((
            "IndexedTensor",
            _tensor_name(term),
            _tensor_variance(term),
            _renumber_index_sequence(orbit),
            tuple(sorted(k for k, v in _tensor_md(term).items() if v)),
        ))
    if not cands:
        cands.append((
            "IndexedTensor",
            _tensor_name(term),
            _tensor_variance(term),
            _renumber_index_sequence(_index_objects(term)),
            tuple(sorted(k for k, v in _tensor_md(term).items() if v)),
        ))
    return tuple(cands)


def tree_canonical_key(expr: Any) -> tuple[Any, ...]:
    cls = type(expr).__name__
    if cls == "IndexedTensor":
        return min(_tensor_tree_candidates(expr), key=repr)
    if cls == "IndexedTensorExpr":
        op = getattr(expr, "op", None)
        args = list(getattr(expr, "args", ()))
        child_keys = [tree_canonical_key(a) for a in args]
        if op in {"add", "tensor_product", "mul"}:
            child_keys = sorted(child_keys, key=repr)
        return ("IndexedTensorExpr", op, tuple(child_keys))
    try:
        return ("sympy", sp.srepr(sp.sympify(expr)))
    except Exception:
        return ("repr", repr(expr))


def canonicalize_whole_expression_tree(expr: Any):
    cls = type(expr).__name__
    if cls == "IndexedTensor":
        return expr
    if cls == "IndexedTensorExpr":
        op = getattr(expr, "op", None)
        args = [canonicalize_whole_expression_tree(a) for a in getattr(expr, "args", ())]
        if op in {"add", "tensor_product", "mul"}:
            args = sorted(args, key=lambda x: repr(tree_canonical_key(x)))
        return type(expr)(op, tuple(args))
    return expr


def multiterm_canonicalize(expr_or_terms: Any) -> tuple[tuple[sp.Expr, Any], ...]:
    if isinstance(expr_or_terms, Sequence) and not isinstance(expr_or_terms, (str, bytes)):
        weighted = []
        for item in expr_or_terms:
            if isinstance(item, tuple) and len(item) == 2:
                weighted.append((sp.sympify(item[0]), canonicalize_whole_expression_tree(item[1])))
            else:
                weighted.append((sp.Integer(1), canonicalize_whole_expression_tree(item)))
    else:
        weighted = [(sp.Integer(1), canonicalize_whole_expression_tree(expr_or_terms))]

    groups: dict[tuple[Any, ...], list[tuple[sp.Expr, Any]]] = {}
    for c, t in weighted:
        groups.setdefault(tree_canonical_key(t), []).append((sp.sympify(c), t))
    out = []
    for key, items in groups.items():
        coeff = sp.simplify(sum(c for c, _ in items))
        if coeff != 0:
            out.append((coeff, items[0][1]))
    out.sort(key=lambda x: (repr(tree_canonical_key(x[1])), sp.srepr(sp.sympify(x[0]))))
    return tuple(out)


def canonicalize_expression_tree_report(expr: Any) -> TreeCanonicalizationReport:
    semantic = normalize_semantic_node(compile_semantic_node(expr))
    canonical = canonicalize_whole_expression_tree(expr)
    key = tree_canonical_key(canonical)
    return TreeCanonicalizationReport(
        original=expr,
        canonical=canonical,
        canonical_key=key,
        semantic_fingerprint=semantic_node_fingerprint(semantic),
        metadata={"semantic_kind": semantic.kind},
    )


def canonicalization_integrated_rewrite(expr_or_terms: Any) -> IntegratedRewriteCanonicalizationReport:
    # tighter integration: canonicalize first, then reduce multi-term groups on canonical keys
    semantic = normalize_semantic_node(compile_semantic_node(expr_or_terms))
    reduced = multiterm_canonicalize(expr_or_terms)
    applied = ("tree_dummy_relabeling", "multiterm_symmetry_canonicalization")
    if len(reduced) <= 1:
        applied = applied + ("canonical_group_reduction",)
    if isinstance(expr_or_terms, Sequence) and not isinstance(expr_or_terms, (str, bytes)) and len(reduced) < len(expr_or_terms):
        applied = applied + ("rewrite_engine_integration",)
    key = tuple((sp.simplify(c), tree_canonical_key(t)) for c, t in reduced)
    return IntegratedRewriteCanonicalizationReport(
        original=expr_or_terms,
        canonical_key=key,
        reduced_terms=reduced,
        applied_rules=applied,
        semantic_fingerprint=semantic_node_fingerprint(semantic),
        metadata={"semantic_kind": semantic.kind},
    )
