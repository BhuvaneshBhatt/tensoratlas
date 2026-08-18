
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import sympy as sp

from .semantic_core import compile_semantic_node, normalize_semantic_node, semantic_node_fingerprint


@dataclass(frozen=True)
class CanonicalizationCoreReport:
    original: Any
    canonical: Any
    canonical_key: tuple[Any, ...]
    semantic_fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _tensor_name(term: Any) -> str:
    return getattr(getattr(term, "tensor", None), "name", "") or getattr(term, "value", "") or ""


def _tensor_variance(term: Any) -> str:
    return getattr(getattr(term, "tensor", None), "variance_spec", "") or ""


def _tensor_md(term: Any) -> Mapping[str, Any]:
    return getattr(getattr(term, "tensor", None), "symmetry_metadata", {}) or {}


def _index_objects(term: Any):
    return tuple(getattr(term, "indices", ()))


def _index_name(idx: Any) -> str:
    return getattr(idx, "name", str(idx))


def _index_var(idx: Any) -> str:
    return getattr(idx, "variance", "")


def _dummy_class_pattern(indices) -> tuple[tuple[str, int], ...]:
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


def _symmetry_flags(term: Any) -> tuple[str, ...]:
    md = _tensor_md(term)
    flags = [k for k, v in md.items() if v]
    return tuple(sorted(flags))


def _orbit_candidates_for_indexed_tensor(term: Any) -> tuple[tuple[Any, ...], ...]:
    idx = list(_index_objects(term))
    if not idx:
        return (tuple(),)
    candidates = {tuple(idx)}
    md = _tensor_md(term)

    # slot symmetries
    if len(idx) == 2 and (md.get("ricci_symmetric") or md.get("symmetric")):
        candidates.add((idx[1], idx[0]))
    if len(idx) == 2 and md.get("antisymmetric"):
        candidates.add((idx[1], idx[0]))

    # Riemann-style permutations
    if len(idx) == 4 and (md.get("riemann") or md.get("bianchi") or md.get("pair_symmetric")):
        a, b, c, d = idx
        candidates.update({
            (a, b, c, d),
            (c, d, a, b),
            (b, a, c, d),
            (a, b, d, c),
            (a, c, d, b),
            (a, d, b, c),
        })
    return tuple(candidates)


def _canonical_index_arrangement(term: Any) -> tuple[tuple[str, int], ...]:
    candidates = []
    for cand in _orbit_candidates_for_indexed_tensor(term):
        candidates.append(_dummy_class_pattern(cand))
    return min(candidates, key=repr) if candidates else tuple()


def indexed_canonical_key(term: Any) -> tuple[Any, ...]:
    return (
        "IndexedTensor",
        _tensor_name(term),
        _tensor_variance(term),
        _canonical_index_arrangement(term),
        _symmetry_flags(term),
    )


def expr_canonical_key(expr: Any) -> tuple[Any, ...]:
    cls = type(expr).__name__
    if cls == "IndexedTensor":
        return indexed_canonical_key(expr)
    if cls == "IndexedTensorExpr":
        op = getattr(expr, "op", None)
        args = getattr(expr, "args", ())
        child_keys = [expr_canonical_key(a) for a in args]
        if op in {"add", "tensor_product", "mul"}:
            child_keys = sorted(child_keys, key=repr)
        return ("IndexedTensorExpr", op, tuple(child_keys))
    try:
        return ("sympy", sp.srepr(sp.sympify(expr)))
    except Exception:
        return ("repr", repr(expr))


def canonicalize_indexed_expression(expr: Any):
    cls = type(expr).__name__
    if cls == "IndexedTensor":
        return expr
    if cls == "IndexedTensorExpr":
        op = getattr(expr, "op", None)
        args = list(getattr(expr, "args", ()))
        canon_args = [canonicalize_indexed_expression(a) for a in args]
        if op in {"add", "tensor_product", "mul"}:
            canon_args = sorted(canon_args, key=lambda x: repr(expr_canonical_key(x)))
        return type(expr)(op, tuple(canon_args))
    return expr


def abstract_index_canonicalize(obj: Any) -> CanonicalizationCoreReport:
    semantic = normalize_semantic_node(compile_semantic_node(obj))
    canonical = canonicalize_indexed_expression(obj)
    key = expr_canonical_key(canonical)
    return CanonicalizationCoreReport(
        original=obj,
        canonical=canonical,
        canonical_key=key,
        semantic_fingerprint=semantic_node_fingerprint(semantic),
        metadata={"semantic_kind": semantic.kind},
    )


def abstract_index_equivalent(left: Any, right: Any) -> bool:
    return expr_canonical_key(canonicalize_indexed_expression(left)) == expr_canonical_key(canonicalize_indexed_expression(right))
