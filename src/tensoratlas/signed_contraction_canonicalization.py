
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Mapping

import sympy as sp

from .semantic_core import compile_semantic_node, normalize_semantic_node, semantic_node_fingerprint
from .canonicalization_core import _tensor_name, _tensor_variance, _tensor_md, _index_objects, _index_name, _index_var


@dataclass(frozen=True)
class SignedContractionCanonicalizationReport:
    original: Any
    canonical_key: tuple[Any, ...]
    semantic_fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


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


def _collect_tree_index_names(expr: Any) -> tuple[str, ...]:
    out: list[str] = []
    def walk(obj: Any):
        cls = type(obj).__name__
        if cls == "IndexedTensor":
            for idx in _index_objects(obj):
                out.append(_index_name(idx))
        elif cls == "IndexedTensorExpr":
            for a in getattr(obj, "args", ()):
                walk(a)
    walk(expr)
    return tuple(out)


def _global_dummy_name_map(expr: Any) -> dict[str, str]:
    names = _collect_tree_index_names(expr)
    uniq = sorted(set(names))
    return {name: f"d{i}" for i, name in enumerate(uniq)}


def _permutation_sign_from_reference(reference: tuple[Any, ...], candidate: tuple[Any, ...]) -> int:
    pos = []
    used = set()
    for r in reference:
        matched = None
        for j, c in enumerate(candidate):
            if j in used:
                continue
            if c is r or (_index_name(c) == _index_name(r) and _index_var(c) == _index_var(r)):
                matched = j
                used.add(j)
                break
        pos.append(matched if matched is not None else 0)
    inv = 0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if pos[i] > pos[j]:
                inv += 1
    return -1 if inv % 2 else 1


def _signed_slot_orbits(term: Any):
    idx = tuple(_index_objects(term))
    if not idx:
        return ((tuple(), 1),)
    md = _tensor_md(term)
    cands: set[tuple[tuple[Any, ...], int]] = {(idx, 1)}
    n = len(idx)

    def add_perm(perm):
        sign = _permutation_sign_from_reference(idx, perm)
        cands.add((tuple(perm), sign))

    if n == 2:
        if md.get("symmetric") or md.get("ricci_symmetric") or md.get("metric"):
            cands.add(((idx[1], idx[0]), 1))
        if md.get("antisymmetric") or md.get("delta"):
            cands.add(((idx[1], idx[0]), -1))

    if n == 3 and (md.get("antisymmetric") or md.get("epsilon")):
        for p in permutations(idx, 3):
            add_perm(p)

    if n == 4 and (md.get("riemann") or md.get("bianchi") or md.get("pair_symmetric") or md.get("antisymmetric") or md.get("weyl")):
        a, b, c, d = idx
        base = [
            (a,b,c,d), (b,a,c,d), (a,b,d,c), (b,a,d,c),
            (c,d,a,b), (d,c,a,b), (c,d,b,a), (d,c,b,a),
            (a,c,d,b), (a,d,b,c), (b,c,d,a), (b,d,a,c),
        ]
        for p in base:
            sign = 1
            if md.get("antisymmetric"):
                first_swap = (p[0] is b and p[1] is a)
                second_swap = (p[2] is d and p[3] is c)
                if first_swap ^ second_swap:
                    sign = -1
            cands.add((tuple(p), sign))

    return tuple(cands)


def _renumber_global(indices, global_map: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple((_index_var(idx), global_map[_index_name(idx)]) for idx in indices)


def _signed_tensor_key(term: Any, *, global_counts: Mapping[str, int], global_map: Mapping[str, str]) -> tuple[Any, ...]:
    candidates = []
    for orbit, sign in _signed_slot_orbits(term):
        base = _renumber_global(orbit, global_map)
        graph_marks = tuple((i, global_counts.get(_index_name(ix), 0)) for i, ix in enumerate(orbit))
        base_key = (
            "IndexedTensor",
            _tensor_name(term),
            _tensor_variance(term),
            base,
            graph_marks,
            tuple(sorted(k for k, v in _tensor_md(term).items() if v)),
        )
        candidates.append((base_key, sign))
    canonical_base = min((bk for bk, _ in candidates), key=repr)
    signs = [sgn for bk, sgn in candidates if bk == canonical_base]
    canonical_sign = min(signs)
    return (
        canonical_base[0],
        canonical_base[1],
        canonical_base[2],
        canonical_sign,
        canonical_base[3],
        canonical_base[4],
        canonical_base[5],
    )


def signed_contraction_canonical_key(expr: Any, *, global_counts: Mapping[str, int] | None = None, global_map: Mapping[str, str] | None = None) -> tuple[Any, ...]:
    if global_counts is None:
        global_counts = _whole_tree_index_histogram(expr)
    if global_map is None:
        global_map = _global_dummy_name_map(expr)
    cls = type(expr).__name__
    if cls == "IndexedTensor":
        return _signed_tensor_key(expr, global_counts=global_counts, global_map=global_map)
    if cls == "IndexedTensorExpr":
        op = getattr(expr, "op", None)
        args = list(getattr(expr, "args", ()))
        child_keys = [signed_contraction_canonical_key(a, global_counts=global_counts, global_map=global_map) for a in args]
        if op in {"add", "tensor_product", "mul"}:
            child_keys = sorted(child_keys, key=repr)
        return ("IndexedTensorExpr", op, tuple(child_keys))
    try:
        return ("sympy", sp.srepr(sp.sympify(expr)))
    except Exception:
        return ("repr", repr(expr))


def signed_contraction_canonicalize_report(expr: Any) -> SignedContractionCanonicalizationReport:
    semantic = normalize_semantic_node(compile_semantic_node(expr))
    counts = _whole_tree_index_histogram(expr)
    gmap = _global_dummy_name_map(expr)
    return SignedContractionCanonicalizationReport(
        original=expr,
        canonical_key=signed_contraction_canonical_key(expr, global_counts=counts, global_map=gmap),
        semantic_fingerprint=semantic_node_fingerprint(semantic),
        metadata={"semantic_kind": semantic.kind, "index_histogram": dict(counts), "global_dummy_map": dict(gmap)},
    )


def signed_contraction_equivalent(left: Any, right: Any) -> bool:
    return signed_contraction_canonical_key(left) == signed_contraction_canonical_key(right)
