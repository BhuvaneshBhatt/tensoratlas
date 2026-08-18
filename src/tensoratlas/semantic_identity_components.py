
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import sympy as sp

from .exterior_geometry import ExteriorFormNF, canonicalize_exterior_form, exterior_derivative_nf
from .semantic_core import compile_semantic_node, normalize_semantic_node, materialize_semantic_node, semantic_execute
from .geometry_components import ComponentTensorField


@dataclass(frozen=True)
class MultiTermIdentityRule:
    name: str
    family: str
    arity: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReducedMultiTermExpression:
    terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticSubsystemExecutionReport:
    subsystem: str
    original: Any
    semantic_kind: str
    normalized: Any
    materialized: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnhancedFrameGeometryContext:
    dimension: int
    metric_matrix: sp.Matrix
    frame_labels: tuple[str, ...]
    coframe_labels: tuple[str, ...]
    orientation_sign: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _flatten_add_like(expr: Any) -> list[Any]:
    if type(expr).__name__ == "IndexedTensorExpr" and getattr(expr, "op", None) == "add":
        out = []
        for arg in getattr(expr, "args", ()):
            out.extend(_flatten_add_like(arg))
        return out
    return [expr]


def _as_weighted_terms(expr_or_terms: Any) -> list[tuple[sp.Expr, Any]]:
    if isinstance(expr_or_terms, Sequence) and not isinstance(expr_or_terms, (str, bytes)):
        out = []
        for item in expr_or_terms:
            if isinstance(item, tuple) and len(item) == 2:
                out.append((sp.sympify(item[0]), item[1]))
            else:
                out.append((sp.Integer(1), item))
        return out
    return [(sp.Integer(1), term) for term in _flatten_add_like(expr_or_terms)]


def _dummy_class_pattern(indexed_tensor: Any) -> tuple[tuple[str, int], ...]:
    names = [getattr(i, "name", str(i)) for i in getattr(indexed_tensor, "indices", ())]
    counts: dict[str, int] = {}
    out = []
    next_id = 0
    remap: dict[str, int] = {}
    for pos, name in enumerate(names):
        if name not in remap:
            remap[name] = next_id
            next_id += 1
        out.append((str(getattr(indexed_tensor.indices[pos], "variance", "")), remap[name]))
    return tuple(out)


def _safe_term_key(term: Any) -> tuple[Any, ...]:
    cls = type(term).__name__
    if cls == "IndexedTensor":
        tensor = getattr(term, "tensor", None)
        name = getattr(tensor, "name", "")
        variance = getattr(tensor, "variance_spec", "")
        md = getattr(tensor, "symmetry_metadata", {}) or {}
        family = tuple(sorted(k for k, v in md.items() if v))
        return ("IndexedTensor", name, variance, _dummy_class_pattern(term), family)
    if cls == "IndexedTensorExpr":
        op = getattr(term, "op", None)
        return ("IndexedTensorExpr", op, tuple(_safe_term_key(a) for a in getattr(term, "args", ())))
    try:
        return ("repr", sp.srepr(sp.sympify(term)))
    except Exception:
        return ("repr", repr(term))


def _canonical_group_reduce(weighted_terms: list[tuple[sp.Expr, Any]]) -> tuple[tuple[sp.Expr, Any], ...]:
    groups: dict[tuple[Any, ...], list[tuple[sp.Expr, Any]]] = {}
    for coeff, term in weighted_terms:
        groups.setdefault(_safe_term_key(term), []).append((sp.sympify(coeff), term))
    reduced: list[tuple[sp.Expr, Any]] = []
    for _, items in groups.items():
        coeff_sum = sp.simplify(sum(c for c, _ in items))
        if coeff_sum != 0:
            reduced.append((coeff_sum, items[0][1]))
    reduced.sort(key=lambda x: repr(_safe_term_key(x[1])))
    return tuple(reduced)


def _tensor_name(obj: Any) -> str:
    return getattr(getattr(obj, "tensor", None), "name", "") or ""


def _tensor_variance(obj: Any) -> str:
    return getattr(getattr(obj, "tensor", None), "variance_spec", "")


def _index_names(indexed_tensor) -> tuple[str, ...]:
    return tuple(getattr(idx, "name", str(idx)) for idx in getattr(indexed_tensor, "indices", ()))


def _is_riemann_like(indexed_tensor) -> bool:
    name = _tensor_name(indexed_tensor).lower()
    md = getattr(getattr(indexed_tensor, "tensor", None), "symmetry_metadata", {}) or {}
    return bool(md.get("riemann") or md.get("bianchi") or name == "r" or "riemann" in name)


def _is_ricci_like(indexed_tensor) -> bool:
    name = _tensor_name(indexed_tensor).lower()
    md = getattr(getattr(indexed_tensor, "tensor", None), "symmetry_metadata", {}) or {}
    return bool(md.get("ricci_symmetric") or "ricci" in name)


def _is_metric_like(indexed_tensor) -> bool:
    name = _tensor_name(indexed_tensor).lower()
    md = getattr(getattr(indexed_tensor, "tensor", None), "symmetry_metadata", {}) or {}
    return bool(md.get("metric") or name == "g" or "metric" in name)


def _is_weyl_like(indexed_tensor) -> bool:
    name = _tensor_name(indexed_tensor).lower()
    md = getattr(getattr(indexed_tensor, "tensor", None), "symmetry_metadata", {}) or {}
    return bool(md.get("weyl") or name == "c" or "weyl" in name)


def _is_epsilon_like(indexed_tensor) -> bool:
    name = _tensor_name(indexed_tensor).lower()
    md = getattr(getattr(indexed_tensor, "tensor", None), "symmetry_metadata", {}) or {}
    return bool(md.get("epsilon") or md.get("levi_civita") or name in {"eps", "epsilon"})


def _is_delta_like(indexed_tensor) -> bool:
    name = _tensor_name(indexed_tensor).lower()
    md = getattr(getattr(indexed_tensor, "tensor", None), "symmetry_metadata", {}) or {}
    return bool(md.get("delta") or name in {"delta", "kronecker"})


def _same_tensor_family(tensors: Iterable[Any], pred) -> bool:
    lst = list(tensors)
    return bool(lst) and all(pred(t) for t in lst) and len({(_tensor_name(t), _tensor_variance(t)) for t in lst}) == 1


def _bianchi_signature(weighted_terms: list[tuple[sp.Expr, Any]]) -> tuple[Any, ...] | None:
    if len(weighted_terms) != 3:
        return None
    terms = [t for _, t in weighted_terms]
    coeffs = tuple(sp.simplify(c) for c, _ in weighted_terms)
    if not _same_tensor_family(terms, _is_riemann_like):
        return None
    tuples = [_index_names(t) for t in terms]
    if not all(len(t) == 4 for t in tuples):
        return None
    for base in tuples:
        a, b, c, d = base
        expected = {(a, b, c, d), (a, c, d, b), (a, d, b, c)}
        if set(tuples) == expected:
            return ("riemann_bianchi_three_term", _tensor_name(terms[0]), tuple(sorted(map(sp.srepr, coeffs))))
    return None


def general_multi_term_tensor_identity_engine(expr_or_terms: Any) -> ReducedMultiTermExpression:
    weighted_terms = _as_weighted_terms(expr_or_terms)
    applied: list[str] = []

    reduced = list(_canonical_group_reduce(weighted_terms))
    if len(reduced) != len(weighted_terms):
        applied.append("coefficient_consolidation")

    bsig = _bianchi_signature(reduced)
    if bsig is not None:
        applied.append("riemann_bianchi_three_term")

    current = tuple(reduced)
    return ReducedMultiTermExpression(current, tuple(applied), {"term_count": len(current), "bianchi_signature": bsig})


def general_multi_term_identity_equivalent(left: Any, right: Any) -> bool:
    lred = general_multi_term_tensor_identity_engine(left)
    rred = general_multi_term_tensor_identity_engine(right)
    if lred.metadata.get("bianchi_signature") is not None and lred.metadata.get("bianchi_signature") == rred.metadata.get("bianchi_signature"):
        return True
    if len(lred.terms) != len(rred.terms):
        return False
    for (lc, lt), (rc, rt) in zip(lred.terms, rred.terms):
        if sp.simplify(lc - rc) != 0:
            return False
        if _safe_term_key(lt) != _safe_term_key(rt):
            return False
    return True


def semantic_core_native_execute(obj: Any, *, subsystem: str = "generic") -> SemanticSubsystemExecutionReport:
    node = compile_semantic_node(obj)
    normalized = normalize_semantic_node(node)
    materialized = materialize_semantic_node(normalized)
    executed = semantic_execute(obj)
    return SemanticSubsystemExecutionReport(
        subsystem=subsystem,
        original=obj,
        semantic_kind=node.kind,
        normalized=normalized,
        materialized=materialized if materialized is not None else executed,
        metadata={"node_kind": node.kind},
    )


def semantic_core_native_execute_many(objs: Iterable[Any], *, subsystem: str = "generic") -> tuple[SemanticSubsystemExecutionReport, ...]:
    return tuple(semantic_core_native_execute(obj, subsystem=subsystem) for obj in objs)


def component_tensor_change_basis(field: ComponentTensorField, transform_matrix: Any) -> ComponentTensorField:
    mat = sp.Matrix(transform_matrix)
    arr = field.components
    rank = len(field.variance_spec)
    dim = field.chart.dimension
    if rank == 1:
        vec = sp.Matrix([arr[i] for i in range(dim)])
        new_vec = sp.simplify(mat * vec if field.variance_spec == "u" else mat.T * vec)
        new_arr = sp.MutableDenseNDimArray([sp.simplify(v) for v in new_vec], (dim,))
    elif rank == 2:
        M = sp.Matrix([[arr[i, j] for j in range(dim)] for i in range(dim)])
        left = mat if field.variance_spec[0] == "u" else mat.T
        right = mat if field.variance_spec[1] == "u" else mat.T
        new_M = sp.simplify(left * M * right.T)
        new_arr = sp.MutableDenseNDimArray([[sp.simplify(new_M[i, j]) for j in range(dim)] for i in range(dim)])
    else:
        new_arr = sp.MutableDenseNDimArray(arr)
    return ComponentTensorField(
        name=field.name,
        chart=field.chart,
        variance_spec=field.variance_spec,
        components=new_arr,
        basis_kind="transformed",
        metadata=dict(field.metadata),
    )


def _minor_det(mat: sp.Matrix, rows: tuple[int, ...], cols: tuple[int, ...]) -> sp.Expr:
    if len(rows) == 0:
        return sp.Integer(1)
    return sp.simplify(mat.extract(rows, cols).det())


def advanced_frame_metric_hodge(form: ExteriorFormNF, context: EnhancedFrameGeometryContext) -> ExteriorFormNF:
    g = sp.Matrix(context.metric_matrix)
    n = context.dimension
    detg = sp.simplify(g.det())
    labels = form.basis_labels or context.coframe_labels or context.frame_labels
    coeffs: dict[tuple[int, ...], sp.Expr] = {}
    for I, coeff in form.terms.items():
        I = tuple(I)
        J = tuple(i for i in range(n) if i not in I)
        perm = list(I) + list(J)
        inv_count = sum(1 for a in range(len(perm)) for b in range(a + 1, len(perm)) if perm[a] > perm[b])
        sign = -1 if inv_count % 2 else 1
        weight = _minor_det(g.inv(), I, I) * sp.sqrt(sp.Abs(detg))
        coeffs[J] = sp.simplify(coeffs.get(J, 0) + context.orientation_sign * sign * weight * coeff)
    return canonicalize_exterior_form(ExteriorFormNF(context.dimension, coeffs, basis_labels=labels, metadata=dict(form.metadata)))


def advanced_frame_metric_codifferential(form: ExteriorFormNF, context: EnhancedFrameGeometryContext, *, coordinates: tuple[Any, ...] | None = None) -> ExteriorFormNF:
    coords = coordinates or tuple(sp.Symbol(f"x{i}") for i in range(context.dimension))
    star1 = advanced_frame_metric_hodge(form, context)
    d_star = exterior_derivative_nf(star1, coords)
    star2 = advanced_frame_metric_hodge(d_star, context)
    sign = (-1) ** (context.dimension * form.degree + context.dimension + 1)
    coeffs = {k: sp.simplify(sign * v) for k, v in star2.terms.items()}
    return canonicalize_exterior_form(ExteriorFormNF(context.dimension, coeffs, basis_labels=star2.basis_labels, metadata=dict(form.metadata)))
