
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import sympy as sp

from .exterior_geometry import ExteriorFormNF, canonicalize_exterior_form, exterior_derivative_nf
from .geometry_components import ComponentTensorField, component_geometry_report
from .semantic_core import compile_semantic_node, normalize_semantic_node, materialize_semantic_node, semantic_execute, semantic_node_fingerprint


@dataclass(frozen=True)
class AlgebraicIdentityRule:
    name: str
    family: str
    arity: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgebraicReductionReport:
    original: Any
    terms: tuple[tuple[sp.Expr, Any], ...]
    basis: tuple[str, ...]
    applied_rules: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedSemanticExecutionReport:
    subsystem: str
    original: Any
    semantic_kind: str
    normalized_node: Any
    materialized: Any
    fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossChartComponentReport:
    source_chart: str
    target_chart: str
    transformed: ComponentTensorField
    jacobian: sp.Matrix
    inverse_jacobian: sp.Matrix | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DerivedGeometryReport:
    base: Any
    gradient: sp.Matrix | None = None
    divergence: sp.Expr | None = None
    laplacian: sp.Expr | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FullFrameHodgeContext:
    dimension: int
    metric_matrix: sp.Matrix
    frame_labels: tuple[str, ...]
    coframe_labels: tuple[str, ...]
    orientation_sign: int = 1
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


def _safe_term_key(term: Any) -> tuple[Any, ...]:
    cls = type(term).__name__
    if cls == "IndexedTensor":
        tensor = getattr(term, "tensor", None)
        name = getattr(tensor, "name", "")
        variance = getattr(tensor, "variance_spec", "")
        md = tuple(sorted(k for k, v in (_tensor_md(term)).items() if v))
        pattern = []
        remap = {}
        next_id = 0
        for idx in getattr(term, "indices", ()):
            nm = getattr(idx, "name", str(idx))
            if nm not in remap:
                remap[nm] = next_id
                next_id += 1
            pattern.append((getattr(idx, "variance", ""), remap[nm]))
        return ("IndexedTensor", name, variance, tuple(pattern), md)
    if cls == "IndexedTensorExpr":
        return ("IndexedTensorExpr", getattr(term, "op", None), tuple(_safe_term_key(a) for a in getattr(term, "args", ())))
    try:
        return ("sympy", sp.srepr(sp.sympify(term)))
    except Exception:
        return ("repr", repr(term))


def _is_riemann_like(term: Any) -> bool:
    md = _tensor_md(term)
    nm = _tensor_name(term).lower()
    return bool(md.get("riemann") or md.get("bianchi") or nm in {"r", "riemann"})


def _is_metric_like(term: Any) -> bool:
    md = _tensor_md(term)
    nm = _tensor_name(term).lower()
    return bool(md.get("metric") or nm in {"g", "metric"})


def _bianchi_basis_key(term: Any) -> tuple[str, tuple[str, ...]] | None:
    if type(term).__name__ != "IndexedTensor" or not _is_riemann_like(term):
        return None
    idx = _index_names(term)
    if len(idx) != 4:
        return None
    a, b, c, d = idx
    orbit = tuple(sorted((f"{a}{b}{c}{d}", f"{a}{c}{d}{b}", f"{a}{d}{b}{c}")))
    return (_tensor_name(term), orbit)


def _metric_raise_key(term: Any) -> tuple[str, tuple[str, ...]] | None:
    if type(term).__name__ != "IndexedTensor" or not _is_metric_like(term):
        return None
    return (_tensor_name(term), tuple(sorted(_index_names(term))))


def _coefficient_consolidate(weighted_terms: list[tuple[sp.Expr, Any]]) -> list[tuple[sp.Expr, Any]]:
    groups: dict[tuple[Any, ...], list[tuple[sp.Expr, Any]]] = {}
    for c, t in weighted_terms:
        groups.setdefault(_safe_term_key(t), []).append((sp.sympify(c), t))
    out: list[tuple[sp.Expr, Any]] = []
    for items in groups.values():
        coeff = sp.simplify(sum(c for c, _ in items))
        if coeff != 0:
            out.append((coeff, items[0][1]))
    out.sort(key=lambda x: repr(_safe_term_key(x[1])))
    return out


def _apply_bianchi_family_reduction(weighted_terms: list[tuple[sp.Expr, Any]]) -> tuple[list[tuple[sp.Expr, Any]], bool]:
    if len(weighted_terms) != 3:
        return weighted_terms, False
    basis = [_bianchi_basis_key(t) for _, t in weighted_terms]
    if basis[0] is None or basis.count(basis[0]) != 3:
        return weighted_terms, False
    coeff_sum = sp.simplify(sum(c for c, _ in weighted_terms))
    if coeff_sum == 0:
        return [], True
    return weighted_terms, False


def _apply_metric_pair_family(weighted_terms: list[tuple[sp.Expr, Any]]) -> tuple[list[tuple[sp.Expr, Any]], bool]:
    # modest algebraic rule: if two metric-identical terms appear with opposite coefficients, cancel.
    metrics = {}
    out = []
    changed = False
    for c, t in weighted_terms:
        key = _metric_raise_key(t)
        if key is None:
            out.append((c, t))
            continue
        prev = metrics.get(key)
        if prev is None:
            metrics[key] = [c, t]
        else:
            pc, pt = prev
            newc = sp.simplify(pc + c)
            metrics[key] = [newc, pt]
            changed = True
    for key, val in metrics.items():
        c, t = val
        if sp.simplify(c) != 0:
            out.append((sp.simplify(c), t))
    out.extend([(sp.simplify(c), t) for c, t in out if False])  # no-op, preserves type stability
    out.sort(key=lambda x: repr(_safe_term_key(x[1])))
    return out, changed


def algebraic_multi_term_tensor_identity_engine(expr_or_terms: Any) -> AlgebraicReductionReport:
    weighted = _weighted_terms(expr_or_terms)
    applied = []

    current = _coefficient_consolidate(weighted)
    if len(current) != len(weighted):
        applied.append("coefficient_consolidation")

    current, changed = _apply_bianchi_family_reduction(current)
    if changed:
        applied.append("riemann_bianchi_three_term")

    current2, changed2 = _apply_metric_pair_family(current)
    if changed2:
        current = _coefficient_consolidate(current2)
        applied.append("metric_family_pair_consolidation")

    basis = tuple(repr(_safe_term_key(t)) for _, t in current)
    return AlgebraicReductionReport(
        original=expr_or_terms,
        terms=tuple((sp.simplify(c), t) for c, t in current),
        basis=basis,
        applied_rules=tuple(applied),
        metadata={"term_count": len(current)},
    )


def algebraic_multi_term_identity_equivalent(left: Any, right: Any) -> bool:
    l = algebraic_multi_term_tensor_identity_engine(left)
    r = algebraic_multi_term_tensor_identity_engine(right)
    if len(l.terms) != len(r.terms):
        return False
    for (lc, lt), (rc, rt) in zip(l.terms, r.terms):
        if sp.simplify(lc - rc) != 0:
            return False
        if _safe_term_key(lt) != _safe_term_key(rt):
            return False
    return True


def unified_semantic_subsystem_execute(obj: Any, *, subsystem: str = "generic") -> UnifiedSemanticExecutionReport:
    node = compile_semantic_node(obj)
    normalized = normalize_semantic_node(node)
    materialized = materialize_semantic_node(normalized)
    executed = semantic_execute(obj)
    materialized = materialized if materialized is not None else executed
    return UnifiedSemanticExecutionReport(
        subsystem=subsystem,
        original=obj,
        semantic_kind=node.kind,
        normalized_node=normalized,
        materialized=materialized,
        fingerprint=semantic_node_fingerprint(normalized),
        metadata={"node_kind": node.kind},
    )


def unified_semantic_subsystem_execute_many(objs: Iterable[Any], *, subsystem: str = "generic") -> tuple[UnifiedSemanticExecutionReport, ...]:
    return tuple(unified_semantic_subsystem_execute(o, subsystem=subsystem) for o in objs)


def component_tensor_change_basis_higher_rank(field: ComponentTensorField, transform_matrix: Any) -> ComponentTensorField:
    mat = sp.Matrix(transform_matrix)
    arr = field.components
    rank = len(field.variance_spec)
    dim = field.chart.dimension
    if rank == 0:
        return field
    out = sp.MutableDenseNDimArray.zeros(*([dim] * rank))
    from itertools import product
    for out_idx in product(range(dim), repeat=rank):
        total = sp.Integer(0)
        for in_idx in product(range(dim), repeat=rank):
            factor = sp.Integer(1)
            for pos, var in enumerate(field.variance_spec):
                factor *= mat[out_idx[pos], in_idx[pos]] if var == "u" else mat[in_idx[pos], out_idx[pos]]
            total += factor * arr[in_idx]
        out[out_idx] = sp.simplify(total)
    return ComponentTensorField(
        name=field.name,
        chart=field.chart,
        variance_spec=field.variance_spec,
        components=out,
        basis_kind="transformed",
        metadata=dict(field.metadata),
    )


def component_tensor_cross_chart_transform(field: ComponentTensorField, mapping) -> CrossChartComponentReport:
    src_coords = tuple(field.chart.symbols())
    J = mapping.jacobian(src_coords) if hasattr(mapping, "jacobian") else sp.eye(field.chart.dimension)
    J = sp.Matrix(J)
    Jinv = None
    try:
        Jinv = sp.simplify(J.inv())
    except Exception:
        Jinv = None
    transformed = component_tensor_change_basis_higher_rank(field, J)
    return CrossChartComponentReport(
        source_chart=getattr(field.chart, "chart_name", "source"),
        target_chart=getattr(getattr(mapping, "target", None), "chart_name", "target"),
        transformed=transformed,
        jacobian=J,
        inverse_jacobian=Jinv,
        metadata={},
    )


def component_tensor_with_derived_geometry(field: ComponentTensorField) -> DerivedGeometryReport:
    rep = component_geometry_report(field.chart, include_curvature=True)
    coords = tuple(field.chart.symbols())
    gradient = None
    divergence = None
    laplacian = None
    if len(field.variance_spec) == 0 and len(field.components.shape) == 0:
        scalar = sp.sympify(field.components[()])
        gradient = sp.Matrix([sp.diff(scalar, c) for c in coords])
        laplacian = sp.simplify(sum(sp.diff(scalar, c, 2) for c in coords))
    elif field.variance_spec == "u" and len(field.components.shape) == 1:
        divergence = sp.simplify(sum(sp.diff(field.components[i], coords[i]) for i in range(field.chart.dimension)))
    return DerivedGeometryReport(base=rep, gradient=gradient, divergence=divergence, laplacian=laplacian, metadata={})


def _minor_det(mat: sp.Matrix, rows: tuple[int, ...], cols: tuple[int, ...]) -> sp.Expr:
    if len(rows) == 0:
        return sp.Integer(1)
    return sp.simplify(mat.extract(rows, cols).det())


def full_frame_metric_hodge(form: ExteriorFormNF, context: FullFrameHodgeContext) -> ExteriorFormNF:
    g = sp.Matrix(context.metric_matrix)
    n = context.dimension
    detg = sp.simplify(g.det())
    labels = form.basis_labels or context.coframe_labels or context.frame_labels
    coeffs: dict[tuple[int, ...], sp.Expr] = {}
    ginv = sp.simplify(g.inv())
    for I, coeff in form.terms.items():
        I = tuple(I)
        J = tuple(i for i in range(n) if i not in I)
        perm = list(I) + list(J)
        inv_count = sum(1 for a in range(len(perm)) for b in range(a + 1, len(perm)) if perm[a] > perm[b])
        sign = -1 if inv_count % 2 else 1
        # more complete non-diagonal weighting via principal minor in inverse metric
        weight = _minor_det(ginv, I, I) * sp.sqrt(sp.Abs(detg))
        coeffs[J] = sp.simplify(coeffs.get(J, 0) + context.orientation_sign * sign * weight * coeff)
    return canonicalize_exterior_form(ExteriorFormNF(context.dimension, coeffs, basis_labels=labels, metadata=dict(form.metadata)))


def full_frame_metric_codifferential(form: ExteriorFormNF, context: FullFrameHodgeContext, *, coordinates: tuple[Any, ...] | None = None) -> ExteriorFormNF:
    coords = coordinates or tuple(sp.Symbol(f"x{i}") for i in range(context.dimension))
    star1 = full_frame_metric_hodge(form, context)
    d_star = exterior_derivative_nf(star1, coords)
    star2 = full_frame_metric_hodge(d_star, context)
    sign = (-1) ** (context.dimension * form.degree + context.dimension + 1)
    coeffs = {k: sp.simplify(sign * v) for k, v in star2.terms.items()}
    return canonicalize_exterior_form(ExteriorFormNF(context.dimension, coeffs, basis_labels=star2.basis_labels, metadata=dict(form.metadata)))
