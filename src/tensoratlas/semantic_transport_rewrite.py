
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import sympy as sp

from .exterior_geometry import ExteriorFormNF, canonicalize_exterior_form, exterior_derivative_nf
from .geometry_components import ComponentTensorField, component_geometry_report
from .semantic_core import (
    SemanticNode,
    compile_semantic_node,
    normalize_semantic_node,
    materialize_semantic_node,
    semantic_execute,
    semantic_node_fingerprint,
)


@dataclass(frozen=True)
class RewriteIdentityRule:
    name: str
    family: str
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewriteReductionReport:
    original: Any
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    iterations: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticCanonicalizationReport:
    subsystem: str
    original: Any
    node: SemanticNode
    normalized_node: SemanticNode
    materialized: Any
    fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportedDerivedGeometryReport:
    transported_tensor: ComponentTensorField
    jacobian: sp.Matrix
    inverse_jacobian: sp.Matrix | None
    connection_report: Any = None
    curvature_report: Any = None
    derived_geometry_report: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InteroperableExteriorReport:
    result: ExteriorFormNF
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MixedBasisMetricContext:
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


def _term_family_signature(term: Any) -> tuple[Any, ...]:
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
        return ("IndexedTensorExpr", getattr(term, "op", None), tuple(_term_family_signature(a) for a in getattr(term, "args", ())))
    try:
        return ("sympy", sp.srepr(sp.sympify(term)))
    except Exception:
        return ("repr", repr(term))


def _group_terms(weighted_terms: list[tuple[sp.Expr, Any]]) -> list[tuple[sp.Expr, Any]]:
    groups: dict[tuple[Any, ...], list[tuple[sp.Expr, Any]]] = {}
    for c, t in weighted_terms:
        groups.setdefault(_term_family_signature(t), []).append((sp.sympify(c), t))
    out = []
    for items in groups.values():
        coeff = sp.simplify(sum(c for c, _ in items))
        if coeff != 0:
            out.append((coeff, items[0][1]))
    out.sort(key=lambda x: repr(_term_family_signature(x[1])))
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


def _metric_family_key(term: Any) -> tuple[str, tuple[str, ...]] | None:
    if type(term).__name__ != "IndexedTensor" or not _is_metric_like(term):
        return None
    return (_tensor_name(term), tuple(sorted(_index_names(term))))


def _apply_orbit_rule(weighted_terms: list[tuple[sp.Expr, Any]], key_fn, threshold: int) -> tuple[list[tuple[sp.Expr, Any]], bool]:
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
    return _group_terms(reduced), changed


_REWRITE_RULES = (
    ("rewrite_bianchi_three_term", lambda terms: _apply_orbit_rule(terms, _bianchi_orbit, 3)),
    ("rewrite_riemann_pair_exchange", lambda terms: _apply_orbit_rule(terms, _pair_exchange_orbit, 2)),
    ("rewrite_ricci_symmetry", lambda terms: _apply_orbit_rule(terms, _ricci_symmetry_orbit, 2)),
    ("rewrite_metric_family", lambda terms: _apply_orbit_rule(terms, _metric_family_key, 2)),
)


def rewrite_system_identity_reduce(expr_or_terms: Any, *, max_iterations: int = 10) -> RewriteReductionReport:
    original_weighted = _weighted_terms(expr_or_terms)
    current = _group_terms(original_weighted)
    applied: list[str] = []
    iterations = 0

    if len(current) != len(original_weighted):
        applied.append("initial_grouping")

    for _ in range(max_iterations):
        iterations += 1
        before = tuple((sp.simplify(c), repr(_term_family_signature(t))) for c, t in current)
        changed_any = False
        for rule_name, rule_fn in _REWRITE_RULES:
            current, changed = rule_fn(current)
            if changed:
                applied.append(rule_name)
                changed_any = True
        current = _group_terms(current)
        after = tuple((sp.simplify(c), repr(_term_family_signature(t))) for c, t in current)
        if (not changed_any) or before == after:
            break

    return RewriteReductionReport(
        original=expr_or_terms,
        reduced_terms=tuple((sp.simplify(c), t) for c, t in current),
        applied_rules=tuple(applied),
        iterations=iterations,
        metadata={"term_count": len(current)},
    )


def rewrite_system_identity_equivalent(left: Any, right: Any) -> bool:
    l = rewrite_system_identity_reduce(left)
    r = rewrite_system_identity_reduce(right)
    if len(l.reduced_terms) != len(r.reduced_terms):
        return False
    lsorted = sorted(l.reduced_terms, key=lambda x: (repr(_term_family_signature(x[1])), sp.simplify(x[0])))
    rsorted = sorted(r.reduced_terms, key=lambda x: (repr(_term_family_signature(x[1])), sp.simplify(x[0])))
    for (lc, lt), (rc, rt) in zip(lsorted, rsorted):
        if sp.simplify(lc - rc) != 0:
            return False
        if _term_family_signature(lt) != _term_family_signature(rt):
            return False
    return True


def semantic_core_native_canonicalize(obj: Any, *, subsystem: str = "generic") -> SemanticCanonicalizationReport:
    node = compile_semantic_node(obj)
    normalized = normalize_semantic_node(node)
    materialized = materialize_semantic_node(normalized)
    if materialized is None:
        materialized = semantic_execute(obj)
    return SemanticCanonicalizationReport(
        subsystem=subsystem,
        original=obj,
        node=node,
        normalized_node=normalized,
        materialized=materialized,
        fingerprint=semantic_node_fingerprint(normalized),
        metadata={"kind": node.kind},
    )


def semantic_core_native_canonicalize_many(objs: Iterable[Any], *, subsystem: str = "generic") -> tuple[SemanticCanonicalizationReport, ...]:
    return tuple(semantic_core_native_canonicalize(o, subsystem=subsystem) for o in objs)


def _as_matrix_or_identity(mapping, chart_dim: int, coords: tuple[Any, ...]) -> tuple[sp.Matrix, sp.Matrix | None]:
    if hasattr(mapping, "jacobian"):
        J = sp.Matrix(mapping.jacobian(coords))
    else:
        J = sp.eye(chart_dim)
    try:
        Jinv = sp.simplify(J.inv())
    except Exception:
        Jinv = None
    return J, Jinv


def deeper_cross_chart_transport_to_derived(field: ComponentTensorField, mapping) -> TransportedDerivedGeometryReport:
    dim = field.chart.dimension
    coords = tuple(field.chart.symbols())
    J, Jinv = _as_matrix_or_identity(mapping, dim, coords)
    arr = field.components
    rank = len(field.variance_spec)
    if rank == 0:
        base_val = arr[()] if hasattr(arr, "__getitem__") else arr
        out = sp.MutableDenseNDimArray([base_val])
    else:
        out = sp.MutableDenseNDimArray.zeros(*([dim] * rank))
        for out_idx in product(range(dim), repeat=rank):
            total = sp.Integer(0)
            for in_idx in product(range(dim), repeat=rank):
                factor = sp.Integer(1)
                for pos, var in enumerate(field.variance_spec):
                    if var == "u":
                        factor *= J[out_idx[pos], in_idx[pos]]
                    else:
                        factor *= (Jinv[in_idx[pos], out_idx[pos]] if Jinv is not None else sp.KroneckerDelta(in_idx[pos], out_idx[pos]))
                total += factor * arr[in_idx]
            out[out_idx] = sp.simplify(total)
    transported = ComponentTensorField(
        name=field.name,
        chart=getattr(mapping, "target", field.chart),
        variance_spec=field.variance_spec,
        components=out,
        basis_kind="transported",
        metadata=dict(field.metadata),
    )
    conn = component_geometry_report(transported.chart, include_curvature=False)
    curv = component_geometry_report(transported.chart, include_curvature=True)
    return TransportedDerivedGeometryReport(
        transported_tensor=transported,
        jacobian=J,
        inverse_jacobian=Jinv,
        connection_report=conn,
        curvature_report=curv,
        derived_geometry_report={"scalar_curvature": curv.scalar_curvature},
        metadata={},
    )


def _minor_det(mat: sp.Matrix, rows: tuple[int, ...], cols: tuple[int, ...]) -> sp.Expr:
    if len(rows) == 0:
        return sp.Integer(1)
    return sp.simplify(mat.extract(rows, cols).det())


def _basis_permutation_sign(labels_from: Sequence[str], labels_to: Sequence[str]) -> int:
    pos = {lab: i for i, lab in enumerate(labels_to)}
    perm = [pos[l] for l in labels_from if l in pos]
    inv = sum(1 for a in range(len(perm)) for b in range(a + 1, len(perm)) if perm[a] > perm[b])
    return -1 if inv % 2 else 1


def _convert_basis(form: ExteriorFormNF, context: MixedBasisMetricContext, *, to_coframe: bool) -> ExteriorFormNF:
    source = form.basis_labels or (context.frame_labels if not to_coframe else context.coframe_labels)
    target = context.coframe_labels if to_coframe else context.frame_labels
    if tuple(source) == tuple(target):
        return form
    sign = _basis_permutation_sign(source, target)
    coeffs = {k: sp.simplify(sign * v) for k, v in form.terms.items()}
    return canonicalize_exterior_form(ExteriorFormNF(context.dimension, coeffs, basis_labels=target, metadata=dict(form.metadata)))


def mixed_basis_hodge(form: ExteriorFormNF, context: MixedBasisMetricContext) -> InteroperableExteriorReport:
    form_cf = _convert_basis(form, context, to_coframe=True)
    g = sp.Matrix(context.metric_matrix)
    n = context.dimension
    detg = sp.simplify(g.det())
    ginv = sp.simplify(g.inv())
    coeffs: dict[tuple[int, ...], sp.Expr] = {}
    for I, coeff in form_cf.terms.items():
        I = tuple(I)
        J = tuple(i for i in range(n) if i not in I)
        perm = list(I) + list(J)
        inv_count = sum(1 for a in range(len(perm)) for b in range(a + 1, len(perm)) if perm[a] > perm[b])
        sign = -1 if inv_count % 2 else 1
        weight = _minor_det(ginv, I, I) * sp.sqrt(sp.Abs(detg))
        coeffs[J] = sp.simplify(coeffs.get(J, 0) + context.orientation_sign * sign * weight * coeff)
    result = canonicalize_exterior_form(ExteriorFormNF(n, coeffs, basis_labels=context.coframe_labels, metadata=dict(form.metadata)))
    return InteroperableExteriorReport(result=result, metadata={"operation": "hodge", "basis": "coframe"})


def mixed_basis_codifferential(form: ExteriorFormNF, context: MixedBasisMetricContext, *, coordinates: tuple[Any, ...] | None = None) -> InteroperableExteriorReport:
    coords = coordinates or tuple(sp.Symbol(f"x{i}") for i in range(context.dimension))
    star1 = mixed_basis_hodge(form, context).result
    d_star = exterior_derivative_nf(star1, coords)
    star2 = mixed_basis_hodge(d_star, context).result
    sign = (-1) ** (context.dimension * form.degree + context.dimension + 1)
    coeffs = {k: sp.simplify(sign * v) for k, v in star2.terms.items()}
    result = canonicalize_exterior_form(ExteriorFormNF(context.dimension, coeffs, basis_labels=context.coframe_labels, metadata=dict(form.metadata)))
    return InteroperableExteriorReport(result=result, metadata={"operation": "codifferential", "basis": "coframe"})


def mixed_basis_interior_product(vector_components: Sequence[Any], form: ExteriorFormNF, context: MixedBasisMetricContext) -> InteroperableExteriorReport:
    form_cf = _convert_basis(form, context, to_coframe=True)
    coeffs: dict[tuple[int, ...], sp.Expr] = {}
    vec = [sp.sympify(v) for v in vector_components]
    for I, coeff in form_cf.terms.items():
        I = tuple(I)
        for pos, idx in enumerate(I):
            J = I[:pos] + I[pos+1:]
            sign = -1 if pos % 2 else 1
            coeffs[J] = sp.simplify(coeffs.get(J, 0) + sign * vec[idx] * coeff)
    result = canonicalize_exterior_form(ExteriorFormNF(context.dimension, coeffs, basis_labels=context.coframe_labels, metadata=dict(form.metadata)))
    return InteroperableExteriorReport(result=result, metadata={"operation": "interior", "basis": "coframe"})


def mixed_basis_lie_derivative(vector_components: Sequence[Any], form: ExteriorFormNF, context: MixedBasisMetricContext, *, coordinates: tuple[Any, ...] | None = None) -> InteroperableExteriorReport:
    coords = coordinates or tuple(sp.Symbol(f"x{i}") for i in range(context.dimension))
    i_part = mixed_basis_interior_product(vector_components, form, context).result
    d_i = exterior_derivative_nf(i_part, coords)
    d_form = exterior_derivative_nf(_convert_basis(form, context, to_coframe=True), coords)
    i_d = mixed_basis_interior_product(vector_components, d_form, context).result
    coeffs = dict(d_i.terms)
    for k, v in i_d.terms.items():
        coeffs[k] = sp.simplify(coeffs.get(k, 0) + v)
    result = canonicalize_exterior_form(ExteriorFormNF(context.dimension, coeffs, basis_labels=context.coframe_labels, metadata=dict(form.metadata)))
    return InteroperableExteriorReport(result=result, metadata={"operation": "lie", "basis": "coframe"})
