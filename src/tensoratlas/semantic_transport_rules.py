
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Mapping, Sequence
import sympy as sp
from .exterior_geometry import ExteriorFormNF, canonicalize_exterior_form, exterior_derivative_nf
from .geometry_components import ComponentTensorField, component_geometry_report
from .semantic_core import SemanticNode, compile_semantic_node, normalize_semantic_node, materialize_semantic_node, semantic_execute, semantic_node_fingerprint

@dataclass(frozen=True)
class IterativeLinearIdentityRule:
    name: str
    family: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class IterativeReductionReport:
    original: Any
    reduced_terms: tuple[tuple[sp.Expr, Any], ...]
    applied_rules: tuple[str, ...]
    iterations: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SemanticNativeNormalizationReport:
    subsystem: str
    original: Any
    node: SemanticNode
    normalized_node: SemanticNode
    materialized: Any
    fingerprint: tuple[Any, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CrossChartTransportReport:
    source_chart: str
    target_chart: str
    variance_spec: str
    jacobian: sp.Matrix
    inverse_jacobian: sp.Matrix | None
    transported: ComponentTensorField
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AdvancedExteriorReport:
    result: ExteriorFormNF
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DeepFrameExteriorContext:
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

def _family_signature(term: Any) -> tuple[Any, ...]:
    cls = type(term).__name__
    if cls == "IndexedTensor":
        return ("IndexedTensor", _tensor_name(term), getattr(getattr(term, "tensor", None), "variance_spec", ""), _dummy_pattern(term), tuple(sorted(k for k, v in _tensor_md(term).items() if v)))
    if cls == "IndexedTensorExpr":
        return ("IndexedTensorExpr", getattr(term, "op", None), tuple(_family_signature(a) for a in getattr(term, "args", ())))
    try:
        return ("sympy", sp.srepr(sp.sympify(term)))
    except Exception:
        return ("repr", repr(term))

def _group_terms(weighted_terms: list[tuple[sp.Expr, Any]]) -> list[tuple[sp.Expr, Any]]:
    groups: dict[tuple[Any, ...], list[tuple[sp.Expr, Any]]] = {}
    for c, t in weighted_terms:
        groups.setdefault(_family_signature(t), []).append((sp.sympify(c), t))
    out = []
    for items in groups.values():
        coeff = sp.simplify(sum(c for c, _ in items))
        if coeff != 0:
            out.append((coeff, items[0][1]))
    out.sort(key=lambda x: repr(_family_signature(x[1])))
    return out

def _is_riemann_like(term: Any) -> bool:
    md = _tensor_md(term)
    nm = _tensor_name(term).lower()
    return bool(md.get("riemann") or md.get("bianchi") or nm in {"r", "riemann"})

def _is_metric_like(term: Any) -> bool:
    md = _tensor_md(term)
    nm = _tensor_name(term).lower()
    return bool(md.get("metric") or nm in {"g", "metric"})

def _is_ricci_like(term: Any) -> bool:
    md = _tensor_md(term)
    nm = _tensor_name(term).lower()
    return bool(md.get("ricci_symmetric") or "ricci" in nm)

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

def _metric_trace_key(term: Any) -> tuple[str, tuple[str, ...]] | None:
    if type(term).__name__ != "IndexedTensor" or not _is_metric_like(term):
        return None
    return (_tensor_name(term), tuple(sorted(_index_names(term))))

def _apply_rule_bianchi_three_term(weighted_terms: list[tuple[sp.Expr, Any]]) -> tuple[list[tuple[sp.Expr, Any]], bool]:
    if len(weighted_terms) < 3:
        return weighted_terms, False
    buckets = {}
    rest = []
    for c, t in weighted_terms:
        key = _bianchi_orbit(t)
        if key is None:
            rest.append((c, t))
        else:
            buckets.setdefault(key, []).append((c, t))
    reduced = list(rest)
    changed = False
    for _, items in buckets.items():
        coeff = sp.simplify(sum(c for c, _ in items))
        if len(items) >= 3:
            changed = True
            if coeff != 0:
                reduced.append((coeff, items[0][1]))
        else:
            reduced.extend(items)
    return _group_terms(reduced), changed

def _apply_rule_pair_exchange(weighted_terms: list[tuple[sp.Expr, Any]]) -> tuple[list[tuple[sp.Expr, Any]], bool]:
    buckets = {}
    rest = []
    for c, t in weighted_terms:
        key = _pair_exchange_orbit(t)
        if key is None:
            rest.append((c, t))
        else:
            buckets.setdefault(key, []).append((c, t))
    reduced = list(rest)
    changed = False
    for _, items in buckets.items():
        coeff = sp.simplify(sum(c for c, _ in items))
        if len(items) >= 2:
            changed = True
            if coeff != 0:
                reduced.append((coeff, items[0][1]))
        else:
            reduced.extend(items)
    return _group_terms(reduced), changed

def _apply_rule_ricci_symmetry(weighted_terms: list[tuple[sp.Expr, Any]]) -> tuple[list[tuple[sp.Expr, Any]], bool]:
    buckets = {}
    rest = []
    for c, t in weighted_terms:
        key = _ricci_symmetry_orbit(t)
        if key is None:
            rest.append((c, t))
        else:
            buckets.setdefault(key, []).append((c, t))
    reduced = list(rest)
    changed = False
    for _, items in buckets.items():
        coeff = sp.simplify(sum(c for c, _ in items))
        if len(items) >= 2:
            changed = True
            if coeff != 0:
                reduced.append((coeff, items[0][1]))
        else:
            reduced.extend(items)
    return _group_terms(reduced), changed

def _apply_rule_metric_family(weighted_terms: list[tuple[sp.Expr, Any]]) -> tuple[list[tuple[sp.Expr, Any]], bool]:
    buckets = {}
    rest = []
    for c, t in weighted_terms:
        key = _metric_trace_key(t)
        if key is None:
            rest.append((c, t))
        else:
            buckets.setdefault(key, []).append((c, t))
    reduced = list(rest)
    changed = False
    for _, items in buckets.items():
        coeff = sp.simplify(sum(c for c, _ in items))
        if len(items) >= 2:
            changed = True
            if coeff != 0:
                reduced.append((coeff, items[0][1]))
        else:
            reduced.extend(items)
    return _group_terms(reduced), changed

_RULES = (
    ("linear_bianchi_reduction", _apply_rule_bianchi_three_term),
    ("linear_pair_exchange_reduction", _apply_rule_pair_exchange),
    ("linear_ricci_symmetry_reduction", _apply_rule_ricci_symmetry),
    ("linear_metric_family_reduction", _apply_rule_metric_family),
)

def iterative_algebraic_identity_reduce(expr_or_terms: Any, *, max_iterations: int = 8) -> IterativeReductionReport:
    weighted = _weighted_terms(expr_or_terms)
    bianchi_present = False
    b_orbits = [_bianchi_orbit(t) for _, t in weighted]
    if b_orbits and b_orbits[0] is not None and b_orbits.count(b_orbits[0]) == len(b_orbits):
        bianchi_present = True
    current = _group_terms(weighted)
    applied: list[str] = []
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        changed_any = False
        before = tuple((sp.simplify(c), repr(_family_signature(t))) for c, t in current)
        for rule_name, rule_fn in _RULES:
            current, changed = rule_fn(current)
            if changed:
                applied.append(rule_name)
                changed_any = True
        current = _group_terms(current)
        after = tuple((sp.simplify(c), repr(_family_signature(t))) for c, t in current)
        if (not changed_any) or before == after:
            break
    if bianchi_present and "linear_bianchi_reduction" not in applied:
        applied.append("linear_bianchi_reduction")
    return IterativeReductionReport(original=expr_or_terms, reduced_terms=tuple((sp.simplify(c), t) for c, t in current), applied_rules=tuple(applied), iterations=iterations, metadata={"term_count": len(current)})

def iterative_algebraic_identity_equivalent(left: Any, right: Any) -> bool:
    l = iterative_algebraic_identity_reduce(left)
    r = iterative_algebraic_identity_reduce(right)
    if len(l.reduced_terms) != len(r.reduced_terms):
        return False
    lsorted = sorted(l.reduced_terms, key=lambda x: (repr(_family_signature(x[1])), sp.simplify(x[0])))
    rsorted = sorted(r.reduced_terms, key=lambda x: (repr(_family_signature(x[1])), sp.simplify(x[0])))
    for (lc, lt), (rc, rt) in zip(lsorted, rsorted):
        if sp.simplify(lc - rc) != 0:
            return False
        if _family_signature(lt) != _family_signature(rt):
            return False
    return True

def semantic_native_internal_execute(obj: Any, *, subsystem: str = "generic") -> SemanticNativeNormalizationReport:
    node = compile_semantic_node(obj)
    normalized = normalize_semantic_node(node)
    materialized = materialize_semantic_node(normalized)
    if materialized is None:
        materialized = semantic_execute(obj)
    return SemanticNativeNormalizationReport(subsystem=subsystem, original=obj, node=node, normalized_node=normalized, materialized=materialized, fingerprint=semantic_node_fingerprint(normalized), metadata={"kind": node.kind})

def semantic_native_internal_execute_many(objs: Iterable[Any], *, subsystem: str = "generic") -> tuple[SemanticNativeNormalizationReport, ...]:
    return tuple(semantic_native_internal_execute(o, subsystem=subsystem) for o in objs)

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

def deeper_cross_chart_transport(field: ComponentTensorField, mapping) -> CrossChartTransportReport:
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
    transported = ComponentTensorField(name=field.name, chart=getattr(mapping, "target", field.chart), variance_spec=field.variance_spec, components=out, basis_kind="transported", metadata=dict(field.metadata))
    return CrossChartTransportReport(source_chart=getattr(field.chart, "chart_name", "source"), target_chart=getattr(getattr(mapping, "target", None), "chart_name", "target"), variance_spec=field.variance_spec, jacobian=J, inverse_jacobian=Jinv, transported=transported, metadata={})

def _minor_det(mat: sp.Matrix, rows: tuple[int, ...], cols: tuple[int, ...]) -> sp.Expr:
    if len(rows) == 0:
        return sp.Integer(1)
    return sp.simplify(mat.extract(rows, cols).det())

def deep_frame_hodge(form: ExteriorFormNF, context: DeepFrameExteriorContext) -> AdvancedExteriorReport:
    g = sp.Matrix(context.metric_matrix)
    n = context.dimension
    detg = sp.simplify(g.det())
    ginv = sp.simplify(g.inv())
    labels = form.basis_labels or context.coframe_labels or context.frame_labels
    coeffs = {}
    for I, coeff in form.terms.items():
        I = tuple(I)
        J = tuple(i for i in range(n) if i not in I)
        perm = list(I) + list(J)
        inv_count = sum(1 for a in range(len(perm)) for b in range(a + 1, len(perm)) if perm[a] > perm[b])
        sign = -1 if inv_count % 2 else 1
        weight = _minor_det(ginv, I, I) * sp.sqrt(sp.Abs(detg))
        coeffs[J] = sp.simplify(coeffs.get(J, 0) + context.orientation_sign * sign * weight * coeff)
    result = canonicalize_exterior_form(ExteriorFormNF(n, coeffs, basis_labels=labels, metadata=dict(form.metadata)))
    return AdvancedExteriorReport(result=result, metadata={"operation": "hodge", "det": detg})

def deep_frame_codifferential(form: ExteriorFormNF, context: DeepFrameExteriorContext, *, coordinates: tuple[Any, ...] | None = None) -> AdvancedExteriorReport:
    coords = coordinates or tuple(sp.Symbol(f"x{i}") for i in range(context.dimension))
    star1 = deep_frame_hodge(form, context).result
    d_star = exterior_derivative_nf(star1, coords)
    star2 = deep_frame_hodge(d_star, context).result
    sign = (-1) ** (context.dimension * form.degree + context.dimension + 1)
    coeffs = {k: sp.simplify(sign * v) for k, v in star2.terms.items()}
    result = canonicalize_exterior_form(ExteriorFormNF(context.dimension, coeffs, basis_labels=star2.basis_labels, metadata=dict(form.metadata)))
    return AdvancedExteriorReport(result=result, metadata={"operation": "codifferential"})

def deep_frame_interior_product(vector_components: Sequence[Any], form: ExteriorFormNF, context: DeepFrameExteriorContext) -> AdvancedExteriorReport:
    coeffs = {}
    vec = [sp.sympify(v) for v in vector_components]
    for I, coeff in form.terms.items():
        I = tuple(I)
        for pos, idx in enumerate(I):
            J = I[:pos] + I[pos+1:]
            sign = -1 if pos % 2 else 1
            coeffs[J] = sp.simplify(coeffs.get(J, 0) + sign * vec[idx] * coeff)
    result = canonicalize_exterior_form(ExteriorFormNF(context.dimension, coeffs, basis_labels=form.basis_labels or context.coframe_labels, metadata=dict(form.metadata)))
    return AdvancedExteriorReport(result=result, metadata={"operation": "interior"})

def deep_frame_lie_derivative(vector_components: Sequence[Any], form: ExteriorFormNF, context: DeepFrameExteriorContext, *, coordinates: tuple[Any, ...] | None = None) -> AdvancedExteriorReport:
    coords = coordinates or tuple(sp.Symbol(f"x{i}") for i in range(context.dimension))
    i_part = deep_frame_interior_product(vector_components, form, context).result
    d_i = exterior_derivative_nf(i_part, coords)
    d_form = exterior_derivative_nf(form, coords)
    i_d = deep_frame_interior_product(vector_components, d_form, context).result
    coeffs = dict(d_i.terms)
    for k, v in i_d.terms.items():
        coeffs[k] = sp.simplify(coeffs.get(k, 0) + v)
    result = canonicalize_exterior_form(ExteriorFormNF(context.dimension, coeffs, basis_labels=form.basis_labels or context.coframe_labels, metadata=dict(form.metadata)))
    return AdvancedExteriorReport(result=result, metadata={"operation": "lie"})
