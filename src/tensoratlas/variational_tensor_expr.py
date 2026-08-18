from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .declarations import DeclarationRegistry, standard_riemannian_registry
from .semantic_ir import (
    TensorExpr,
    canonical_ir_key,
    covariant_derivative_ir,
    ir_node,
    normalize_tensor_expr,
    scalar_ir,
    to_tensor_expr,
)
try:  # keep this module import-cheap if the curvature layer is partially unavailable
    from .connection_curvature import (
        ensure_connection_curvature_declarations,
        riemann_tensor_expr,
        ricci_tensor_expr,
        scalar_curvature_tensor_expr,
        einstein_tensor_expr,
    )
except Exception:  # pragma: no cover
    ensure_connection_curvature_declarations = None  # type: ignore
    riemann_tensor_expr = None  # type: ignore
    ricci_tensor_expr = None  # type: ignore
    scalar_curvature_tensor_expr = None  # type: ignore
    einstein_tensor_expr = None  # type: ignore


# ---------------------------------------------------------------------------
# Reports and light-weight symbolic terms


@dataclass(frozen=True)
class BoundaryTerm:
    expression: TensorExpr
    reason: str
    normal_index: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegrationByPartsReport:
    original: TensorExpr
    bulk: TensorExpr
    boundary_terms: tuple[BoundaryTerm, ...] = ()
    steps: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EulerLagrangeReport:
    lagrangian: TensorExpr
    field: str
    bulk_variation: TensorExpr
    euler_lagrange: TensorExpr
    boundary_terms: tuple[BoundaryTerm, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricVariationReport:
    action_density: TensorExpr
    raw_variation: TensorExpr
    after_integration_by_parts: TensorExpr
    euler_lagrange: TensorExpr
    boundary_terms: tuple[BoundaryTerm, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerturbationExpansionReport:
    expression: TensorExpr
    background_metric: str
    perturbation: str
    order: int
    expanded: TensorExpr
    terms_by_order: Mapping[int, TensorExpr] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Basic TensorExpr constructors


def zero_ir() -> TensorExpr:
    return ir_node("zero")


def add_ir(*children: TensorExpr) -> TensorExpr:
    return normalize_tensor_expr(ir_node("add", *(c for c in children if c.kind != "zero")))


def mul_ir(*children: TensorExpr, coefficient: Any | None = None) -> TensorExpr:
    md = {} if coefficient is None else {"coefficient": sp.sympify(coefficient)}
    return normalize_tensor_expr(ir_node("mul", *children, metadata=md))


def neg_ir(expr: TensorExpr) -> TensorExpr:
    return normalize_tensor_expr(ir_node("neg", expr))


def boundary_divergence_ir(expr: TensorExpr, *, index: Any = "a") -> TensorExpr:
    return ir_node("boundary_divergence", expr, index=index, total_divergence=True, provenance={"origin": "variational_tensor_expr"})


def _metric_name(registry: DeclarationRegistry, connection: str = "CD", metric: str | None = None) -> str:
    if metric is not None:
        return metric
    conn = registry.require_connection(connection)
    if conn.metric:
        return conn.metric
    if registry.metrics:
        return sorted(registry.metrics)[0]
    return "g"


def _metric_decl(registry: DeclarationRegistry, metric: str):
    return registry.require_metric(metric)


def _default_registry(registry: DeclarationRegistry | None) -> DeclarationRegistry:
    return registry if registry is not None else standard_riemannian_registry()


def metric_ir(registry: DeclarationRegistry | None = None, indices: Sequence[Any] = ("a", "b"), *, metric: str | None = None, connection: str = "CD") -> TensorExpr:
    reg = _default_registry(registry)
    name = _metric_name(reg, connection, metric)
    return reg.tensor_expr(name, tuple(indices)).with_metadata(role="metric", family="Metric")


def inverse_metric_ir(registry: DeclarationRegistry | None = None, indices: Sequence[Any] = ("a", "b"), *, metric: str | None = None, connection: str = "CD") -> TensorExpr:
    reg = _default_registry(registry)
    name = _metric_name(reg, connection, metric)
    decl = _metric_decl(reg, name)
    inv = decl.inverse_name or f"{name}_inv"
    if inv not in reg.tensors and inv not in reg.metrics:
        # Use an abstract TensorExpr node instead of mutating the registry.
        return ir_node(
            "indexed_tensor",
            payload=inv,
            tensor_name=inv,
            indices=tuple(indices),
            variance_spec="up" * len(tuple(indices)),
            role="inverse_metric",
            family="Metric",
            metric=name,
            provenance={"origin": "variational_tensor_expr"},
        )
    return reg.tensor_expr(inv, tuple(indices)).with_metadata(role="inverse_metric", family="Metric", metric=name)


def determinant_density_ir(registry: DeclarationRegistry | None = None, *, metric: str | None = None, connection: str = "CD") -> TensorExpr:
    reg = _default_registry(registry)
    name = _metric_name(reg, connection, metric)
    decl = _metric_decl(reg, name)
    return ir_node(
        "metric_density",
        payload=decl.determinant_name or f"det_{name}",
        metric=name,
        density_weight=1,
        family="MetricDensity",
        provenance={"origin": "variational_tensor_expr"},
    )


def metric_variation_ir(registry: DeclarationRegistry | None = None, indices: Sequence[Any] = ("a", "b"), *, metric: str | None = None, connection: str = "CD", inverse: bool = False) -> TensorExpr:
    base = inverse_metric_ir(registry, indices, metric=metric, connection=connection) if inverse else metric_ir(registry, indices, metric=metric, connection=connection)
    return ir_node(
        "metric_variation",
        base,
        payload="delta_g_inv" if inverse else "delta_g",
        indices=tuple(indices),
        inverse=inverse,
        field=_metric_name(_default_registry(registry), connection, metric),
        provenance={"origin": "variational_tensor_expr"},
    )


# ---------------------------------------------------------------------------
# Metric, inverse metric, determinant, connection, and curvature variations


def variation_of_metric(registry: DeclarationRegistry | None = None, indices: Sequence[Any] = ("a", "b"), *, metric: str | None = None, connection: str = "CD") -> TensorExpr:
    return metric_variation_ir(registry, indices, metric=metric, connection=connection, inverse=False)


def variation_of_inverse_metric(registry: DeclarationRegistry | None = None, indices: Sequence[Any] = ("a", "b"), *, metric: str | None = None, connection: str = "CD") -> TensorExpr:
    reg = _default_registry(registry)
    a, b = tuple(indices)
    c, d = "c", "d"
    return neg_ir(mul_ir(
        inverse_metric_ir(reg, (a, c), metric=metric, connection=connection),
        inverse_metric_ir(reg, (b, d), metric=metric, connection=connection),
        variation_of_metric(reg, (c, d), metric=metric, connection=connection),
    )).with_metadata(variation_rule="delta_inverse_metric")


def variation_of_metric_determinant(registry: DeclarationRegistry | None = None, *, metric: str | None = None, connection: str = "CD", inverse_metric_variation: bool = False) -> TensorExpr:
    reg = _default_registry(registry)
    density = determinant_density_ir(reg, metric=metric, connection=connection)
    if inverse_metric_variation:
        return mul_ir(
            density,
            metric_ir(reg, ("a", "b"), metric=metric, connection=connection),
            metric_variation_ir(reg, ("a", "b"), metric=metric, connection=connection, inverse=True),
            coefficient=sp.Rational(-1, 2),
        ).with_metadata(variation_rule="delta_sqrt_det_metric_inverse")
    return mul_ir(
        density,
        inverse_metric_ir(reg, ("a", "b"), metric=metric, connection=connection),
        variation_of_metric(reg, ("a", "b"), metric=metric, connection=connection),
        coefficient=sp.Rational(1, 2),
    ).with_metadata(variation_rule="delta_sqrt_det_metric")


def variation_of_connection(registry: DeclarationRegistry | None = None, *, connection: str = "CD", metric: str | None = None, indices: Sequence[Any] = ("a", "b", "c")) -> TensorExpr:
    """Levi-Civita metric variation of connection coefficients.

    Encodes δΓ^a_bc = 1/2 g^ad (∇_b δg_cd + ∇_c δg_bd - ∇_d δg_bc).
    For non-Levi-Civita connections the same node is marked as a formal
    affine variation rather than pretending metric compatibility holds.
    """
    reg = _default_registry(registry)
    conn = reg.require_connection(connection)
    a, b, c = tuple(indices)
    d = "d"
    if not (conn.metric and conn.is_torsion_free() and conn.is_metric_compatible()):
        return ir_node(
            "connection_variation",
            payload=f"delta_{connection}",
            connection=connection,
            indices=tuple(indices),
            formal_affine=True,
            provenance={"origin": "variational_tensor_expr"},
        )
    terms = add_ir(
        covariant_derivative_ir(variation_of_metric(reg, (c, d), metric=metric, connection=connection), index=b, connection=connection),
        covariant_derivative_ir(variation_of_metric(reg, (b, d), metric=metric, connection=connection), index=c, connection=connection),
        neg_ir(covariant_derivative_ir(variation_of_metric(reg, (b, c), metric=metric, connection=connection), index=d, connection=connection)),
    )
    return mul_ir(inverse_metric_ir(reg, (a, d), metric=metric, connection=connection), terms, coefficient=sp.Rational(1, 2)).with_metadata(
        variation_rule="delta_levi_civita_connection", connection=connection, indices=tuple(indices)
    )


def variation_of_riemann(registry: DeclarationRegistry | None = None, *, connection: str = "CD", indices: Sequence[Any] = ("a", "b", "c", "d")) -> TensorExpr:
    reg = _default_registry(registry)
    a, b, c, d = tuple(indices)
    delta_gamma_abd = variation_of_connection(reg, connection=connection, indices=(a, b, d))
    delta_gamma_abc = variation_of_connection(reg, connection=connection, indices=(a, b, c))
    return add_ir(
        covariant_derivative_ir(delta_gamma_abd, index=c, connection=connection),
        neg_ir(covariant_derivative_ir(delta_gamma_abc, index=d, connection=connection)),
    ).with_metadata(variation_rule="delta_riemann_from_delta_connection", connection=connection, indices=tuple(indices))


def variation_of_ricci(registry: DeclarationRegistry | None = None, *, connection: str = "CD", indices: Sequence[Any] = ("a", "b")) -> TensorExpr:
    reg = _default_registry(registry)
    a, b = tuple(indices)
    return ir_node(
        "contract",
        variation_of_riemann(reg, connection=connection, indices=("c", a, "c", b)),
        pattern=("c", a, "c", b),
        result_indices=tuple(indices),
        variation_rule="delta_ricci_from_delta_riemann",
        provenance={"origin": "variational_tensor_expr"},
    )


def variation_of_scalar_curvature(registry: DeclarationRegistry | None = None, *, connection: str = "CD") -> TensorExpr:
    reg = _default_registry(registry)
    ric = ricci_tensor_expr(reg, connection, ("a", "b")) if ricci_tensor_expr else ir_node("curvature", payload="Ricci", indices=("a", "b"))
    return add_ir(
        mul_ir(inverse_metric_ir(reg, ("a", "b"), connection=connection), variation_of_ricci(reg, connection=connection, indices=("a", "b"))),
        mul_ir(ric, variation_of_inverse_metric(reg, ("a", "b"), connection=connection)),
    ).with_metadata(variation_rule="delta_scalar_curvature")


def variation_of_curvature(registry: DeclarationRegistry | None = None, family: str = "Riemann", *, connection: str = "CD", indices: Sequence[Any] = ()) -> TensorExpr:
    if family == "Riemann":
        return variation_of_riemann(registry, connection=connection, indices=indices or ("a", "b", "c", "d"))
    if family == "Ricci":
        return variation_of_ricci(registry, connection=connection, indices=indices or ("a", "b"))
    if family == "ScalarCurvature":
        return variation_of_scalar_curvature(registry, connection=connection)
    return ir_node("curvature_variation", payload=f"delta_{family}", family=family, connection=connection, indices=tuple(indices), provenance={"origin": "variational_tensor_expr"})


# ---------------------------------------------------------------------------
# Integration by parts, boundary tracking, and Euler-Lagrange extraction


def _contains_field_variation(expr: TensorExpr, field: str) -> bool:
    if expr.kind in {"metric_variation", "variation"} and expr.metadata.get("field", field) == field:
        return True
    return any(_contains_field_variation(ch, field) for ch in expr.children)


def _is_derivative_of_variation(expr: TensorExpr, field: str) -> bool:
    return expr.kind == "covariant_derivative" and len(expr.children) == 1 and _contains_field_variation(expr.children[0], field)


def _ibp_once_product(expr: TensorExpr, *, field: str) -> tuple[TensorExpr, tuple[BoundaryTerm, ...], bool]:
    if expr.kind != "mul":
        return expr, (), False
    factors = list(expr.children)
    for pos, factor in enumerate(factors):
        if not _is_derivative_of_variation(factor, field):
            continue
        idx = factor.metadata.get("index", "a")
        varied = factor.children[0]
        coeff_factors = factors[:pos] + factors[pos + 1:]
        coeff = mul_ir(*coeff_factors, coefficient=expr.metadata.get("coefficient")) if coeff_factors else scalar_ir(expr.metadata.get("coefficient", 1))
        bulk = neg_ir(mul_ir(covariant_derivative_ir(coeff, index=idx, connection=factor.metadata.get("connection")), varied))
        boundary_expr = boundary_divergence_ir(mul_ir(coeff, varied), index=idx)
        return bulk, (BoundaryTerm(boundary_expr, "integration_by_parts", idx),), True
    return expr, (), False


def integrate_by_parts_tensor_expr(expr: Any, *, field: str = "g", max_passes: int = 8, discard_boundary: bool = False) -> IntegrationByPartsReport:
    current = normalize_tensor_expr(to_tensor_expr(expr))
    boundaries: list[BoundaryTerm] = []
    steps: list[str] = []
    for _ in range(max_passes):
        changed = False

        def visit(node: TensorExpr) -> TensorExpr:
            nonlocal changed
            if node.kind == "add":
                return add_ir(*(visit(ch) for ch in node.children))
            if node.kind == "mul":
                new_node, bterms, did = _ibp_once_product(node, field=field)
                if did:
                    boundaries.extend(bterms)
                    steps.append("move_covariant_derivative_off_variation")
                    changed = True
                    return new_node
            if node.kind == "covariant_derivative" and len(node.children) == 1 and _contains_field_variation(node.children[0], field):
                idx = node.metadata.get("index", "a")
                boundary = boundary_divergence_ir(node.children[0], index=idx)
                boundaries.append(BoundaryTerm(boundary, "pure_total_divergence", idx))
                steps.append("extract_total_divergence")
                changed = True
                return zero_ir() if discard_boundary else boundary
            if node.children:
                return TensorExpr(node.kind, node.payload, tuple(visit(ch) for ch in node.children), dict(node.metadata), node.provenance)
            return node

        current = normalize_tensor_expr(visit(current))
        if not changed:
            break
    bulk = eliminate_boundary_terms_tensor_expr(current) if discard_boundary else current
    return IntegrationByPartsReport(expr if isinstance(expr, TensorExpr) else to_tensor_expr(expr), bulk, tuple(boundaries), tuple(steps))


def eliminate_boundary_terms_tensor_expr(expr: Any) -> TensorExpr:
    ir = normalize_tensor_expr(to_tensor_expr(expr))
    if ir.kind == "boundary_divergence" or bool(ir.metadata.get("total_divergence", False)):
        return zero_ir()
    if ir.kind == "add":
        kept = [eliminate_boundary_terms_tensor_expr(ch) for ch in ir.children]
        return add_ir(*(ch for ch in kept if ch.kind != "zero"))
    if ir.children:
        return TensorExpr(ir.kind, ir.payload, tuple(eliminate_boundary_terms_tensor_expr(ch) for ch in ir.children), dict(ir.metadata), ir.provenance)
    return ir


def extract_euler_lagrange_tensor_expr(expr: Any, *, field: str = "g", variation_indices: Sequence[Any] = ("a", "b")) -> EulerLagrangeReport:
    lagrangian = normalize_tensor_expr(to_tensor_expr(expr))
    ibp = integrate_by_parts_tensor_expr(lagrangian, field=field, discard_boundary=True)
    delta = metric_variation_ir(indices=variation_indices, metric=field)

    def strip_delta(node: TensorExpr) -> TensorExpr | None:
        if node.kind == "mul":
            keep = [ch for ch in node.children if not (ch.kind == "metric_variation" and ch.metadata.get("field") == field)]
            if len(keep) != len(node.children):
                return mul_ir(*keep, coefficient=node.metadata.get("coefficient")) if keep else scalar_ir(node.metadata.get("coefficient", 1))
        if node.kind == "metric_variation" and node.metadata.get("field") == field:
            return scalar_ir(1)
        return None

    if ibp.bulk.kind == "add":
        pieces = [strip_delta(ch) or ch for ch in ibp.bulk.children]
        euler = add_ir(*pieces)
    else:
        euler = strip_delta(ibp.bulk) or ir_node("euler_lagrange_derivative", ibp.bulk, delta, field=field)
    return EulerLagrangeReport(lagrangian, field, ibp.bulk, euler.with_metadata(field=field, role="euler_lagrange"), ibp.boundary_terms)


# ---------------------------------------------------------------------------
# Einstein-Hilbert, f(R), curvature-squared, and perturbation examples


def einstein_hilbert_density_ir(registry: DeclarationRegistry | None = None, *, connection: str = "CD") -> TensorExpr:
    reg = _default_registry(registry)
    scalar = scalar_curvature_tensor_expr(reg, connection) if scalar_curvature_tensor_expr else ir_node("curvature_scalar", payload="R")
    return mul_ir(determinant_density_ir(reg, connection=connection), scalar).with_metadata(action="einstein_hilbert")


def einstein_hilbert_variation(registry: DeclarationRegistry | None = None, *, connection: str = "CD", include_boundary: bool = True) -> MetricVariationReport:
    reg = _default_registry(registry)
    density = determinant_density_ir(reg, connection=connection)
    einstein = einstein_tensor_expr(reg, connection, ("a", "b")) if einstein_tensor_expr else ir_node("curvature", payload="Einstein", indices=("a", "b"))
    raw = add_ir(
        mul_ir(variation_of_metric_determinant(reg, connection=connection), scalar_curvature_tensor_expr(reg, connection) if scalar_curvature_tensor_expr else ir_node("curvature_scalar", payload="R")),
        mul_ir(density, variation_of_scalar_curvature(reg, connection=connection)),
    )
    boundary = BoundaryTerm(
        boundary_divergence_ir(ir_node("symplectic_potential", payload="Theta_EH", connection=connection), index="a"),
        "einstein_hilbert_boundary_term",
        "a",
        {"name": "Gibbons-Hawking-York-compatible total divergence"},
    )
    final = mul_ir(density, einstein, variation_of_metric(reg, ("a", "b"), connection=connection), coefficient=sp.Rational(-1, 1)).with_metadata(
        variation_result="einstein_hilbert_bulk", convention="delta_g_covariant"
    )
    after = add_ir(final, boundary.expression) if include_boundary else final
    return MetricVariationReport(einstein_hilbert_density_ir(reg, connection=connection), raw, after, final, (boundary,))


def f_of_r_density_ir(function_name: str = "f", registry: DeclarationRegistry | None = None, *, connection: str = "CD") -> TensorExpr:
    reg = _default_registry(registry)
    scalar = scalar_curvature_tensor_expr(reg, connection) if scalar_curvature_tensor_expr else ir_node("curvature_scalar", payload="R")
    return mul_ir(determinant_density_ir(reg, connection=connection), ir_node("function_of_scalar_curvature", scalar, payload=function_name)).with_metadata(action="f_of_R")


def f_of_r_variation_example(function_name: str = "f", registry: DeclarationRegistry | None = None, *, connection: str = "CD") -> MetricVariationReport:
    reg = _default_registry(registry)
    density = determinant_density_ir(reg, connection=connection)
    fprime = ir_node("function_derivative", scalar_curvature_tensor_expr(reg, connection), payload=f"{function_name}'")
    ric = ricci_tensor_expr(reg, connection, ("a", "b")) if ricci_tensor_expr else ir_node("curvature", payload="Ricci", indices=("a", "b"))
    metric = metric_ir(reg, ("a", "b"), connection=connection)
    scalar = scalar_curvature_tensor_expr(reg, connection) if scalar_curvature_tensor_expr else ir_node("curvature_scalar", payload="R")
    fscalar = ir_node("function_of_scalar_curvature", scalar, payload=function_name)
    box_f = ir_node("covariant_laplacian", fprime, connection=connection)
    nabla_nabla_f = covariant_derivative_ir(covariant_derivative_ir(fprime, index="b", connection=connection), index="a", connection=connection)
    eom = add_ir(
        mul_ir(fprime, ric),
        neg_ir(mul_ir(metric, fscalar, coefficient=sp.Rational(1, 2))),
        mul_ir(metric, box_f),
        neg_ir(nabla_nabla_f),
    ).with_metadata(role="f_R_euler_lagrange")
    boundary = BoundaryTerm(boundary_divergence_ir(ir_node("symplectic_potential", payload="Theta_fR", connection=connection), index="a"), "f_R_boundary_term", "a")
    final = mul_ir(density, eom, variation_of_metric(reg, ("a", "b"), connection=connection), coefficient=-1)
    return MetricVariationReport(f_of_r_density_ir(function_name, reg, connection=connection), ir_node("variation", f_of_r_density_ir(function_name, reg, connection=connection), field="g"), add_ir(final, boundary.expression), eom, (boundary,))


def curvature_squared_density_ir(kind: str = "Riemann2", registry: DeclarationRegistry | None = None, *, connection: str = "CD") -> TensorExpr:
    reg = _default_registry(registry)
    if kind == "Ricci2":
        factor = ir_node("curvature_invariant", payload="Ricci_ab Ricci^ab", family="RicciSquared", connection=connection)
    elif kind == "Scalar2":
        factor = ir_node("curvature_invariant", payload="R^2", family="ScalarCurvatureSquared", connection=connection)
    else:
        factor = ir_node("curvature_invariant", payload="R_abcd R^abcd", family="RiemannSquared", connection=connection)
    return mul_ir(determinant_density_ir(reg, connection=connection), factor).with_metadata(action="curvature_squared", invariant=kind)


def curvature_squared_variation_example(kind: str = "Riemann2", registry: DeclarationRegistry | None = None, *, connection: str = "CD") -> MetricVariationReport:
    reg = _default_registry(registry)
    invariant = curvature_squared_density_ir(kind, reg, connection=connection)
    if kind == "Scalar2":
        eom = ir_node("higher_curvature_eom", payload="2 R R_ab - 1/2 g_ab R^2 + 2(g_ab Box - nabla_a nabla_b)R", invariant=kind, connection=connection)
    elif kind == "Ricci2":
        eom = ir_node("higher_curvature_eom", payload="RicciSquaredMetricVariation", invariant=kind, connection=connection)
    else:
        eom = ir_node("higher_curvature_eom", payload="RiemannSquaredMetricVariation", invariant=kind, connection=connection)
    boundary = BoundaryTerm(boundary_divergence_ir(ir_node("symplectic_potential", payload=f"Theta_{kind}", connection=connection), index="a"), "curvature_squared_boundary_term", "a")
    final = mul_ir(determinant_density_ir(reg, connection=connection), eom, variation_of_metric(reg, ("a", "b"), connection=connection), coefficient=-1)
    return MetricVariationReport(invariant, ir_node("variation", invariant, field="g"), add_ir(final, boundary.expression), eom, (boundary,))


def perturbative_metric_expansion(registry: DeclarationRegistry | None = None, expr: Any | None = None, *, background_metric: str = "gbar", perturbation: str = "h", parameter: str = "epsilon", order: int = 2, connection: str = "CD") -> PerturbationExpansionReport:
    reg = _default_registry(registry)
    source = to_tensor_expr(expr) if expr is not None else metric_ir(reg, ("a", "b"), connection=connection)
    eps = scalar_ir(sp.Symbol(parameter))
    h = ir_node("indexed_tensor", payload=perturbation, tensor_name=perturbation, indices=source.metadata.get("indices", ("a", "b")), perturbation=True)
    gbar = ir_node("indexed_tensor", payload=background_metric, tensor_name=background_metric, indices=source.metadata.get("indices", ("a", "b")), background=True)
    terms: dict[int, TensorExpr] = {0: gbar}
    if order >= 1:
        terms[1] = mul_ir(eps, h)
    for k in range(2, order + 1):
        terms[k] = mul_ir(scalar_ir(sp.Symbol(parameter) ** k), ir_node("perturbative_coefficient", payload=f"{perturbation}{k}", order=k))
    expanded = add_ir(*(terms[k] for k in sorted(terms))).with_metadata(expansion="background_metric", order=order, background_metric=background_metric, perturbation=perturbation)
    return PerturbationExpansionReport(source, background_metric, perturbation, order, expanded, terms)


__all__ = [
    "BoundaryTerm", "IntegrationByPartsReport", "EulerLagrangeReport", "MetricVariationReport", "PerturbationExpansionReport",
    "zero_ir", "add_ir", "mul_ir", "neg_ir", "boundary_divergence_ir",
    "metric_ir", "inverse_metric_ir", "determinant_density_ir", "metric_variation_ir",
    "variation_of_metric", "variation_of_inverse_metric", "variation_of_metric_determinant",
    "variation_of_connection", "variation_of_riemann", "variation_of_ricci", "variation_of_scalar_curvature", "variation_of_curvature",
    "integrate_by_parts_tensor_expr", "eliminate_boundary_terms_tensor_expr", "extract_euler_lagrange_tensor_expr",
    "einstein_hilbert_density_ir", "einstein_hilbert_variation", "f_of_r_density_ir", "f_of_r_variation_example",
    "curvature_squared_density_ir", "curvature_squared_variation_example", "perturbative_metric_expansion",
]
